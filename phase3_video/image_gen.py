"""
Per-scene image generation.

Two paths:
1. If OPENAI_API_KEY is set + IMAGE_BACKEND=openai, call DALL-E 3 / gpt-image-1.
2. Otherwise (default), render a stylised PIL placeholder driven by mood +
   visual_prompt + character names. Looks deliberately like a film still —
   gradient sky, vignette, scene heading typeset on top — so the demo video
   looks intentional even without paid APIs.

Public: `generate_scene_image(scene, story, out_path) -> path`
        `apply_filter(path, filter_name, out_path) -> path`  (used by Phase 5)
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from schemas.pipeline import Scene, Story


WIDTH, HEIGHT = 1280, 720


_MOOD_PALETTE = {
    "tense": [(35, 12, 18), (98, 22, 32), (200, 70, 60)],
    "hopeful": [(255, 198, 132), (255, 142, 89), (95, 60, 130)],
    "melancholy": [(34, 50, 86), (78, 98, 138), (162, 180, 210)],
    "reflective": [(48, 68, 92), (110, 142, 168), (220, 220, 220)],
    "joyful": [(255, 215, 110), (255, 130, 110), (130, 200, 220)],
    "noir": [(15, 15, 18), (44, 44, 52), (220, 220, 220)],
    "neutral": [(40, 50, 70), (90, 110, 140), (210, 210, 210)],
}


def _palette_for(mood: str) -> list[tuple[int, int, int]]:
    return _MOOD_PALETTE.get((mood or "neutral").lower(), _MOOD_PALETTE["neutral"])


def _seeded_rng(scene: Scene, story_title: str) -> random.Random:
    h = hashlib.sha1(f"{story_title}|{scene.scene_number}|{scene.heading}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        os.environ.get("PHASE3_FONT"),
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _gradient(palette: list[tuple[int, int, int]]) -> Image.Image:
    """Vertical 3-stop gradient using the mood palette."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    px = img.load()
    top, mid, bot = palette
    for y in range(HEIGHT):
        if y < HEIGHT // 2:
            t = y / (HEIGHT // 2)
            r = int(top[0] + (mid[0] - top[0]) * t)
            g = int(top[1] + (mid[1] - top[1]) * t)
            b = int(top[2] + (mid[2] - top[2]) * t)
        else:
            t = (y - HEIGHT // 2) / (HEIGHT // 2)
            r = int(mid[0] + (bot[0] - mid[0]) * t)
            g = int(mid[1] + (bot[1] - mid[1]) * t)
            b = int(mid[2] + (bot[2] - mid[2]) * t)
        for x in range(WIDTH):
            px[x, y] = (r, g, b)
    return img


def _add_silhouettes(img: Image.Image, count: int, rng: random.Random) -> None:
    """Draw simple character silhouettes in the foreground."""
    draw = ImageDraw.Draw(img, "RGBA")
    base_y = int(HEIGHT * 0.78)
    for i in range(count):
        cx = int(WIDTH * (0.3 + 0.4 * (i / max(1, count - 1)) if count > 1 else 0.5))
        h = int(rng.uniform(0.18, 0.28) * HEIGHT)
        w = int(h * 0.5)
        # body
        draw.ellipse(
            (cx - w // 2, base_y - h, cx + w // 2, base_y),
            fill=(0, 0, 0, 220),
        )
        # head
        head_r = int(w * 0.45)
        draw.ellipse(
            (cx - head_r, base_y - h - head_r, cx + head_r, base_y - h + head_r),
            fill=(0, 0, 0, 220),
        )


def _add_vignette(img: Image.Image, strength: float = 0.65) -> None:
    """Darken the corners — gives the still a cinematic feel."""
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for i in range(60):
        alpha = int(255 * strength * (i / 60))
        draw.rectangle((i, i, img.size[0] - i, img.size[1] - i), outline=alpha)
    blurred = mask.filter(ImageFilter.GaussianBlur(40))
    black = Image.new("RGB", img.size, (0, 0, 0))
    img.paste(black, (0, 0), blurred)


def _typeset_caption(img: Image.Image, scene: Scene) -> None:
    """Burn the scene heading onto the image like a movie title card."""
    draw = ImageDraw.Draw(img, "RGBA")
    title = scene.heading.upper()
    sub = (scene.action or scene.location or "")[:90]

    title_font = _pick_font(38)
    sub_font = _pick_font(22)

    # Translucent bottom bar.
    bar_h = 130
    draw.rectangle((0, HEIGHT - bar_h, WIDTH, HEIGHT), fill=(0, 0, 0, 140))
    draw.text((40, HEIGHT - bar_h + 18), title, fill=(255, 240, 220), font=title_font)
    if sub:
        draw.text((40, HEIGHT - bar_h + 70), sub, fill=(220, 220, 220), font=sub_font)


def _render_placeholder(scene: Scene, story: Story, out_path: str) -> str:
    palette = _palette_for(scene.mood)
    img = _gradient(palette)
    rng = _seeded_rng(scene, story.title)
    _add_silhouettes(img, max(1, len(scene.characters or [])), rng)
    _add_vignette(img)
    _typeset_caption(img, scene)
    img.save(out_path, "PNG")
    return out_path


def _render_openai(scene: Scene, story: Story, out_path: str) -> Optional[str]:
    """Try DALL-E 3 / gpt-image-1 via OpenAI Images API. Returns None on failure."""
    try:
        import requests
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        size = os.getenv("OPENAI_IMAGE_SIZE", "1792x1024")
        prompt = scene.visual_prompt or f"{story.style} still: {scene.action}"
        if model == "dall-e-3":
            response = client.images.generate(
                model=model, prompt=prompt, size=size, quality="standard", n=1
            )
            url = response.data[0].url
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
    except Exception:
        return None
    return None


def generate_scene_image(scene: Scene, story: Story, out_path: str) -> str:
    """Generate an image for a scene and write it to `out_path`."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    backend = os.getenv("IMAGE_BACKEND", "placeholder").lower()
    if backend == "openai":
        result = _render_openai(scene, story, out_path)
        if result:
            return result
    return _render_placeholder(scene, story, out_path)


# --------------------------------------------------------------------------- #
# Filters used by Phase 5 ("apply_filter" intent — darken / brighten / sepia / noir)
# --------------------------------------------------------------------------- #


def apply_filter(in_path: str, filter_name: str, out_path: str) -> str:
    """Apply a colour filter to an existing image and save to out_path."""
    img = Image.open(in_path).convert("RGB")
    name = (filter_name or "").lower()

    if name == "darken":
        img = ImageEnhance.Brightness(img).enhance(0.55)
    elif name == "brighten":
        img = ImageEnhance.Brightness(img).enhance(1.4)
    elif name == "sepia":
        gray = img.convert("L")
        sepia = Image.merge(
            "RGB",
            (
                gray.point(lambda p: min(255, int(p * 1.07))),
                gray.point(lambda p: min(255, int(p * 0.74))),
                gray.point(lambda p: min(255, int(p * 0.43))),
            ),
        )
        img = sepia
    elif name == "noir":
        gray = img.convert("L")
        img = ImageEnhance.Contrast(gray.convert("RGB")).enhance(1.6)
    elif name == "blur":
        img = img.filter(ImageFilter.GaussianBlur(4))
    elif name == "saturate":
        img = ImageEnhance.Color(img).enhance(1.6)
    else:
        # Unknown filter — copy through.
        pass

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


__all__ = ["generate_scene_image", "apply_filter", "WIDTH", "HEIGHT"]
