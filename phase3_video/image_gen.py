"""
Per-scene image generation.

Three-tier backend chain, picked by `IMAGE_BACKEND` env var (default `auto`):

1. **sdxl**   — local Stable Diffusion XL Turbo via `diffusers`. Lazy-loaded
                on first call; subsequent calls reuse the cached pipeline.
                Falls through if `diffusers` / `torch` aren't installed or the
                model can't load.
2. **openai** — DALL-E 3 / gpt-image-1 (existing path). Skipped if no API key.
3. **placeholder** — PIL stylised gradient + silhouettes + caption (existing
                path). Always works.

Two public renderers:
    `generate_scene_image(scene, story, out_path)`   — establishing shot
    `generate_speaker_image(scene, character, story, out_path)` — close-up
                                                         portrait for Wav2Lip
Plus the unchanged `apply_filter(...)` used by Phase 5.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from phase3_video import _http
from schemas.pipeline import Character, Scene, Story


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


def _render_openai(prompt: str, out_path: str) -> Optional[str]:
    """Try DALL-E 3 / gpt-image-1 via OpenAI Images API. Returns None on failure."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        import requests
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        size = os.getenv("OPENAI_IMAGE_SIZE", "1792x1024")
        if model == "dall-e-3":
            response = client.images.generate(
                model=model, prompt=prompt, size=size, quality="standard", n=1
            )
            url = response.data[0].url
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            # Normalise to our 1280x720 working size.
            try:
                Image.open(out_path).convert("RGB").resize((WIDTH, HEIGHT)).save(out_path, "PNG")
            except Exception:
                pass
            return out_path
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Local SDXL-Turbo backend (lazy-loaded — never imports torch/diffusers at    #
# module import time, so the placeholder path stays zero-dependency).         #
# --------------------------------------------------------------------------- #


_SDXL_PIPELINE = None    # populated on first successful load
_SDXL_FAILED = False     # latches True after a load failure to avoid retries


def _sdxl_pipeline():
    """Return a cached SDXL-Turbo pipeline, or None if it can't be loaded."""
    global _SDXL_PIPELINE, _SDXL_FAILED
    if _SDXL_PIPELINE is not None:
        return _SDXL_PIPELINE
    if _SDXL_FAILED:
        return None
    try:
        import torch  # type: ignore
        from diffusers import AutoPipelineForText2Image  # type: ignore

        model_id = os.getenv("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")
        # CPU build — float32 is required (float16 silently misbehaves on CPU).
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float32, variant=None
        )
        pipe.to("cpu")
        # Disable safety checker to save RAM on CPU; outputs are story stills.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        _SDXL_PIPELINE = pipe
        return pipe
    except Exception:
        _SDXL_FAILED = True
        return None


def _render_sdxl(prompt: str, out_path: str, *, seed: int = 0) -> Optional[str]:
    """Generate one image via SDXL-Turbo at 1 inference step (LOCAL CPU)."""
    pipe = _sdxl_pipeline()
    if pipe is None:
        return None
    try:
        import torch  # type: ignore
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            num_inference_steps=int(os.getenv("SDXL_STEPS", "1")),
            guidance_scale=0.0,
            generator=gen,
        )
        img = result.images[0].convert("RGB").resize((WIDTH, HEIGHT))
        img.save(out_path, "PNG")
        return out_path
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Remote SDXL backend — calls the Kaggle FastAPI server's /sdxl endpoint.     #
# Preferred over local SDXL whenever KAGGLE_ENDPOINT is set, because the      #
# model is 13 GB and inference on CPU is glacial.                             #
# --------------------------------------------------------------------------- #


def _render_remote_sdxl(prompt: str, out_path: str, *, seed: int = 0) -> Optional[str]:
    """Generate via the remote SDXL endpoint. Returns None if anything fails."""
    if not _http.have_endpoint():
        return None
    payload = {
        "prompt": prompt,
        "seed": int(seed),
        "width": WIDTH,
        "height": HEIGHT,
    }
    resp = _http.post_endpoint("sdxl", payload, timeout=120.0)
    if not resp or "image_b64" not in resp:
        return None
    try:
        _http.b64_to_file(resp["image_b64"], out_path)
        return out_path
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public renderers — both walk the same three-tier backend chain.             #
# --------------------------------------------------------------------------- #


def _backend_chain(prompt: str, out_path: str, *, seed: int) -> Optional[str]:
    """
    Try the image backends in priority order. Returns the path on success or
    None if every tier failed (caller falls back to PIL placeholder).

    IMAGE_BACKEND values:
        auto         (default) remote SDXL (if KAGGLE_ENDPOINT set) → local SDXL
                     (if no endpoint) → DALL-E (if OPENAI_API_KEY) → placeholder
        remote-sdxl  remote only — fails if endpoint is unset/unreachable
        local-sdxl   local diffusers only — slow on CPU but works offline
        sdxl         alias of `auto` for back-compat
        openai       DALL-E 3 only
        placeholder  PIL stylised gradient (always works)
    """
    backend = os.getenv("IMAGE_BACKEND", "auto").lower()

    # Tier 1 — remote SDXL via Kaggle (T4 GPU). Preferred when available.
    if backend in ("auto", "sdxl", "remote-sdxl") and _http.have_endpoint():
        result = _render_remote_sdxl(prompt, out_path, seed=seed)
        if result:
            return result
        if backend == "remote-sdxl":
            return None  # explicit remote request — don't silently fall back

    # Tier 2 — local SDXL via diffusers (CPU, 13 GB model). Only attempted
    # in `auto` mode when no endpoint is configured, to avoid a multi-hour
    # download on machines that don't actually want it.
    if backend == "local-sdxl" or (
        backend in ("auto", "sdxl") and not _http.have_endpoint()
    ):
        result = _render_sdxl(prompt, out_path, seed=seed)
        if result:
            return result
        if backend == "local-sdxl":
            return None

    # Tier 3 — DALL-E 3 (paid API).
    if backend in ("auto", "openai"):
        result = _render_openai(prompt, out_path)
        if result:
            return result
        if backend == "openai":
            return None

    return None


def generate_scene_image(scene: Scene, story: Story, out_path: str) -> str:
    """
    Generate an establishing-shot image for a scene.

    We deliberately don't blindly trust `scene.visual_prompt` from Phase 1 —
    the LLM (and the stub) often write generic prompts ("cinematic still,
    <user-prompt>, beat N of 3") that produce SDXL hallucinations unrelated
    to what's actually happening in the scene. Instead we always compose a
    fresh prompt from concrete scene metadata (location, action, characters,
    mood, time-of-day) and append the Phase 1 visual_prompt as supplementary
    detail when present.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Style prefix — same lookup as generate_speaker_image, but for wide shots.
    style_key = (story.style or "cinematic").lower()
    STYLE_PREFIX = {
        "cinematic":   "cinematic film still, wide establishing shot",
        "noir":        "high-contrast black-and-white noir film still, wide shot",
        "anime":       "anime-style wide establishing shot",
        "cartoon":     "cartoon animation cel, wide establishing shot",
        "realistic":   "photorealistic wide establishing shot",
        "documentary": "documentary photograph, wide establishing shot",
    }
    style_prefix = STYLE_PREFIX.get(style_key, f"{style_key} wide establishing shot")

    # Concrete scene anchors — these dominate the prompt so the image
    # actually reflects this scene's content rather than a generic mood piece.
    location  = (scene.location or "a dramatic setting")[:80]
    action    = (scene.action or "")[:120]
    chars     = ", ".join((scene.characters or [])[:4])[:80]
    time_of_day = (scene.time_of_day or "day").lower()
    mood      = scene.mood or "neutral"

    parts = [style_prefix, location]
    if chars:
        parts.append(f"with {chars}")
    if action:
        parts.append(action)
    parts.extend([
        f"{time_of_day} lighting",
        f"{mood} mood",
        "ultra detailed, 16:9, dramatic composition",
    ])
    # Append the LLM's visual_prompt only as supplementary atmosphere — it
    # can sharpen the look but doesn't get to override the scene anchors.
    if scene.visual_prompt and len(scene.visual_prompt) > 30:
        # Strip the LLM's own style prefix to avoid duplication.
        extra = scene.visual_prompt[:120]
        parts.append(extra)

    prompt = ", ".join(parts)
    seed = int(_seeded_rng(scene, story.title).randrange(2**31))

    rendered = _backend_chain(prompt, out_path, seed=seed)
    if rendered:
        return rendered
    return _render_placeholder(scene, story, out_path)


def generate_speaker_image(
    scene: Scene, character_name: str, story: Story, out_path: str
) -> str:
    """
    Close-up portrait of one named speaker — composed for Wav2Lip's face
    detector (centred face, looking at camera, sharp focus).

    Falls back to the same scene placeholder if neither SDXL nor DALL-E
    is reachable. Wav2Lip then can't find a face on the placeholder
    silhouette and the lip-sync layer drops to passthrough — still a
    valid output, just no mouth motion.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    char: Optional[Character] = next(
        (c for c in story.characters if c.name == character_name), None
    )
    appearance = ((char.appearance if char else "") or "calm expression").strip()
    # CLIP truncates at 77 tokens. Trim appearance + drop redundant adjectives
    # so the face-detectability suffix survives intact.
    appearance = appearance[:80]

    # Wav2Lip uses S3FD which is trained on real, unobscured human faces.
    # Strip / replace common appearance terms that produce face-detector-hostile
    # outputs (helmets, masks, hoods, veils). For an astronaut, we still want
    # them recognisable as an astronaut — just visor up, face exposed.
    OBSCURING = {
        "helmet": "helmet visor open, face fully visible",
        "mask":   "mask lowered, face fully visible",
        "hood":   "hood pulled back, face fully visible",
        "veil":   "veil lifted, face fully visible",
        "balaclava": "face fully visible",
    }
    lower = appearance.lower()
    for term, replacement in OBSCURING.items():
        if term in lower:
            appearance = appearance + f", {replacement}"
            break

    location = (scene.location or "evocative setting")[:30]

    # Map the user-chosen story.style to a SDXL-recognised style phrase.
    # The leading word still anchors S3FD detection (it needs to see a "face"-
    # shaped output) but we can pivot from photo to drawn-style for non-realistic
    # styles. WARNING: heavy stylization (anime, cartoon, noir-illustration)
    # makes S3FD's job harder — Wav2Lip then falls through to passthrough on
    # those lines (still + audio, no mouth motion).
    style_key = (story.style or "cinematic").lower()
    STYLE_PROMPTS = {
        "cinematic":  "cinematic close-up portrait of a human face",
        "noir":       "high-contrast black-and-white noir close-up of a human face",
        "anime":      "anime-style close-up portrait of a character face, expressive eyes",
        "cartoon":    "cartoon close-up portrait of a stylised human-like face, animation cel",
        "realistic":  "photorealistic headshot of a human face",
        "documentary":"documentary photograph headshot of a human face",
    }
    style_prefix = STYLE_PROMPTS.get(style_key, f"{style_key} close-up portrait of a human face")

    prompt = (
        f"{style_prefix}, {character_name}, "
        f"{appearance}, "
        f"front-facing, looking directly at camera, eye contact, "
        f"clear unobscured face, sharp focus on eyes nose mouth, "
        f"{location}, {scene.mood} mood"
    )
    seed = _seed_for_character(character_name)

    rendered = _backend_chain(prompt, out_path, seed=seed)
    if rendered:
        return rendered
    return _render_placeholder(scene, story, out_path)


def _seed_for_character(name: str) -> int:
    """Deterministic seed per character so the same person looks consistent across scenes."""
    h = hashlib.sha1((name or "unknown").encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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


__all__ = [
    "generate_scene_image",
    "generate_speaker_image",
    "apply_filter",
    "WIDTH",
    "HEIGHT",
]
