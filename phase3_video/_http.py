"""
Shared HTTP helper for the Phase 3 remote model clients.

Both `svd.py` and `lipsync.py` POST b64-encoded media to a Kaggle FastAPI
server tunnelled through ngrok. The server URL lives in `KAGGLE_ENDPOINT`.
If the env var is unset, every call returns `None` and the caller falls
through to its local passthrough path. This means tests stay fully offline
and the local pipeline keeps working when the Kaggle session is down.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional


def _endpoint() -> Optional[str]:
    base = os.getenv("KAGGLE_ENDPOINT", "").strip().rstrip("/")
    return base or None


def have_endpoint() -> bool:
    return _endpoint() is not None


def file_to_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def b64_to_file(b64: str, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))
    return str(out)


# ngrok's free tier serves an HTML interstitial ("you are about to visit...")
# to browser clients on first hit from a given IP. Sending this magic header
# tells ngrok the client is a script and to skip the warning page.
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


def post_endpoint(path: str, payload: dict, *, timeout: float = 180.0) -> Optional[dict[str, Any]]:
    """POST JSON to {KAGGLE_ENDPOINT}/{path}. Returns parsed dict or None.

    Failures are printed to stderr (with the server's `detail` field when
    present) so silent fall-through to passthrough is visible in the log.

    Retries once on `RemoteProtocolError` — ngrok free tier occasionally cuts
    multi-megabyte response bodies mid-transfer. The response is fully
    rendered on the server side, just dropped on the wire.
    """
    base = _endpoint()
    if not base:
        return None
    try:
        import httpx  # local: required dep; remote-side: irrelevant
    except ImportError:
        return None
    url = f"{base}/{path.lstrip('/')}"

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout, headers=_NGROK_HEADERS) as client:
                r = client.post(url, json=payload)
                if r.status_code >= 400:
                    # Try to surface FastAPI's `detail` field — useful when Wav2Lip
                    # raises on face-detection or audio decoding errors.
                    detail: Any = r.text
                    try:
                        detail = r.json().get("detail", detail)
                    except Exception:
                        pass
                    detail_str = str(detail)
                    # Long tracebacks are useful — print up to 3000 chars.
                    if len(detail_str) > 3000:
                        detail_str = detail_str[:3000] + "...[truncated]"
                    print(f"[Phase3] {path} → HTTP {r.status_code}: {detail_str}",
                          file=__import__("sys").stderr, flush=True)
                    return None
                return r.json()
        except httpx.RemoteProtocolError as exc:
            last_exc = exc
            if attempt == 0:
                print(f"[Phase3] {path} → mid-stream disconnect, retrying once: {exc!r}",
                      file=__import__("sys").stderr, flush=True)
                continue
            print(f"[Phase3] {path} → request failed after retry: {exc!r}",
                  file=__import__("sys").stderr, flush=True)
            return None
        except Exception as exc:
            print(f"[Phase3] {path} → request failed: {exc!r}",
                  file=__import__("sys").stderr, flush=True)
            return None
    return None


def endpoint_health(timeout: float = 5.0) -> Optional[dict[str, Any]]:
    """GET {KAGGLE_ENDPOINT}/health. Returns parsed dict or None."""
    base = _endpoint()
    if not base:
        return None
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=timeout, headers=_NGROK_HEADERS) as client:
            r = client.get(f"{base}/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


def endpoint_unload(timeout: float = 30.0) -> Optional[dict[str, Any]]:
    """
    POST {KAGGLE_ENDPOINT}/unload — ask the server to free GPU memory.

    Returns the parsed dict (with `freed_mb` / `vram_used_mb`) or None when
    the endpoint is missing / unreachable / older than this client. Safe to
    call before every pipeline run; a missing endpoint just no-ops.
    """
    base = _endpoint()
    if not base:
        return None
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=timeout, headers=_NGROK_HEADERS) as client:
            r = client.post(f"{base}/unload")
            if r.status_code == 404:
                return None  # server doesn't have /unload yet — fine, ignore
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


__all__ = [
    "have_endpoint",
    "file_to_b64",
    "b64_to_file",
    "post_endpoint",
    "endpoint_health",
]
