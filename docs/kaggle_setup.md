# Phase 3 — Kaggle inference server runbook

Phase 3's rich path (Stable Video Diffusion ambient motion + Wav2Lip
lip-sync) runs on a Kaggle GPU notebook exposed through one ngrok
tunnel. The local pipeline calls those endpoints; if anything is down
or unreachable, it falls back to a plain FFmpeg passthrough so the
project keeps producing a valid (less rich) MP4.

This is **opt-in** — every test runs offline, every demo works without
the server, and the rich path is the layer of polish you turn on for
the final demo recording.

## One-time setup

1. **ngrok account + auth token.** Sign up at
   [ngrok.com](https://dashboard.ngrok.com/signup) (free tier is fine),
   open *Your Authtoken* in the dashboard, and copy the value.
2. **Kaggle account** with phone verification (required for GPU + Internet).

## Per-session workflow

### 1. Spin up the notebook

- Go to [kaggle.com/code/new](https://kaggle.com/code/new). Name it
  `agentic-phase3-server`.
- **Settings → Accelerator** → `GPU T4 x2`.
- **Settings → Internet** → `On`.
- **Settings → Add-ons → Secrets** → add **two** secrets:
  - `NGROK_AUTH_TOKEN` — from ngrok dashboard.
  - `HF_TOKEN` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    (a free Read-scope token is enough). This raises the rate limit
    when SDXL-Turbo (~7 GB) and SVD-XT (~9 GB) get downloaded into the
    Kaggle session, and avoids the "unauthenticated request" warning.

### 2. Paste in the server code

Open [`docs/kaggle_phase3_server.py`](kaggle_phase3_server.py) and copy
the contents into a single Kaggle code cell, **or** split at the
`# %%` markers into one cell per section. Either works.

### 3. Run all

The first run takes ~2 minutes (apt + pip + cloning Wav2Lip +
downloading two checkpoints). Subsequent runs in the same session
are fast.

When the last cell prints:

```
======================================================================
  Public URL:  https://<random>.ngrok-free.app
  Set this in your local .env:  KAGGLE_ENDPOINT=https://<random>.ngrok-free.app
======================================================================
```

…you're ready.

### 4. Wire the URL into your local `.env`

```dotenv
# .env (local)
KAGGLE_ENDPOINT=https://<random>.ngrok-free.app
```

### 5. (Optional) sanity-check from your local machine

ngrok free-tier shows an HTML interstitial to "browser-like" clients on
first hit. To bypass it from a script or `curl`, send the
`ngrok-skip-browser-warning` header — the local Phase 3 client already
does this automatically, but for a manual probe:

```powershell
curl -H "ngrok-skip-browser-warning: true" https://<your-url>.ngrok-free.dev/health
# expected: {"models_loaded":["wav2lip"],"gpu":"Tesla T4","vram_used_mb":...}
```

`models_loaded` lists `wav2lip` immediately. `sdxl` is added after the
first `/sdxl` request and `svd` after the first `/svd` request — both
are lazy-loaded so an unused model never costs you VRAM. All three use
diffusers' `enable_model_cpu_offload()` so weights swap between CPU
RAM and GPU on demand; only the actively-running pipeline sits in VRAM.

### 6. Run the local pipeline

```powershell
python main.py --prompt "A young astronaut discovers a hidden ocean on Mars"
```

In the progress log you should see:

```
[Phase3] remote endpoint reachable — wav2lip
```

…instead of the offline fallback line.

## When the session times out (~9 hours)

Kaggle stops the kernel after ~9 hours of compute. Re-run the notebook
(steps 3 and 4 — the URL changes), update `.env`, and you're back in
business.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `models_loaded: []` after a few minutes | The server is still loading — `wav2lip` should appear within ~30s, `svd` lazy-loads on first /svd call. Re-check `/health` later. |
| `502 Bad Gateway` from ngrok | Kaggle session probably ended. Restart the notebook + update the URL. |
| Local log says `KAGGLE_ENDPOINT set but unreachable` | Either the Kaggle kernel stopped, or `Internet` is off in the notebook settings. |
| `403 Forbidden` from `/svd` or `/lipsync` | First request after a long idle — Kaggle may have paused the kernel. Run any cell briefly to wake it, then retry. |
| ngrok complains about session limit on free tier | Free ngrok allows one concurrent tunnel per account. Kill any existing tunnel from the [ngrok dashboard](https://dashboard.ngrok.com/cloud-edge/endpoints) before re-running. |
| Wav2Lip fails with `Face not detected` | The speaker portrait isn't centred or face is too small. Re-render with a tighter crop, or fall back to passthrough by clearing `KAGGLE_ENDPOINT` for that run. |

## Cost / quota

| Item | Free tier |
|---|---|
| Kaggle GPU hours | 30 hrs/week per account |
| Kaggle session length | up to 9 hrs continuous |
| ngrok concurrent tunnels | 1 per account |
| ngrok request rate | ~120 req/min |
| ngrok domain | random `*.ngrok-free.app` URL (changes per session) |

A 3-scene demo render hits the endpoints ~10 times — well under all
quotas. Recording the demo video is the only time you actually need
the rich path running.
