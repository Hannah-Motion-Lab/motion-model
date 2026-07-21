# hannah-motion-lab

Research repo for **Hannah's next motion model**: a unified, text-conditioned, generative motion model that replaces EMAGE (audio→gesture) with **text→motion directly** — and accepts action prompts too. One model, two conditioning modes, both generative (never clip retrieval).

> Part of the Hannah-Motion workspace but fully standalone: nothing here touches
> `hannah-backend` / `hannah-frontend` until the model is good enough to swap in.

## Why

- EMAGE's output is subtle TED-talk gesturing (single-speaker checkpoint, audio-only conditioning, 2024 architecture). Hannah's vision is anime/VTuber expressiveness.
- Today's pipeline is `text → TTS → audio → motion`; here it becomes `text → motion` (parallel with TTS, lower latency, and she can move without speaking — e.g. reacting to YOLO vision events phrased as text).

## Architecture

Two-stage latent generative model (the current BEAT2-SOTA recipe — flow-matching lineage: GestureLSM, SemConFlow, LiveGesture — with text conditioning):

1. **Stage A — part-wise motion VAE** (`src/motionlab/models/vae.py`)
   - Body-part decomposition from EMAGE (upper / hands / lower+trans), 6D rotations internally, **continuous** latents (no VQ).
   - Trains on ALL motion data regardless of pairing: BEAT2 + AMASS + Motion-X++ → the movement vocabulary grows far beyond "person giving a talk".
2. **Stage B — latent flow-matching DiT** (`src/motionlab/models/flow.py`)
   - Dual text slots, each independently droppable (classifier-free):
     - **transcript slot** — word-level frozen-T5 embeddings with word↔frame timing (BEAT2 TextGrids at training; words spread over TTS duration at inference).
     - **action slot** — caption embedding ("waves hand excitedly"), trained on captioned mocap (HumanML3D / Motion-X++).
   - Plus: emotion (BEAT2's 8 labels), speaker/style id, **intensity scalar** and per-slot guidance weights (the "otaku dial"), and a **4-frame motion prefix** for seamless streaming across sentences and into/out of actions.
   - Few-step sampling → real-time on an RTX 5070 Ti.
3. Output contract: SMPL-X `poses (T×165 axis-angle, 30 fps)` + `trans (T×3)` — exactly what Hannah's avatar already plays; `serve/main.py` mirrors the EMAGE sidecar schema (dev port 8005).

## Data

| Dataset | Role | Status |
|---|---|---|
| BEAT2-English (all ~25 speakers × 8 emotions, transcripts) | Stage B co-text pairs + Stage A | `scripts/download_beat2.py` (public HF; audio skipped on purpose) |
| AMASS | Stage A motion prior | **needs user registration** at amass.is.tue.mpg.de → drop archives in `data/amass/`, run `scripts/prepare_amass.py` |
| HumanML3D captions | Stage B action pairs | captions public; motions derive from AMASS (same registration) |
| Motion-X++ | Stage A + action pairs (SMPL-X native) | **needs access request** |
| Mined video (yt-dlp → GVHMR → whisper) | style fine-tune (phase 2) | `scripts/mine_video.py` skeleton |

Expressiveness without anime mocap: emotion-weighted sampling, exaggeration augmentation (`data/augment.py`), CFG/intensity at inference; LoRA fine-tune on mined data later.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_beat2.py                  # ~motion+textgrid, resumable
.venv/bin/python -m pytest tests -q
.venv/bin/python -m motionlab.train.train_vae --config configs/vae.yaml
.venv/bin/python -m motionlab.train.train_flow --config configs/flow.yaml   # after VAE
.venv/bin/python demo/generate.py --text "hola, ¿cómo estás?" --duration 3  # → mp4
```

## Status / roadmap

- [x] Scaffold, venv (py3.12 + torch cu128 — mandatory for Blackwell GPUs)
- [x] BEAT2 data pipeline + tests (17 passing; 1 skip until textgrids finish downloading)
- [x] Part-wise VAE + flow DiT implemented, smoke-trained (loss falls, checkpoints, sampler works)
- [x] End-to-end demo pipeline runs: `demo/generate.py --text ... --action ...` → mp4
- [x] Serve module (`serve/main.py`, :8005) schema-compatible with the EMAGE sidecar
- [x] Stage A VAE trained (100k steps, 0 NaN, 0.024 rad test recon)
- [x] Stage B flow trained on BEAT2 co-text (400k steps, loss 1.11→0.57; emotion conditioning verified)
- [ ] HumanML3D/AMASS action pairs (**user: register at amass.is.tue.mpg.de**, drop archives in `data/amass/raw/`)
- [ ] Beat EMAGE on FGD + eyeball test → swap into Hannah (user's call; `MOTION_SIDECAR_URL` flip)
- [ ] Phase 2: video-mining fine-tune ("otaku" style; `scripts/mine_video.py` + GVHMR), Motion-X++, streaming serve

## Setup / assets not included

Excluded from the repo (obtain locally):
- `assets/smplx/SMPLX_NEUTRAL_2020.npz` — the SMPL-X model (~160MB, **non-commercial
  license**). Get it from the official SMPL-X site.
- `data/` corpora (BEAT2, AMASS, GRAB, HumanML3D) — see the training scripts.
- `runs/` trained checkpoints, and the `uv` venv (Python 3.12, torch cu128).
