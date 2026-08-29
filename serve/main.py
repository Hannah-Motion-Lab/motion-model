# serve/main.py — motion-lab sidecar (dev port 8005; NOT wired to Hannah yet).
#
# Response schema is IDENTICAL to the EMAGE sidecar so the future swap is a
# one-line env change on the backend:
#   {fps, num_frames, poses_b64, trans_b64, inference_ms}
#
# Request extends the contract additively: text and/or action, emotion,
# intensity, session_id (keeps a pose-prefix per session for continuity).
import asyncio
import base64
import logging
import os
import time
from pathlib import Path

# Apple Silicon: let unsupported MPS ops fall back to the CPU instead of failing. Must be set
# before torch initialises its MPS backend (i.e. before anything imports torch).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from motionlab.pipeline import MotionPipeline
from motionlab.data.beat2 import EMOTIONS
from motionlab.postprocess import plant_legs, smooth_poses, get_mean_pose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hannah Motion-Lab Sidecar", version="0.1.0")

# Live model = v1 (clean BEAT2 co-speech: forward-facing, upright, natural).
# v4/v4b REGRESSED: exaggeration aug + action-data contamination made even speech
# turn sideways / lean / over-gesture. Named ACTIONS come from a clean clip library
# (director), not the generative action slot. v4b still reachable via env for study.
VAE_CKPT = os.environ.get("VAE_CKPT", "runs/vae/latest.pt")
FLOW_CKPT = os.environ.get("FLOW_CKPT", "runs/flow/latest.pt")



def pick_device() -> str:
    """MOTION_DEVICE=auto|cuda|mps|cpu. `auto` = CUDA if there is one, else Apple's MPS, else CPU.
    The model is small (56M + 46M params, 8 sampling steps): on a CPU a sentence takes from half
    a second to a few seconds instead of ~0.2 s — slower, but the gestures are the point."""
    want = os.environ.get("MOTION_DEVICE", "auto").lower()
    if want in ("cuda", "mps", "cpu"):
        if want == "cuda" and not torch.cuda.is_available():
            logger.warning("MOTION_DEVICE=cuda but CUDA is not available; using CPU")
            return "cpu"
        if want == "mps" and not torch.backends.mps.is_available():
            logger.warning("MOTION_DEVICE=mps but MPS is not available; using CPU")
            return "cpu"
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
if DEVICE == "cpu":
    # half the cores: Whisper and Kokoro share this CPU, and a warm-up that pins every core made
    # the listening time out on Windows/macOS laptops
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
logger.info(f"Loading pipeline (vae={VAE_CKPT}, flow={FLOW_CKPT}) on {DEVICE}...")
pipeline = MotionPipeline(VAE_CKPT, FLOW_CKPT, device=DEVICE)
# Warm-up in the BACKGROUND: pulls the text encoder from the Hub if needed (a minute on a slow
# link) and runs one short window. /health answers at once with ready=false so a launcher can say
# "warming up" instead of "missing"; /motion waits for it (gestures are never skipped).
import threading
_READY = threading.Event()
_LOCK = threading.Lock()
# _READY only means "the warm-up finished" — it is what /motion waits on, so it has to be set even
# when the warm-up blew up, or the first request would hang for ever. It is NOT proof that the
# pipeline works, so keep the failure reason separately: a dead pipeline that still answered
# {"status": "ok", "ready": true} is exactly how a torch/numpy ABI mismatch stayed invisible while
# every single /motion returned 500.
_WARM_UP_ERROR: str | None = None


def _warm_up() -> None:
    global _WARM_UP_ERROR
    try:
        _t0 = time.time()
        with _LOCK:
            pipeline.generate(text="hello", duration_s=1.0, emotion=0)
        logger.info(f"Motion-lab pipeline ready on {DEVICE} (warm-up {time.time() - _t0:.1f}s).")
    except Exception as exc:  # noqa: BLE001 — not fatal: /motion still answers, with the real error
        _WARM_UP_ERROR = f"{type(exc).__name__}: {exc}"
        logger.error(f"warm-up FAILED ({_WARM_UP_ERROR}); /health reports status=error, ready=false")
    finally:
        _READY.set()


threading.Thread(target=_warm_up, name="motion-warm-up", daemon=True).start()

_session_prefix: dict[str, object] = {}

# ── Named-action clip library ─────────────────────────────────────────────
# Deliberate gestures (wave, point, raise hand) come from CURATED real-mocap
# clips, not the generative action slot — that produced contorted, turned-away
# motion. Clips are canonicalized (camera-facing, planted) and play deterministically.
_LIB_DIR = Path(__file__).resolve().parents[1] / "assets" / "action_library"
_ACTION_LIB: dict[str, tuple] = {}
if _LIB_DIR.is_dir():
    for _f in sorted(_LIB_DIR.glob("*.npz")):
        with np.load(_f) as _d:
            _ACTION_LIB[_f.stem] = (_d["poses"].astype(np.float32), _d["trans"].astype(np.float32))
    logger.info(f"Action library: {list(_ACTION_LIB)}")

# free-text action caption (from the LLM's [MOTION:]) -> library key
_ACTION_KEYWORDS = [
    ("wave", ("wav", "hello", "hi ", "greet", "goodbye", "bye")),
    ("point", ("point",)),
    ("raise_hand", ("rais", "hand up", "volunteer", "put up")),
]


def _match_action(action: str) -> str | None:
    a = action.lower()
    for key, kws in _ACTION_KEYWORDS:
        if key in _ACTION_LIB and any(k in a for k in kws):
            return key
    return None


def _play_clip(key: str, n_frames: int):
    """Play a library gesture, made seamless with the co-speech that surrounds it:
      - ROOT frozen to the co-speech neutral orientation. The clip's root is already
        yaw-0, but the frontend VRoid retarget spins on the mocap root's residual
        pitch/roll; pinning root = neutral (which the retarget handles fine for
        speech) kills the 'gira como loco'. Gesture lives in arms/spine, not root.
      - Both ends cross-faded to the co-speech neutral pose (mean_pose) so the arm
        lowers back to talking rest and action↔speech has NO visible cut.
    Held to fill the sentence; trimmed if shorter."""
    poses, _ = _ACTION_LIB[key]
    T = poses.shape[0]
    if n_frames <= T:
        p = poses[:n_frames].astype(np.float32).copy()
    else:  # hold the (now-neutral, post-fade) tail to fill a longer sentence
        p = np.concatenate([poses, np.repeat(poses[-1:], n_frames - T, 0)]).astype(np.float32)
    L = len(p)
    aa = p.reshape(L, 55, 3)
    mean = get_mean_pose().reshape(55, 3)

    aa[:, 0, :] = mean[0]                      # freeze root to co-speech neutral (no spin)

    fi = min(6, L)                             # ease in from neutral (~0.2s)
    for j in range(fi):
        w = j / max(1, fi - 1)
        aa[j, 1:] = mean[1:] * (1 - w) + aa[j, 1:] * w
    fo = min(12, L)                            # ease out to neutral (~0.4s): arm lowers
    for j in range(fo):
        i = L - fo + j
        w = j / max(1, fo - 1)                 # ends fully neutral -> seamless into speech
        aa[i, 1:] = aa[i, 1:] * (1 - w) + mean[1:] * w

    p = plant_legs(aa.reshape(L, 165))
    p = smooth_poses(p, window=5)
    return p.astype(np.float32), np.zeros((L, 3), np.float32)


class MotionRequest(BaseModel):
    text: str = ""
    action: str = ""
    duration_s: float = 3.0
    emotion: str = "neutral"
    intensity: float = 1.0
    session_id: str = ""


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


@app.post("/motion")
async def motion(req: MotionRequest):
    if not req.text and not req.action:
        raise HTTPException(status_code=400, detail="text or action required")
    if not _READY.is_set():
        # first sentences of the first run: wait for the warm-up rather than answer without motion
        await asyncio.get_event_loop().run_in_executor(None, _READY.wait)
    emotion = EMOTIONS.index(req.emotion) if req.emotion in EMOTIONS else 0

    start = time.time()

    # Named action → play a curated library clip (deterministic, human, camera-facing).
    lib_key = _match_action(req.action) if req.action else None
    if lib_key:
        n_frames = max(pipeline.window, round(req.duration_s * pipeline.fps))
        poses, trans = _play_clip(lib_key, n_frames)
        elapsed_ms = round((time.time() - start) * 1000)
        logger.info(f"{poses.shape[0]} frames from library[{lib_key}] in {elapsed_ms}ms")
        return {
            "fps": pipeline.fps, "num_frames": int(poses.shape[0]),
            "poses_b64": _b64(poses), "trans_b64": _b64(trans), "inference_ms": elapsed_ms,
        }

    prefix = _session_prefix.get(req.session_id) if req.session_id else None
    try:
        result, last_latent = pipeline.generate(
            text=req.text, action=req.action, duration_s=req.duration_s,
            emotion=emotion, intensity=req.intensity, prefix_latent=prefix)
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if req.session_id:
        _session_prefix[req.session_id] = last_latent
    elapsed_ms = round((time.time() - start) * 1000)

    logger.info(f"{result.poses.shape[0]} frames in {elapsed_ms}ms "
                f"(text={bool(req.text)}, action={bool(req.action)})")
    return {
        "fps": result.fps,
        "num_frames": int(result.poses.shape[0]),
        "poses_b64": _b64(result.poses),
        "trans_b64": _b64(result.trans),
        "inference_ms": elapsed_ms,
    }


# Existing keys keep their exact meaning — the launcher reads `ready` to tell "warming up" from
# "missing" and `device` for its status line — and `warm_up_error` is purely additive. A warm-up
# that FAILED is neither ready nor ok: it stays HTTP 200 so the launcher's port probe still finds
# the sidecar, but the body has to say that gestures are broken instead of showing a false green.
@app.get("/health")
async def health() -> dict:
    failed = _WARM_UP_ERROR is not None
    return {
        "status": "error" if failed else "ok",
        "model": "motionlab-flow",
        "fps": pipeline.fps,
        "device": DEVICE,
        "ready": _READY.is_set() and not failed,
        "warm_up_error": _WARM_UP_ERROR,
    }
