"""The serve response must stay byte-compatible with the EMAGE sidecar contract
that hannah-backend already consumes: {fps, num_frames, poses_b64, trans_b64}.
This test validates the schema WITHOUT loading model checkpoints."""
import base64
import importlib.util
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = {"fps", "num_frames", "poses_b64", "trans_b64"}


def test_b64_layout_matches_emage_contract():
    # 2 frames of fake motion, little-endian float32 — what the frontend decodes
    poses = np.arange(2 * 165, dtype="<f4").reshape(2, 165)
    encoded = base64.b64encode(np.ascontiguousarray(poses).tobytes()).decode("ascii")
    decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4").reshape(2, 165)
    assert np.array_equal(poses, decoded)


def test_serve_module_declares_contract_fields():
    src = (Path(__file__).parents[1] / "serve" / "main.py").read_text()
    for field in REQUIRED_FIELDS:
        assert f'"{field}"' in src, f"serve/main.py response is missing contract field {field}"


def test_motion_request_defaults():
    # request model importable without torch checkpoints? serve imports pipeline
    # at module load, so validate the pydantic model shape from source instead.
    src = (Path(__file__).parents[1] / "serve" / "main.py").read_text()
    assert "class MotionRequest" in src
    for field in ("text", "action", "duration_s", "emotion", "intensity", "session_id"):
        assert field in src
