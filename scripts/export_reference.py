#!/usr/bin/env python
"""Generate EMAGE baseline outputs on chosen audio for side-by-side comparison.

Uses ~/PantoMatrix read-only (its venv is the workspace one at Hannah-Motion/.venv,
which already runs EMAGE for the current Hannah sidecar).

  <workspace>/.venv/bin/python scripts/export_reference.py --wav path/to.wav --out data/reference/
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PANTOMATRIX = os.path.expanduser(os.environ.get("PANTOMATRIX_DIR", "~/PantoMatrix"))
WORKSPACE_VENV = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, nargs="+")
    p.add_argument("--out", default="data/reference")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for w in args.wav:
            os.symlink(os.path.abspath(w), os.path.join(tmp, os.path.basename(w)))
        subprocess.run(
            [str(WORKSPACE_VENV), "test_emage_audio.py",
             "--audio_folder", tmp, "--save_folder", str(out)],
            cwd=PANTOMATRIX, check=True)
    print(f"EMAGE reference npz written to {out}")


if __name__ == "__main__":
    main()
