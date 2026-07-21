#!/usr/bin/env python
"""Convert AMASS archives (user-downloaded, registration required) into the
lab's format for Stage A training + HumanML3D building.

1. Register at https://amass.is.tue.mpg.de and download SMPL-X packages
   (neutral where offered; gendered is fine — we only use poses/trans).
2. Drop the .tar.bz2 archives into data/amass/raw/
3. Run:  .venv/bin/python scripts/prepare_amass.py

Output: data/amass/motions/{path_flattened}.npz  poses (T,165)@30fps, trans (T,3).

Implementation notes:
- Archives are EXTRACTED to a temp dir first, then loaded from disk. np.load
  directly on tar-member streams seeks, which on bz2 re-decompresses from the
  start of the stream (quadratic — the naive version took hours per GB).
- Already-converted sequences are skipped, so re-running after adding new
  archives only processes the new ones.
- Archives are processed in parallel (one worker per archive, up to 4).
"""
import multiprocessing as mp
import tarfile
import tempfile
from pathlib import Path

import numpy as np

RAW = Path("data/amass/raw")
OUT = Path("data/amass/motions")


def convert_array(poses_in: np.ndarray, trans_in: np.ndarray, fps: float):
    step = fps / 30.0
    idx = np.arange(0, poses_in.shape[0], step).astype(int)
    idx = idx[idx < poses_in.shape[0]]
    if len(idx) < 32:
        return None
    p, t = poses_in[idx], trans_in[idx]
    T = p.shape[0]
    out = np.zeros((T, 165), dtype=np.float32)
    out[:, :3] = p[:, :3]
    out[:, 3:66] = p[:, 3:66]
    if p.shape[1] == 156:                      # SMPL+H: [root][body][lhand45][rhand45]
        out[:, 66:156] = p[:, 66:156]
    elif p.shape[1] == 165:                    # SMPL-X: [root][body][jaw][eyes][hands]
        out[:, 156:159] = p[:, 66:69]
        out[:, 66:156] = p[:, 75:165]
    else:
        return None
    return out, t.astype(np.float32)


def out_name(rel_path: str) -> str:
    return rel_path.replace("/", "_").removesuffix(".npz") + ".npz"


TMP_BASE = Path("data/amass/tmp")  # system tmp has a quota; extract on /home instead


def process_archive(archive: Path) -> tuple[str, int, int]:
    written = skipped = 0
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amass_", dir=TMP_BASE) as tmp:
        with tarfile.open(archive) as tf:
            tf.extractall(tmp, filter="data")
        for npz_path in Path(tmp).rglob("*.npz"):
            rel = str(npz_path.relative_to(tmp))
            dest = OUT / out_name(rel)
            if dest.exists():
                skipped += 1
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            if "poses" not in data or "trans" not in data:
                continue
            fps = float(data.get("mocap_framerate", data.get("mocap_frame_rate", 0)))
            if fps <= 0:
                continue
            result = convert_array(data["poses"], data["trans"], fps)
            if result is None:
                continue
            poses, trans = result
            np.savez(dest, poses=poses, trans=trans, mocap_frame_rate=30)
            written += 1
    return archive.name, written, skipped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archives = sorted(RAW.glob("*.tar.bz2"))
    if not archives:
        raise SystemExit(f"No archives in {RAW} — download AMASS first (see docstring).")

    total = 0
    with mp.Pool(min(2, len(archives))) as pool:
        for name, written, skipped in pool.imap_unordered(process_archive, archives):
            total += written
            print(f"{name}: +{written} sequences ({skipped} already converted)", flush=True)
    print(f"done: {total} new sequences in {OUT}")


if __name__ == "__main__":
    main()
