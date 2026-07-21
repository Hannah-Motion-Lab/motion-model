#!/usr/bin/env python
"""Build the HumanML3D caption->motion dataset from converted AMASS motions.

Inputs:
  data/humanml3d/index.csv       (official mapping: AMASS source, frame range, clip id)
  data/humanml3d/texts.zip       (official captions, incl. M-prefixed mirrors)
  data/amass/motions/*.npz       (scripts/prepare_amass.py output, 30fps SMPL-X)

Output (what src/motionlab/data/humanml3d.py loads):
  data/humanml3d/motions/{id}.npz   poses (T,165) @30fps, trans (T,3)
  data/humanml3d/texts/{id}.txt     captions (one per line)
  + mirrored M{id} pairs via augment.mirror when captions exist

Frame ranges in index.csv are in the official 20fps processing space -> we map
via seconds onto our 30fps conversions. humanact12 rows (not AMASS) are skipped.
"""
import csv
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motionlab.data.augment import mirror  # noqa: E402

ROOT = Path("data/humanml3d")
MOTIONS_IN = Path("data/amass/motions")
OUT_M = ROOT / "motions"
OUT_T = ROOT / "texts"

# index.csv dataset dir -> our archive naming (normalization removes case/_/-)
ALIASES = {
    "biomotionlabntroje": "bmlrub",
    "mpihdm05": "hdm05",
    "mpimosh": "mosh",
    "mpilimits": "poseprior",
    "tcdhandmocap": "tcdhands",
    "dfaust67": "dfaust",
    "ssmsynced": "ssm",
    "transitionsmocap": "transitions",
    "eyesjapandataset": "eyesjapandataset",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_T.mkdir(parents=True, exist_ok=True)

    # our converted files indexed by normalized name (minus stageii/npz)
    ours = {}
    for p in MOTIONS_IN.glob("*.npz"):
        key = norm(p.stem.replace("_stageii", ""))
        ours[key] = p
    print(f"{len(ours)} converted AMASS sequences indexed")

    texts = zipfile.ZipFile(ROOT / "texts.zip")
    text_names = set(texts.namelist())

    def caption_for(clip_id: str) -> str | None:
        for cand in (f"texts/{clip_id}.txt", f"{clip_id}.txt"):
            if cand in text_names:
                return texts.read(cand).decode("utf-8", errors="ignore")
        return None

    matched = missed = written = 0
    with open(ROOT / "index.csv") as f:
        for row in csv.DictReader(f):
            src = row["source_path"].strip()
            parts = src.replace("./pose_data/", "").split("/")
            dataset, rest = parts[0], parts[1:]
            if norm(dataset) == "humanact12":
                continue
            dnorm = ALIASES.get(norm(dataset), norm(dataset))
            key = dnorm + norm("".join(rest)).replace("poses" + "npy", "")
            key = re.sub(r"posesnpy$", "", key)
            p = ours.get(key)
            if p is None:
                missed += 1
                continue
            matched += 1

            clip_id = row["new_name"].replace(".npy", "")
            cap = caption_for(clip_id)
            if cap is None:
                continue

            with np.load(p) as d:
                poses_full = d["poses"]
                trans_full = d["trans"]
            start20, end20 = int(row["start_frame"]), int(row["end_frame"])
            s30 = max(0, round(start20 / 20 * 30))
            e30 = poses_full.shape[0] if end20 < 0 else min(poses_full.shape[0], round(end20 / 20 * 30))
            if e30 - s30 < 32:
                continue
            poses = poses_full[s30:e30].astype(np.float32)
            trans = trans_full[s30:e30].astype(np.float32)
            trans = trans - trans[0:1]

            np.savez(OUT_M / f"{clip_id}.npz", poses=poses, trans=trans, mocap_frame_rate=30)
            (OUT_T / f"{clip_id}.txt").write_text(
                "\n".join(l.split("#")[0].strip() for l in cap.splitlines() if l.strip()))
            written += 1

            mcap = caption_for("M" + clip_id)
            if mcap is not None:
                mp, mt = mirror(torch.from_numpy(poses), torch.from_numpy(trans))
                np.savez(OUT_M / f"M{clip_id}.npz", poses=mp.numpy(), trans=mt.numpy(),
                         mocap_frame_rate=30)
                (OUT_T / f"M{clip_id}.txt").write_text(
                    "\n".join(l.split("#")[0].strip() for l in mcap.splitlines() if l.strip()))
                written += 1

    print(f"index rows matched: {matched}, missed: {missed}, clips written: {written}")


if __name__ == "__main__":
    main()
