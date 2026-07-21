#!/usr/bin/env python
"""Download the BEAT2-English portion needed for text->motion training.

We deliberately SKIP wave16k (~audio, not needed for text conditioning) and
the pretrained weights folder. Resumable: re-running continues where it left off.
"""
import argparse
from huggingface_hub import snapshot_download

PATTERNS = [
    "beat_english_v2.0.0/smplxflame_30/*",   # SMPL-X motion npz @30fps
    "beat_english_v2.0.0/textgrid/*",        # word-level alignments
    "beat_english_v2.0.0/sem/*",             # semantic gesture annotations
    "beat_english_v2.0.0/train_test_split.csv",
    "beat_english_v2.0.0/readme.md",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/beat2")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id="H-Liu1997/BEAT2",
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=PATTERNS,
        max_workers=8,
    )
    print(f"BEAT2-English ready at {path}")
