"""BEAT2-English dataset: windows of SMPL-X motion + word-aligned transcript + emotion.

File layout (data/beat2/beat_english_v2.0.0):
  smplxflame_30/{seq}.npz   poses (T,165) axis-angle 55 joints, trans (T,3), betas, 30fps
  textgrid/{seq}.TextGrid   word tier with start/end times
  train_test_split.csv      id,type (train/val/test)

Filename convention `{spk}_{name}_{r0}_{a}_{b}`: the recording index encodes the
emotion session (BEAT paper): 0-64 neutral, 65-72 happiness, 73-80 anger,
81-86 sadness, 87-94 contempt, 95-102 surprise, 103-110 fear, 111-118 disgust.
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EMOTIONS = ["neutral", "happiness", "anger", "sadness", "contempt", "surprise", "fear", "disgust"]

_EMOTION_RANGES = [
    (0, 64, 0), (65, 72, 1), (73, 80, 2), (81, 86, 3),
    (87, 94, 4), (95, 102, 5), (103, 110, 6), (111, 118, 7),
]


def emotion_from_name(stem: str) -> int:
    """Best-effort emotion id from the BEAT sequence number; neutral on failure."""
    try:
        seq = int(stem.split("_")[3])
    except (IndexError, ValueError):
        return 0
    for lo, hi, emo in _EMOTION_RANGES:
        if lo <= seq <= hi:
            return emo
    return 0


def speaker_from_name(stem: str) -> int:
    try:
        return int(stem.split("_")[0])
    except (IndexError, ValueError):
        return 0


def load_textgrid_words(path: str) -> list[tuple[float, float, str]]:
    """[(start_s, end_s, word), ...] from the first interval tier, silences dropped."""
    import textgrid
    tg = textgrid.TextGrid.fromFile(path)
    tier = next((t for t in tg.tiers if hasattr(t, "intervals")), None)
    if tier is None:
        return []
    out = []
    for iv in tier.intervals:
        w = iv.mark.strip()
        if w and w.lower() not in {"sil", "sp", "<unk>", ""}:
            out.append((float(iv.minTime), float(iv.maxTime), w))
    return out


@dataclass
class Beat2Item:
    poses: torch.Tensor       # (window, 165) axis-angle float32
    trans: torch.Tensor       # (window, 3)
    words: list               # [(start_frame, end_frame, word)] relative to window
    emotion: int
    speaker: int
    name: str
    start_frame: int


class Beat2Dataset(Dataset):
    def __init__(self, root: str, window: int = 64, stride: int = 20, fps: int = 30,
                 split: str = "train", require_textgrid: bool = True, max_files: int | None = None):
        self.root = Path(root)
        self.window, self.stride, self.fps = window, stride, fps
        self.require_textgrid = require_textgrid

        split_ids = None
        csv_path = self.root / "train_test_split.csv"
        if csv_path.exists() and split:
            split_ids = set()
            with open(csv_path) as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[1].strip() == split:
                        split_ids.add(row[0].strip())

        self.index = []  # (npz_path, tg_path|None, start_frame, n_frames, emotion, speaker)
        files = sorted((self.root / "smplxflame_30").glob("*.npz"))
        if max_files:
            files = files[:max_files]
        for npz_path in files:
            stem = npz_path.stem
            if split_ids is not None and stem not in split_ids:
                continue
            tg_path = self.root / "textgrid" / f"{stem}.TextGrid"
            if require_textgrid and not tg_path.exists():
                continue
            try:
                with np.load(npz_path, allow_pickle=True) as d:
                    n = int(d["poses"].shape[0])
            except Exception:
                continue
            emo, spk = emotion_from_name(stem), speaker_from_name(stem)
            for start in range(0, max(1, n - window + 1), stride):
                if start + window <= n:
                    self.index.append((npz_path, tg_path if tg_path.exists() else None, start, n, emo, spk))

        self._word_cache: dict[str, list] = {}

    def __len__(self):
        return len(self.index)

    def _words_for(self, tg_path, start_frame):
        if tg_path is None:
            return []
        key = str(tg_path)
        if key not in self._word_cache:
            self._word_cache[key] = load_textgrid_words(key)
        window_s, window_e = start_frame / self.fps, (start_frame + self.window) / self.fps
        out = []
        for s, e, w in self._word_cache[key]:
            if e > window_s and s < window_e:
                out.append((max(0, int(s * self.fps) - start_frame),
                            min(self.window, int(e * self.fps) - start_frame), w))
        return out

    def __getitem__(self, i) -> Beat2Item:
        npz_path, tg_path, start, _, emo, spk = self.index[i]
        with np.load(npz_path, allow_pickle=True) as d:
            poses = d["poses"][start:start + self.window].astype(np.float32)
            trans = d["trans"][start:start + self.window].astype(np.float32)
        return Beat2Item(
            poses=torch.from_numpy(poses),
            trans=torch.from_numpy(trans),
            words=self._words_for(tg_path, start),
            emotion=emo,
            speaker=spk,
            name=npz_path.stem,
            start_frame=start,
        )


def collate(items: list[Beat2Item]):
    return {
        "poses": torch.stack([it.poses for it in items]),
        "trans": torch.stack([it.trans for it in items]),
        "words": [it.words for it in items],
        "emotion": torch.tensor([it.emotion for it in items], dtype=torch.long),
        "speaker": torch.tensor([it.speaker for it in items], dtype=torch.long),
        "names": [it.name for it in items],
    }
