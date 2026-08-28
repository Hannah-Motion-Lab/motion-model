"""AMASS converted motions (scripts/prepare_amass.py output) — motion-only
windows for Stage A. Same npz schema as BEAT2 (poses (T,165) @30fps, trans)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AmassDataset(Dataset):
    def __init__(self, root: str, window: int = 64, stride: int = 64, max_files: int | None = None):
        self.root = Path(root)
        self.window = window
        self.index = []
        files = sorted(self.root.glob("*.npz"))
        if max_files:
            files = files[:max_files]
        for p in files:
            try:
                with np.load(p) as d:
                    n = int(d["poses"].shape[0])
            except Exception:
                continue
            for start in range(0, max(1, n - window + 1), stride):
                if start + window <= n:
                    self.index.append((p, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        p, start = self.index[i]
        with np.load(p) as d:
            poses = d["poses"][start:start + self.window].astype(np.float32)
            trans = d["trans"][start:start + self.window].astype(np.float32)
        trans = trans - trans[0:1]  # AMASS clips roam; anchor windows at origin
        return {"poses": torch.from_numpy(poses), "trans": torch.from_numpy(trans)}
