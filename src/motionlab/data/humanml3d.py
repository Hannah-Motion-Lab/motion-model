"""HumanML3D captioned motions for the action-prompt slot.

Motion source is AMASS (registration required — see README). Until the user
drops AMASS archives and runs scripts/prepare_amass.py, this dataset reports
len()==0 and training simply proceeds with BEAT2 alone.

Expected prepared layout (produced by prepare_amass.py, phase 2):
  data/humanml3d/motions/{id}.npz   poses (T,165) axis-angle SMPL-X body (hands neutral), trans (T,3), 30fps
  data/humanml3d/texts/{id}.txt     one caption per line (HumanML3D text release)
"""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

# Body joints (no fingers) whose motion defines "where the action happens".
_ACTION_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def _canonicalize(poses: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make an action clip camera-facing and planted so the model learns the
    GESTURE, not the locomotion it happened to be embedded in:
      - remove world-yaw drift of the root (no turning away) — arm/body-relative
        articulation is untouched, so the action itself is preserved;
      - zero horizontal translation (no walking across the room).
    Height (y) is kept so crouch/reach still read."""
    aa = poses.reshape(len(poses), 55, 3).copy()
    Rm = R.from_rotvec(aa[:, 0, :])
    fwd = Rm.apply(np.array([0.0, 0.0, 1.0]))
    heading = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    delta = heading - heading[0]
    corr = R.from_rotvec(np.outer(-delta, np.array([0.0, 1.0, 0.0])))
    aa[:, 0, :] = (corr * Rm).as_rotvec()
    t = trans.copy()
    t[:, 0] -= t[0, 0]
    t[:, 2] -= t[0, 2]
    return aa.reshape(poses.shape).astype(np.float32), t.astype(np.float32)


class HumanML3DDataset(Dataset):
    def __init__(self, root: str, window: int = 64, fps: int = 30, motion_centered: bool = True):
        self.root = Path(root)
        self.window, self.fps = window, fps
        self.motion_centered = motion_centered
        self.items = []
        motions = self.root / "motions"
        texts = self.root / "texts"
        if motions.is_dir():
            for npz_path in sorted(motions.glob("*.npz")):
                txt = texts / f"{npz_path.stem}.txt"
                if txt.exists():
                    self.items.append((npz_path, txt))

    def __len__(self):
        return len(self.items)

    def _pick_window(self, poses: np.ndarray) -> int:
        """Choose a window centered on where the action actually happens (peak
        body-joint velocity) instead of a uniform-random crop — a random 2s slice
        of a captioned clip often misses the gesture, teaching the model a blurry
        average. A little jitter keeps variety."""
        T = poses.shape[0]
        if T <= self.window:
            return 0
        if not self.motion_centered:
            return np.random.randint(0, T - self.window + 1)
        aa = poses.reshape(T, 55, 3)[:, _ACTION_JOINTS, :]
        speed = np.linalg.norm(np.diff(aa, axis=0), axis=-1).sum(-1)   # (T-1,)
        csum = np.concatenate([[0.0], np.cumsum(speed)])
        win_motion = csum[self.window:] - csum[:T - self.window]       # motion per window start
        best = int(win_motion.argmax())
        jit = self.window // 4
        start = best + np.random.randint(-jit, jit + 1)
        return int(np.clip(start, 0, T - self.window))

    def __getitem__(self, i):
        npz_path, txt = self.items[i]
        with np.load(npz_path) as d:
            poses = d["poses"].astype(np.float32)
            trans = d["trans"].astype(np.float32)
        poses, trans = _canonicalize(poses, trans)
        T = poses.shape[0]
        if T >= self.window:
            start = self._pick_window(poses)
            poses, trans = poses[start:start + self.window], trans[start:start + self.window]
        else:  # pad short clips by holding the last frame
            pad = self.window - T
            poses = np.concatenate([poses, np.repeat(poses[-1:], pad, 0)])
            trans = np.concatenate([trans, np.repeat(trans[-1:], pad, 0)])
        captions = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        caption = captions[np.random.randint(len(captions))] if captions else ""
        return {
            "poses": torch.from_numpy(poses),
            "trans": torch.from_numpy(trans),
            "caption": caption,
        }
