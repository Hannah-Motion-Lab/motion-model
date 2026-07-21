#!/usr/bin/env python
"""Canonicalize converted AMASS motions into BEAT2's coordinate frame.

BUG this fixes: AMASS is Z-up with arbitrary facing; BEAT2 is Y-up facing +Z
(camera). Training the VAE on the raw mix collapsed the pose space. Here we,
per sequence, IN PLACE on data/amass/motions/*.npz:
  1. Rotate the root orient from Z-up to Y-up (fixed -90deg about X).
  2. Remove the frame-0 heading (yaw about the vertical Y axis) so every clip
     starts facing the camera, matching BEAT2 speakers.
  3. Apply the same global rotation to `trans`, then anchor at the origin.
Body/hand joint rotations are LOCAL to the root, so they are untouched — only
the global placement changes, preserving the actual motion.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motionlab.rotations import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402
import torch  # noqa: E402

MOT = Path("data/amass/motions")

# Z-up -> Y-up: -90deg about X.  (x,y,z)_zup -> (x,z,-y)_yup
RX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def yaw_only(R: np.ndarray) -> np.ndarray:
    """Rotation about Y (vertical) that matches R's heading — its projection of
    the forward axis onto the ground plane."""
    fwd = R @ np.array([0, 0, 1.0], dtype=np.float32)   # body forward in world
    theta = np.arctan2(fwd[0], fwd[2])                  # heading angle about Y
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def canonicalize(poses: np.ndarray, trans: np.ndarray):
    root_m = axis_angle_to_matrix(torch.from_numpy(poses[:, :3])).numpy()  # (T,3,3)
    yup = RX[None] @ root_m                                                # to Y-up
    yaw0 = yaw_only(yup[0])                                                # frame-0 heading
    align = yaw0.T @ RX                                                    # remove heading + up-fix
    new_root = align[None] @ root_m
    poses = poses.copy()
    poses[:, :3] = matrix_to_axis_angle(torch.from_numpy(new_root)).numpy()
    new_trans = (align[None] @ trans[:, :, None])[:, :, 0]
    new_trans = new_trans - new_trans[0:1]
    return poses.astype(np.float32), new_trans.astype(np.float32)


def main():
    files = sorted(MOT.glob("*.npz"))
    done = 0
    for p in files:
        with np.load(p) as d:
            if d.get("canon", np.array(0)).item() if "canon" in d else False:
                continue
            poses, trans = d["poses"], d["trans"]
        if poses.shape[0] < 2:
            continue
        poses, trans = canonicalize(poses, trans)
        np.savez(p, poses=poses, trans=trans, mocap_frame_rate=30, canon=True)
        done += 1
        if done % 1000 == 0:
            print(f"{done}/{len(files)}", flush=True)
    print(f"canonicalized {done} sequences")


if __name__ == "__main__":
    main()
