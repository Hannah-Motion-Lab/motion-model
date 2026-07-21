"""Post-processing for generated motion — the cheap fixes that don't need retraining.

Applied by MotionPipeline before returning:
  1. Trans de-drift: subtract a slow moving average so the character sways
     naturally but never wanders off (or toward the camera).
  2. Temporal smoothing in 6D rotation space (kills flow-sampling jitter).
  3. Hand relaxation: blend fingers toward the dataset's mean hand pose —
     the continuous latent space is weakest on high-dim low-variance fingers.
  4. Pose clamping: axis-angle norms capped to keep joints in human range
     (prevents arm-through-body extremes from off-manifold latents).
"""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .rotations import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle

_mean_pose = None


def get_mean_pose():
    global _mean_pose
    if _mean_pose is None:
        p = Path(__file__).resolve().parents[2] / "assets" / "mean_pose.npy"
        _mean_pose = np.load(p) if p.exists() else np.zeros(165, dtype=np.float32)
    return _mean_pose


def dedrift_trans(trans: np.ndarray, fps: int = 30, window_s: float = 1.5,
                  keep: float = 0.35) -> np.ndarray:
    """Remove the slow drift component of global translation, keep `keep` of
    the fast sway. Anchored so frame 0 stays where playback expects it."""
    T = trans.shape[0]
    w = max(3, int(window_s * fps) | 1)
    pad = w // 2
    padded = np.pad(trans, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w) / w
    slow = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1)[:T]
    out = (trans - slow) * keep
    return out - out[0:1]  # start at origin of the chunk


def smooth_poses(poses: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing in 6D rotation space (safe for interpolation)."""
    T = poses.shape[0]
    if T < window:
        return poses
    r6 = axis_angle_to_rotation_6d(torch.from_numpy(poses.reshape(T, 55, 3))).numpy()
    pad = window // 2
    padded = np.pad(r6, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    sm = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)[:T]
    return rotation_6d_to_axis_angle(torch.from_numpy(sm)).numpy().reshape(T, 165).astype(np.float32)


def relax_hands(poses: np.ndarray, blend: float = 0.45, jaw_scale: float = 0.55) -> np.ndarray:
    """Blend finger joints toward the dataset mean hand pose (blend=1 keeps the
    generated fingers fully; 0 pins them to the relaxed pose) and damp the jaw
    so the mouth doesn't gape."""
    mean = get_mean_pose()
    out = poses.copy()
    out[:, 66:156] = mean[66:156] + blend * (poses[:, 66:156] - mean[66:156])
    out[:, 156:159] *= jaw_scale
    return out


# Per-joint sane axis-angle limits (radians). Generous — only extreme
# off-manifold poses get pulled back.
_BODY_LIMIT = 2.2
_HAND_LIMIT = 1.6


def clamp_pose_range(poses: np.ndarray) -> np.ndarray:
    aa = poses.reshape(-1, 55, 3).copy()
    norms = np.linalg.norm(aa, axis=-1, keepdims=True)
    limits = np.full((55, 1), _BODY_LIMIT, dtype=np.float32)
    limits[25:] = _HAND_LIMIT
    scale = np.minimum(1.0, limits[None] / np.maximum(norms, 1e-8))
    aa[:, 1:] *= scale[:, 1:]  # never touch global orient
    return aa.reshape(poses.shape).astype(np.float32)


# Lower-body joints (legs/feet) — anchored in "talking" mode so a standing
# avatar never walks or bends to the floor mid-sentence.
_LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
# Spine joints — forward bend limited in talking mode.
_SPINE_JOINTS = [3, 6, 9]


def plant_legs(poses: np.ndarray, leg_blend: float = 0.15) -> np.ndarray:
    """Pull the legs toward the dataset mean stance so the avatar stays planted
    (no stepping/walking), leaving spine, arms, hands and head free."""
    mean = get_mean_pose().reshape(55, 3)
    aa = poses.reshape(-1, 55, 3).copy()
    for j in _LEG_JOINTS:
        aa[:, j] = mean[j] + leg_blend * (aa[:, j] - mean[j])
    return aa.reshape(poses.shape).astype(np.float32)


def anchor_talking(poses: np.ndarray, leg_blend: float = 0.15, spine_bend_limit: float = 0.35) -> np.ndarray:
    """Keep a talking avatar upright and planted: pull legs toward the dataset
    mean stance and cap forward spine bend, leaving arms/hands/head free."""
    out = plant_legs(poses, leg_blend)
    aa = out.reshape(-1, 55, 3)
    # limit the X-axis (forward/back bend) component of the spine
    for j in _SPINE_JOINTS:
        aa[:, j, 0] = np.clip(aa[:, j, 0], -spine_bend_limit, spine_bend_limit)
    return aa.reshape(poses.shape).astype(np.float32)


def anchor_yaw_to_start(poses: np.ndarray) -> np.ndarray:
    """Cancel world-yaw *drift* of the global orient so the avatar keeps facing
    the way it started — kills the 'turns around and shows her back' artifact
    that leaks in from locomotion-heavy action data — while preserving lean
    (pitch/roll) and every body-relative joint (the raised arm still raises).
    Frame 0 is left untouched; later frames are de-rotated about +Y by their
    heading delta from frame 0."""
    aa = poses.reshape(-1, 55, 3).copy()
    Rm = R.from_rotvec(aa[:, 0, :])
    fwd = Rm.apply(np.array([0.0, 0.0, 1.0]))          # body forward in world
    heading = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))   # yaw about +Y, continuous
    delta = heading - heading[0]                        # drift from the start pose
    corr = R.from_rotvec(np.outer(-delta, np.array([0.0, 1.0, 0.0])))
    aa[:, 0, :] = (corr * Rm).as_rotvec()
    return aa.reshape(poses.shape).astype(np.float32)


def postprocess(poses: np.ndarray, trans: np.ndarray, fps: int = 30,
                smooth_window: int = 5, hand_blend: float = 0.45,
                trans_keep: float = 0.35, talking: bool = True):
    poses = clamp_pose_range(poses)
    if talking:
        poses = anchor_talking(poses)
        trans = np.zeros_like(trans)          # planted; no wandering while speaking
    else:
        # ACTION mode: she performs a deliberate gesture but is still a
        # camera-facing upper-body avatar — so keep her facing forward (no
        # turning away) and planted (no walking), leaving spine/arms/hands free.
        poses = anchor_yaw_to_start(poses)
        poses = plant_legs(poses)
        trans = np.zeros_like(trans)
    poses = relax_hands(poses, blend=hand_blend)
    poses = smooth_poses(poses, window=smooth_window)
    return poses, trans
