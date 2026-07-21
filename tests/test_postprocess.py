import numpy as np

from motionlab.postprocess import clamp_pose_range, dedrift_trans, relax_hands, smooth_poses


def test_dedrift_removes_linear_drift():
    T = 150
    drift = np.linspace(0, 2.0, T)[:, None] * np.array([1.0, 0.0, 1.0])
    sway = 0.02 * np.sin(np.linspace(0, 30, T))[:, None] * np.array([1.0, 0, 0])
    out = dedrift_trans((drift + sway).astype(np.float32))
    assert np.abs(out).max() < 0.15          # drift gone, only small sway remains
    assert np.allclose(out[0], 0, atol=1e-6)  # anchored at chunk origin


def test_smooth_preserves_shape_and_reduces_jitter():
    rng = np.random.default_rng(0)
    base = np.zeros((60, 165), dtype=np.float32)
    noisy = base + rng.normal(0, 0.1, base.shape).astype(np.float32)
    sm = smooth_poses(noisy)
    assert sm.shape == (60, 165)
    jitter = lambda p: np.abs(np.diff(p, axis=0)).mean()
    assert jitter(sm) < jitter(noisy) * 0.7


def test_relax_hands_only_touches_fingers():
    poses = np.random.default_rng(1).normal(0, 0.5, (10, 165)).astype(np.float32)
    out = relax_hands(poses, blend=0.0, jaw_scale=1.0)
    assert np.allclose(out[:, :66], poses[:, :66])
    assert np.allclose(out[:, 156:], poses[:, 156:])
    assert not np.allclose(out[:, 66:156], poses[:, 66:156])


def test_jaw_damping():
    poses = np.zeros((4, 165), dtype=np.float32)
    poses[:, 156] = 1.0
    out = relax_hands(poses, blend=1.0, jaw_scale=0.5)
    assert np.allclose(out[:, 156], 0.5)


def test_clamp_caps_extremes_but_not_root():
    poses = np.zeros((5, 165), dtype=np.float32)
    poses[:, :3] = 3.0        # root orient — must survive
    poses[:, 30:33] = 5.0     # a body joint way out of range
    out = clamp_pose_range(poses)
    assert np.allclose(out[:, :3], 3.0)
    assert np.linalg.norm(out[:, 30:33], axis=1).max() <= 2.2 + 1e-4
