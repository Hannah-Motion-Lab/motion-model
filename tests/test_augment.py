import torch

from motionlab.data.augment import mirror, tempo_jitter, exaggerate


def test_mirror_is_involution():
    torch.manual_seed(0)
    poses = torch.randn(64, 165) * 0.4
    trans = torch.randn(64, 3) * 0.1
    p2, t2 = mirror(*mirror(poses, trans))
    assert torch.allclose(p2, poses, atol=1e-6)
    assert torch.allclose(t2, trans, atol=1e-6)


def test_tempo_jitter_shapes():
    poses = torch.randn(64, 165)
    trans = torch.randn(64, 3)
    p, t = tempo_jitter(poses, trans, 1.2)
    assert p.shape == (64, 165) and t.shape == (64, 3)
    assert torch.isfinite(p).all()


def test_exaggerate_caps_angle():
    poses = torch.randn(64, 165) * 2.0
    p = exaggerate(poses, 2.5)
    norms = p.reshape(64, 55, 3)[:, 1:].norm(dim=-1)  # root is exempt from scaling/capping
    assert norms.max() <= 3.1 + 1e-4
    # root orient untouched
    assert torch.allclose(p.reshape(64, 55, 3)[:, 0], poses.reshape(64, 55, 3)[:, 0])
