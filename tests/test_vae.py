import torch

from motionlab.models.vae import PartwiseMotionVAE, VAEConfig


def small_vae():
    return PartwiseMotionVAE(VAEConfig(latent_dim=32, channels=64, down_t=2))


def test_roundtrip_shapes():
    torch.manual_seed(0)
    vae = small_vae()
    poses = torch.randn(2, 64, 165) * 0.3
    trans = torch.randn(2, 64, 3) * 0.1
    out = vae(poses, trans)
    assert out["rec_poses"].shape == (2, 64, 165)
    assert out["rec_trans"].shape == (2, 64, 3)
    assert torch.isfinite(out["rec_poses"]).all()
    assert out["kl"].item() >= 0


def test_latent_tensor_roundtrip():
    vae = small_vae()
    poses = torch.randn(2, 64, 165) * 0.3
    trans = torch.zeros(2, 64, 3)
    zs, _, _ = vae.encode(poses, trans, sample=False)
    x = vae.latents_to_tensor(zs)          # (B, t, L*parts)
    assert x.shape == (2, 16, 32 * 4)      # 64/2**2 = 16 latent frames
    zs2 = vae.tensor_to_latents(x)
    for name in zs:
        assert torch.allclose(zs[name], zs2[name])


def test_split_join_identity():
    vae = small_vae()
    poses = torch.randn(2, 64, 165) * 0.3
    trans = torch.randn(2, 64, 3) * 0.1
    feats = vae.split_parts(poses, trans)
    poses2, trans2 = vae.join_parts(feats, 64)
    # join(split(x)) must reproduce the rotations (via matrices) and trans
    from motionlab.rotations import axis_angle_to_matrix
    m1 = axis_angle_to_matrix(poses.reshape(2, 64, 55, 3))
    m2 = axis_angle_to_matrix(poses2.reshape(2, 64, 55, 3))
    assert torch.allclose(m1, m2, atol=1e-4)
    assert torch.allclose(trans, trans2, atol=1e-5)
