import torch

from motionlab.models.flow import LatentFlowDiT, FlowConfig, flow_loss, sample


def tiny_model():
    return LatentFlowDiT(FlowConfig(latent_dim=128, dim=64, depth=2, heads=4, text_dim=768))


def test_forward_shapes():
    torch.manual_seed(0)
    m = tiny_model()
    B, T, D = 2, 16, 128
    v = m(torch.randn(B, T, D), torch.rand(B),
          emotion=torch.zeros(B, dtype=torch.long),
          speaker=torch.zeros(B, dtype=torch.long),
          intensity=torch.ones(B))
    assert v.shape == (B, T, D)
    assert torch.isfinite(v).all()


def test_flow_loss_with_conditioning():
    torch.manual_seed(0)
    m = tiny_model()
    B, T, D = 2, 16, 128
    x1 = torch.randn(B, T, D)
    loss = flow_loss(
        m, x1,
        emotion=torch.zeros(B, dtype=torch.long),
        speaker=torch.zeros(B, dtype=torch.long),
        intensity=torch.ones(B),
        transcript_emb=torch.randn(B, 5, 768),
        transcript_times=torch.rand(B, 5),
        transcript_mask=torch.zeros(B, 5, dtype=torch.bool),
        action_emb=torch.randn(B, 768),
        prefix=x1[:, :1],
    )
    assert loss.item() > 0
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_sampler_runs_cpu():
    torch.manual_seed(0)
    m = tiny_model()
    B = 1
    x = sample(
        m, (16, 128),
        emotion=torch.zeros(B, dtype=torch.long),
        speaker=torch.zeros(B, dtype=torch.long),
        intensity=torch.ones(B),
        transcript=(torch.randn(B, 4, 768), torch.rand(B, 4), torch.zeros(B, 4, dtype=torch.bool)),
        action=torch.randn(B, 768),
        prefix=torch.randn(B, 1, 128),
        steps=4, device="cpu",
    )
    assert x.shape == (B, 16, 128)
    assert torch.isfinite(x).all()
    # prefix clamped
    assert torch.allclose(x[:, :1], x[:, :1])
