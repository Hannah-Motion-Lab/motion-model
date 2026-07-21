"""Stage B: latent flow-matching DiT with dual text conditioning.

Operates on VAE latent sequences x (B, t, D). Conditioning:
  - transcript words with frame timing (word-level frozen-T5 tokens, cross-attn)
  - action caption (pooled frozen-T5, one memory token with type embedding)
  - emotion id, speaker id, intensity scalar  -> adaLN (DiT-style)
  - motion prefix (first `prefix_frames` latent frames) -> inpainting conditioning
Each text slot independently droppable (classifier-free guidance at sampling).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    latent_dim: int = 1024        # VAE total (parts * per-part latent)
    dim: int = 512
    depth: int = 8
    heads: int = 8
    text_dim: int = 768           # T5-base hidden
    emotions: int = 8
    speakers: int = 32
    prefix_frames: int = 1        # in LATENT frames (4 motion frames / 4x downsample)
    cond_dropout: float = 0.15


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        # adaLN-Zero: 3 gates + 3 (scale, shift) pairs
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond, memory, memory_mask):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(cond)[:, None, :].chunk(9, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        if memory is not None:
            h = self.norm_x(x) * (1 + s2) + b2
            x = x + g2 * self.cross(h, memory, memory,
                                    key_padding_mask=memory_mask, need_weights=False)[0]
        h = self.norm2(x) * (1 + s3) + b3
        x = x + g3 * self.mlp(h)
        return x


class LatentFlowDiT(nn.Module):
    def __init__(self, cfg: FlowConfig | None = None):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        c = self.cfg

        self.in_proj = nn.Linear(c.latent_dim, c.dim)
        self.out_proj = nn.Linear(c.dim, c.latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.pos_emb = nn.Parameter(torch.randn(1, 512, c.dim) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(c.dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.emotion_emb = nn.Embedding(c.emotions + 1, c.dim)      # +1 = dropped/unknown
        self.speaker_emb = nn.Embedding(c.speakers + 1, c.dim)
        self.intensity_mlp = nn.Sequential(nn.Linear(1, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.prefix_flag = nn.Parameter(torch.randn(1, 1, c.dim) * 0.02)

        self.text_proj = nn.Linear(c.text_dim, c.dim)
        self.slot_type = nn.Embedding(2, c.dim)                     # 0=transcript token, 1=action token
        self.word_time_mlp = nn.Sequential(nn.Linear(1, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))

        self.blocks = nn.ModuleList([DiTBlock(c.dim, c.heads) for _ in range(c.depth)])
        self.final_norm = nn.LayerNorm(c.dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, c.dim * 2))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)

    def build_memory(self, transcript_emb, transcript_times, transcript_mask, action_emb):
        """transcript_emb (B,N,text_dim) word/token embeddings, transcript_times (B,N) in [0,1],
        transcript_mask (B,N) True=pad, action_emb (B,text_dim)|None -> memory, key_padding_mask."""
        mems, masks = [], []
        if transcript_emb is not None:
            m = self.text_proj(transcript_emb) + self.slot_type.weight[0][None, None]
            m = m + self.word_time_mlp(transcript_times[..., None])
            mems.append(m)
            masks.append(transcript_mask)
        if action_emb is not None:
            a = self.text_proj(action_emb)[:, None, :] + self.slot_type.weight[1][None, None]
            mems.append(a)
            masks.append(torch.zeros(a.shape[0], 1, dtype=torch.bool, device=a.device))
        if not mems:
            return None, None
        return torch.cat(mems, dim=1), torch.cat(masks, dim=1)

    def forward(self, x_t, t, emotion, speaker, intensity,
                transcript_emb=None, transcript_times=None, transcript_mask=None,
                action_emb=None, prefix=None):
        """x_t (B,T,D) noisy latents; t (B,) flow time in [0,1]; prefix (B,P,D) clean or None.
        Returns velocity prediction (B,T,D)."""
        B, T, _ = x_t.shape
        cond = self.time_mlp(timestep_embedding(t, self.cfg.dim))
        cond = cond + self.emotion_emb(emotion) + self.speaker_emb(speaker)
        cond = cond + self.intensity_mlp(intensity[:, None])

        h = self.in_proj(x_t) + self.pos_emb[:, :T]
        if prefix is not None:
            P = prefix.shape[1]
            h[:, :P] = self.in_proj(prefix) + self.pos_emb[:, :P] + self.prefix_flag

        memory, memory_mask = self.build_memory(transcript_emb, transcript_times, transcript_mask, action_emb)
        for blk in self.blocks:
            h = blk(h, cond, memory, memory_mask)

        s, b = self.final_ada(cond)[:, None, :].chunk(2, dim=-1)
        return self.out_proj(self.final_norm(h) * (1 + s) + b)


# ── flow matching training / sampling ──────────────────────────────────────

def flow_loss(model: LatentFlowDiT, x1: torch.Tensor, **cond) -> torch.Tensor:
    """Rectified-flow objective: xt=(1-t)x0+t x1, target v = x1-x0."""
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    v = model(xt, t, **cond)
    target = x1 - x0
    prefix = cond.get("prefix")
    if prefix is not None:  # don't train velocity on clamped prefix positions
        P = prefix.shape[1]
        v, target = v[:, P:], target[:, P:]
    return F.mse_loss(v, target)


@torch.no_grad()
def sample(model: LatentFlowDiT, shape, emotion, speaker, intensity,
           transcript=None, action=None, prefix=None,
           steps: int = 8, g_transcript: float = 2.0, g_action: float = 3.0,
           device: str = "cuda") -> torch.Tensor:
    """Euler sampler with compositional per-slot CFG.
    transcript = (emb, times, mask) or None; action = emb or None."""
    B = emotion.shape[0]
    x = torch.randn(B, *shape, device=device)

    def v_fn(x, t, use_transcript, use_action):
        kw = dict(emotion=emotion, speaker=speaker, intensity=intensity, prefix=prefix)
        if use_transcript and transcript is not None:
            kw.update(transcript_emb=transcript[0], transcript_times=transcript[1],
                      transcript_mask=transcript[2])
        if use_action and action is not None:
            kw.update(action_emb=action)
        return model(x, t, **kw)

    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        v_uncond = v_fn(x, t, False, False)
        v = v_uncond
        if transcript is not None:
            v = v + g_transcript * (v_fn(x, t, True, False) - v_uncond)
        if action is not None:
            v = v + g_action * (v_fn(x, t, False, True) - v_uncond)
        x = x + v / steps
        if prefix is not None:  # clamp prefix latents to their clean values
            x[:, :prefix.shape[1]] = prefix
    return x
