"""Stage A: part-wise motion VAE.

Input:  poses (B,T,165) axis-angle + trans (B,T,3)
Split into parts (upper/hands/lower/face; config-driven), each encoded by a
temporal-conv VAE into continuous latents at T/4 resolution. `trans` rides with
the lower-body part. Decoding reassembles full (B,T,165)+(B,T,3).

Continuous latents (no VQ) per the flow-matching lineage; the flow model
(Stage B) operates on the concatenation of part latents.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..rotations import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle

DEFAULT_PARTS = {
    "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "hands": list(range(25, 55)),
    "lower": [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "face": [22, 23, 24],
}


@dataclass
class VAEConfig:
    parts: dict = field(default_factory=lambda: dict(DEFAULT_PARTS))
    latent_dim: int = 256
    channels: int = 512
    down_t: int = 2          # number of stride-2 downsamples: T -> T / 2**down_t
    kl_weight: float = 1e-4


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class _PartEncoder(nn.Module):
    def __init__(self, in_dim, ch, latent_dim, down_t):
        super().__init__()
        layers = [nn.Conv1d(in_dim, ch, 3, padding=1), nn.SiLU()]
        for _ in range(down_t):
            layers += [nn.Conv1d(ch, ch, 4, stride=2, padding=1), nn.SiLU(), _ResBlock(ch)]
        self.net = nn.Sequential(*layers)
        self.to_mu = nn.Conv1d(ch, latent_dim, 1)
        self.to_logvar = nn.Conv1d(ch, latent_dim, 1)

    def forward(self, x):           # x: (B, C_in, T)
        h = self.net(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10, 10)


class _PartDecoder(nn.Module):
    def __init__(self, out_dim, ch, latent_dim, down_t):
        super().__init__()
        layers = [nn.Conv1d(latent_dim, ch, 3, padding=1), nn.SiLU()]
        for _ in range(down_t):
            layers += [nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1), nn.SiLU(), _ResBlock(ch)]
        layers += [nn.Conv1d(ch, out_dim, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):           # z: (B, latent, T/4)
        return self.net(z)


class PartwiseMotionVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.part_names = list(self.cfg.parts.keys())
        self.part_joints = {k: list(v) for k, v in self.cfg.parts.items()}

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name in self.part_names:
            dim = len(self.part_joints[name]) * 6 + (3 if name == "lower" else 0)
            self.encoders[name] = _PartEncoder(dim, self.cfg.channels, self.cfg.latent_dim, self.cfg.down_t)
            self.decoders[name] = _PartDecoder(dim, self.cfg.channels, self.cfg.latent_dim, self.cfg.down_t)

    # ── representation helpers ────────────────────────────────────────────
    def split_parts(self, poses: torch.Tensor, trans: torch.Tensor) -> dict:
        """poses (B,T,165), trans (B,T,3) -> {part: (B, C, T)}"""
        B, T, _ = poses.shape
        r6 = axis_angle_to_rotation_6d(poses.reshape(B, T, 55, 3))     # (B,T,55,6)
        out = {}
        for name in self.part_names:
            feat = r6[:, :, self.part_joints[name], :].reshape(B, T, -1)
            if name == "lower":
                feat = torch.cat([feat, trans], dim=-1)
            out[name] = feat.transpose(1, 2)                            # (B,C,T)
        return out

    def join_parts(self, part_feats: dict, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """{part: (B,C,T)} -> poses (B,T,165), trans (B,T,3)"""
        B = next(iter(part_feats.values())).shape[0]
        r6 = torch.zeros(B, T, 55, 6, device=next(iter(part_feats.values())).device,
                         dtype=next(iter(part_feats.values())).dtype)
        r6[..., 0] = 1.0  # identity-ish init for any joint not covered
        r6[..., 4] = 1.0
        trans = torch.zeros(B, T, 3, device=r6.device, dtype=r6.dtype)
        for name in self.part_names:
            feat = part_feats[name].transpose(1, 2)                     # (B,T,C)
            joints = self.part_joints[name]
            if name == "lower":
                trans = feat[..., -3:]
                feat = feat[..., :-3]
            r6[:, :, joints, :] = feat.reshape(B, T, len(joints), 6)
        poses = rotation_6d_to_axis_angle(r6).reshape(B, T, 165)
        return poses, trans

    # ── VAE interface ─────────────────────────────────────────────────────
    def encode(self, poses, trans, sample: bool = True):
        feats = self.split_parts(poses, trans)
        zs, mus, logvars = {}, {}, {}
        for name in self.part_names:
            mu, logvar = self.encoders[name](feats[name])
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
            zs[name], mus[name], logvars[name] = z, mu, logvar
        return zs, mus, logvars

    def decode(self, zs: dict, T: int):
        part_feats = {name: self.decoders[name](zs[name]) for name in self.part_names}
        return self.join_parts(part_feats, T)

    def forward(self, poses, trans):
        zs, mus, logvars = self.encode(poses, trans)
        rec_poses, rec_trans = self.decode(zs, poses.shape[1])
        kl = sum(
            (-0.5 * (1 + logvars[n] - mus[n].pow(2) - logvars[n].exp())).mean()
            for n in self.part_names
        ) / len(self.part_names)
        return {"rec_poses": rec_poses, "rec_trans": rec_trans, "kl": kl, "zs": zs}

    # latent layout used by the flow model
    @property
    def total_latent_dim(self):
        return self.cfg.latent_dim * len(self.part_names)

    def latents_to_tensor(self, zs: dict) -> torch.Tensor:
        """{part: (B,L,t)} -> (B, t, L*parts) sequence-first for the transformer."""
        return torch.cat([zs[n] for n in self.part_names], dim=1).transpose(1, 2)

    def tensor_to_latents(self, x: torch.Tensor) -> dict:
        """(B, t, L*parts) -> {part: (B,L,t)}"""
        x = x.transpose(1, 2)
        L = self.cfg.latent_dim
        return {n: x[:, i * L:(i + 1) * L, :] for i, n in enumerate(self.part_names)}


def load_vae(ckpt_path: str, device: str) -> PartwiseMotionVAE:
    """The frozen VAE from a checkpoint. Lives next to the model it builds so that serving
    (pipeline.py) never imports the training stack (omegaconf, tensorboard, the loaders).
    weights_only=True: the checkpoint is plain tensors + a dict cfg, never code."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vcfg = ckpt["cfg"]["model"]
    vae = PartwiseMotionVAE(VAEConfig(
        parts=vcfg["parts"], latent_dim=vcfg["latent_dim"],
        channels=vcfg["channels"], down_t=vcfg["down_t"],
    )).to(device).eval()
    vae.load_state_dict(ckpt["model"])
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae
