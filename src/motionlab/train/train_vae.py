"""Stage A training: part-wise motion VAE on BEAT2 (+AMASS when present).

  python -m motionlab.train.train_vae --config configs/vae.yaml [--steps N] [--max-files N] [--no-fk]
"""
import argparse
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..data.beat2 import Beat2Dataset, collate
from ..models.vae import PartwiseMotionVAE, VAEConfig
from ..models.losses import vae_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-fk", action="store_true", help="skip smplx FK loss (fast smoke tests)")
    parser.add_argument("--out-dir", default=None, help="override cfg.train.out_dir (checkpoint versioning)")
    parser.add_argument("--batch", type=int, default=None, help="override cfg.train.batch_size")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.batch:
        cfg.train.batch_size = args.batch
    steps = args.steps or cfg.train.steps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Beat2Dataset(cfg.data.beat2_root, window=cfg.data.window, stride=cfg.data.stride,
                      split="train", require_textgrid=False, max_files=args.max_files)
    if len(ds) == 0:
        raise SystemExit("No BEAT2 windows found — is the download finished? (data/beat2)")

    from ..data.amass import AmassDataset
    from torch.utils.data import ConcatDataset
    from ..data.beat2 import Beat2Item

    class _MotionAsBeat(AmassDataset):
        """Adapter: emit Beat2Item-compatible objects for the shared collate."""
        def __getitem__(self, i):
            d = super().__getitem__(i)
            return Beat2Item(poses=d["poses"], trans=d["trans"], words=[],
                             emotion=0, speaker=0, name="extra", start_frame=0)

    # Extra motion sources for the codec (GRAB fingers + HumanML3D actions).
    # Pure-locomotion raw AMASS is left OUT to keep the latent space clean.
    parts = [ds]
    for d in cfg.data.get("extra_motion_dirs", []) or []:
        if Path(d).is_dir():
            ads = _MotionAsBeat(str(d), window=cfg.data.window,
                                stride=cfg.data.window, max_files=args.max_files)
            if len(ads) > 0:
                print(f"extra motion: +{len(ads)} windows from {d}")
                parts.append(ads)
    if len(parts) > 1:
        ds = ConcatDataset(parts)

    # Amplitude (exaggeration) + mirror augmentation so the codec covers the
    # range the intensity dial will ask for at inference.
    if cfg.train.get("augment", True):
        from ..data.augment import exaggerate, mirror

        class _Aug:
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                it = self.base[i]
                p, tr = it.poses, it.trans
                if torch.rand(1).item() < 0.5:
                    p, tr = mirror(p, tr)
                s = 0.7 + torch.rand(1).item() * 0.7          # [0.7, 1.4]
                p = exaggerate(p, s)
                return Beat2Item(poses=p, trans=tr, words=[], emotion=it.emotion,
                                 speaker=it.speaker, name=it.name, start_frame=it.start_frame)
        ds = _Aug(ds)

    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True, persistent_workers=True)
    print(f"dataset: {len(ds)} windows from {cfg.data.beat2_root}")

    model = PartwiseMotionVAE(VAEConfig(
        parts=OmegaConf.to_container(cfg.model.parts),
        latent_dim=cfg.model.latent_dim,
        channels=cfg.model.channels,
        down_t=cfg.model.down_t,
        kl_weight=cfg.model.kl_weight,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, cfg.train.warmup)))

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter  # lazy: serving imports load_vae from here without tensorboard
    writer = SummaryWriter(out_dir)

    step, t0, nan_streak = 0, time.time(), 0
    ckpt_path = out_dir / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", 0)
        for _ in range(min(step, cfg.train.warmup)):
            sched.step()
        print(f"resumed from {ckpt_path} at step {step}")
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            poses = batch["poses"].to(device, non_blocking=True)
            trans = batch["trans"].to(device, non_blocking=True)

            # bf16 autocast: same range as fp32 (no overflow-NaNs like fp16), no GradScaler needed
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
                out = model(poses, trans)
                losses = vae_losses(
                    poses, trans, out["rec_poses"], out["rec_trans"],
                    fk_weight=0.0 if args.no_fk else cfg.train.fk_loss_weight,
                    vel_weight=cfg.train.vel_loss_weight,
                    use_fk=not args.no_fk,
                )
                loss = losses["total"] + cfg.model.kl_weight * out["kl"]

            if not torch.isfinite(loss) or loss.item() > 1e4:
                nan_streak += 1
                opt.zero_grad(set_to_none=True)
                if nan_streak >= 50:
                    raise SystemExit(f"ABORT: {nan_streak} consecutive bad losses at step {step}")
                continue
            nan_streak = 0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % cfg.train.log_every == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"[{step}/{steps}] {msg} kl={out['kl'].item():.4f} "
                      f"({(time.time()-t0)/cfg.train.log_every:.2f}s/it-avg)")
                for k, v in losses.items():
                    writer.add_scalar(f"loss/{k}", v.item(), step)
                writer.add_scalar("loss/kl", out["kl"].item(), step)
                t0 = time.time()

            if step % cfg.train.ckpt_every == 0 or step == steps:
                torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "step": step},
                           out_dir / "latest.pt")

    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "step": step},
               out_dir / "latest.pt")
    print(f"done at step {step}; checkpoint: {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
