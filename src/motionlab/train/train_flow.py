"""Stage B training: latent flow-matching DiT on BEAT2 (transcript slot) and,
when available, HumanML3D (action slot).

  python -m motionlab.train.train_flow --config configs/flow.yaml [--steps N] [--max-files N]
"""
import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data.beat2 import Beat2Dataset, collate
from ..models.vae import PartwiseMotionVAE, VAEConfig
from ..models.flow import LatentFlowDiT, FlowConfig, flow_loss
from ..models.text import encode_transcripts


def load_vae(ckpt_path: str, device: str) -> PartwiseMotionVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vcfg = ckpt["cfg"]["model"]
    vae = PartwiseMotionVAE(VAEConfig(
        parts=vcfg["parts"], latent_dim=vcfg["latent_dim"],
        channels=vcfg["channels"], down_t=vcfg["down_t"],
    )).to(device).eval()
    vae.load_state_dict(ckpt["model"])
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--vae", default=None, help="override cfg.vae_ckpt")
    parser.add_argument("--batch", type=int, default=None, help="override cfg.train.batch_size")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.batch:
        cfg.train.batch_size = args.batch
    if args.vae:
        cfg.vae_ckpt = args.vae
    steps = args.steps or cfg.train.steps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = load_vae(cfg.vae_ckpt, device)

    ds = Beat2Dataset(cfg.data.beat2_root, window=cfg.data.window, stride=cfg.data.stride,
                      split="train", require_textgrid=True, max_files=args.max_files)
    if len(ds) == 0:  # textgrids may not be downloaded yet — fall back for smoke tests
        ds = Beat2Dataset(cfg.data.beat2_root, window=cfg.data.window, stride=cfg.data.stride,
                          split="train", require_textgrid=False, max_files=args.max_files)
        print("WARNING: no textgrids found; training with empty transcripts (smoke mode)")
    if len(ds) == 0:
        raise SystemExit("No BEAT2 windows found")
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True, persistent_workers=True)
    print(f"dataset: {len(ds)} windows")

    # Captioned action batches (the ACTION slot): HumanML3D (body actions) +
    # GRAB (finger-articulated hand actions). Kept SEPARATE from the transcript
    # (speech) slot on purpose — action motion must never bleed into talking.
    from ..data.humanml3d import HumanML3DDataset
    from torch.utils.data import ConcatDataset as _Concat
    action_sets = []
    for key in ("humanml3d_root", "grab_root"):
        root = cfg.data.get(key)
        if root:
            ds_a = HumanML3DDataset(root, window=cfg.data.window)
            if len(ds_a) > 0:
                action_sets.append(ds_a)
                print(f"action data [{key}]: {len(ds_a)} captioned clips")
    hml_iter, hml_dl = None, None
    if action_sets:
        act_ds = action_sets[0] if len(action_sets) == 1 else _Concat(action_sets)
        hml_dl = DataLoader(act_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            num_workers=4, drop_last=True, persistent_workers=True)
        hml_iter = iter(hml_dl)
        print(f"ACTION slot ACTIVE: {len(act_ds)} total captioned clips")
    hml_ratio = float(cfg.train.get("hml_ratio", 0.35)) if hml_dl is not None else 0.0

    model = LatentFlowDiT(FlowConfig(
        latent_dim=vae.total_latent_dim,
        dim=cfg.model.dim, depth=cfg.model.depth, heads=cfg.model.heads,
        emotions=cfg.model.emotions, speakers=cfg.model.speakers,
        prefix_frames=max(1, cfg.model.prefix_frames // (2 ** vae.cfg.down_t)),
        cond_dropout=cfg.model.cond_dropout,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, cfg.train.warmup)))
    nan_streak = 0

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)

    from ..models.text import encode_captions

    def next_hml_batch():
        nonlocal hml_iter
        try:
            return next(hml_iter)
        except StopIteration:
            hml_iter = iter(hml_dl)
            return next(hml_iter)

    drop = cfg.model.cond_dropout
    step, t0 = 0, time.time()
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

            use_hml = hml_ratio > 0 and torch.rand(1).item() < hml_ratio
            action_emb = None
            if use_hml:
                hb = next_hml_batch()
                poses = hb["poses"].to(device, non_blocking=True)
                trans = hb["trans"].to(device, non_blocking=True)
                B = poses.shape[0]
                emotion = torch.full((B,), cfg.model.emotions, dtype=torch.long, device=device)
                speaker = torch.zeros(B, dtype=torch.long, device=device)
                words = [[] for _ in range(B)]
            else:
                poses = batch["poses"].to(device, non_blocking=True)
                trans = batch["trans"].to(device, non_blocking=True)
                emotion = batch["emotion"].to(device)
                speaker = batch["speaker"].clamp(max=cfg.model.speakers - 1).to(device)
                B = poses.shape[0]
                words = batch["words"]

            # ── Intensity/energy dial ──────────────────────────────────────
            # Sample a per-example amplitude in [0.5, 1.5] and exaggerate the
            # target motion by it, so the model learns intensity ↔ amplitude.
            # At inference this becomes the "energy" knob (calm ↔ energetic).
            intensity = torch.rand(B, device=device) * 1.0 + 0.5
            with torch.no_grad():
                aa = poses.reshape(B, poses.shape[1], 55, 3).clone()
                mean = aa.mean(dim=1, keepdim=True)                     # temporal mean pose
                s = intensity[:, None, None, None]
                # scale the deviation from the mean (amplitude), not the raw
                # angle from rest — preserves posture, avoids folding limbs in
                sc = mean[:, :, 1:, :] + (aa[:, :, 1:, :] - mean[:, :, 1:, :]) * s
                nrm = sc.norm(dim=-1, keepdim=True)
                sc = torch.where(nrm > 3.1, sc / nrm.clamp(min=1e-8) * 3.1, sc)
                aa[:, :, 1:, :] = sc
                poses = aa.reshape(B, poses.shape[1], 165)

            with torch.no_grad():
                zs, _, _ = vae.encode(poses, trans, sample=False)
                x1 = vae.latents_to_tensor(zs)                       # (B, t, D)
                t_emb, t_times, t_mask = encode_transcripts(
                    words, window=cfg.data.window,
                    model_name=cfg.model.text_encoder, device=device)
                if use_hml:
                    action_emb = encode_captions(hb["caption"], cfg.model.text_encoder, device)

            # per-slot CFG dropout
            if drop > 0:
                drop_transcript = torch.rand(B, device=device) < drop
                t_mask = t_mask.clone()
                t_mask[drop_transcript] = True
                t_mask[drop_transcript, 0] = False   # keep one (zeroed) token to avoid NaN
                t_emb = t_emb.clone()
                t_emb[drop_transcript] = 0
                emotion = torch.where(torch.rand(B, device=device) < drop,
                                      torch.full_like(emotion, cfg.model.emotions), emotion)
                if action_emb is not None:
                    action_emb = action_emb.clone()
                    action_emb[torch.rand(B, device=device) < drop] = 0

            prefix = x1[:, :model.cfg.prefix_frames].clone()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
                loss = flow_loss(model, x1,
                                 emotion=emotion, speaker=speaker, intensity=intensity,
                                 transcript_emb=t_emb, transcript_times=t_times,
                                 transcript_mask=t_mask, action_emb=action_emb, prefix=prefix)

            # Guard both NaN/inf AND finite explosions (flow loss is normally ~1;
            # a huge value = an out-of-distribution batch). Skip the step either way.
            if not torch.isfinite(loss) or loss.item() > 1e3:
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
                print(f"[{step}/{steps}] flow={loss.item():.4f} "
                      f"({(time.time()-t0)/cfg.train.log_every:.2f}s/it-avg)")
                writer.add_scalar("loss/flow", loss.item(), step)
                t0 = time.time()

            if step % cfg.train.ckpt_every == 0 or step == steps:
                torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg),
                            "step": step}, out_dir / "latest.pt")

    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "step": step},
               out_dir / "latest.pt")
    print(f"done at step {step}; checkpoint: {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
