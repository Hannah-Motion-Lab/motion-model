"""End-to-end generation pipeline: text/action -> SMPL-X motion.

Shared by demo/generate.py and serve/main.py. Loads the frozen VAE + flow
checkpoints once and generates windows autoregressively (prefix chaining) to
reach any requested duration.
"""
from dataclasses import dataclass

import numpy as np
import torch

import os

from .models.flow import LatentFlowDiT, FlowConfig, sample
from .models.text import encode_transcripts, encode_captions
from .postprocess import postprocess
from .models.vae import load_vae


@dataclass
class GenerationResult:
    poses: np.ndarray   # (T,165) axis-angle float32
    trans: np.ndarray   # (T,3)
    fps: int = 30


class MotionPipeline:
    def __init__(self, vae_ckpt: str, flow_ckpt: str, device: str = "cuda"):
        self.device = device
        self.vae = load_vae(vae_ckpt, device)
        ckpt = torch.load(flow_ckpt, map_location=device, weights_only=True)  # checkpoints are plain tensors + dict cfg; never unpickle code
        fcfg = ckpt["cfg"]["model"]
        self.window = ckpt["cfg"]["data"]["window"]
        self.fps = ckpt["cfg"]["data"].get("fps", 30)
        self.text_encoder = fcfg["text_encoder"]
        self.sample_cfg = ckpt["cfg"].get("sample", {})
        self.model = LatentFlowDiT(FlowConfig(
            latent_dim=self.vae.total_latent_dim,
            dim=fcfg["dim"], depth=fcfg["depth"], heads=fcfg["heads"],
            emotions=fcfg["emotions"], speakers=fcfg["speakers"],
            prefix_frames=max(1, fcfg["prefix_frames"] // (2 ** self.vae.cfg.down_t)),
        )).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.latent_frames = self.window // (2 ** self.vae.cfg.down_t)
        self.n_emotions = fcfg["emotions"]   # index == the null/unconditional emotion slot

    def _spread_words(self, text: str, n_frames: int):
        """Uniformly spread transcript words over n_frames (inference-time timing)."""
        words = [w for w in text.split() if w]
        if not words:
            return []
        per = n_frames / len(words)
        return [(int(i * per), int((i + 1) * per), w) for i, w in enumerate(words)]

    @torch.no_grad()
    def generate(self, text: str = "", action: str = "", duration_s: float = 3.0,
                 emotion: int = 0, speaker: int = 0, intensity: float = 1.0,
                 prefix_latent: torch.Tensor | None = None,
                 steps: int | None = None) -> tuple[GenerationResult, torch.Tensor]:
        """Returns (result, last_window_latent) — pass the latent back in as
        prefix_latent on the next call for seamless continuation."""
        total_frames = max(self.window, int(round(duration_s * self.fps)))
        n_windows = -(-total_frames // self.window)

        # Sampling knobs, env-tunable without restart-level code changes.
        # Lower guidance + more steps = poses stay on the human manifold
        # (high CFG was the main source of arm-through-body artifacts).
        steps = steps or int(os.getenv("SAMPLE_STEPS", "16"))
        g_t = float(os.getenv("GUIDANCE_TRANSCRIPT", "2.0"))
        g_a = float(os.getenv("GUIDANCE_ACTION", "3.5"))

        # Pure-action mode (action prompt, no speech) was trained with the null
        # emotion — pass a talking emotion here and she co-gestures like she's
        # speaking on top of the action. Match training: null emotion for actions.
        pure_action = bool(action) and not text
        emo = self.n_emotions if pure_action else emotion
        emotion_t = torch.tensor([emo], device=self.device)
        speaker_t = torch.tensor([speaker], device=self.device)
        intensity_t = torch.tensor([float(intensity)], device=self.device)

        action_emb = encode_captions([action], self.text_encoder, self.device) if action else None

        all_poses, all_trans = [], []
        # split transcript words across windows proportionally
        words_full = self._spread_words(text, total_frames)
        prev_latent = prefix_latent

        for wi in range(n_windows):
            w_start = wi * self.window
            words_win = [(s - w_start, e - w_start, w) for s, e, w in words_full
                         if e > w_start and s < w_start + self.window]
            words_win = [(max(0, s), min(self.window, e), w) for s, e, w in words_win]

            t_emb, t_times, t_mask = encode_transcripts(
                [words_win], window=self.window, model_name=self.text_encoder, device=self.device)

            prefix = None
            if prev_latent is not None:
                prefix = prev_latent[:, -self.model.cfg.prefix_frames:, :]

            x = sample(
                self.model, (self.latent_frames, self.vae.total_latent_dim),
                emotion=emotion_t, speaker=speaker_t, intensity=intensity_t,
                transcript=(t_emb, t_times, t_mask) if text else None,
                action=action_emb, prefix=prefix,
                steps=steps, g_transcript=g_t, g_action=g_a, device=self.device)

            zs = self.vae.tensor_to_latents(x)
            poses, trans = self.vae.decode(zs, self.window)
            all_poses.append(poses[0])
            all_trans.append(trans[0])
            prev_latent = x

        poses = torch.cat(all_poses)[:total_frames].float().cpu().numpy()
        trans = torch.cat(all_trans)[:total_frames].float().cpu().numpy()
        # An action prompt means she should be free to move (and use her fingers);
        # pure co-text means she's a standing talker → anchor legs, keep hands lively.
        is_action = bool(action)
        poses, trans = postprocess(
            poses, trans, fps=self.fps,
            smooth_window=int(os.getenv("SMOOTH_WINDOW", "5")),
            hand_blend=float(os.getenv("HAND_BLEND", "0.85" if is_action else "0.5")),
            trans_keep=float(os.getenv("TRANS_KEEP", "0.35")),
            talking=not is_action,
        )
        return GenerationResult(poses=poses, trans=trans, fps=self.fps), prev_latent
