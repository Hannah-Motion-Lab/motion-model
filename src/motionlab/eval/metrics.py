"""Evaluation metrics.

- MPJPE / recon errors (self-contained).
- L1 diversity (self-contained).
- FGD: follows the emage_evaltools protocol (AESKConv feature extractor). The
  evaluator weights download via `emage_evaltools` from HuggingFace
  (H-Liu1997/emage_evaltools) — kept optional so the repo works without them.
"""
import numpy as np
import torch


def mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """(N,J,3) vs (N,J,3) mean per-joint position error (meters)."""
    return (pred_joints - gt_joints).norm(dim=-1).mean().item()


def l1_diversity(poses_batch: np.ndarray) -> float:
    """Mean pairwise L1 distance across a set of generated clips (N,T,165)."""
    N = poses_batch.shape[0]
    if N < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(N):
        for j in range(i + 1, N):
            total += np.abs(poses_batch[i] - poses_batch[j]).mean()
            count += 1
    return total / count


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    """Fréchet distance between two Gaussians (numerically robust sqrtm)."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))


def fgd_from_features(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """FGD given (N,D) feature matrices from the AESKConv extractor."""
    mu1, cov1 = feats_a.mean(0), np.cov(feats_a, rowvar=False)
    mu2, cov2 = feats_b.mean(0), np.cov(feats_b, rowvar=False)
    return frechet_distance(mu1, cov1, mu2, cov2)
