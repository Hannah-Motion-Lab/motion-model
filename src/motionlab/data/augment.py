"""Motion augmentation. Operates on (T,165) axis-angle poses + (T,3) trans tensors."""
import torch

from ..rotations import axis_angle_to_matrix, matrix_to_axis_angle

# SMPL-X left<->right joint swaps (55-joint order)
_SWAP = {
    1: 2, 4: 5, 7: 8, 10: 11, 13: 14, 16: 17, 18: 19, 20: 21, 23: 24,
}
_SWAP.update({v: k for k, v in _SWAP.items()})
# hands: left 25-39 <-> right 40-54
for k in range(15):
    _SWAP[25 + k] = 40 + k
    _SWAP[40 + k] = 25 + k

_PERM = [_SWAP.get(j, j) for j in range(55)]


def mirror(poses: torch.Tensor, trans: torch.Tensor):
    """Mirror motion across the sagittal (x=0) plane."""
    aa = poses.reshape(*poses.shape[:-1], 55, 3)[..., _PERM, :].clone()
    aa[..., 1] *= -1  # negate y and z rotation components
    aa[..., 2] *= -1
    t = trans.clone()
    t[..., 0] *= -1
    return aa.reshape(*poses.shape[:-1], 165), t


def tempo_jitter(poses: torch.Tensor, trans: torch.Tensor, factor: float):
    """Resample time by `factor` (0.8-1.25 sensible), keeping the original length
    by cropping/padding at the end. Linear interp on axis-angle is acceptable for
    small factors at 30fps."""
    T = poses.shape[0]
    src = torch.clamp(torch.arange(T, dtype=torch.float32) * factor, max=T - 1)
    i0 = src.floor().long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    w = (src - i0.float()).unsqueeze(-1)
    return (poses[i0] * (1 - w) + poses[i1] * w,
            trans[i0] * (1 - w) + trans[i1] * w)


def exaggerate(poses: torch.Tensor, scale: float):
    """Amplitude-scale gestures around the clip's mean pose (the 'otaku dial').

    Scaling the raw axis-angle (deviation from REST) folds the arms into the body
    at high scale and collapses to a stiff T-pose at low scale. Instead we scale
    the deviation from each joint's TEMPORAL MEAN: the natural standing/talking
    posture is preserved and only the *movement amplitude* grows or shrinks, so
    low intensity still looks alive and high intensity doesn't clip through the body.
    Root orient (joint 0) is left untouched. Expects a time axis (T, 165)."""
    aa = poses.reshape(*poses.shape[:-1], 55, 3).clone()
    t_axis = aa.dim() - 3                       # the frame dimension
    mean = aa.mean(dim=t_axis, keepdim=True)
    scaled = mean[..., 1:, :] + (aa[..., 1:, :] - mean[..., 1:, :]) * scale
    norms = scaled.norm(dim=-1, keepdim=True)   # keep angles on the human manifold
    scaled = torch.where(norms > 3.1, scaled / norms.clamp(min=1e-8) * 3.1, scaled)
    aa[..., 1:, :] = scaled
    return aa.reshape(*poses.shape[:-1], 165)
