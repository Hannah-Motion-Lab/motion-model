"""Rotation representation utilities (torch). Standard 6D-rotation recipe
(Zhou et al. 2019), matching the conventions used across the gesture literature."""
import torch
import torch.nn.functional as F


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix (Rodrigues).
    Computed in fp32: trig + normalization overflow/lose precision in fp16 autocast."""
    aa = aa.float()
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * K + (1 - cos) * (K @ K)


def matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two rows, flattened."""
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. fp32 for numerical stability."""
    r6 = r6.float()
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_axis_angle(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle. fp32: acos is edge-unstable in fp16."""
    m = m.float()
    # rotation angle
    cos = ((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos)
    # rotation axis from the skew-symmetric part
    axis = torch.stack([
        m[..., 2, 1] - m[..., 1, 2],
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 1, 0] - m[..., 0, 1],
    ], dim=-1)
    axis = axis / (2 * torch.sin(angle)[..., None]).clamp(min=1e-8)
    return axis * angle[..., None]


def axis_angle_to_rotation_6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(r6: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(rotation_6d_to_matrix(r6))
