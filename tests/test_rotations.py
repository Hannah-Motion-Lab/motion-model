import torch

from motionlab.rotations import (
    axis_angle_to_matrix, axis_angle_to_rotation_6d,
    rotation_6d_to_axis_angle, rotation_6d_to_matrix,
)


def test_roundtrip_axis_angle_6d():
    torch.manual_seed(0)
    aa = torch.randn(100, 3) * 0.8
    r6 = axis_angle_to_rotation_6d(aa)
    aa2 = rotation_6d_to_axis_angle(r6)
    # compare as matrices (axis-angle has sign/2pi ambiguities)
    m1, m2 = axis_angle_to_matrix(aa), axis_angle_to_matrix(aa2)
    assert torch.allclose(m1, m2, atol=1e-4)


def test_matrices_are_orthonormal():
    torch.manual_seed(1)
    r6 = torch.randn(50, 6)
    m = rotation_6d_to_matrix(r6)
    eye = torch.eye(3).expand(50, 3, 3)
    assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(m), torch.ones(50), atol=1e-5)


def test_zero_rotation():
    aa = torch.zeros(4, 3)
    m = axis_angle_to_matrix(aa)
    assert torch.allclose(m, torch.eye(3).expand(4, 3, 3), atol=1e-6)
