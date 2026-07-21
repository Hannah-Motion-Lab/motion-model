"""Losses for VAE training: 6D-rotation recon, FK joint-position recon,
velocity, KL (computed in the model). FK uses the SMPL-X body model on a
subsample of frames to keep it affordable."""
import torch
import torch.nn.functional as F

from ..rotations import axis_angle_to_rotation_6d

_smplx_model = None


def get_smplx(model_dir: str = "assets", device: str = "cuda"):
    global _smplx_model
    if _smplx_model is None:
        import smplx
        _smplx_model = smplx.create(
            model_dir, model_type="smplx", gender="NEUTRAL_2020",
            use_face_contour=False, num_betas=300, num_expression_coeffs=100,
            use_pca=False, ext="npz",
        ).to(device).eval()
        for p in _smplx_model.parameters():
            p.requires_grad_(False)
    return _smplx_model


def fk_positions(poses: torch.Tensor, trans: torch.Tensor, model_dir="assets") -> torch.Tensor:
    """poses (N,165), trans (N,3) -> joint positions (N,55,3)."""
    model = get_smplx(model_dir, poses.device)
    N = poses.shape[0]
    zeros = torch.zeros(N, 100, device=poses.device, dtype=poses.dtype)
    betas = torch.zeros(N, 300, device=poses.device, dtype=poses.dtype)
    out = model(
        betas=betas,
        transl=trans,
        expression=zeros,
        jaw_pose=poses[:, 156:159],
        global_orient=poses[:, :3],
        body_pose=poses[:, 3:66],
        left_hand_pose=poses[:, 66:111],
        right_hand_pose=poses[:, 111:156],
        leye_pose=poses[:, 159:162],
        reye_pose=poses[:, 162:165],
        return_joints=True,
    )
    return out.joints[:, :55]


def vae_losses(batch_poses, batch_trans, rec_poses, rec_trans,
               fk_frames: int = 8, fk_weight: float = 1.0, vel_weight: float = 1.0,
               model_dir: str = "assets", use_fk: bool = True):
    B, T, _ = batch_poses.shape

    r6_gt = axis_angle_to_rotation_6d(batch_poses.reshape(B, T, 55, 3))
    r6_rec = axis_angle_to_rotation_6d(rec_poses.reshape(B, T, 55, 3))
    loss_rot = F.mse_loss(r6_rec, r6_gt)
    loss_trans = F.mse_loss(rec_trans, batch_trans)

    vel_gt = r6_gt[:, 1:] - r6_gt[:, :-1]
    vel_rec = r6_rec[:, 1:] - r6_rec[:, :-1]
    loss_vel = F.mse_loss(vel_rec, vel_gt) + F.mse_loss(
        rec_trans[:, 1:] - rec_trans[:, :-1], batch_trans[:, 1:] - batch_trans[:, :-1])

    loss_fk = torch.tensor(0.0, device=batch_poses.device)
    if use_fk and fk_weight > 0:
        idx = torch.randint(0, T, (fk_frames,), device=batch_poses.device)
        gt_j = fk_positions(batch_poses[:, idx].reshape(-1, 165), batch_trans[:, idx].reshape(-1, 3), model_dir)
        rec_j = fk_positions(rec_poses[:, idx].reshape(-1, 165), rec_trans[:, idx].reshape(-1, 3), model_dir)
        loss_fk = F.mse_loss(rec_j, gt_j)

    total = loss_rot + loss_trans + vel_weight * loss_vel + fk_weight * loss_fk
    return {"total": total, "rot": loss_rot, "trans": loss_trans, "vel": loss_vel, "fk": loss_fk}
