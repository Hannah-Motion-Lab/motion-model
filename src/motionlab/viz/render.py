"""Render SMPL-X motion (poses T×165 + trans T×3) to an mp4 with pyrender.

Headless: uses EGL if available, else OSMesa. Simplified from PantoMatrix's
fast_render (single process, single sequence — fine for eyeballing)."""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import torch
import trimesh

from ..models.losses import get_smplx


@torch.no_grad()
def render_motion(poses: np.ndarray, trans: np.ndarray, out_path: str,
                  fps: int = 30, size: tuple = (720, 720), model_dir: str = "assets",
                  every: int = 1, device: str = "cuda"):
    """poses (T,165) axis-angle, trans (T,3) -> out_path mp4."""
    import pyrender

    model = get_smplx(model_dir, device)
    T = poses.shape[0]
    poses_t = torch.from_numpy(poses.astype(np.float32)).to(device)
    trans_t = torch.from_numpy(trans.astype(np.float32)).to(device)

    zeros = torch.zeros(T, 100, device=device)
    betas = torch.zeros(T, 300, device=device)
    out = model(
        betas=betas, transl=trans_t, expression=zeros,
        global_orient=poses_t[:, :3], body_pose=poses_t[:, 3:66],
        left_hand_pose=poses_t[:, 66:111], right_hand_pose=poses_t[:, 111:156],
        jaw_pose=poses_t[:, 156:159], leye_pose=poses_t[:, 159:162],
        reye_pose=poses_t[:, 162:165],
    )
    verts = out.vertices.cpu().numpy()
    faces = model.faces

    center = verts[0].mean(axis=0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = center + np.array([0.0, 0.2, 2.6])

    renderer = pyrender.OffscreenRenderer(*size)
    frames = []
    for i in range(0, T, every):
        mesh = trimesh.Trimesh(verts[i], faces, process=False)
        mesh.visual.vertex_colors = [180, 160, 150, 255]
        scene = pyrender.Scene(bg_color=[18, 22, 34, 255], ambient_light=[0.25] * 3)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
        scene.add(cam, pose=cam_pose)
        light = pyrender.DirectionalLight(intensity=3.0)
        scene.add(light, pose=cam_pose)
        color, _ = renderer.render(scene)
        frames.append(color)
    renderer.delete()

    imageio.mimwrite(out_path, frames, fps=fps // every, quality=7)
    return out_path
