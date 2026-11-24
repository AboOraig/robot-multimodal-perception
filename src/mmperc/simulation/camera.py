"""Pinhole projection of world points through a MuJoCo camera (used for calibration tests and
by the classical vision baseline)."""
from __future__ import annotations

import mujoco
import numpy as np


def intrinsics(fovy_deg: float, height: int, width: int) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) in pixels. MuJoCo uses square pixels: fx = fy = (h/2) / tan(fovy/2)."""
    f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2.0)
    return f, f, 0.5 * width, 0.5 * height


def project(model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, points_w: np.ndarray,
            height: int, width: int) -> np.ndarray:
    """Project world points (N,3) to pixel coordinates (N,2) as (u right, v down).

    MuJoCo cameras look along their own -z axis, with +x right and +y up, so a point with camera
    coordinates (x, y, z) lies at depth -z and maps to u = cx + f x/(-z), v = cy - f y/(-z).
    """
    cid = model.camera(cam_name).id
    pos = data.cam_xpos[cid]
    R = data.cam_xmat[cid].reshape(3, 3)        # columns: camera axes in world coordinates
    pc = (np.atleast_2d(points_w) - pos) @ R    # world -> camera coordinates
    fx, fy, cx, cy = intrinsics(float(model.cam_fovy[cid]), height, width)
    depth = -pc[:, 2]
    return np.stack([cx + fx * pc[:, 0] / depth, cy - fy * pc[:, 1] / depth], axis=1)
