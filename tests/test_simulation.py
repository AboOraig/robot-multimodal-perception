"""Simulation tests: camera calibration, wrench sign/scale, determinism, synchronisation."""
import mujoco
import numpy as np
import pytest

from mmperc.common.config import load_config
from mmperc.simulation.camera import project
from mmperc.simulation.episode import run_episode
from mmperc.simulation.kinematics import ArmIK
from mmperc.simulation.scene import CUBE_JOINT, build_model


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/simulation.yaml")


@pytest.fixture(scope="module")
def model(cfg):
    return build_model(cfg)


@pytest.fixture(scope="module")
def ik(model):
    return ArmIK(model)


@pytest.fixture(scope="module")
def ep(cfg, model, ik):
    return run_episode(cfg, 3, model=model, ik=ik, render=False)


def test_camera_projection_matches_render(cfg, model):
    """Projected cube centre must coincide with the centroid of red pixels in the rendered image."""
    data = mujoco.MjData(model)
    size = cfg["image_size"]
    cj = model.joint(CUBE_JOINT)
    for (x, y) in [(0.45, -0.15), (0.55, 0.12), (0.5, 0.0)]:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = cfg["robot"]["home_qpos"]
        data.qpos[7:9] = 0.04
        a = int(cj.qposadr[0])
        data.qpos[a:a + 3] = [x, y, 0.02]
        data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, size, size) as r:
            r.update_scene(data, camera=cfg["camera"]["name"])
            img = r.render().astype(int)
        red = (img[..., 0] > 150) & (img[..., 1] < 90) & (img[..., 2] < 90)
        assert red.sum() > 20, "cube not visible"
        vs, us = np.nonzero(red)
        uv = project(model, data, cfg["camera"]["name"], np.array([[x, y, 0.02]]), size, size)[0]
        # centroid of visible pixels is biased by the cube's visible top/side faces: allow 2 px
        assert abs(us.mean() + 0.5 - uv[0]) < 2.0 and abs(vs.mean() + 0.5 - uv[1]) < 2.0, (uv, us.mean(), vs.mean())


def test_determinism(cfg, model, ik, ep):
    again = run_episode(cfg, 3, model=model, ik=ik, render=False)
    for k in ("q", "wrench", "cube_pos", "tcp_pos"):
        assert np.array_equal(ep[k], again[k]), k
    other = run_episode(cfg, 4, model=model, ik=ik, render=False)
    assert not np.array_equal(ep["cube_pos"], other["cube_pos"])


def test_time_grid_and_phases(cfg, ep):
    dt = np.diff(ep["t"])
    assert np.allclose(dt, 1.0 / cfg["log_hz"], atol=1e-9)
    assert np.all(np.diff(ep["phase"]) >= 0), "phases must be non-decreasing"
    assert set(ep["phase"]) == set(range(7))


def test_forward_kinematics_matches_logged_tcp(ik, ep):
    """Logged joint angles and TCP position come from the same instant (synchronisation check)."""
    for k in range(0, len(ep["t"]), 7):
        p, _ = ik.fk(ep["q"][k])
        assert np.linalg.norm(p - ep["tcp_pos"][k]) < 1e-6


def test_grasp_succeeds_and_cube_follows_hand(ep):
    assert ep["meta"]["success"]
    hold = ep["phase"] == 3
    assert ep["cube_pos"][hold, 2].min() > 0.02 + 0.08
    assert (ep["n_finger_contacts"][hold] == 2).all()


def test_wrench_scale_equals_cube_weight(cfg, ep):
    """Static wrench magnitude rises by the cube weight when the cube is held (sensor sign/scale)."""
    g = 9.81
    m = cfg["cube"]["mass"]
    hold = np.where(ep["phase"] == 3)[0][-5:]
    f_hold = np.linalg.norm(ep["wrench"][hold, :3], axis=1).mean()
    f_free = np.linalg.norm(ep["wrench"][0, :3])
    assert f_hold - f_free == pytest.approx(m * g, rel=0.15)
    assert 0.9 * (0.76) * g < f_free < 1.1 * 0.76 * g * 1.05   # hand + fingers weight


def test_cube_yaw_symmetry_representation(ep):
    phi = ep["cube_phi"]
    assert np.all(np.abs(phi) <= np.pi + 1e-9)
    assert np.allclose(np.sin(phi), np.sin(4 * ep["cube_yaw"]))
