"""MuJoCo scene construction: Franka Panda + table + one cube + one fixed RGB camera.

The Menagerie Panda model is loaded unmodified from disk; everything we add (table,
cube, camera, wrist force/torque sensor, TCP site) is attached programmatically with
`mujoco.MjSpec`, so the vendored asset folder stays pristine.

Frames and conventions
----------------------
* World W: origin at the robot base, x forward, z up, table surface = plane z = 0.
* Hand/sensor frame S ("hand" body, at the flange): z_h points along the fingers
  (down when grasping from above), y_h is the finger-closing axis, x_h = y_h x z_h.
* MuJoCo cameras look along their own -z axis, with +x to the image right and +y up.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mmperc.common.config import repo_root

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
ROBOT_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                "hand", "left_finger", "right_finger")

CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
CUBE_JOINT = "cube_free"
TABLE_GEOM = "table"
TCP_SITE = "tcp"
FT_SITE = "wrist_ft"


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo `xyaxes` (6 numbers) for a camera at `pos` looking at `target`.

    Camera x = image right, camera y = image up, camera z = -(viewing direction).
    """
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    f = target - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    return np.concatenate([right, cam_up])


def build_spec(cfg: dict[str, Any]) -> mujoco.MjSpec:
    """Assemble the scene as an (uncompiled) MjSpec."""
    robot_xml = repo_root() / cfg["robot"]["xml"]
    if not robot_xml.exists():
        raise FileNotFoundError(
            f"{robot_xml} not found. Fetch the Panda asset first: scripts/fetch_panda.ps1 "
            "(or see assets/robots/PROVENANCE.md)."
        )
    spec = mujoco.MjSpec.from_file(str(robot_xml))

    # --- physics options -------------------------------------------------------------
    phys = cfg["physics"]
    spec.option.timestep = float(phys["timestep"])
    spec.option.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC if phys["cone"] == "elliptic"
                        else mujoco.mjtCone.mjCONE_PYRAMIDAL)
    spec.option.impratio = float(phys["impratio"])

    # --- offscreen framebuffer must be at least as large as the rendered image --------
    size = int(cfg["image_size"])
    spec.visual.global_.offwidth = max(size, 640)
    spec.visual.global_.offheight = max(size, 480)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]

    # --- gripper strength (Menagerie default is weak for a 0.2 kg cube) ---------------
    scale = float(cfg["robot"]["grip_kp_scale"])
    grip = spec.actuator("actuator8")
    grip.gainprm[0] = grip.gainprm[0] * scale        # keeps ctrl=255 <-> 0.04 m per finger
    grip.biasprm[1] = grip.biasprm[1] * scale
    grip.forcerange = [-100.0 * max(scale, 1.0), 100.0 * max(scale, 1.0)]

    # --- hand: TCP site and wrist F/T site --------------------------------------------
    hand = spec.body("hand")
    hand.add_site(name=TCP_SITE, pos=[0.0, 0.0, float(cfg["robot"]["tcp_offset_z"])],
                  size=[0.005, 0.0, 0.0], rgba=[0, 1, 0, 0.0])
    hand.add_site(name=FT_SITE, pos=[0.0, 0.0, 0.0], size=[0.005, 0.0, 0.0], rgba=[0, 0, 1, 0.0])
    # Force first, torque second -> sensordata[adr:adr+6] = [Fx Fy Fz Tx Ty Tz] in the site frame
    for name, typ in (("ft_force", mujoco.mjtSensor.mjSENS_FORCE),
                      ("ft_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        spec.add_sensor(name=name, type=typ, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=FT_SITE)

    # --- table (plane z = 0) and light -------------------------------------------------
    world = spec.worldbody
    world.add_geom(name=TABLE_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05],
                   rgba=cfg["table"]["rgba"], friction=[1.0, 0.005, 0.0001])
    world.add_light(pos=[0.5, 0.0, 2.0], dir=[0.0, 0.0, -1.0], diffuse=[0.5, 0.5, 0.5])

    # --- cube: free body resting on the table ------------------------------------------
    c = cfg["cube"]
    half = 0.5 * float(c["size"])
    cube = world.add_body(name=CUBE_BODY, pos=[0.5, 0.0, half])
    cube.add_joint(name=CUBE_JOINT, type=mujoco.mjtJoint.mjJNT_FREE)
    cube.add_geom(name=CUBE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, half],
                  mass=float(c["mass"]), rgba=c["rgba"], friction=c["friction"], condim=4)

    # --- fixed RGB camera ---------------------------------------------------------------
    cam = cfg["camera"]
    world.add_camera(name=cam["name"], pos=cam["pos"], fovy=float(cam["fovy"]),
                     xyaxes=look_at_xyaxes(cam["pos"], cam["lookat"]).tolist())
    return spec


def build_model(cfg: dict[str, Any]) -> mujoco.MjModel:
    return build_spec(cfg).compile()
