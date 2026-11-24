"""M0 environment verification: MuJoCo sim + offscreen render, HDF5 I/O, CUDA step."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

XML = """
<mujoco>
  <visual><global offwidth="96" offheight="96"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <camera name="cam" pos="0 -0.8 0.6" xyaxes="1 0 0 0 0.6 0.8"/>
    <body name="cube" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def check_mujoco() -> None:
    import mujoco

    print(f"mujoco {mujoco.__version__}")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(1000):  # 2 s of simulated time: the cube should fall and rest
        mujoco.mj_step(model, data)
    z = data.qpos[2]
    assert abs(z - 0.02) < 5e-3, f"cube not resting on plane, z={z}"
    with mujoco.Renderer(model, height=96, width=96) as renderer:
        renderer.update_scene(data, camera="cam")
        img = renderer.render()
    assert img.shape == (96, 96, 3) and img.dtype == np.uint8
    assert img.std() > 1.0, "rendered image looks blank"
    
    cube_px = ((img[..., 0] > 150) & (img[..., 1] < 90) & (img[..., 2] < 90)).sum()
    assert cube_px > 3, "cube not visible"
    print(f"  sim OK (cube z={z:.4f} m), render OK {img.shape}, cube pixels={cube_px}")

def check_h5() -> None:
    import h5py

    arr = np.random.default_rng(0).integers(0, 255, (10, 96, 96, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=arr, compression="gzip")
        with h5py.File(path, "r") as f:
            assert np.array_equal(f["images"][:], arr)
    print("  hdf5 OK")


def check_cuda() -> None:
    import torch
    import torch.nn as nn

    print(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    net = nn.Sequential(
        nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU(),
        nn.Flatten(), nn.LazyLinear(4),
    ).to(dev)
    x = torch.randn(128, 3, 96, 96, device=dev)
    net(x)  # materialize the lazy layer
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = net(x).float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  train step OK (AMP, batch 128), peak VRAM {peak:.0f} MiB")


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    check_mujoco()
    check_h5()
    check_cuda()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())