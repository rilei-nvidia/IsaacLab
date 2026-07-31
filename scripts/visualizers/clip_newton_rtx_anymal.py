#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clip 1 — Newton RTX full UI capture, AnymalD Rough (10 s @ 15 fps).

Captures the entire Newton RTX window including the path-traced viewport AND
the Isaac Lab ImGui HUD sidebar, using pyglet's framebuffer readback.
Warms up for 300 steps so OVRTX accumulates clean samples, then records 150 frames.

Output::

    docs / source / _static / visualizers / streaming / clip_newton_rtx_anymal.mp4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
import numpy as np
import torch  # noqa: E402
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

OUT_VIDEO = "docs/source/_static/visualizers/streaming/clip_newton_rtx_anymal.mp4"
WARMUP_STEPS = 300
RECORD_FRAMES = 150
VIDEO_FPS = 15

env_cfg = AnymalDRoughEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp
env_cfg.sim.visualizer_cfgs = [
    NewtonRTXVisualizerCfg(
        rtx_environment="studio",
        eye=(1.5, -1.5, 1.5),
    )
]

env = gym.make("Isaac-Velocity-Rough-AnymalD", cfg=env_cfg, render_mode=None)
env.reset()
viz = env.unwrapped.sim._visualizers[0] if env.unwrapped.sim._visualizers else None
rtx_viewer = getattr(viz, "_viewer", None) if viz else None
print("[OK] Newton RTX AnymalD — warming up ...")


def _capture_window() -> np.ndarray | None:
    """Read the full Newton RTX window framebuffer (scene + HUD) via glReadPixels."""
    if rtx_viewer is None:
        return None
    win = getattr(rtx_viewer, "_window", None)
    if win is None:
        return None
    try:
        import ctypes

        from pyglet.gl import GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels

        win.switch_to()
        w, h = win.width, win.height
        buf = (ctypes.c_ubyte * (w * h * 4))()
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(arr[::-1, :, :3])  # flip Y, drop alpha
    except Exception as exc:
        print(f"  [WARN] capture: {exc}")
        return None


for i in range(WARMUP_STEPS):
    env.step(torch.tensor(env.action_space.sample(), device="cuda:0"))
    if (i + 1) % 100 == 0:
        print(f"  warm-up {i + 1}/{WARMUP_STEPS}")

print(f"[OK] Recording {RECORD_FRAMES} frames → {OUT_VIDEO}")
writer = imageio.get_writer(OUT_VIDEO, fps=VIDEO_FPS, codec="libx264", quality=8)
try:
    for step in range(RECORD_FRAMES):
        env.step(torch.tensor(env.action_space.sample(), device="cuda:0"))
        frame = _capture_window()
        if frame is not None:
            writer.append_data(frame)
        if (step + 1) % 50 == 0:
            shape = frame.shape if frame is not None else "none"
            print(f"  frame {step + 1}/{RECORD_FRAMES}  shape={shape}")
finally:
    writer.close()

env.close()
simulation_app.close()
print(f"[DONE] {OUT_VIDEO}")
