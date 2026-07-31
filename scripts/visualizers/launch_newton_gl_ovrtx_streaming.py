#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Newton GL visualizer with OVRTX streaming panel.

Shows the AnymalD rough env in the Newton GL interactive window while
streaming RGB | depth | segmentation from an OVRTX-rendered camera into
the HUD image panel.

Usage::

    uv run python scripts/visualizers/launch_newton_gl_ovrtx_streaming.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

env_cfg = AnymalDRoughEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp

# Newton GL interactive viewer + OVRTX streaming panel
env_cfg.sim.visualizer_cfgs = [
    NewtonGLVisualizerCfg(
        streaming_view=True,
        streaming_envs=3,
        streaming_gt_types=["rgb", "depth", "segmentation"],
        streaming_depth_min=0.5,
        streaming_depth_max=10.0,
        streaming_cam_renderer="ovrtx",  # falls back to newton_warp if OVRTX SDK unavailable
    )
]

env = gym.make("Isaac-Velocity-Rough-AnymalD", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()
print("[OK] Environment ready — Newton GL + OVRTX streaming panel active")

visualizers = getattr(env.unwrapped.sim, "_visualizers", [])
print(f"[OK] Visualizers: {[type(v).__name__ for v in visualizers]}")
print("[INFO] Running — close the Newton GL window or press Ctrl-C to stop.")

try:
    while True:
        action = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated.any() or truncated.any():
            obs, _ = env.reset()
except (KeyboardInterrupt, SystemExit):
    print("[INFO] Stopping.")

env.close()
simulation_app.close()
