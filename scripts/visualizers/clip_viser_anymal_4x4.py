#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clip 2 — Viser streaming viewport, AnymalD 4×4 grid (10 s @ 15 fps).

16 AnymalD environments, streaming_envs=16 → auto-layout gives a 4×4 grid
of RGB | depth cells (4 env-cols × 4 env-rows × 2 GT-cols = 32 cells).
Captures the streaming composite (what Viser shows as its background image),
letterboxed to 16:9.

Output::

    docs / source / _static / visualizers / streaming / clip_viser_anymal_4x4.mp4

Usage::

    uv run python scripts/visualizers/clip_viser_anymal_4x4.py
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402

from isaaclab.envs.utils.camera_colorizer import CameraFrameColorizer, sensor_key_for_gt_type  # noqa: E402
from isaaclab.envs.utils.camera_view import (  # noqa: E402
    apply_camera_target_positions,
    camera_gt_batch,
    compose_streaming_grid,
    create_visualizer_camera,
    prim_world_positions,
)
from isaaclab.sim import get_current_stage  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

OUT_DIR = "docs/source/_static/visualizers/streaming"
OUT_VIDEO = os.path.join(OUT_DIR, "clip_viser_anymal_4x4.mp4")
GT_TYPES = ["rgb"]
DEPTH_MIN, DEPTH_MAX = 0.5, 8.0
N_ENVS = 16
WARMUP_STEPS = 150
RECORD_FRAMES = 150
VIDEO_FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)

env_cfg = AnymalDRoughEnvCfg()
env_cfg.scene.num_envs = 16  # more envs so we can sample 16 varied ones
env_cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp

env = gym.make("Isaac-Velocity-Rough-AnymalD", cfg=env_cfg, render_mode=None)
env.reset()
num_envs = env.unwrapped.num_envs
print(f"[OK] AnymalD — {num_envs} envs, streaming 16 in 4×4 grid")

# Streaming camera: 16 envs, 2 GT types → 4 env-cols × 4 env-rows × 2 GT-cols
from isaaclab.envs.utils.camera_colorizer import sensor_keys_for_gt_types  # noqa: E402

cam, _ = create_visualizer_camera(
    num_envs=num_envs,
    width=240,
    height=180,
    renderer_cfg=NewtonWarpRendererCfg(),
    data_types=sensor_keys_for_gt_types(GT_TYPES),
)

# Sample 16 evenly-spaced env indices for a clean 4×4 grid
stride = max(1, num_envs // N_ENVS)
env_ids = list(range(0, min(N_ENVS * stride, num_envs), stride))[:N_ENVS]
print(f"  env_ids={env_ids}")

stage = get_current_stage()
scene = env.unwrapped.scene
target_positions = prim_world_positions(stage, "/World/envs/*/Robot", env_ids, scene=scene)
apply_camera_target_positions(cam, target_positions, (0.8, -0.8, 0.8), env_ids)

available: frozenset | None = None


def _composite() -> np.ndarray:
    cam.update(dt=0.0, force_recompute=True)
    global available
    if available is None:
        available = frozenset(cam.data.output.keys())
    frames = []
    for env_idx in env_ids:
        for gt in GT_TYPES:
            key = sensor_key_for_gt_type(gt, available)
            raw = camera_gt_batch(cam, [env_idx], key)[0]
            frames.append(CameraFrameColorizer.colorize(raw, gt, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX))
    composite = compose_streaming_grid(frames, len(env_ids), len(GT_TYPES))
    # Letterbox to 16:9 (matches Viser's canvas)
    h, w = composite.shape[:2]
    target_w = max(w, int(h * 16 / 9))
    target_h = max(h, int(w * 9 / 16))
    if target_w != w or target_h != h:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0, x0 = (target_h - h) // 2, (target_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = composite
        composite = canvas
    return composite


print(f"[OK] Warming up {WARMUP_STEPS} steps ...")
for i in range(WARMUP_STEPS):
    env.step(torch.tensor(env.action_space.sample(), device="cuda:0"))
    if (i + 1) % 100 == 0:
        print(f"  warm-up {i + 1}/{WARMUP_STEPS}")

print(f"[OK] Recording {RECORD_FRAMES} frames → {OUT_VIDEO}")
writer = imageio.get_writer(OUT_VIDEO, fps=VIDEO_FPS, codec="libx264", quality=8)

try:
    for step in range(RECORD_FRAMES):
        env.step(torch.tensor(env.action_space.sample(), device="cuda:0"))
        writer.append_data(_composite())
        if (step + 1) % 50 == 0:
            print(f"  frame {step + 1}/{RECORD_FRAMES}")
finally:
    writer.close()

env.close()
simulation_app.close()
print(f"[DONE] {OUT_VIDEO}")
