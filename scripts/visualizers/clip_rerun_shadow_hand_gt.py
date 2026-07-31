#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clip 3 — Rerun streaming viewport, Shadow Hand 4×4 depth|seg|normals (10 s @ 15 fps).

4 Shadow Hand DexCube environments, GT types = depth | segmentation | normals
(no RGB), aimed at the cube.  This highlights the non-RGB ground-truth streaming
capabilities: turbo depth, cyan-cube segmentation, and XYZ-normals surface detail.

The composite (4 envs × 3 GTs → 1 col × 4 rows × 3 GT-cols) is what Rerun
shows as its primary full-screen Spatial2DView.

Output::

    docs / source / _static / visualizers / streaming / clip_rerun_shadow_hand_gt.mp4

Usage::

    uv run python scripts/visualizers/clip_rerun_shadow_hand_gt.py
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

from isaaclab.envs.utils.camera_colorizer import (  # noqa: E402
    CameraFrameColorizer,
    sensor_key_for_gt_type,
    sensor_keys_for_gt_types,
)
from isaaclab.envs.utils.camera_view import (  # noqa: E402
    apply_camera_target_positions,
    camera_gt_batch,
    compose_streaming_grid,
    create_visualizer_camera,
    prim_world_positions,
    resolve_streaming_envs,
)
from isaaclab.sim import get_current_stage  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import PhysicsCfg  # noqa: E402
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import (  # noqa: E402
    ShadowHandEnvCfg,
)

OUT_DIR = "docs/source/_static/visualizers/streaming"
OUT_VIDEO = os.path.join(OUT_DIR, "clip_rerun_shadow_hand_gt.mp4")
GT_TYPES = ["rgb", "depth", "segmentation", "normals"]  # full 4-channel: rgb, depth, seg, normals
DEPTH_MIN, DEPTH_MAX = 0.15, 0.5
N_ENVS_STREAM = 4
WARMUP_STEPS = 100  # shadow hand is direct env, fewer warm-up steps needed
RECORD_FRAMES = 150
VIDEO_FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)

env_cfg = ShadowHandEnvCfg()
env_cfg.sim.physics = PhysicsCfg().newton_mjwarp
for preset_name in ["default", "newton_mjwarp", "newton_kamino", "physx", "ovphysx"]:
    variant = getattr(env_cfg.scene, preset_name, None)
    if variant is not None and hasattr(variant, "num_envs"):
        variant.num_envs = 16

env = gym.make("Isaac-Reorient-Cube-Shadow-Direct", cfg=env_cfg, render_mode=None)
env.reset()
num_envs = env.unwrapped.num_envs
print(f"[OK] Shadow Hand DexCube — {num_envs} envs, 4 streamed, GT={GT_TYPES}")

cam, _ = create_visualizer_camera(
    num_envs=num_envs,
    width=160,
    height=120,
    renderer_cfg=NewtonWarpRendererCfg(),
    data_types=sensor_keys_for_gt_types(GT_TYPES),
)

env_ids = resolve_streaming_envs(num_envs, N_ENVS_STREAM)
stage = get_current_stage()
scene = env.unwrapped.scene
target_positions = prim_world_positions(stage, "/World/envs/*/object", env_ids, scene=scene)
apply_camera_target_positions(cam, target_positions, (0.2, -0.2, 0.2), env_ids)

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
    return compose_streaming_grid(frames, len(env_ids), len(GT_TYPES))


print(f"[OK] Warming up {WARMUP_STEPS} steps ...")
for i in range(WARMUP_STEPS):
    env.step(torch.tensor(env.action_space.sample(), device=env.unwrapped.device))
    if (i + 1) % 50 == 0:
        print(f"  warm-up {i + 1}/{WARMUP_STEPS}")

print(f"[OK] Recording {RECORD_FRAMES} frames → {OUT_VIDEO}")
writer = imageio.get_writer(OUT_VIDEO, fps=VIDEO_FPS, codec="libx264", quality=8)

try:
    for step in range(RECORD_FRAMES):
        env.step(torch.tensor(env.action_space.sample(), device=env.unwrapped.device))
        writer.append_data(_composite())
        if (step + 1) % 50 == 0:
            print(f"  frame {step + 1}/{RECORD_FRAMES}")
finally:
    writer.close()

env.close()
simulation_app.close()
print(f"[DONE] {OUT_VIDEO}")
