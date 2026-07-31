#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the streaming view pipeline and produce a recorded video.

Runs 120 steps of Isaac-Velocity-Rough-AnymalD (Newton MJWarp), composites
RGB | depth (turbo, depth_min=0.5 depth_max=2.0) | segmentation for 3 envs
into a 3-row × 3-col grid at every step, and writes the result to an mp4.

Output::

    docs / source / _static / visualizers / streaming / streaming_view_anymal_3x3.mp4

Usage::

    uv run python scripts/visualizers/test_streaming_view_shadow_hand.py
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
from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
GT_TYPES = ["rgb", "depth", "segmentation"]
DEPTH_MIN = 0.5  # [m] maps to blue end of turbo
DEPTH_MAX = 2.0  # [m] maps to red end of turbo
N_ENVS_STREAM = 3
N_STEPS = 120
TILE_W, TILE_H = 160, 120
EYE_OFFSET = (0.8, -0.8, 0.8)  # ~1.4m from robot center; keeps robot body in [0.5, 2.0]m range
OUT_DIR = "docs/source/_static/visualizers/streaming"
VIDEO_FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)
out_video = os.path.join(OUT_DIR, "streaming_view_anymal_3x3.mp4")

# ── Environment ───────────────────────────────────────────────────────────────
env_cfg = AnymalDRoughEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp

env = gym.make("Isaac-Velocity-Rough-AnymalD", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()
print(f"[OK] Environment reset — AnymalD Rough / Newton MJWarp  ({env.unwrapped.num_envs} envs)")

# ── Streaming camera ──────────────────────────────────────────────────────────
sensor_keys = sensor_keys_for_gt_types(GT_TYPES)
cam, _ = create_visualizer_camera(
    num_envs=env.unwrapped.num_envs,
    width=TILE_W,
    height=TILE_H,
    renderer_cfg=NewtonWarpRendererCfg(),
    data_types=sensor_keys,
)

env_ids = resolve_streaming_envs(env.unwrapped.num_envs, N_ENVS_STREAM)
print(f"[OK] Streaming env_ids={env_ids}  GT={GT_TYPES}  depth=[{DEPTH_MIN},{DEPTH_MAX}]m")

stage = get_current_stage()
scene = env.unwrapped.scene
target_positions = prim_world_positions(stage, "/World/envs/*/Robot", env_ids, scene=scene)
apply_camera_target_positions(cam, target_positions, EYE_OFFSET, env_ids)

cam.update(dt=0.0, force_recompute=True)
available = frozenset(cam.data.output.keys())
print(f"[OK] Camera output keys: {sorted(available)}")


# ── Recording ─────────────────────────────────────────────────────────────────
def _composite_frame() -> np.ndarray:
    """Capture and colorize one composite frame from the streaming camera."""
    cam.update(dt=0.0, force_recompute=True)
    frames = []
    for env_idx in env_ids:
        for gt in GT_TYPES:
            key = sensor_key_for_gt_type(gt, available)
            raw = camera_gt_batch(cam, [env_idx], key)[0]
            frames.append(
                CameraFrameColorizer.colorize(
                    raw,
                    gt,
                    depth_min=DEPTH_MIN,
                    depth_max=DEPTH_MAX,
                )
            )
    return compose_streaming_grid(frames, len(env_ids), len(GT_TYPES))


writer = imageio.get_writer(out_video, fps=VIDEO_FPS, codec="libx264", quality=8)
print(f"[OK] Writing video → {out_video}  ({N_STEPS} frames @ {VIDEO_FPS} fps)")

try:
    for step in range(N_STEPS):
        action = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
        env.step(action)
        composite = _composite_frame()
        writer.append_data(composite)
        if (step + 1) % 30 == 0:
            print(f"      frame {step + 1}/{N_STEPS}  composite={composite.shape}")
finally:
    writer.close()

env.close()
simulation_app.close()
print(f"[DONE] Video saved → {out_video}")
