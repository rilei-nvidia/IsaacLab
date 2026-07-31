#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming view validation for the Shadow Hand env with segmentation.

Uses Isaac-Reorient-Cube-Shadow-Direct (Newton MJWarp).  The DexCube object
has semantic_tags=[("class", "cube")] so it shows up in segmentation.
Camera is aimed at the cube for a close-up view.

Output::

    docs / source / _static / visualizers / streaming / streaming_view_shadow_hand_3x3.mp4

Usage::

    uv run python scripts/visualizers/test_streaming_view_shadow_hand_seg.py
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

# ── Config ────────────────────────────────────────────────────────────────────
GT_TYPES = ["rgb", "depth", "segmentation", "normals"]
DEPTH_MIN = 0.15
DEPTH_MAX = 0.5
N_ENVS_STREAM = 4
N_STEPS = 120
TILE_W, TILE_H = 160, 120
# Aim at the cube (object) — it has semantic_tags=[("class", "cube")]
TARGET_PRIM = "/World/envs/*/object"
EYE_OFFSET = (0.2, -0.2, 0.2)  # close-up on the hand/cube area
OUT_DIR = "docs/source/_static/visualizers/streaming"
VIDEO_FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)
out_video = os.path.join(OUT_DIR, "streaming_view_shadow_hand_3x3.mp4")

# ── Environment ───────────────────────────────────────────────────────────────
env_cfg = ShadowHandEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.sim.physics = PhysicsCfg().newton_mjwarp

env = gym.make("Isaac-Reorient-Cube-Shadow-Direct", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()
num_envs = env.unwrapped.num_envs
print(f"[OK] Environment reset — Shadow Hand / Newton MJWarp  ({num_envs} envs)")

# ── Streaming camera ──────────────────────────────────────────────────────────
sensor_keys = sensor_keys_for_gt_types(GT_TYPES)
cam, _ = create_visualizer_camera(
    num_envs=num_envs,
    width=TILE_W,
    height=TILE_H,
    renderer_cfg=NewtonWarpRendererCfg(),
    data_types=sensor_keys,
)

env_ids = resolve_streaming_envs(num_envs, N_ENVS_STREAM)
print(f"[OK] Streaming env_ids={env_ids}  GT={GT_TYPES}  depth=[{DEPTH_MIN},{DEPTH_MAX}]m")

stage = get_current_stage()
scene = env.unwrapped.scene
target_positions = prim_world_positions(stage, TARGET_PRIM, env_ids, scene=scene)
apply_camera_target_positions(cam, target_positions, EYE_OFFSET, env_ids)

# Warm-up: run a few steps so the scene settles before checking segmentation
for _ in range(5):
    action = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
    env.step(action)

cam.update(dt=0.0, force_recompute=True)
available = frozenset(cam.data.output.keys())
print(f"[OK] Camera output keys: {sorted(available)}")

# Probe segmentation to confirm cube label is visible
import warp as wp  # noqa: E402

seg = cam.data.output.get("semantic_segmentation")
if seg is not None:
    if isinstance(seg, wp.array):
        seg = wp.to_torch(seg)
    t = seg[env_ids[0]]
    # decode packed RGBA → int ID
    ids = t[..., 0].long() + t[..., 1].long() * 256 + t[..., 2].long() * 65536
    unique = ids.unique()
    print(f"     Seg unique IDs: {unique[:10].tolist()}  (0=background, non-zero=cube)")


# ── Recording ─────────────────────────────────────────────────────────────────
def _composite_frame() -> np.ndarray:
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
            print(f"      frame {step + 1}/{N_STEPS}  shape={composite.shape}")
finally:
    writer.close()

env.close()
simulation_app.close()
print(f"[DONE] Video saved → {out_video}")
