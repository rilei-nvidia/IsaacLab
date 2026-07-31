#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch all available visualizers with streaming view on Shadow Hand DexCube.

Runs Isaac-Reorient-Cube-Shadow-Direct (Newton MJWarp, 16 envs) with:
- Newton GL   — interactive OpenGL window + Streaming View panel
- Rerun       — streaming composite as primary view (80%) + small 3D (20%)
- Viser       — streaming composite as background image

Each visualizer streams RGB | depth (turbo, [0.1, 0.6] m) | segmentation
(DexCube shows in cyan, semantic label "cube") for 3 envs aimed at the cube.

Video output (10 s @ 15 fps = 150 frames):
    docs/source/_static/visualizers/streaming/streaming_view_shadow_hand_all_viz.mp4

Usage::

    uv run python scripts/visualizers/launch_all_visualizers_streaming.py
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg  # noqa: E402
from isaaclab_visualizers.rerun import RerunVisualizerCfg  # noqa: E402
from isaaclab_visualizers.viser import ViserVisualizerCfg  # noqa: E402

from isaaclab.envs.utils.camera_colorizer import (  # noqa: E402
    CameraFrameColorizer,
    sensor_key_for_gt_type,
)
from isaaclab.envs.utils.camera_view import camera_gt_batch, compose_streaming_grid  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import PhysicsCfg  # noqa: E402
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import (  # noqa: E402
    ShadowHandEnvCfg,
)

# ── Config ────────────────────────────────────────────────────────────────────
GT_TYPES = ["rgb", "depth", "segmentation", "normals"]
N_STEPS = 150  # 10 s @ 15 fps
VIDEO_FPS = 15
OUT_DIR = "docs/source/_static/visualizers/streaming"
OUT_VIDEO = os.path.join(OUT_DIR, "streaming_view_shadow_hand_all_viz.mp4")

_STREAMING = dict(
    streaming_view=True,
    streaming_envs=4,
    streaming_gt_types=GT_TYPES,
    streaming_depth_min=0.15,
    streaming_depth_max=0.5,
    streaming_cam_target_prim_path="/World/envs/*/object",  # DexCube with "cube" label
    streaming_cam_eye=(0.2, -0.2, 0.2),
)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Environment — Shadow Hand DexCube ─────────────────────────────────────────
env_cfg = ShadowHandEnvCfg()
env_cfg.sim.physics = PhysicsCfg().newton_mjwarp

# ShadowHandEnvCfg.scene uses a PresetCfg; the num_envs must be set on each
# concrete variant so the resolved scene respects the override.
for _preset_name in ["default", "newton_mjwarp", "newton_kamino", "physx", "ovphysx"]:
    _variant = getattr(env_cfg.scene, _preset_name, None)
    if _variant is not None and hasattr(_variant, "num_envs"):
        _variant.num_envs = 16

env_cfg.sim.visualizer_cfgs = [
    NewtonGLVisualizerCfg(**_STREAMING),
    NewtonRTXVisualizerCfg(**_STREAMING),
    RerunVisualizerCfg(**_STREAMING),
    ViserVisualizerCfg(**_STREAMING),
]

env = gym.make("Isaac-Reorient-Cube-Shadow-Direct", cfg=env_cfg, render_mode=None)
obs, _ = env.reset()

visualizers = getattr(env.unwrapped.sim, "_visualizers", [])
active = [type(v).__name__ for v in visualizers]
print(f"[OK] Active visualizers ({len(active)}): {active}")
print(f"[OK] num_envs = {env.unwrapped.num_envs}")
print("[INFO] Rerun:  http://127.0.0.1:9090")
print("[INFO] Viser:  http://localhost:8080")

# Find a streaming camera from the first visualizer that has one.
streaming_cam = None
streaming_env_ids: list[int] = []
streaming_depth_min = _STREAMING["streaming_depth_min"]
streaming_depth_max = _STREAMING["streaming_depth_max"]
for viz in visualizers:
    if getattr(viz, "_camera_sensor", None) is not None:
        streaming_cam = viz._camera_sensor
        streaming_env_ids = viz._camera_sensor_indices
        streaming_depth_min = viz.cfg.streaming_depth_min
        streaming_depth_max = viz.cfg.streaming_depth_max
        print(f"[OK] Recording from {type(viz).__name__} camera  env_ids={streaming_env_ids}")
        break

if streaming_cam is None:
    print("[WARNING] No streaming camera found — video will not be recorded.")

# ── Step loop + video recording ───────────────────────────────────────────────
writer = imageio.get_writer(OUT_VIDEO, fps=VIDEO_FPS, codec="libx264", quality=8) if streaming_cam else None
print(f"[OK] Running {N_STEPS} steps ({N_STEPS / VIDEO_FPS:.0f} s) — recording to {OUT_VIDEO}")

try:
    for step in range(N_STEPS):
        # Shadow Hand is a direct env; actions are numpy arrays but env.step expects torch.
        action = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated.any() or truncated.any():
            obs, _ = env.reset()

        if writer is not None and streaming_cam is not None:
            streaming_cam.update(dt=0.0, force_recompute=True)
            available = frozenset(streaming_cam.data.output.keys())
            frames = []
            for env_idx in streaming_env_ids:
                for gt in GT_TYPES:
                    key = sensor_key_for_gt_type(gt, available)
                    raw = camera_gt_batch(streaming_cam, [env_idx], key)[0]
                    frames.append(
                        CameraFrameColorizer.colorize(
                            raw,
                            gt,
                            depth_min=streaming_depth_min,
                            depth_max=streaming_depth_max,
                        )
                    )
            composite = compose_streaming_grid(frames, len(streaming_env_ids), len(GT_TYPES))
            writer.append_data(composite)

        if (step + 1) % 50 == 0:
            print(f"      step {step + 1}/{N_STEPS}")

except (KeyboardInterrupt, SystemExit):
    print("[INFO] Interrupted.")
finally:
    if writer is not None:
        writer.close()

env.close()
simulation_app.close()
print(f"[DONE] Video saved → {OUT_VIDEO}")
