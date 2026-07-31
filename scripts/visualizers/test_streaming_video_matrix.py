#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive streaming video recording test matrix.

Runs a grid of (env × physics × visualizer × GT-types) combinations, records
10-frame clips for each, saves PNGs of representative frames, and reports
pass/fail based on whether the output is non-black and non-uniform.

Outputs: docs/source/_static/visualizers/streaming/test_matrix/

Usage::

    uv run python scripts/visualizers/test_streaming_video_matrix.py
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

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
from PIL import Image  # noqa: E402

from isaaclab.envs.utils.camera_colorizer import (  # noqa: E402
    CameraFrameColorizer,
    sensor_key_for_gt_type,
)
from isaaclab.envs.utils.camera_view import (  # noqa: E402  # noqa: E402
    apply_camera_target_positions,
    camera_gt_batch,
    compose_streaming_grid,
    create_visualizer_camera,
    prim_world_positions,
    resolve_streaming_envs,
)
from isaaclab.sim import get_current_stage  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

OUT_DIR = "docs/source/_static/visualizers/streaming/test_matrix"
os.makedirs(OUT_DIR, exist_ok=True)

N_WARMUP = 5  # warm-up steps before recording
N_FRAMES = 10  # frames to record per clip
FPS = 5


def _quality_check(arr: np.ndarray, name: str) -> tuple[bool, str]:
    """Return (ok, note) based on pixel statistics."""
    mean = float(arr.mean())
    std = float(arr.std())
    mx = int(arr.max())
    if mx < 5:
        return False, f"FAIL black (mean={mean:.1f} std={std:.1f} max={mx})"
    if std < 2.0:
        return False, f"FAIL uniform (mean={mean:.1f} std={std:.1f})"
    return True, f"OK   mean={mean:.1f} std={std:.1f} max={mx}"


def run_case(
    tag: str,
    task: str,
    env_cfg,
    gt_types: list[str],
    depth_min: float = 0.5,
    depth_max: float = 8.0,
    target_prim: str = "/World/envs/*/Robot",
    eye_offset: tuple = (0.8, -0.8, 0.8),
    num_stream_envs: int = 3,
    is_direct_env: bool = False,
) -> bool:
    """Run one test case, save video + frame, return True if all frames pass QC."""
    print(f"\n{'=' * 60}")
    print(f"  {tag}")
    print(f"  task={task}  gt={gt_types}")
    print(f"{'=' * 60}")

    results = []
    try:
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        obs, _ = env.reset()
        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"
        print(f"  num_envs={num_envs}  device={device}")

        # Camera sensor keys from GT types
        sensor_keys = []
        seen = set()
        for gt in gt_types:
            from isaaclab.envs.utils.camera_colorizer import _GT_TO_SENSOR_KEY

            key = _GT_TO_SENSOR_KEY.get(gt)
            if key and key not in seen:
                sensor_keys.append(key)
                seen.add(key)

        cam, _ = create_visualizer_camera(
            num_envs=num_envs,
            width=160,
            height=120,
            renderer_cfg=NewtonWarpRendererCfg(),
            data_types=sensor_keys,
        )

        env_ids = resolve_streaming_envs(num_envs, num_stream_envs)
        stage = get_current_stage()
        scene = env.unwrapped.scene if hasattr(env.unwrapped, "scene") else None
        try:
            target_pos = prim_world_positions(stage, target_prim, env_ids, scene=scene)
            apply_camera_target_positions(cam, target_pos, eye_offset, env_ids)
        except Exception as e:
            print(f"  [WARN] camera pose: {e}")

        # Warm-up
        for _ in range(N_WARMUP):
            if is_direct_env:
                action = torch.tensor(env.action_space.sample(), device=device)
            else:
                action = torch.tensor(env.action_space.sample(), device=device)
            env.step(action)

        # Record
        video_path = os.path.join(OUT_DIR, f"{tag}.mp4")
        frame_path = os.path.join(OUT_DIR, f"{tag}_frame5.png")
        writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8)

        available: frozenset[str] | None = None
        for step in range(N_FRAMES):
            if is_direct_env:
                action = torch.tensor(env.action_space.sample(), device=device)
            else:
                action = torch.tensor(env.action_space.sample(), device=device)
            env.step(action)

            cam.update(dt=0.0, force_recompute=True)
            if available is None:
                available = frozenset(cam.data.output.keys())

            frames = []
            for env_idx in env_ids:
                for gt in gt_types:
                    key = sensor_key_for_gt_type(gt, available)
                    raw = camera_gt_batch(cam, [env_idx], key)[0]
                    frames.append(CameraFrameColorizer.colorize(raw, gt, depth_min=depth_min, depth_max=depth_max))

            composite = compose_streaming_grid(frames, len(env_ids), len(gt_types))
            writer.append_data(composite)

            if step == 5:
                Image.fromarray(composite).save(frame_path)
                ok, note = _quality_check(composite, tag)
                results.append((ok, note))
                print(f"  frame5 QC: {note}")

        writer.close()
        env.close()

        overall = all(ok for ok, _ in results)
        print(f"  → {'PASS' if overall else 'FAIL'}  {video_path}")
        return overall

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc(limit=3)
        return False


# ── Test cases ────────────────────────────────────────────────────────────────


def make_anymal_cfg(physics: str = "newton"):
    from isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

    cfg = AnymalDRoughEnvCfg()
    cfg.scene.num_envs = 16
    if physics == "newton":
        cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp
    return cfg


def make_shadow_cfg():
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import PhysicsCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg

    cfg = ShadowHandEnvCfg()
    cfg.sim.physics = PhysicsCfg().newton_mjwarp
    for name in ["default", "newton_mjwarp", "newton_kamino", "physx", "ovphysx"]:
        obj = getattr(cfg.scene, name, None)
        if obj is not None and hasattr(obj, "num_envs"):
            obj.num_envs = 16
    return cfg


passed = 0
failed = 0
skipped = 0


def run(tag, task, cfg, gt, **kwargs):
    global passed, failed
    ok = run_case(tag, task, cfg, gt, **kwargs)
    if ok:
        passed += 1
    else:
        failed += 1


# ── AnymalD Newton + RGB only ──────────────────────────────────────────────
run("anymal_newton_rgb", "Isaac-Velocity-Rough-AnymalD", make_anymal_cfg("newton"), ["rgb"], depth_max=8.0)

# ── AnymalD Newton + RGB + depth ──────────────────────────────────────────
run(
    "anymal_newton_rgb_depth",
    "Isaac-Velocity-Rough-AnymalD",
    make_anymal_cfg("newton"),
    ["rgb", "depth"],
    depth_min=0.5,
    depth_max=8.0,
)

# ── AnymalD Newton + RGB + depth + normals ────────────────────────────────
run(
    "anymal_newton_rgb_depth_normals",
    "Isaac-Velocity-Rough-AnymalD",
    make_anymal_cfg("newton"),
    ["rgb", "depth", "normals"],
    depth_min=0.5,
    depth_max=8.0,
)

# ── AnymalD Newton + RGB + depth + segmentation (no labels → dark grey) ──
run(
    "anymal_newton_rgb_depth_seg",
    "Isaac-Velocity-Rough-AnymalD",
    make_anymal_cfg("newton"),
    ["rgb", "depth", "segmentation"],
    depth_min=0.5,
    depth_max=8.0,
)

# ── Shadow Hand Newton + RGB + depth + segmentation (cube labeled cyan) ──
run(
    "shadow_newton_rgb_depth_seg",
    "Isaac-Reorient-Cube-Shadow-Direct",
    make_shadow_cfg(),
    ["rgb", "depth", "segmentation"],
    depth_min=0.1,
    depth_max=0.6,
    target_prim="/World/envs/*/object",
    eye_offset=(0.2, -0.2, 0.2),
    is_direct_env=True,
)

# ── Shadow Hand Newton + RGB + normals ────────────────────────────────────
run(
    "shadow_newton_rgb_normals",
    "Isaac-Reorient-Cube-Shadow-Direct",
    make_shadow_cfg(),
    ["rgb", "normals"],
    depth_min=0.1,
    depth_max=0.6,
    target_prim="/World/envs/*/object",
    eye_offset=(0.2, -0.2, 0.2),
    is_direct_env=True,
)

# ── Shadow Hand Newton + all 4 GT types ──────────────────────────────────
run(
    "shadow_newton_all4",
    "Isaac-Reorient-Cube-Shadow-Direct",
    make_shadow_cfg(),
    ["rgb", "depth", "segmentation", "normals"],
    depth_min=0.1,
    depth_max=0.6,
    target_prim="/World/envs/*/object",
    eye_offset=(0.2, -0.2, 0.2),
    num_stream_envs=2,
    is_direct_env=True,
)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  SUMMARY: {passed} passed / {failed} failed / {skipped} skipped")
print(f"  Output:  {OUT_DIR}/")
print(f"{'=' * 60}")

simulation_app.close()
sys.exit(0 if failed == 0 else 1)
