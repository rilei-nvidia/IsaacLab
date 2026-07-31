# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for visualizer and recorder camera image views."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim.views import FrameView

_GENERATED_CAMERA_NAME = "VisualizerCamera"
VISUALIZER_TILED_CAMERA_MAX_TILES = 100


def resolve_tiled_env_indices(
    num_envs: int,
    tiled_cam_num: int,
    env_indices: list[int] | None,
    max_tiles: int | None = None,
    sample_from: list[int] | None = None,
) -> list[int]:
    """Resolve env ids for tiled camera view once at visualizer initialization."""
    if num_envs <= 0:
        return []
    if env_indices is not None:
        max_count = min(max(1, int(tiled_cam_num)), num_envs)
        if max_tiles is not None:
            max_count = min(max_count, max(1, int(max_tiles)))
        return [idx for idx in env_indices if 0 <= int(idx) < num_envs][:max_count]
    candidates = [
        idx for idx in (sample_from if sample_from is not None else range(num_envs)) if 0 <= int(idx) < num_envs
    ]
    if not candidates:
        return []
    max_count = min(max(1, int(tiled_cam_num)), len(candidates))
    if max_tiles is not None:
        max_count = min(max_count, max(1, int(max_tiles)))
    return sorted(random.sample(candidates, max_count))


def resolve_mono_env_index(num_envs: int) -> list[int]:
    """Return env_0 for mono sensor camera views."""
    return [0] if num_envs > 0 else []


def env_path_from_template(path_template: str, env_id: int) -> str:
    """Resolve common env wildcard/template spellings to a concrete env path."""
    path = path_template
    if "%d" in path:
        return path % env_id
    if "{}" in path:
        return path.format(env_id)
    path = path.replace("/World/envs/*", f"/World/envs/env_{env_id}")
    path = path.replace("/World/envs/env_.*", f"/World/envs/env_{env_id}")
    path = path.replace("/World/envs/env_.*/", f"/World/envs/env_{env_id}/")
    return path


def _camera_concrete_paths(camera: Camera) -> list[str]:
    view = getattr(camera, "_view", None)
    prims = getattr(view, "prims", None)
    if not prims:
        return []
    return [prim.GetPath().pathString for prim in prims]


def find_camera_by_prim_path(camera_sensors: dict[str, Camera], cam_prim_path: str, env_indices: list[int]) -> Camera:
    """Find a scene-owned Camera by config template or concrete camera prim paths."""
    wanted = {env_path_from_template(cam_prim_path, env_id) for env_id in env_indices}
    for camera in camera_sensors.values():
        if getattr(camera.cfg, "prim_path", None) == cam_prim_path:
            return camera
        concrete = set(_camera_concrete_paths(camera))
        if wanted and wanted.issubset(concrete):
            return camera
    stage = sim_utils.get_current_stage()
    stage_matches = [path for path in wanted if stage.GetPrimAtPath(path).IsValid()] if stage is not None else []
    if stage_matches:
        raise RuntimeError(
            f"cam_prim_path={cam_prim_path!r} matched USD camera prims, but no Isaac Lab Camera sensor owns them. "
            "Add the camera to scene.sensors or leave tiled_cam_prim_path unset to use generated tiled cameras."
        )
    if not camera_sensors:
        raise RuntimeError(
            f"No Isaac Lab Camera sensors are registered in the scene, so tiled_cam_prim_path={cam_prim_path!r} "
            "cannot be used. Use an environment that defines Camera sensors, or leave tiled_cam_prim_path unset "
            "to use generated tiled cameras."
        )
    available_paths = {
        getattr(camera.cfg, "prim_path", None) for camera in camera_sensors.values() if getattr(camera, "cfg", None)
    }
    raise RuntimeError(
        f"No Isaac Lab Camera sensor matched cam_prim_path={cam_prim_path!r}. "
        f"Available Camera sensor prim paths: {sorted(path for path in available_paths if path)}. "
        "Leave tiled_cam_prim_path unset to use generated tiled cameras."
    )


def ensure_camera_initialized(camera: Camera) -> None:
    """Initialize a visualizer-owned Camera created after the normal physics-ready callback."""
    if not camera.is_initialized:
        camera._initialize_callback(None)


def resolve_streaming_envs(
    num_envs: int,
    streaming_envs: int | list[int],
    max_tiles: int = VISUALIZER_TILED_CAMERA_MAX_TILES,
    sample_from: list[int] | None = None,
) -> list[int]:
    """Resolve ``streaming_envs`` to a concrete list of env indices.

    Args:
        num_envs: Total number of simulation environments.
        streaming_envs: ``int`` → randomly sample that many envs;
            ``list[int]`` → use exactly those indices (capped at ``max_tiles``).
        max_tiles: Hard cap on the number of returned indices.
        sample_from: When ``streaming_envs`` is an ``int``, sample from this
            subset rather than all envs (e.g. visible env indices).

    Returns:
        Sorted list of env indices, length ≤ ``max_tiles``.
    """
    if isinstance(streaming_envs, list):
        indices = [i for i in streaming_envs if 0 <= i < num_envs]
        return sorted(indices[:max_tiles])
    pool = sample_from if sample_from is not None else list(range(num_envs))
    count = min(int(streaming_envs), max_tiles, len(pool))
    return sorted(random.sample(pool, count))


def camera_gt_batch(camera: Camera, env_indices: list[int], sensor_key: str) -> torch.Tensor:
    """Return GT output for selected env indices from a camera sensor.

    Args:
        camera: Isaac Lab :class:`~isaaclab.sensors.camera.Camera` sensor.
        env_indices: Env indices to select (must be valid indices into the
            camera's tiled output).
        sensor_key: Key in ``camera.data.output``, e.g. ``"rgb"``,
            ``"depth"``, or ``"semantic_segmentation"``.

    Returns:
        Tensor of shape ``(len(env_indices), H, W, C)`` on the camera's device.
    """
    raw = camera.data.output[sensor_key]
    if isinstance(raw, wp.array):
        raw = wp.to_torch(raw)
    elif hasattr(raw, "torch"):
        raw = raw.torch
    if env_indices:
        idx = torch.tensor(env_indices, dtype=torch.long, device=raw.device)
        return raw.index_select(0, idx)
    return raw


def compose_streaming_grid(
    frames: list[np.ndarray],
    n_envs: int,
    n_gt: int,
) -> np.ndarray:
    """Composite streaming frames into a tiled output image.

    Layout minimises ``|log(W/H)|`` subject to the constraint that all GT
    columns for one env remain on the same row.  For a single GT type, envs
    are packed into a near-square grid (matching the legacy tiled camera
    behaviour).

    Args:
        frames: Flat list of ``uint8 (H, W, 3)`` arrays ordered as
            ``[env0_gt0, env0_gt1, ..., env0_gtM-1, env1_gt0, ...]``.
        n_envs: Number of environments represented in ``frames``.
        n_gt: Number of GT types per environment.

    Returns:
        Single ``uint8 (total_H, total_W, 3)`` composite image, or a 1×1 black
        pixel if ``frames`` is empty.
    """
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = frames[0].shape[:2]
    env_cols = _best_streaming_cols(n_envs, n_gt, h, w)
    env_rows = math.ceil(n_envs / env_cols)
    canvas = np.zeros((env_rows * h, env_cols * n_gt * w, 3), dtype=np.uint8)
    for env_idx in range(n_envs):
        ec = env_idx % env_cols
        er = env_idx // env_cols
        for gt_idx in range(n_gt):
            frame = frames[env_idx * n_gt + gt_idx]
            y0, x0 = er * h, (ec * n_gt + gt_idx) * w
            canvas[y0 : y0 + h, x0 : x0 + w] = frame[..., :3]
    return canvas


def _best_streaming_cols(n_envs: int, n_gt: int, frame_h: int, frame_w: int) -> int:
    """Env-column count that minimises ``|log(total_W / total_H)|``."""
    best_cols, best_score = 1, float("inf")
    for cols in range(1, n_envs + 1):
        rows = math.ceil(n_envs / cols)
        score = abs(math.log((cols * n_gt * frame_w) / (rows * frame_h)))
        if score < best_score:
            best_score, best_cols = score, cols
    return best_cols


def create_visualizer_camera(
    *,
    num_envs: int,
    camera_name: str = _GENERATED_CAMERA_NAME,
    width: int,
    height: int,
    renderer_cfg: Any,
    data_types: list[str] | None = None,
) -> tuple[Camera, list[str]]:
    """Create an internal Camera sensor for visualizer image views."""
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    generated_paths = [f"/World/envs/env_{env_id}/{camera_name}" for env_id in range(int(num_envs))]
    for path in generated_paths:
        if len(sim_utils.find_matching_prims(path)) == 0:
            spawn.func(path, spawn, translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))

    stage = sim_utils.get_current_stage()
    for path in generated_paths:
        cam_prim = stage.GetPrimAtPath(path)
        attr = cam_prim.GetAttribute("omni:scenePartition")
        if not attr.IsValid():
            attr = cam_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
        attr.Set(path.split("/")[-2])
    cfg = CameraCfg(
        prim_path=f"/World/envs/env_.*/{camera_name}",
        update_period=0.0,
        height=int(height),
        width=int(width),
        data_types=data_types if data_types is not None else ["rgb"],
        spawn=None,
        renderer_cfg=renderer_cfg,
    )
    camera = Camera(cfg)
    ensure_camera_initialized(camera)
    return camera, generated_paths


def remove_generated_prims(prim_paths: list[str] | None) -> None:
    """Remove visualizer-owned camera prims from the current stage."""
    if not prim_paths:
        return
    stage = sim_utils.get_current_stage()
    if stage is None:
        return
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


def camera_rgb_batch(camera: Camera, env_indices: list[int]) -> torch.Tensor:
    """Return RGB output for selected env indices."""
    rgb = camera.data.output["rgb"]
    if isinstance(rgb, wp.array):
        rgb = wp.to_torch(rgb)
    elif hasattr(rgb, "torch"):
        rgb = rgb.torch
    if env_indices:
        index = torch.tensor(env_indices, dtype=torch.long, device=rgb.device)
        return rgb.index_select(0, index)
    return rgb


def compose_rgb_grid_tensor(rgb_batch: torch.Tensor) -> torch.Tensor:
    """Compose an RGB batch into a near-square uint8 image grid without leaving its device."""
    if rgb_batch.ndim == 3:
        return rgb_batch[..., :3].contiguous()
    n, h, w, _ = rgb_batch.shape
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    rgb = rgb_batch[..., :3]
    pad = rows * cols - n
    if pad > 0:
        rgb = torch.cat([rgb, torch.zeros((pad, h, w, 3), dtype=rgb.dtype, device=rgb.device)], dim=0)
    return rgb.reshape(rows, cols, h, w, 3).permute(0, 2, 1, 3, 4).reshape(rows * h, cols * w, 3).contiguous()


def compute_tile_resolution(window_width: int, window_height: int, num_tiles: int) -> tuple[int, int]:
    """Derive a conservative per-tile resolution from the visualizer window."""
    if window_width <= 0 or window_height <= 0:
        raise ValueError(f"Window dimensions must be positive, got {window_width}x{window_height}.")
    cols = max(1, math.ceil(math.sqrt(max(1, num_tiles))))
    rows = math.ceil(max(1, num_tiles) / cols)
    return max(1, int(window_width) // cols), max(1, int(window_height) // rows)


def _normalize_env0_path(path_template: str) -> str:
    """Resolve env template spellings to env_0 for path comparison."""
    return env_path_from_template(path_template, 0)


def _scene_articulation_positions(scene: Any, prim_path_template: str, env_indices: list[int]) -> torch.Tensor | None:
    """Resolve follow positions from scene articulation state when the path targets an asset/body."""
    follow_env0 = _normalize_env0_path(prim_path_template)
    for asset in getattr(scene, "articulations", {}).values():
        asset_path = _normalize_env0_path(getattr(asset.cfg, "prim_path", ""))
        if not asset_path:
            continue
        if follow_env0 == asset_path:
            return asset.data.root_pos_w.torch[env_indices].detach().cpu()
        prefix = asset_path + "/"
        if not follow_env0.startswith(prefix):
            continue
        body_name = follow_env0.removeprefix(prefix).split("/")[-1]
        if body_name not in asset.body_names:
            continue
        body_ids, _ = asset.find_bodies(body_name)
        if not body_ids:
            continue
        return asset.data.body_pos_w.torch[env_indices, int(body_ids[0])].detach().cpu()
    return None


def prim_world_positions(
    stage: Any, prim_path_template: str, env_indices: list[int], scene: Any | None = None
) -> torch.Tensor:
    """Return world-space translations for concrete prim paths resolved from env ids.

    Uses scene articulation state first when the target is an asset/body path,
    then falls back to ``FrameView`` and USD for arbitrary prim paths.
    """
    if scene is not None:
        positions_tensor = _scene_articulation_positions(scene, prim_path_template, env_indices)
        if positions_tensor is not None:
            return positions_tensor

    xform_cache = UsdGeom.XformCache()
    positions = []
    try:
        for env_id in env_indices:
            prim_path = env_path_from_template(prim_path_template, env_id)
            view = FrameView(prim_path, device="cpu", stage=stage)
            if view.count != 1:
                raise RuntimeError(f"expected one prim, got {view.count}")
            pos_w, _ = view.get_world_poses()
            pos = pos_w.torch[0].detach().cpu()
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
        return torch.tensor(positions, dtype=torch.float32)
    except Exception:
        positions.clear()

    for env_id in env_indices:
        prim_path = env_path_from_template(prim_path_template, env_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"tiled_cam_target_prim_path resolved to missing prim: {prim_path!r}.")
        transform = xform_cache.GetLocalToWorldTransform(prim)
        translation = transform.ExtractTranslation()
        positions.append((float(translation[0]), float(translation[1]), float(translation[2])))
    return torch.tensor(positions, dtype=torch.float32)


def apply_camera_view_from_origins(
    camera: Camera,
    origins: torch.Tensor,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    env_ids: list[int] | None = None,
) -> None:
    """Set camera poses from origins plus relative eye/lookat offsets."""
    device = camera.device
    origins = origins.to(device=device)
    eye_offset = torch.tensor(eye, dtype=torch.float32, device=device).unsqueeze(0)
    lookat_offset = torch.tensor(lookat, dtype=torch.float32, device=device).unsqueeze(0)
    camera.set_world_poses_from_view(origins + eye_offset, origins + lookat_offset, env_ids=env_ids)
    camera._update_poses(None)


def apply_camera_target_positions(
    camera: Camera,
    target_positions: torch.Tensor,
    eye: tuple[float, float, float],
    env_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set generated tiled camera poses as target-relative eye offsets."""
    device = camera.device
    target_positions = target_positions.to(device=device)
    eye_offset = torch.tensor(eye, dtype=torch.float32, device=device).unsqueeze(0)
    eyes = target_positions + eye_offset
    targets = target_positions
    camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)
    camera._update_poses(None)
    return eyes.detach().cpu(), targets.detach().cpu()
