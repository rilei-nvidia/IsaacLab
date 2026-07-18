# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan resolution and OVRTX-side cloning."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXRenderer,
        _create_homogeneous_clone_plan,
        _resolve_clone_plan,
        _write_file,
        ovrtx_use_ovstage_enabled,
    )

    from pxr import Gf, Usd, UsdGeom  # noqa: E402
else:
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    Gf = None
    Usd = None
    UsdGeom = None
    _create_homogeneous_clone_plan = None
    _resolve_clone_plan = None

    def ovrtx_use_ovstage_enabled():
        return False

    _write_file = None


_PRE_OVRTX_STAGE_FILE = "pre_ovrtx_renderer_stage.usda"
_OVRTX_STAGE_FILE = "ovrtx_renderer_stage.usda"

_skip_if_not_ovstage = pytest.mark.skipif(
    not ovrtx_use_ovstage_enabled(), reason="requires ISAAC_LAB_OVRTX_USE_OVSTAGE=1"
)
_skip_if_not_legacy = pytest.mark.skipif(ovrtx_use_ovstage_enabled(), reason="requires ISAAC_LAB_OVRTX_USE_OVSTAGE=0")


def _make_multi_env_stage(num_envs: int) -> Usd.Stage:
    """Build an in-memory stage with distinguishable content per environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")

    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        UsdGeom.Xform.Define(stage, env_path)
        UsdGeom.Xform.Define(stage, f"{env_path}/Robot")
        UsdGeom.Xform.Define(stage, f"{env_path}/Object_env{env_idx}_only")
        UsdGeom.Camera.Define(stage, f"{env_path}/Camera")

    return stage


def _assert_export_contains_env_roots_and_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environment roots appear in the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' in exported

    assert exported.count('def Xform "Robot"') == len(env_indices)
    assert exported.count('def Camera "Camera"') == len(env_indices)


def _assert_export_contains_env_roots_but_omits_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environments and their unique children are omitted from the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def _assert_export_omits_env_roots(exported: str, env_indices: range | list[int]) -> None:
    """Listed environment roots and their children are absent from the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' not in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def _patch_simulation_context(monkeypatch: pytest.MonkeyPatch, clone_plan: ClonePlan | None) -> None:
    mock_ctx = SimpleNamespace(get_clone_plan=lambda: clone_plan)
    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.SimulationContext",
        SimpleNamespace(instance=lambda: mock_ctx),
    )


class _FakeOperation:
    """Minimal ovstage Operation stub that succeeds immediately."""

    def wait(self):
        pass


class _FakeQuery:
    """Minimal ovstage Query stub."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class _FakeStage:
    """Minimal ovstage Stage stub for clone-plan tests."""

    def __init__(self):
        self.clone_calls: list[tuple[str, list[str], int]] = []
        self.write_calls: list[dict] = []
        self.advance_floor_calls: list[int] = []
        self.release_query_calls: list = []

    def clone(self, source: str, targets: list[str], ordinal: int) -> None:
        self.clone_calls.append((source, targets, ordinal))

    def write_attribute(self, query, attr, ordinal, tensors, **kwargs):
        self.write_calls.append({"query": query, "attr": attr, "ordinal": ordinal, "tensors": tensors, **kwargs})
        return _FakeOperation()

    def advance_write_floor(self, ordinal: int):
        self.advance_floor_calls.append(ordinal)
        return _FakeOperation()

    def query_from_path_list(self, path_list):
        return _FakeQuery()

    def release_query(self, query):
        self.release_query_calls.append(query)
        return _FakeOperation()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class _FakePathDictionary:
    """Minimal ovstage PathDictionary stub."""

    def __init__(self):
        self._token_counter = 1
        self._tokens: dict[str, int] = {}
        self._path_counter = 1
        self._paths: dict[str, int] = {}

    def create_path_list_from_strings(self, paths: list[str]):
        return paths  # return the list as-is for simplicity

    def destroy_path_list(self, path_list):
        pass

    def intern_token(self, s: str) -> int:
        if s not in self._tokens:
            self._tokens[s] = self._token_counter
            self._token_counter += 1
        return self._tokens[s]

    def intern_path(self, path: str) -> int:
        if path not in self._paths:
            self._paths[path] = self._path_counter
            self._path_counter += 1
        return self._paths[path]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class _FakeLegacyRenderer:
    """Minimal legacy ovrtx Renderer stub for clone-plan tests."""

    def __init__(self):
        self.clone_usd_calls: list[tuple[str, list[str]]] = []
        self.write_attribute_calls: list[dict] = []
        self.open_usd_calls: list[str] = []

    def open_usd_from_string(self, usd: str) -> None:
        self.open_usd_calls.append(usd)

    def clone_usd(self, source: str, targets: list[str]) -> None:
        self.clone_usd_calls.append((source, targets))

    def read_attribute(self, attr, paths, **kwargs) -> np.ndarray:
        return np.tile(np.eye(4, dtype=np.float64), (len(paths), 1, 1))

    def write_attribute(self, prim_paths, attribute_name, tensor, **kwargs) -> None:
        self.write_attribute_calls.append({"attr": attribute_name, "prim_paths": prim_paths})

    def bind_attribute(self, prim_paths, attribute_name, **kwargs):
        return SimpleNamespace(map=lambda **kw: contextlib.nullcontext(), unbind=lambda: None)

    def reset_stage(self) -> None:
        pass

    def step(self, **kwargs) -> dict:
        return {}

    def detach_ovstage(self) -> None:
        pass


def _make_ovrtx_renderer_without_backend(fake_stage: _FakeStage | None = None) -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._renderer = SimpleNamespace(
        attach_ovstage=lambda stage: None,
        detach_ovstage=lambda: None,
        step=lambda **kwargs: {},
    )
    renderer._use_ovstage = True
    renderer._stage = fake_stage or _FakeStage()
    renderer._stage_paths = _FakePathDictionary()
    renderer._ovstage_exit_stack = contextlib.ExitStack()
    renderer._current_ordinal = 1
    renderer._clone_plan = None
    renderer._camera_rel_path = "Camera"
    renderer._render_product_paths = []
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
    renderer._camera_xform_query = None
    renderer._camera_paths_list = None
    renderer._object_xform_query = None
    renderer._object_paths_list = None
    renderer._deformable_points_query = None
    renderer._deformable_paths_list = None
    renderer._object_newton_indices = None
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    renderer._output_id_color_buffers = {}
    renderer._env_root_xforms = None
    return renderer


def _make_ovrtx_renderer_without_backend_legacy() -> OVRTXRenderer:
    legacy_renderer = _FakeLegacyRenderer()
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._renderer = legacy_renderer
    renderer._use_ovstage = False
    renderer._camera_xform_binding = None
    renderer._object_xform_binding = None
    renderer._deformable_points_binding = None
    renderer._clone_plan = None
    renderer._camera_rel_path = "Camera"
    renderer._render_product_paths = []
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
    renderer._object_newton_indices = None
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    renderer._output_id_color_buffers = {}
    return renderer


def _make_camera_render_spec(num_envs: int = 1) -> CameraRenderSpec:
    spawn = PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    cfg = CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=spawn,
        data_types=["rgb"],
    )
    camera_paths = tuple(f"/World/envs/env_{env_idx}/Camera" for env_idx in range(num_envs))
    return CameraRenderSpec(
        cfg=cfg,
        device="cpu",
        num_instances=num_envs,
        camera_prim_paths=camera_paths,
        view_count=num_envs,
        camera_path_relative_to_env_0="Camera",
    )


# ---------------------------------------------------------------------------
# Path-agnostic tests
# ---------------------------------------------------------------------------


def test_resolve_clone_plan_returns_homogeneous_when_unpublished(monkeypatch: pytest.MonkeyPatch):
    """Missing published plan falls back to env_0 replication."""
    _patch_simulation_context(monkeypatch, None)

    num_envs = 4
    resolved = _resolve_clone_plan(num_envs)
    expected = _create_homogeneous_clone_plan(num_envs)

    assert resolved.sources == expected.sources
    assert resolved.destinations == expected.destinations
    assert torch.equal(resolved.clone_mask, expected.clone_mask)


def test_resolve_clone_plan_returns_homogeneous_when_no_active_rows(monkeypatch: pytest.MonkeyPatch):
    """Plan with no active rows falls back to homogeneous replication."""
    published = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    _patch_simulation_context(monkeypatch, published)

    num_envs = 4
    resolved = _resolve_clone_plan(num_envs)
    expected = _create_homogeneous_clone_plan(num_envs)

    assert resolved.sources == expected.sources
    assert resolved.destinations == expected.destinations
    assert torch.equal(resolved.clone_mask, expected.clone_mask)


def test_resolve_clone_plan_filters_inactive_rows(monkeypatch: pytest.MonkeyPatch):
    """Inactive clone-plan rows are removed before OVRTX uses the plan."""
    published = ClonePlan(
        sources=(
            "/World/envs/env_0/Robot",
            "/World/envs/env_1/Object",
            "/World/envs/env_0/table",
        ),
        destinations=(
            "/World/envs/env_{}/Robot",
            "/World/envs/env_{}/Object",
            "/World/envs/env_{}/table",
        ),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, False, False],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    _patch_simulation_context(monkeypatch, published)

    resolved = _resolve_clone_plan(4)

    assert resolved.sources == (published.sources[0], published.sources[2])
    assert resolved.destinations == (published.destinations[0], published.destinations[2])
    assert torch.equal(resolved.clone_mask, published.clone_mask[[0, 2]])


def test_resolve_clone_plan_returns_published_plan_when_all_active(monkeypatch: pytest.MonkeyPatch):
    """Fully active published plans are reused without copying."""
    published = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    _patch_simulation_context(monkeypatch, published)

    resolved = _resolve_clone_plan(3)

    assert resolved is published


def test_write_file_creates_parent_directory_and_writes_utf8(tmp_path: Path):
    """_write_file creates nested directories and writes UTF-8 content."""
    output_dir = tmp_path / "nested" / "usd"

    _write_file(output_dir, "stage.usda", "#usda 1.0\n")

    output_path = output_dir / "stage.usda"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_prepare_stage_writes_pre_ovrtx_stage_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_stage writes the raw stage before OVRTX-specific preparation."""
    _patch_simulation_context(monkeypatch, None)

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    expected_pre_export = stage.ExportToString()

    renderer.prepare_stage(stage, 2)

    pre_stage_path = tmp_path / _PRE_OVRTX_STAGE_FILE
    assert pre_stage_path.is_file()
    assert pre_stage_path.read_text(encoding="utf-8") == expected_pre_export
    assert (tmp_path / _OVRTX_STAGE_FILE).exists() is False


def test_prepare_stage_skips_temp_usd_write_when_temp_usd_dir_unset(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage does not write debug dumps when temp_usd_dir is None."""
    _patch_simulation_context(monkeypatch, None)
    write_calls: list[tuple[Path, str, str]] = []

    def _record_write(output_dir: Path, file_name: str, content: str) -> None:
        write_calls.append((output_dir, file_name, content))

    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer._write_file", _record_write)

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = None

    renderer.prepare_stage(stage, 2)

    assert write_calls == []


# ---------------------------------------------------------------------------
# Cloning tests — ovstage path
# ---------------------------------------------------------------------------


@_skip_if_not_ovstage
def test_clone_sources_in_ovrtx_homogeneous_row_ovstage():
    """Homogeneous env_0 row clones only env_1..env_{N-1} (env_0 is the source) — ovstage path."""
    fake_stage = _FakeStage()
    num_envs = 4
    identity_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))

    renderer = _make_ovrtx_renderer_without_backend(fake_stage)
    renderer._clone_plan = _create_homogeneous_clone_plan(4)
    renderer._env_root_xforms = identity_xforms

    renderer._clone_sources_ovstage()

    clone_calls = [(source, targets) for source, targets, _ in fake_stage.clone_calls]
    assert clone_calls == [
        (
            "/World/envs/env_0",
            ["/World/envs/env_1", "/World/envs/env_2", "/World/envs/env_3"],
        )
    ]


@_skip_if_not_ovstage
def test_clone_sources_in_ovrtx_heterogeneous_rows_ovstage():
    """Each active clone-plan row issues its own clone call — ovstage path."""
    num_envs = 4
    identity_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
    fake_stage = _FakeStage()

    renderer = _make_ovrtx_renderer_without_backend(fake_stage)
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    renderer._env_root_xforms = identity_xforms

    renderer._clone_sources_ovstage()

    clone_calls = [(source, targets) for source, targets, _ in fake_stage.clone_calls]
    assert clone_calls == [
        (
            "/World/envs/env_0/Robot",
            [
                "/World/envs/env_1/Robot",
                "/World/envs/env_2/Robot",
                "/World/envs/env_3/Robot",
            ],
        ),
        (
            "/World/envs/env_1/Object",
            ["/World/envs/env_2/Object", "/World/envs/env_3/Object"],
        ),
    ]


@_skip_if_not_ovstage
def test_clone_sources_in_ovrtx_skips_empty_target_rows_ovstage():
    """Rows with no clone targets do not call clone — ovstage path."""
    num_envs = 4
    identity_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
    fake_stage = _FakeStage()

    renderer = _make_ovrtx_renderer_without_backend(fake_stage)
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_7/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    )
    renderer._env_root_xforms = identity_xforms

    renderer._clone_sources_ovstage()

    assert fake_stage.clone_calls == []


@_skip_if_not_ovstage
def test_clone_sources_in_ovrtx_raises_on_clone_failure_ovstage():
    """Clone failures surface as RuntimeError with the row index — ovstage path."""
    num_envs = 2
    identity_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
    fake_stage = _FakeStage()

    def _failing_clone(source, targets, ordinal):
        raise OSError("clone failed")

    fake_stage.clone = _failing_clone

    renderer = _make_ovrtx_renderer_without_backend(fake_stage)
    renderer._clone_plan = _create_homogeneous_clone_plan(2)
    renderer._env_root_xforms = identity_xforms

    with pytest.raises(RuntimeError, match="Failed to clone row 0"):
        renderer._clone_sources_ovstage()


@_skip_if_not_ovstage
def test_clone_sources_in_ovrtx_restores_env_root_transforms_ovstage():
    """Pre-captured omni:xform snapshots are written back after cloning — ovstage path.

    The env-root transforms are captured from the live USD stage in ``prepare_stage``
    (see :meth:`_capture_env_root_xforms_ovstage`) before export strips non-source
    envs, so cloning only clones and writes — it does not read back from ovstage.
    """
    num_envs = 4
    captured_transforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
    captured_transforms[:, 3, :3] = np.array([[i * 2.0, 0.0, 0.0] for i in range(num_envs)])

    fake_stage = _FakeStage()

    call_order: list[str] = []

    original_clone = fake_stage.clone

    def _tracked_clone(source, targets, ordinal):
        call_order.append("clone")
        original_clone(source, targets, ordinal)

    original_write = fake_stage.write_attribute

    def _tracked_write(query, attr, ordinal, tensors, **kwargs):
        call_order.append("write")
        return original_write(query, attr, ordinal, tensors, **kwargs)

    fake_stage.clone = _tracked_clone
    fake_stage.write_attribute = _tracked_write

    renderer = _make_ovrtx_renderer_without_backend(fake_stage)
    renderer._clone_plan = _create_homogeneous_clone_plan(num_envs)
    renderer._env_root_xforms = captured_transforms

    renderer._clone_sources_ovstage()

    # Restore must happen after cloning so the source-env-root xform does not persist
    # on the cloned envs; the pre-captured snapshot is not read back from the stage.
    assert call_order == ["clone", "write"]

    xform_writes = [c for c in fake_stage.write_calls if c["attr"] == "omni:xform"]
    assert len(xform_writes) == 1

    # The snapshot is consumed and cleared so it cannot be reused on a later clone.
    assert renderer._env_root_xforms is None


@_skip_if_not_ovstage
def test_capture_env_root_xforms_reads_live_stage_transforms_ovstage():
    """_capture_env_root_xforms_ovstage snapshots per-env root local-to-world transforms.

    This runs before export strips non-source envs, so it reads the live USD stage
    rather than ovstage; :meth:`_clone_sources_ovstage` later consumes the snapshot.
    """
    num_envs = 3
    stage = _make_multi_env_stage(num_envs)
    # Give each env root a distinct translation so the capture is verifiable.
    for i in range(num_envs):
        env_xform = UsdGeom.Xform.Get(stage, f"/World/envs/env_{i}")
        env_xform.AddTranslateOp().Set(Gf.Vec3d(i * 2.0, 0.0, 0.0))

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._capture_env_root_xforms_ovstage(stage, num_envs)

    captured = renderer._env_root_xforms
    assert captured is not None
    assert captured.shape == (num_envs, 4, 4)
    # USD stores translation in the last row (row-vector convention).
    for i in range(num_envs):
        np.testing.assert_allclose(captured[i, 3, :3], [i * 2.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Cloning tests — legacy path
# ---------------------------------------------------------------------------


@_skip_if_not_legacy
def test_clone_sources_in_ovrtx_homogeneous_row_legacy():
    """Homogeneous env_0 row clones only env_1..env_{N-1} (env_0 is the source) — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()
    renderer._clone_plan = _create_homogeneous_clone_plan(4)

    renderer._clone_sources_legacy()

    clone_calls = renderer._renderer.clone_usd_calls
    assert clone_calls == [
        (
            "/World/envs/env_0",
            ["/World/envs/env_1", "/World/envs/env_2", "/World/envs/env_3"],
        )
    ]


@_skip_if_not_legacy
def test_clone_sources_in_ovrtx_heterogeneous_rows_legacy():
    """Each active clone-plan row issues its own clone_usd call — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        ),
    )

    renderer._clone_sources_legacy()

    clone_calls = renderer._renderer.clone_usd_calls
    assert clone_calls == [
        (
            "/World/envs/env_0/Robot",
            [
                "/World/envs/env_1/Robot",
                "/World/envs/env_2/Robot",
                "/World/envs/env_3/Robot",
            ],
        ),
        (
            "/World/envs/env_1/Object",
            ["/World/envs/env_2/Object", "/World/envs/env_3/Object"],
        ),
    ]


@_skip_if_not_legacy
def test_clone_sources_in_ovrtx_skips_empty_target_rows_legacy():
    """Rows with no clone targets do not call clone_usd — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_7/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    )

    renderer._clone_sources_legacy()

    assert renderer._renderer.clone_usd_calls == []


@_skip_if_not_legacy
def test_clone_sources_in_ovrtx_raises_on_clone_failure_legacy():
    """Clone failures surface as RuntimeError with the row index — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()

    def _failing_clone(source, targets):
        raise OSError("clone failed")

    renderer._renderer.clone_usd = _failing_clone
    renderer._clone_plan = _create_homogeneous_clone_plan(2)

    with pytest.raises(RuntimeError, match="Failed to clone row 0"):
        renderer._clone_sources_legacy()


@_skip_if_not_legacy
def test_clone_sources_in_ovrtx_restores_env_root_transforms_legacy():
    """Pre-clone xform snapshots are written back after cloning — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()
    renderer._clone_plan = _create_homogeneous_clone_plan(4)

    call_order: list[str] = []

    original_clone = renderer._renderer.clone_usd

    def _tracked_clone(source, targets):
        call_order.append("clone")
        original_clone(source, targets)

    original_write = renderer._renderer.write_attribute

    def _tracked_write(prim_paths, attribute_name, tensor, **kwargs):
        call_order.append("write")
        original_write(prim_paths, attribute_name, tensor, **kwargs)

    original_read = renderer._renderer.read_attribute

    def _tracked_read(attr, paths, **kwargs):
        call_order.append("read")
        return original_read(attr, paths, **kwargs)

    renderer._renderer.clone_usd = _tracked_clone
    renderer._renderer.write_attribute = _tracked_write
    renderer._renderer.read_attribute = _tracked_read

    renderer._clone_sources_legacy()

    assert call_order == ["read", "clone", "write"]

    xform_writes = [c for c in renderer._renderer.write_attribute_calls if c["attr"] == "omni:xform"]
    assert len(xform_writes) == 1


# ---------------------------------------------------------------------------
# initialize_from_spec — writes combined stage dump
# ---------------------------------------------------------------------------


@_skip_if_not_ovstage
def test_initialize_from_spec_writes_combined_stage_dump_ovstage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """_initialize_from_spec writes the combined stage when temp_usd_dir is set — ovstage path."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    renderer._exported_usd_string = "#usda 1.0\n"

    fake_stage = _FakeStage()
    fake_paths = _FakePathDictionary()

    open_calls: list[str] = []

    def _fake_open_usd_from_string(stage, usd_string, ordinal, domains):
        open_calls.append(usd_string)

    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.population.open_usd_from_string", _fake_open_usd_from_string
    )

    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer.ovstage.Stage", lambda name: fake_stage)
    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer.ovstage.PathDictionary", lambda stage: fake_paths)

    renderer._initialize_from_spec(_make_camera_render_spec(num_envs=1))

    combined_path = tmp_path / _OVRTX_STAGE_FILE
    combined_text = combined_path.read_text(encoding="utf-8")
    assert combined_text.startswith("#usda 1.0")
    assert 'def RenderProduct "RenderProduct"' in combined_text
    assert open_calls == [combined_text]
    assert renderer._exported_usd_string is None


@_skip_if_not_legacy
def test_initialize_from_spec_writes_combined_stage_dump_legacy(tmp_path: Path):
    """_initialize_from_spec writes the combined stage when temp_usd_dir is set — legacy path."""
    renderer = _make_ovrtx_renderer_without_backend_legacy()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    renderer._exported_usd_string = "#usda 1.0\n"

    renderer._initialize_from_spec(_make_camera_render_spec(num_envs=1))

    combined_path = tmp_path / _OVRTX_STAGE_FILE
    combined_text = combined_path.read_text(encoding="utf-8")
    assert combined_text.startswith("#usda 1.0")
    assert 'def RenderProduct "RenderProduct"' in combined_text
    assert renderer._renderer.open_usd_calls == [combined_text]
    assert renderer._exported_usd_string is None


# ---------------------------------------------------------------------------
# prepare_stage — clone plan and export
# ---------------------------------------------------------------------------


@_skip_if_not_ovstage
def test_prepare_stage_stores_clone_plan_and_exports_ovstage(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage resolves the clone plan and strips non-source env roots — ovstage path."""
    num_envs = 4

    published = _create_homogeneous_clone_plan(num_envs)
    _patch_simulation_context(monkeypatch, published)

    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend()

    renderer.prepare_stage(stage, 4)

    assert renderer._clone_plan is not None
    assert renderer._clone_plan.sources == published.sources

    # ovstage export strips non-source env roots entirely so stage.clone can repopulate them.
    _assert_export_contains_env_roots_and_children(renderer._exported_usd_string, [0])
    _assert_export_omits_env_roots(renderer._exported_usd_string, [1, 2, 3])


@_skip_if_not_legacy
def test_prepare_stage_stores_clone_plan_and_exports_legacy(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage resolves the clone plan and deactivates non-source env children — legacy path."""
    num_envs = 4

    published = _create_homogeneous_clone_plan(num_envs)
    _patch_simulation_context(monkeypatch, published)

    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend_legacy()

    renderer.prepare_stage(stage, 4)

    assert renderer._clone_plan is not None
    assert renderer._clone_plan.sources == published.sources

    # Legacy export keeps env root prims but deactivates their non-source descendants.
    _assert_export_contains_env_roots_and_children(renderer._exported_usd_string, [0])
    _assert_export_contains_env_roots_but_omits_children(renderer._exported_usd_string, [1, 2, 3])
