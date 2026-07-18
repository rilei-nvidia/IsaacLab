# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX deformable mesh point bindings."""

from __future__ import annotations

import contextlib
import importlib.util
from types import SimpleNamespace

import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    import numpy as np  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer, ovrtx_use_ovstage_enabled  # noqa: E402
else:
    np = None
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None

    def ovrtx_use_ovstage_enabled():
        return False


_skip_if_not_ovstage = pytest.mark.skipif(
    not ovrtx_use_ovstage_enabled(), reason="requires ISAAC_LAB_OVRTX_USE_OVSTAGE=1"
)
_skip_if_not_legacy = pytest.mark.skipif(ovrtx_use_ovstage_enabled(), reason="requires ISAAC_LAB_OVRTX_USE_OVSTAGE=0")


class _FakeOperation:
    """Fake ovstage Operation that succeeds immediately."""

    def wait(self):
        pass


class _FakeQuery:
    """Fake ovstage Query."""

    pass


class _FakeOvstageStage:
    """Minimal ovstage Stage stub for deformable binding tests."""

    def __init__(self):
        self.write_calls: list[dict] = []
        self.query_calls: list[list[str]] = []
        self.advance_floor_calls: list[int] = []
        self.release_query_calls: list = []
        self.write_calls_per_query: dict = {}
        self._last_query = None

    def query_from_path_list(self, path_list: list[str]) -> _FakeQuery:
        self.query_calls.append(list(path_list))
        q = _FakeQuery()
        self._last_query = q
        self.write_calls_per_query[id(q)] = []
        return q

    def write_attribute(self, query, attr, ordinal, tensors, **kwargs):
        call = {"query": query, "attr": attr, "ordinal": ordinal, "tensors": tensors, **kwargs}
        self.write_calls.append(call)
        if id(query) in self.write_calls_per_query:
            self.write_calls_per_query[id(query)].append(call)
        return _FakeOperation()

    def advance_write_floor(self, ordinal: int) -> _FakeOperation:
        self.advance_floor_calls.append(ordinal)
        return _FakeOperation()

    def release_query(self, query) -> _FakeOperation:
        self.release_query_calls.append(query)
        return _FakeOperation()


class _FakeOvstagePathDictionary:
    """Minimal ovstage PathDictionary stub."""

    def __init__(self):
        self._created_lists: list[list[str]] = []

    def create_path_list_from_strings(self, paths: list[str]) -> list[str]:
        pl = list(paths)
        self._created_lists.append(pl)
        return pl

    def destroy_path_list(self, path_list):
        pass

    def intern_token(self, s: str) -> int:
        return hash(s) & 0xFFFFFFFF


class _FakeLegacyRenderer:
    """Minimal legacy ovrtx Renderer stub for deformable binding tests."""

    def __init__(self):
        self.write_calls: list[dict] = []
        self.bind_array_calls: list[dict] = []
        self.bind_calls: list[dict] = []

    def write_attribute(self, prim_paths, attribute_name, tensor, **kwargs) -> None:
        self.write_calls.append({"attr": attribute_name, "prim_paths": list(prim_paths)})

    def bind_array_attribute(self, prim_paths, attribute_name, **kwargs):
        self.bind_array_calls.append({"attr": attribute_name, "prim_paths": list(prim_paths)})
        return SimpleNamespace(write=lambda *a, **kw: None, unbind=lambda: None)

    def bind_attribute(self, prim_paths, attribute_name, **kwargs):
        self.bind_calls.append({"attr": attribute_name, "prim_paths": list(prim_paths)})
        return SimpleNamespace(map=lambda **kw: contextlib.nullcontext(), unbind=lambda: None)

    def reset_stage(self) -> None:
        pass

    def detach_ovstage(self) -> None:
        pass


def _make_renderer_ovstage(device: str = "cpu") -> tuple[OVRTXRenderer, _FakeOvstageStage]:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._device = device
    renderer._camera_rel_path = "Camera"
    renderer._use_ovstage = True
    fake_stage = _FakeOvstageStage()
    renderer._stage = fake_stage
    renderer._stage_paths = _FakeOvstagePathDictionary()
    renderer._ovstage_exit_stack = contextlib.ExitStack()
    renderer._current_ordinal = 1
    renderer._renderer = SimpleNamespace(detach_ovstage=lambda: None)
    renderer._deformable_points_query = None
    renderer._deformable_paths_list = None
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    return renderer, fake_stage


def _make_renderer_legacy(device: str = "cpu") -> tuple[OVRTXRenderer, _FakeLegacyRenderer]:
    legacy_renderer = _FakeLegacyRenderer()
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._device = device
    renderer._camera_rel_path = "Camera"
    renderer._use_ovstage = False
    renderer._renderer = legacy_renderer
    renderer._camera_xform_binding = None
    renderer._object_xform_binding = None
    renderer._deformable_points_binding = None
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    return renderer, legacy_renderer


# ---------------------------------------------------------------------------
# Setup deformable bindings — ovstage path
# ---------------------------------------------------------------------------


@_skip_if_not_ovstage
def test_setup_deformable_bindings_binds_surface_mesh_points_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries create OVRTX ``points`` attribute queries — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert renderer._deformable_points_query is not None
    assert len(renderer._stage_paths._created_lists) == 1
    assert renderer._stage_paths._created_lists[0] == ["/World/envs/env_0/Deformable/mesh"]
    assert len(fake_stage.write_calls) == 2
    assert fake_stage.write_calls[0]["attr"] == "omni:resetXformStack"
    assert list(fake_stage.query_calls[0]) == ["/World/envs/env_0/Deformable/mesh"]
    assert fake_stage.write_calls[1]["attr"] == "omni:xform"
    assert len(renderer._deformable_particle_counts) == 1
    assert renderer._deformable_particle_counts[0] == 3
    assert renderer._deformable_particle_offsets == [7]


@_skip_if_not_ovstage
def test_setup_deformable_bindings_binds_volume_mesh_points_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Volume deformable registry entries create OVRTX ``points`` queries — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="volume",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert renderer._deformable_points_query is not None
    assert renderer._stage_paths._created_lists[0] == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_particle_offsets == [7]


@_skip_if_not_ovstage
def test_setup_deformable_bindings_binds_mixed_surface_and_volume_entries_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Surface and volume deformable registry entries bind together with distinct offsets — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    surface_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableSurface",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableSurface/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3],
        particles_per_body=3,
    )
    volume_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableVolume",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableVolume/mesh",
        deformable_type="volume",
        particle_offsets=[6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [surface_entry, volume_entry])

    renderer._setup_deformable_bindings(num_envs=2)

    assert renderer._stage_paths._created_lists[0] == [
        "/World/envs/env_0/DeformableSurface/mesh",
        "/World/envs/env_1/DeformableSurface/mesh",
        "/World/envs/env_0/DeformableVolume/mesh",
        "/World/envs/env_1/DeformableVolume/mesh",
    ]
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


@_skip_if_not_ovstage
def test_setup_deformable_bindings_works_without_stage_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Deformable bindings are created from registry metadata without a USD stage — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: None)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert renderer._deformable_points_query is not None
    assert renderer._stage_paths._created_lists[0] == ["/World/envs/env_0/Deformable/mesh"]


@_skip_if_not_ovstage
def test_setup_deformable_bindings_binds_all_surface_mesh_instances_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries bind every cloned visual mesh instance — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=4)

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert renderer._stage_paths._created_lists[0] == expected_paths
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


@_skip_if_not_ovstage
def test_update_deformable_points_writes_world_particle_positions_ovstage(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` slices are handed to OVRTX through update_geometries — ovstage path."""
    renderer, fake_stage = _make_renderer_ovstage()
    renderer._deformable_points_query = _FakeQuery()
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particle_counts = [3]
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(1.0, 2.0, 3.0),
            wp.vec3f(4.0, 5.0, 6.0),
            wp.vec3f(7.0, 8.0, 9.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    monkeypatch.setattr(ovrtx_renderer_module.wp, "synchronize_device", lambda device: None)

    renderer.update_geometries()

    assert len(fake_stage.write_calls) == 1
    call = fake_stage.write_calls[0]
    assert call["attr"] == "points"
    assert call["is_array"] is True
    # points must be written as a lanes=3 (vec3) column with POINT semantic to match ``point3f[] points``;
    # a flat lanes=1 tensor is rejected by ovstage as a type mismatch against the existing column.
    assert call["semantic"] == ovrtx_renderer_module.AttributeSemantic.POINT
    written_tensors = call["tensors"]
    assert len(written_tensors) == 1
    tensor = written_tensors[0]
    assert tensor.dtype.lanes == 3
    assert tensor.shape_tuple == (3,)
    # Only env-1's particle slice [1:4] is written, in world space.
    assert np.asarray(tensor._array).reshape(-1, 3).tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


# ---------------------------------------------------------------------------
# Setup deformable bindings — legacy path
# ---------------------------------------------------------------------------


@_skip_if_not_legacy
def test_setup_deformable_bindings_binds_surface_mesh_points_legacy(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries create bind_array_attribute call — legacy path."""
    renderer, fake_renderer = _make_renderer_legacy()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert renderer._deformable_points_binding is not None
    assert len(fake_renderer.write_calls) == 2
    assert fake_renderer.write_calls[0]["attr"] == "omni:resetXformStack"
    assert fake_renderer.write_calls[1]["attr"] == "omni:xform"
    assert len(fake_renderer.bind_array_calls) == 1
    assert fake_renderer.bind_array_calls[0]["attr"] == "points"
    assert fake_renderer.bind_array_calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_particle_offsets == [7]
    assert renderer._deformable_particle_counts == [3]


@_skip_if_not_legacy
def test_setup_deformable_bindings_binds_volume_mesh_points_legacy(monkeypatch: pytest.MonkeyPatch):
    """Volume deformable registry entries create bind_array_attribute call — legacy path."""
    renderer, fake_renderer = _make_renderer_legacy()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="volume",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=1)

    assert renderer._deformable_points_binding is not None
    assert fake_renderer.bind_array_calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_particle_offsets == [7]


@_skip_if_not_legacy
def test_setup_deformable_bindings_binds_mixed_surface_and_volume_entries_legacy(monkeypatch: pytest.MonkeyPatch):
    """Surface and volume registry entries bind together with distinct offsets — legacy path."""
    renderer, fake_renderer = _make_renderer_legacy()
    surface_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableSurface",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableSurface/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3],
        particles_per_body=3,
    )
    volume_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableVolume",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableVolume/mesh",
        deformable_type="volume",
        particle_offsets=[6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [surface_entry, volume_entry])

    renderer._setup_deformable_bindings(num_envs=2)

    assert fake_renderer.bind_array_calls[0]["prim_paths"] == [
        "/World/envs/env_0/DeformableSurface/mesh",
        "/World/envs/env_1/DeformableSurface/mesh",
        "/World/envs/env_0/DeformableVolume/mesh",
        "/World/envs/env_1/DeformableVolume/mesh",
    ]
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


@_skip_if_not_legacy
def test_setup_deformable_bindings_binds_all_surface_mesh_instances_legacy(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries bind every cloned visual mesh instance — legacy path."""
    renderer, fake_renderer = _make_renderer_legacy()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings(num_envs=4)

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert fake_renderer.bind_array_calls[0]["prim_paths"] == expected_paths
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


@_skip_if_not_legacy
def test_update_deformable_points_writes_world_particle_positions_legacy(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` slices are passed to the binding's write() — legacy path."""
    renderer, fake_renderer = _make_renderer_legacy()
    write_calls: list[dict] = []
    renderer._deformable_points_binding = SimpleNamespace(
        write=lambda tensors, **kw: write_calls.append({"tensors": tensors}),
        unbind=lambda: None,
    )
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particle_counts = [3]
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(1.0, 2.0, 3.0),
            wp.vec3f(4.0, 5.0, 6.0),
            wp.vec3f(7.0, 8.0, 9.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    monkeypatch.setattr(ovrtx_renderer_module.wp, "synchronize_device", lambda device: None)
    # Patch get_stream so the legacy path can call wp.get_stream without a real CUDA device.
    monkeypatch.setattr(
        ovrtx_renderer_module.wp,
        "get_stream",
        lambda device: SimpleNamespace(cuda_stream=0),
    )

    renderer.update_geometries()

    assert len(write_calls) == 1
    written_tensors = write_calls[0]["tensors"]
    assert len(written_tensors) == 1
    assert written_tensors[0].ptr == particle_q[1:4].ptr


# ---------------------------------------------------------------------------
# Shared validation tests (path-agnostic error cases)
# ---------------------------------------------------------------------------


def test_setup_deformable_bindings_rejects_offset_count_mismatch(monkeypatch: pytest.MonkeyPatch):
    """Registry entries must provide one particle offset per environment, listing every bad entry."""
    renderer, _ = _make_renderer_ovstage() if ovrtx_use_ovstage_enabled() else _make_renderer_legacy()
    bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )
    other_bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableOther",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableOther/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [bad_entry, other_bad_entry])

    with pytest.raises(RuntimeError, match="one particle offset per environment") as excinfo:
        renderer._setup_deformable_bindings(num_envs=2)

    message = str(excinfo.value)
    assert bad_entry.prim_path in message
    assert other_bad_entry.prim_path in message


def test_update_geometries_rejects_inconsistent_deformable_mapping(monkeypatch: pytest.MonkeyPatch):
    """Geometry sync fails fast when offset and count metadata drift out of alignment."""
    if ovrtx_use_ovstage_enabled():
        renderer, fake_stage = _make_renderer_ovstage()
        renderer._deformable_points_query = _FakeQuery()
    else:
        renderer, _ = _make_renderer_legacy()
        renderer._deformable_points_binding = SimpleNamespace(write=lambda *a, **kw: None, unbind=lambda: None)

    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [3, 3]
    particle_q = wp.array(
        [wp.vec3f(float(i), 0.0, 0.0) for i in range(4)],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    monkeypatch.setattr(ovrtx_renderer_module.wp, "synchronize_device", lambda device: None)
    if not ovrtx_use_ovstage_enabled():
        monkeypatch.setattr(
            ovrtx_renderer_module.wp,
            "get_stream",
            lambda device: SimpleNamespace(cuda_stream=0),
        )

    with pytest.raises(ValueError, match="zip"):
        renderer.update_geometries()
