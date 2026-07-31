# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .base_visualizer import BaseVisualizer


_VISUALIZER_EXTRAS = {
    "kit": "isaacsim",
    "rerun": "rerun",
    "viser": "viser",
}


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use specific configs from isaaclab_visualizers: KitVisualizerCfg, NewtonVisualizerCfg,
        RerunVisualizerCfg, or ViserVisualizerCfg (from isaaclab_visualizers.kit/.newton/.rerun/.viser).
    """

    # Primary interactive camera settings
    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Interactive visualizer camera eye position in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Interactive visualizer camera look-at target in world coordinates."""

    focal_length: float = 12.0
    """Camera focal length in millimeters for visualizer camera views."""

    # ── Streaming view ────────────────────────────────────────────────────────
    # Captures pixels from a camera sensor (existing or auto-created), tiles them
    # across envs and GT types, and shows the result as an image panel in interactive
    # visualizers (Newton GL, Kit) or pushes it per-step to sink-based ones (Rerun, Viser).

    streaming_view: bool = False
    """Enable the streaming camera image view."""

    # Source — existing sensor (takes priority when set)
    streaming_sensor_prim_path: str | None = None
    """Prim path of an existing TiledCamera sensor to stream from.

    When set, all ``streaming_cam_*`` fields are ignored.  Should point to an
    existing camera sensor, e.g. ``"/World/envs/*/Camera"``.
    """

    # Source — auto-created camera (used when streaming_sensor_prim_path is None)
    streaming_cam_target_prim_path: str = "/World/envs/*/Robot"
    """Target prim for the auto-created streaming camera (ignored when
    :attr:`streaming_sensor_prim_path` is set)."""

    streaming_cam_eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Eye offset [m] for the auto-created streaming camera relative to the target prim."""

    streaming_cam_renderer: str | None = None
    """Renderer for the auto-created streaming camera.

    One of ``"newton_warp"``, ``"ovrtx"``, ``"isaac_rtx"``, or ``None`` to use
    the backend default.  Ignored when :attr:`streaming_sensor_prim_path` is set.
    Validated at initialisation time.
    """

    # Shared settings
    streaming_envs: int | list[int] = 1
    """Environments to capture.

    * ``int`` — randomly sample this many envs each reset (from visible envs).
    * ``list[int]`` — capture exactly these env indices.
    """

    streaming_gt_types: list[str] = ("rgb",)
    """GT data types displayed left-to-right per environment row.

    Valid values: ``"rgb"``, ``"depth"``, ``"segmentation"``.
    Validated against :data:`~isaaclab.envs.utils.camera_colorizer.SUPPORTED_GT_TYPES`
    at initialisation time (only when :attr:`streaming_view` is ``True``).
    """

    streaming_depth_min: float = 0.1
    """Near-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    streaming_depth_max: float = 10.0
    """Far-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    # Partial visualization settings
    max_visible_envs: int | None = None
    """Upper bound on how many envs are shown.

    * If visible_env_indices is not None, then this field will apply also
      to the explicit env indices set to the visible_env_indices.
    """

    visible_env_indices: list[int] | None = None
    """env indices to visualize in order (out-of-range indices are dropped)."""

    randomly_sample_visible_envs: bool = True
    """If ``max_visible_envs`` is provided, when enabled, selected visible envs are randomly sampled.
       If disabled, the first ``max_visible_envs`` envs are selected.

    * Note: ``visible_env_indices`` overrides this field.
    """

    # Visualization Markers
    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    # Live Plots
    enable_live_plots: bool = True
    """Stream per-step scalar data (manager terms, episode reward, episode length) into the visualizer.

    Plot windows start hidden or collapsed by default and can be toggled open at runtime.
    Set to ``False`` to disable live plots entirely and avoid any collection overhead.
    """

    live_plots_update_interval: int = 5
    """Collect and push live plot data every ``N`` simulation steps (default: every 5 steps)."""

    # Internal
    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'viser', 'kit'). Must be overridden by subclasses."""

    def get_visualizer_type(self) -> str | None:
        """Get the visualizer type identifier.

        Returns:
            The visualizer type string, or None if not set (base class).
        """
        return self.visualizer_type

    def create_visualizer(self) -> BaseVisualizer:
        """Create visualizer instance from this config using factory pattern.

        Loads the matching backend from isaaclab_visualizers (e.g. isaaclab_visualizers.rerun).

        Raises:
            ValueError: If visualizer_type is None (base class used directly) or not registered.
            ImportError: If isaaclab_visualizers or the requested backend extra is not installed.
        """
        from .visualizer import Visualizer

        if self.visualizer_type is None:
            raise ValueError(
                "Cannot create visualizer from base VisualizerCfg class. "
                "Use a specific config from isaaclab_visualizers "
                "(e.g. KitVisualizerCfg, NewtonVisualizerCfg, RerunVisualizerCfg, ViserVisualizerCfg)."
            )

        try:
            return Visualizer(self)
        except (ValueError, ImportError, ModuleNotFoundError) as exc:
            if self.visualizer_type in ("newton", "rerun", "viser", "kit"):
                extra = _VISUALIZER_EXTRAS.get(self.visualizer_type)
                if extra is None:
                    install_hint = "Synchronize the project environment with: uv sync."
                else:
                    install_hint = f"Run your command with: uv run --extra {extra} <command>."
                raise ImportError(
                    f"Could not import visualizer '{self.visualizer_type}' from isaaclab_visualizers. "
                    f"{install_hint}\nOriginal error: {exc}"
                ) from exc
            raise
