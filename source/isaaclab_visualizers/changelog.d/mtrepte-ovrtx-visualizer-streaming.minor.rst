Added
^^^^^

* Added :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg` (``--viz newton_rtx``) backed
  by Newton's OVRTX path-tracer.  Both Newton GL and Newton RTX now share a common
  ``_NewtonViewerUIMixin`` that provides the same Isaac Lab HUD (Isaac Lab section,
  Streaming View controls, Live Plots, Rendering Options).

* Added **streaming view** to all five visualizer backends (Newton GL, Newton RTX, Rerun,
  Viser, Kit), replacing the old ``tiled_cam_*`` fields on :class:`~isaaclab.visualizers.VisualizerCfg`
  with a unified ``streaming_*`` API.  Streaming supports multiple ground-truth channels —
  ``"rgb"``, ``"depth"`` (turbo colormap with ``streaming_depth_min``/``streaming_depth_max``),
  ``"segmentation"`` (stable golden-ratio palette), and ``"normals"`` (XYZ→RGB) — composited
  into a single tiled image whose layout minimises ``|log(W/H)|``.

* Added :class:`~isaaclab.envs.utils.camera_colorizer.CameraFrameColorizer` in
  :mod:`isaaclab.envs.utils.camera_colorizer` with per-GT colorization helpers and a
  :data:`~isaaclab.envs.utils.camera_colorizer.SUPPORTED_GT_TYPES` registry.

* Added ``newton_rtx`` as a valid ``--viz`` CLI type in
  :class:`~isaaclab.sim.SimulationContext` and
  :class:`~isaaclab.physics.scene_data_requirements`.

Changed
^^^^^^^

* ``tiled_cam_*`` fields on :class:`~isaaclab.visualizers.VisualizerCfg` have been
  **replaced** by ``streaming_*`` fields.  Migrate by renaming:
  ``tiled_cam_view`` → ``streaming_view``, ``tiled_cam_num`` → ``streaming_envs``,
  ``tiled_cam_prim_path`` → ``streaming_sensor_prim_path``,
  ``tiled_cam_eye`` → ``streaming_cam_eye``,
  ``tiled_cam_target_prim_path`` → ``streaming_cam_target_prim_path``.

* Rerun blueprint now uses a full-screen ``Spatial2DView`` for the streaming composite
  as the primary view when streaming is active.

* Viser now skips ``log_state()`` when ``streaming_view=True``, showing the streaming
  composite as a letterboxed 16:9 background without 3D scene overlay.
