Added
^^^^^

* Added :class:`~isaaclab.envs.utils.camera_colorizer.CameraFrameColorizer` with
  support for ``"rgb"``, ``"depth"`` (turbo colormap), ``"segmentation"``
  (stable golden-ratio palette), and ``"normals"`` (XYZ→RGB) ground-truth types.

* Added streaming layout helpers to :mod:`isaaclab.envs.utils.camera_view`:
  :func:`~isaaclab.envs.utils.camera_view.resolve_streaming_envs`,
  :func:`~isaaclab.envs.utils.camera_view.camera_gt_batch`, and
  :func:`~isaaclab.envs.utils.camera_view.compose_streaming_grid` with an
  auto-layout algorithm that minimises ``|log(W/H)|``.  Also added
  ``data_types`` parameter to
  :func:`~isaaclab.envs.utils.camera_view.create_visualizer_camera`.

* Added ``newton_rtx`` to :data:`~isaaclab.sim.simulation_context._VISUALIZER_TYPES`
  and :data:`~isaaclab.physics.scene_data_requirements._VISUALIZER_REQUIREMENTS`.

Changed
^^^^^^^

* Replaced ``tiled_cam_*`` fields on :class:`~isaaclab.visualizers.VisualizerCfg`
  with ``streaming_*`` fields.  See :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg`
  release notes for the migration guide.
