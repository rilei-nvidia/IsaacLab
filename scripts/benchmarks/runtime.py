# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Supported launcher for :mod:`isaaclab.benchmark.entrypoints.runtime`.

Beyond the entrypoint's own arguments, this launcher accepts::

    --ovrtx_cpu_profile <path.json>

which enables the Carbonite CPU profiler for an OVRTX run and writes its trace to
``<path.json>`` at process shutdown. It is handled here rather than in the entrypoint
because the profiler plugin reads its settings when the Carbonite framework starts, so
they must be queued before any Isaac Lab module is imported. Profiling instrumentation
adds per-scope overhead, so throughput measured with it enabled is not comparable to an
un-profiled run.

Usage example::

    .venv/bin/python scripts/benchmarks/runtime.py \\
        --task Isaac-Lift-KukaAllegro-Camera --num_envs 1024 --num_steps 150 \\
        presets=newton_mjwarp,ovrtx_renderer,rgb128,single_camera \\
        --ovrtx_cpu_profile profiles/ovrtx_lift.json
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

_PROFILE_ARG = "--ovrtx_cpu_profile"

# Profiler mask bit 1 covers ``kSceneRendererContextProfilerMask``; bit 0 is the default
# application scope. Both are needed to see renderer-side scopes.
_PROFILER_MASK = 3

# Location of the ovrtx shared library relative to the installed package root.
_LIB_RELPATH = pathlib.PurePath("bin", "plugins", "rtx", "libovrtx.dylib.so")


class _OvxString(ctypes.Structure):
    """Mirror of the ovrtx ``ovx_string_t`` ABI struct (pointer plus explicit length)."""

    _fields_ = [("ptr", ctypes.c_char_p), ("length", ctypes.c_size_t)]


class _OvrtxResult(ctypes.Structure):
    """Mirror of the ovrtx ``ovrtx_result_t`` ABI struct (zero status means success)."""

    _fields_ = [("status", ctypes.c_int32)]


def _pop_profile_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract ``--ovrtx_cpu_profile`` from *argv* so it never reaches the entrypoint parser.

    Args:
        argv: Command-line arguments excluding the script path.

    Returns:
        The requested trace path (``None`` when the flag is absent) and the remaining arguments.

    Raises:
        SystemExit: If the flag is given without a value.
    """
    remaining: list[str] = []
    path: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == _PROFILE_ARG:
            if index + 1 >= len(argv):
                raise SystemExit(f"{_PROFILE_ARG} requires a path to write the JSON trace to.")
            path = argv[index + 1]
            index += 2
            continue
        if arg.startswith(f"{_PROFILE_ARG}="):
            path = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return path, remaining


def _library_path() -> pathlib.Path:
    """Locate ``libovrtx.dylib.so`` inside the installed ``ovrtx`` package.

    The path is derived from ``ovrtx.__file__`` rather than the interpreter prefix: under a
    virtual environment the stdlib resolves to the base interpreter, not to the environment's
    ``site-packages``. Importing ``ovrtx`` also sets up the native library search paths.

    Returns:
        Absolute path to the ovrtx shared library.

    Raises:
        SystemExit: If ``ovrtx`` is not installed or the shared library is missing.
    """
    try:
        import ovrtx
    except ImportError as err:
        raise SystemExit(f"{_PROFILE_ARG} requires the 'ovrtx' package, which is not installed.") from err

    path = pathlib.Path(ovrtx.__file__).parent / _LIB_RELPATH
    if not path.exists():
        raise SystemExit(f"ovrtx shared library not found at {path}.")
    return path


def _enable_ovrtx_cpu_profiler(output_path: str) -> None:
    """Queue the Carbonite CPU-profiler settings so the plugin picks them up at framework startup.

    Args:
        output_path: Destination for the uncompressed JSON trace. Parent directories are
            created if needed, since the profiler does not create them itself.

    Raises:
        SystemExit: If the ovrtx library rejects the profiler settings.
    """
    resolved = pathlib.Path(output_path).expanduser().absolute()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    library = ctypes.CDLL(str(_library_path()), mode=ctypes.RTLD_GLOBAL)
    library.applyRTXSettings.argtypes = [_OvxString]
    library.applyRTXSettings.restype = _OvrtxResult

    settings = " ".join([
        "--/app/profileFromStart=true",
        "--/app/profilerBackend=cpu",
        f"--/app/profilerMask={_PROFILER_MASK}",
        "--/plugins/carb.profiler-cpu.plugin/saveProfile=true",
        f"--/plugins/carb.profiler-cpu.plugin/filePath={resolved}",
        "--/plugins/carb.profiler-cpu.plugin/compressProfile=false",
    ]).encode()

    result = library.applyRTXSettings(_OvxString(settings, len(settings)))
    if result.status != 0:
        raise SystemExit(f"Failed to apply OVRTX profiler settings (status={result.status}).")
    print(f"[benchmark] OVRTX CPU profiler enabled -> {resolved}", flush=True)


if __name__ == "__main__":
    profile_path, benchmark_argv = _pop_profile_arg(sys.argv[1:])
    if profile_path is not None:
        _enable_ovrtx_cpu_profiler(profile_path)

    # Imported after the profiler is configured so no Isaac Lab module loads first.
    from isaaclab.benchmark.entrypoints.runtime import run

    run(benchmark_argv)
