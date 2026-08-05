# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Supported launcher for :mod:`isaaclab.benchmark.entrypoints.runtime`.

Beyond the entrypoint's own arguments, this launcher accepts::

    --ovrtx_cpu_profile <path.json>     enable the Carbonite CPU profiler
    --ovrtx_profile_steps <n>           steady-state steps to keep (default 20, 0 = keep all)
    --ovrtx_profile_keep_raw            keep the full dump alongside the slice

Profiling is handled here rather than in the entrypoint because the profiler plugin reads
its settings when the Carbonite framework starts, so they must be queued before any Isaac
Lab module is imported. Profiling instrumentation adds per-scope overhead, so throughput
measured with it enabled is not comparable to an un-profiled run.

The raw dump is large (hundreds of MB) and its final flush does not close the JSON array,
which some viewers reject. So the profiler writes to ``<path>.raw.json`` and this launcher
slices a well-formed window of steady-state ``ovrtx_step_execute`` zones out of it into
``<path>``, small enough to open in https://ui.perfetto.dev. The window is taken from the
middle of the run so startup and teardown are excluded, and it scales with
``--ovrtx_profile_steps``. Pass ``--ovrtx_profile_steps 0`` to skip slicing and keep the
full dump.

Usage example::

    .venv/bin/python scripts/benchmarks/runtime.py \\
        --task Isaac-Lift-KukaAllegro-Camera --num_envs 1024 --num_steps 150 \\
        presets=newton_mjwarp,ovrtx_renderer,rgb128,single_camera \\
        --ovrtx_cpu_profile profiles/carb_profile.json
"""

from __future__ import annotations

import atexit
import ctypes
import gzip
import json
import pathlib
import sys
from collections.abc import Iterator
from typing import IO, Any

_PROFILE_ARG = "--ovrtx_cpu_profile"
_STEPS_ARG = "--ovrtx_profile_steps"
_KEEP_RAW_ARG = "--ovrtx_profile_keep_raw"

# Profiler mask bit 1 covers ``kSceneRendererContextProfilerMask``; bit 0 is the default
# application scope. Both are needed to see renderer-side scopes.
_PROFILER_MASK = 3

# Location of the ovrtx shared library relative to the installed package root.
_LIB_RELPATH = pathlib.PurePath("bin", "plugins", "rtx", "libovrtx.dylib.so")

# Zone that brackets one OVRTX submit, used to cut the trace on step boundaries.
_STEP_ZONE = "ovrtx_step_execute"

_DEFAULT_STEPS = 20


class _OvxString(ctypes.Structure):
    """Mirror of the ovrtx ``ovx_string_t`` ABI struct (pointer plus explicit length)."""

    _fields_ = [("ptr", ctypes.c_char_p), ("length", ctypes.c_size_t)]


class _OvrtxResult(ctypes.Structure):
    """Mirror of the ovrtx ``ovrtx_result_t`` ABI struct (zero status means success)."""

    _fields_ = [("status", ctypes.c_int32)]


"""
Argument handling.
"""


def _pop_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    """Extract ``--name <value>`` (or ``--name=<value>``) from *argv*.

    Args:
        argv: Command-line arguments excluding the script path.
        name: Long option to extract, including the leading dashes.

    Returns:
        The option value (``None`` when absent) and the remaining arguments.

    Raises:
        SystemExit: If the option is given without a value.
    """
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value.")
            value = argv[index + 1]
            index += 2
            continue
        if arg.startswith(f"{name}="):
            value = arg.split("=", 1)[1]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return value, remaining


def _pop_flag(argv: list[str], name: str) -> tuple[bool, list[str]]:
    """Extract a valueless ``--name`` flag from *argv*.

    Args:
        argv: Command-line arguments excluding the script path.
        name: Long option to extract, including the leading dashes.

    Returns:
        Whether the flag was present, and the remaining arguments.
    """
    present = name in argv
    return present, [arg for arg in argv if arg != name]


"""
Profiler setup. Must run before any Isaac Lab import.
"""


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


def _enable_ovrtx_cpu_profiler(output_path: pathlib.Path) -> None:
    """Queue the Carbonite CPU-profiler settings so the plugin picks them up at framework startup.

    Args:
        output_path: Destination for the uncompressed JSON trace. Parent directories are
            created if needed, since the profiler does not create them itself.

    Raises:
        SystemExit: If the ovrtx library rejects the profiler settings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    library = ctypes.CDLL(str(_library_path()), mode=ctypes.RTLD_GLOBAL)
    library.applyRTXSettings.argtypes = [_OvxString]
    library.applyRTXSettings.restype = _OvrtxResult

    settings = " ".join([
        "--/app/profileFromStart=true",
        "--/app/profilerBackend=cpu",
        f"--/app/profilerMask={_PROFILER_MASK}",
        "--/plugins/carb.profiler-cpu.plugin/saveProfile=true",
        f"--/plugins/carb.profiler-cpu.plugin/filePath={output_path}",
        "--/plugins/carb.profiler-cpu.plugin/compressProfile=false",
    ]).encode()

    result = library.applyRTXSettings(_OvxString(settings, len(settings)))
    if result.status != 0:
        raise SystemExit(f"Failed to apply OVRTX profiler settings (status={result.status}).")
    print(f"[benchmark] OVRTX CPU profiler enabled -> {output_path}", flush=True)


"""
Trace slicing. Runs after the profiler has flushed its dump.
"""


def _open_trace(path: pathlib.Path) -> IO[str]:
    """Open a trace dump, transparently handling gzip."""
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _iter_events(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Yield parsed trace events, tolerating a truncated final line.

    The profiler's last flush does not close the JSON array, so the file is parsed one
    object per line rather than as a whole document.

    Args:
        path: Trace dump to read.

    Yields:
        One decoded event per well-formed line.
    """
    with _open_trace(path) as stream:
        for line in stream:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue  # truncated tail


def _slice_trace(src: pathlib.Path, dst: pathlib.Path, steps: int) -> bool:
    """Write a well-formed slice of *steps* steady-state steps from *src* to *dst*.

    The window is taken from the middle of the run so that startup and teardown are
    excluded. Metadata events are always kept so the viewer can label its tracks.

    Args:
        src: Full profiler dump.
        dst: Destination for the sliced trace.
        steps: Number of ``ovrtx_step_execute`` zones to keep.

    Returns:
        Whether a slice was written.
    """
    step_zones = sorted(
        (event["ts"], event.get("dur", 0.0))
        for event in _iter_events(src)
        if event.get("ph") == "X" and event.get("name") == _STEP_ZONE
    )
    if not step_zones:
        print(
            f"[benchmark] no '{_STEP_ZONE}' zones found in {src}; keeping the full dump.",
            file=sys.stderr,
            flush=True,
        )
        return False

    start_index = max(0, len(step_zones) // 2 - steps // 2)
    window = step_zones[start_index : start_index + steps]
    first_ts = window[0][0]
    last_ts = window[-1][0] + window[-1][1]

    kept = [
        event
        for event in _iter_events(src)
        # "M" events are the thread/process names; keep them so tracks stay labelled.
        if event.get("ph") == "M" or first_ts <= event.get("ts", -1) <= last_ts
    ]
    with open(dst, "w") as stream:
        json.dump(kept, stream)

    span_ms = (last_ts - first_ts) / 1000
    print(
        f"[benchmark] {dst}: {len(kept)} events, {len(window)} steps ({span_ms:.1f} ms)"
        " -- open at https://ui.perfetto.dev",
        flush=True,
    )
    return True


def _finalize_profile(raw_path: pathlib.Path, dst: pathlib.Path, steps: int, keep_raw: bool) -> None:
    """Slice the profiler dump down to *steps* steps and drop the raw file.

    Args:
        raw_path: Path the profiler was told to write.
        dst: Destination for the sliced trace.
        steps: Number of steps to keep; zero keeps the full dump.
        keep_raw: Whether to keep the full dump after slicing.
    """
    if not raw_path.exists():
        print(
            f"[benchmark] profiler dump {raw_path} was not written; nothing to slice.",
            file=sys.stderr,
            flush=True,
        )
        return

    if steps == 0:
        raw_path.replace(dst)
        print(f"[benchmark] profiler dump -> {dst} (unsliced)", flush=True)
        return

    if _slice_trace(raw_path, dst, steps) and not keep_raw:
        raw_path.unlink()


if __name__ == "__main__":
    argv = sys.argv[1:]
    profile_path, argv = _pop_option(argv, _PROFILE_ARG)
    steps_value, argv = _pop_option(argv, _STEPS_ARG)
    keep_raw, argv = _pop_flag(argv, _KEEP_RAW_ARG)

    if profile_path is not None:
        profile_dst = pathlib.Path(profile_path).expanduser().absolute()
        profile_steps = _DEFAULT_STEPS if steps_value is None else int(steps_value)
        if profile_steps < 0:
            raise SystemExit(f"{_STEPS_ARG} must be zero or positive.")
        # The profiler writes the full dump here; the sliced result lands at profile_dst.
        profile_raw = profile_dst.with_name(f"{profile_dst.stem}.raw{profile_dst.suffix or '.json'}")
        _enable_ovrtx_cpu_profiler(profile_raw)
        # The dump is flushed during framework shutdown, which may not happen until the
        # interpreter tears down, so slice at exit rather than right after the run.
        atexit.register(_finalize_profile, profile_raw, profile_dst, profile_steps, keep_raw)

    # Imported after the profiler is configured so no Isaac Lab module loads first.
    from isaaclab.benchmark.entrypoints.runtime import run

    run(argv)
