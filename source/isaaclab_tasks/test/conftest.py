# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the isaaclab_tasks test suite.

Adds this directory to ``sys.path`` so tests located in the ``core/`` and ``contrib/``
sub-directories can import the shared helpers (``env_test_utils``, ``rendering_test_utils``)
that live at the test-suite root.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def enable_scene_partition(monkeypatch):
    """Set ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` for the duration of one test."""
    monkeypatch.setenv("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", "1")


@pytest.fixture(autouse=True)
def enable_ovstage_for_rendering_tests(request, monkeypatch):
    """Enable the OVRTX ovstage code path for the rendering-correctness suite only.

    The ``test_rendering_*.py`` tests exercise the ovstage scene-ownership path (ovstage owns the
    scene, ovrtx renders) by setting ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``. All other tests are left
    untouched (ovstage stays opt-in and disabled by default). The variable is restored after each
    test via ``monkeypatch``.
    """
    if request.path.name.startswith("test_rendering_"):
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
