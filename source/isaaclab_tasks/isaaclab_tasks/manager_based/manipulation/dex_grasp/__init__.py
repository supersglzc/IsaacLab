# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dex-Grasp manipulation task -- UFactory 850 + Allegro right hand.

Single-arm 22-DoF (6 arm + 16 hand) robot grasps and lifts a `dog` object off
the lab table. Scene + actuator stack + init pose mirror the bidex
`InsertDrawer` task RIGHT-ROBOT half byte-for-byte (USD, init pos, joint qpos,
stiffness/damping per joint group). The left robot and drawer from the bidex
scene are dropped.
"""

from . import mdp  # noqa: F401 -- re-exports task-local helpers
