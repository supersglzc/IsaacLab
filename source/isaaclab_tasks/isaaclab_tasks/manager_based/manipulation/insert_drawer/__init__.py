# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert-into-Drawer manipulation task — Franka FR3 single-arm.

Long-horizon: open drawer -> drop DexCube inside -> close drawer.

Mirrors the structural conventions of `manipulation/stack_cube/` (scene
constants, EMA EE-delta action stack, FrameTransformer EE sensor, DexCube
spawn) and adds one new articulation (the prismatic drawer at
`nautilus/assets/drawer/drawer.usd`).
"""

from . import mdp  # re-export task-local action / observation / event / reward / termination helpers
