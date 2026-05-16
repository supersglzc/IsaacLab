# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event-manager terms for the stack_cup task.

There are no task-specific event terms. The success geometry (cube_0 stacked
on cube_1) is hardcoded in ``mdp.terminations.cube_0_stacked_on_cube_1``.
Per-cube XY/Z spawn pose comes from the standard ``mdp.reset_root_state_uniform``
terms wired in ``stack_cup_env_cfg.EventCfg``.

This module is kept (rather than deleted) so ``mdp/__init__.py``'s
``from .events import *`` keeps working without producing import errors.
``dr-generator`` may add startup / interval DR helpers here in §7.
"""
