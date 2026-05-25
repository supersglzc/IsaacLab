# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event-manager terms for the stack_cube task.

There are no task-specific event terms. Per-cube XY/Z spawn pose comes from
the standard ``mdp.reset_root_state_uniform`` terms wired in
``stack_cube_env_cfg.EventCfg``.

This module is kept (rather than deleted) so ``mdp/__init__.py``'s
``from .events import *`` keeps working without producing import errors.
``dr-generator`` may add startup / interval DR helpers here in §7.
"""
