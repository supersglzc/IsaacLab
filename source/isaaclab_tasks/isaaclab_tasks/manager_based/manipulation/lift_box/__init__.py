# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm cooperative box-lift task (Triton-Lift-Box).

Two FR3 + Franka-hand robots cooperate to lift a eurobox (40 x 30 x 22 cm,
0.5 kg) off a lab table. Scene + sim timing + actuation pattern mirror
`manipulation/insert_drawer/`.
"""

from . import mdp  # noqa: F401 — re-exports task-local helpers
