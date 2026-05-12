# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the stack_cube task.

Success: `cube_0_stacked_on_cube_1` — cube_0 is on top of cube_1.
    |cube_0.xy - cube_1.xy| < xy_threshold
    |cube_0.z  - cube_1.z - CUBE_SIZE| < z_threshold

Failure: `cube_0_dropping` — cube_0 falls off the table (z below table_height
by more than `drop_margin` metres).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Module-level constant: cube edge length used to compute the expected
# stacked z-gap between cube_0 (top) and cube_1 (bottom).
CUBE_SIZE = 0.043


def cube_0_stacked_on_cube_1(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True when cube_0 is stacked on top of cube_1.

    Conditions (both must hold per env):
        |cube_0.xy - cube_1.xy| < xy_threshold
        |cube_0.z  - cube_1.z - CUBE_SIZE| < z_threshold
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]

    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]

    xy_distance = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]

    aligned_xy = xy_distance < xy_threshold
    on_top_z = torch.abs(z_gap - CUBE_SIZE) < z_threshold
    return aligned_xy & on_top_z


def cube_0_dropping(
    env: "ManagerBasedRLEnv",
    drop_margin: float = 0.05,
    table_height: float = 0.0,
) -> torch.Tensor:
    """True when cube_0 falls more than `drop_margin` metres below `table_height`.

    Mirror of LiftCube's `object_dropping` term, specialized to cube_0.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    z_floor = table_height - drop_margin
    return cube_0.data.root_pos_w[:, 2] < z_floor
