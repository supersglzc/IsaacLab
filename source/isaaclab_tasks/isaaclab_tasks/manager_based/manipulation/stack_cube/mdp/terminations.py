# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the stack_cube task.

The only active termination wired into `TerminationsCfg` is `time_out` from
the shared `isaaclab.envs.mdp` namespace; this module only defines the
cross-module helper `_no_contact_between_cube_and_gripper_or_ee`, used by
`mdp.rewards.three_tier_tower_bonus_once_per_episode`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _no_contact_between_cube_and_gripper_or_ee(
    env: "ManagerBasedRLEnv",
    cube_idx: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """True per env if neither fingertip NOR the panda_hand body has contact
    force vs `cube_<cube_idx>`. `cube_idx` ∈ {0, 1, 2} matches the filter list
    declared on each contact sensor in StackCubeSceneCfg.
    """
    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    hand = env.scene["hand_contact"]
    left_f = torch.norm(left.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    hand_f = torch.norm(hand.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    in_contact = (left_f > eps) | (right_f > eps) | (hand_f > eps)
    return ~in_contact


