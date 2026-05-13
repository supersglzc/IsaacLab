# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the stack_cube task.

Edit_mode_012 (3-tier tower):

  Success: `three_tier_tower_stacked` — cube_0 is on cube_1 AND cube_1 is on cube_2.
    AND of two per-pair stacking checks:
      |Δxy| < xy_threshold AND |Δz - CUBE_SIZE| < z_threshold

  Failure: `any_cube_dropping` — ANY of cube_0/cube_1/cube_2 falls below
    `table_height - drop_margin` metres.

Legacy 2-cube helpers `cube_0_stacked_on_cube_1` and `cube_0_dropping` are
PRESERVED — the §6 reward (`mdp.cube_0_stacked_bonus`) imports the same-named
function indirectly via `from .rewards import *`, but reward-generator will
rewire that in the very next phase. Keeping them removes the cross-section
coupling for this edit pass.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Module-level constant: cube edge length used to compute the expected
# stacked z-gap between consecutive layers.
CUBE_SIZE = 0.043


def _pair_stacked(
    top_pos: torch.Tensor,
    bottom_pos: torch.Tensor,
    xy_threshold: float,
    z_threshold: float,
) -> torch.Tensor:
    """Shared helper: True per env when `top` cube is stacked on `bottom` cube.

    `|top.xy - bottom.xy| < xy_threshold` AND `|Δz - CUBE_SIZE| < z_threshold`.
    """
    xy_distance = torch.norm(top_pos[:, :2] - bottom_pos[:, :2], dim=-1)
    z_gap = top_pos[:, 2] - bottom_pos[:, 2]
    aligned_xy = xy_distance < xy_threshold
    on_top_z = torch.abs(z_gap - CUBE_SIZE) < z_threshold
    return aligned_xy & on_top_z


def cube_0_stacked_on_cube_1(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True when cube_0 is stacked on top of cube_1 (2-cube legacy helper).

    Kept after edit_mode_012 so the §6 reward `mdp.cube_0_stacked_bonus`
    import stays valid; the actual TerminationsCfg.success is now
    `three_tier_tower_stacked`.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    return _pair_stacked(
        cube_0.data.root_pos_w[:, :3],
        cube_1.data.root_pos_w[:, :3],
        xy_threshold,
        z_threshold,
    )


def _gripper_far_from_cube_0(
    env: "ManagerBasedRLEnv",
    min_distance: float = 0.04,
) -> torch.Tensor:
    """True per env if EE (ee_frame target 0) is at least `min_distance` away from cube_0."""
    cube_0 = env.scene["cube_0"]
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube_0.data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    return distance > min_distance


def _no_contact_between_cube_0_and_gripper_or_ee(
    env: "ManagerBasedRLEnv",
    eps: float = 1e-3,
) -> torch.Tensor:
    """True per env if neither fingertip NOR the panda_hand body has contact force vs cube_0."""
    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    hand = env.scene["hand_contact"]
    left_f0 = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f0 = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    hand_f0 = torch.norm(hand.data.force_matrix_w[:, 0, 0, :], dim=-1)
    in_contact = (left_f0 > eps) | (right_f0 > eps) | (hand_f0 > eps)
    return ~in_contact


def three_tier_tower_stacked(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    gripper_away_min_distance: float = 0.04,
) -> torch.Tensor:
    """True when 3-tier tower is assembled: cube_0 on cube_1 (released + no contact) AND cube_2 on cube_0.

    Tower order (bottom-up):  cube_1 (base on table) → cube_0 → cube_2 (top).

    cube_0-on-cube_1 leg requires ALL THREE:
      - geometric pair stack (|Δxy| < xy_threshold AND |Δz - CUBE_SIZE| < z_threshold)
      - EE at least `gripper_away_min_distance` (default 4 cm) away from cube_0
      - no contact force between cube_0 and either fingertip OR the panda_hand body
    cube_2-on-cube_0 leg is geometric only.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    cube_0_on_cube_1 = (
        _pair_stacked(pos_0, pos_1, xy_threshold, z_threshold)
        & _gripper_far_from_cube_0(env, min_distance=gripper_away_min_distance)
        & _no_contact_between_cube_0_and_gripper_or_ee(env)
    )
    cube_2_on_cube_0 = _pair_stacked(pos_2, pos_0, xy_threshold, z_threshold)
    return cube_0_on_cube_1 & cube_2_on_cube_0


def cube_0_dropping(
    env: "ManagerBasedRLEnv",
    drop_margin: float = 0.05,
    table_height: float = 0.0,
) -> torch.Tensor:
    """True when cube_0 falls more than `drop_margin` metres below `table_height`.

    Mirror of LiftCube's `object_dropping` term, specialized to cube_0.
    Preserved alongside `any_cube_dropping` for backwards compatibility.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    z_floor = table_height - drop_margin
    return cube_0.data.root_pos_w[:, 2] < z_floor


def any_cube_dropping(
    env: "ManagerBasedRLEnv",
    drop_margin: float = 0.05,
    table_height: float = 0.0,
) -> torch.Tensor:
    """True when ANY of cube_0/cube_1/cube_2 falls below `table_height - drop_margin`."""
    z_floor = table_height - drop_margin
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    return (
        (cube_0.data.root_pos_w[:, 2] < z_floor)
        | (cube_1.data.root_pos_w[:, 2] < z_floor)
        | (cube_2.data.root_pos_w[:, 2] < z_floor)
    )


def cube_0_stack_broken(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True when cube_0 had been successfully stacked at some point this episode
    AND is currently no longer geometrically stacked on cube_1.

    Reads the per-env `cube_0_stacked_once` latch maintained by
    `mdp.rewards.cube_0_stacked_bonus_once_per_episode`. CRITICAL: IsaacLab
    runs TerminationManager BEFORE RewardManager each step, so this function
    sees the latch state from the PREVIOUS step — including the stale value
    that survives an env reset until the reward function clears it. To avoid
    firing this termination on step 1 of a new episode (when the latch is
    stale from the prior episode), gate on `episode_length_buf > 1`.

    Mirrors the trigger condition for
    `mdp.cube_0_stack_broken_penalty_once_per_episode` (modulo the
    just-reset gate) so the −500 penalty fires the same step the episode
    terminates.
    """
    from .rewards import _LATCH_BUFFERS  # avoid module-level cross-import cycle
    key = (id(env), "cube_0_stacked_once")
    if key not in _LATCH_BUFFERS:
        # No env has stacked cube_0 yet this episode — termination cannot fire.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stacked_once = _LATCH_BUFFERS[key]

    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    currently_stacked = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)
    just_reset = env.episode_length_buf <= 1
    return stacked_once & (~currently_stacked) & (~just_reset)


def three_tier_tower_broken(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True when the FULL 3-tier tower was assembled at some point this episode
    AND is currently no longer assembled.

    Reads the per-env `three_tier_tower_once` latch maintained by
    `mdp.rewards.three_tier_tower_bonus_once_per_episode`. Same just-reset
    gate as `cube_0_stack_broken` to avoid spurious step-1 fires from a stale
    latch (TerminationManager runs before RewardManager).
    """
    from .rewards import _LATCH_BUFFERS
    key = (id(env), "three_tier_tower_once")
    if key not in _LATCH_BUFFERS:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tower_once = _LATCH_BUFFERS[key]

    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    # Upper pair: cube_0 on cube_1
    xy_01 = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_01 = pos_0[:, 2] - pos_1[:, 2]
    upper = (xy_01 < xy_threshold) & (torch.abs(z_01 - CUBE_SIZE) < z_threshold)
    # Top pair: cube_2 on cube_0
    xy_20 = torch.norm(pos_2[:, :2] - pos_0[:, :2], dim=-1)
    z_20 = pos_2[:, 2] - pos_0[:, 2]
    top = (xy_20 < xy_threshold) & (torch.abs(z_20 - CUBE_SIZE) < z_threshold)
    currently_built = upper & top

    just_reset = env.episode_length_buf <= 1
    return tower_once & (~currently_built) & (~just_reset)
