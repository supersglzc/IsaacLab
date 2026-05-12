# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the stack_cube task — LiftCube recipe retargeted.

Five named stages + two regularizers (regularizers come from the shared
`isaaclab.envs.mdp` namespace, not defined here):

    s1  reaching_object       — 1 - tanh(||ee_w - cube_0||/std)  (LiftCube mirror)
    s2  lifting_object        — 1.0 if cube_0.z_w > minimal_height else 0.0
    s3  goal_tracking_coarse  — (cube_0 lifted) * (1 - tanh(d/std)), std=0.3
    s4  goal_tracking_fine    — (cube_0 lifted) * (1 - tanh(d/std)), std=0.05
    s5  success_bonus         — 1.0 when (xy<thresh AND |Δz - CUBE_SIZE|<thresh)

Goal pose: `cube_1.pos_w + [0, 0, CUBE_SIZE]`. No command_manager dependency;
the target is read directly from the cube_1 scene entity.

EE pose is read from the FrameTransformer scene entity `ee_frame` (LiftCube
convention). The Franka per-robot cfg installs this sensor pointing at
`panda_hand` with an offset of [0, 0, 0.1034] (fingertip).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Cube edge length — the expected z-gap between cube_0 (top) and cube_1 (bottom)
# when stacked. Mirrors the constant in `mdp/terminations.py`.
CUBE_SIZE = 0.043


# ---------------------------------------------------------------------------
# s1 — reach cube_0 (LiftCube `object_ee_distance` mirror)
# ---------------------------------------------------------------------------


def cube_0_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||cube_0 - ee_w|| / std)` — LiftCube `object_ee_distance` mirror, cube_0 retarget."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(d / std)


# ---------------------------------------------------------------------------
# s2 — lift cube_0 (LiftCube `object_is_lifted` mirror, binary)
# ---------------------------------------------------------------------------


def cube_0_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """`1.0 if cube_0.z_w > minimal_height else 0.0`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return torch.where(cube.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# ---------------------------------------------------------------------------
# s3 / s4 — coarse / fine goal tracking (LiftCube `object_goal_distance` mirror)
# ---------------------------------------------------------------------------


def cube_0_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    goal_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """`(cube_0 lifted) * (1 - tanh(d / std))` — LiftCube `object_goal_distance` mirror.

    Target is `cube_1.pos_w + [0, 0, CUBE_SIZE]` (the stack top). No
    command_manager dependency: the goal is implicit in the scene.
    """
    cube_0: RigidObject = env.scene[cube_cfg.name]
    cube_1: RigidObject = env.scene[goal_cube_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    target_w = pos_1.clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    distance = torch.norm(target_w - pos_0, dim=1)
    lifted = pos_0[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# s5 — success bonus: sparse +1 when cube_0 is stacked on cube_1
# ---------------------------------------------------------------------------


def cube_0_stacked_bonus(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """`1.0 if (|xy_0 - xy_1|<xy_threshold AND |Δz - CUBE_SIZE|<z_threshold) else 0.0`.

    Same geometric condition as `cube_0_stacked_on_cube_1` in
    `mdp/terminations.py` — this is the dense per-step success indicator and
    the termination wraps the very same check.
    """
    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    aligned_xy = xy_dist < xy_threshold
    on_top_z = torch.abs(z_gap - CUBE_SIZE) < z_threshold
    return (aligned_xy & on_top_z).float()


# ---------------------------------------------------------------------------
# IsaacGymEnvs FrankaCubeStack `compute_franka_reward` re-implementation
# ---------------------------------------------------------------------------


def franka_cube_stack_reward(
    env: "ManagerBasedRLEnv",
    r_dist_scale: float = 0.1,
    r_lift_scale: float = 1.5,
    r_align_scale: float = 2.0,
    r_stack_scale: float = 16.0,
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    ),
) -> torch.Tensor:
    """Verbatim re-implementation of IsaacGymEnvs
    `franka_cube_stack.compute_franka_reward` with FrankaCubeStack.yaml scales:

        r_dist_scale = 0.1, r_lift_scale = 1.5, r_align_scale = 2.0, r_stack_scale = 16.0

    Mapping:
      cubeA → `cube_0` (the cube being moved)
      cubeB → `cube_1` (the destination cube)
      eef / eef_lf / eef_rf → robot bodies `panda_hand`, `panda_leftfinger`,
                              `panda_rightfinger`

    Composition (per IsaacGymEnvs):

        dist  = 1 − tanh(10 · (||cubeA−eef|| + ||cubeA−lf|| + ||cubeA−rf||) / 3)
        lift  = (cubeA.z − cubeA_size − table_height) > 0.04
        d_ab  = ||(cubeB − cubeA) + [0, 0, (sA+sB)/2]||
        align = (1 − tanh(10 · d_ab)) · lift
        dist  = max(dist, align)
        stack = (xy_dist < 2cm) AND (|height − target_h| < 2cm) AND (||eef−cubeA|| > 4cm)
        reward = where(stack, r_stack_scale · 1,
                              r_dist_scale·dist + r_lift_scale·lift + r_align_scale·align)
    """
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Body indices for hand + fingertips
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    lf_idx = robot.find_bodies("panda_leftfinger")[0][0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0][0]

    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    lf_pos = robot.data.body_state_w[:, lf_idx, :3]
    rf_pos = robot.data.body_state_w[:, rf_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]

    cube_a_size = CUBE_SIZE
    cube_b_size = CUBE_SIZE

    # distance: hand + both fingers to cubeA (sharp tanh slope=10)
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    d_lf = torch.norm(cube_a_pos - lf_pos, dim=-1)
    d_rf = torch.norm(cube_a_pos - rf_pos, dim=-1)
    dist_reward = 1.0 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3.0)

    # lift indicator (cubeA bottom > 4 cm above table top)
    cube_a_height = cube_a_pos[:, 2] - table_height
    cube_a_lifted = (cube_a_height - cube_a_size) > 0.04
    lift_reward = cube_a_lifted.float()

    # align: distance from cubeA to (cubeB + stack-offset) — only when lifted
    cube_a_to_b = cube_b_pos - cube_a_pos
    offset = torch.zeros_like(cube_a_to_b)
    offset[:, 2] = (cube_a_size + cube_b_size) / 2.0
    d_ab = torch.norm(cube_a_to_b + offset, dim=-1)
    align_reward = (1.0 - torch.tanh(10.0 * d_ab)) * lift_reward

    # IsaacGymEnvs: dist takes the max of (hand-to-cube) and align
    dist_reward = torch.max(dist_reward, align_reward)

    # stack indicator — xy aligned AND on top AND gripper RELEASED (>4cm away)
    target_height = cube_b_size + cube_a_size / 2.0
    aligned_xy = torch.norm(cube_a_to_b[:, :2], dim=-1) < 0.02
    on_top = torch.abs(cube_a_height - target_height) < 0.02
    gripper_away = d > 0.04
    stack_indicator = aligned_xy & on_top & gripper_away

    # compose: stack overrides; otherwise dist + lift + align (scaled)
    rewards = torch.where(
        stack_indicator,
        r_stack_scale * stack_indicator.float(),
        r_dist_scale * dist_reward
        + r_lift_scale * lift_reward
        + r_align_scale * align_reward,
    )
    return rewards


# ---------------------------------------------------------------------------
# IsaacGymEnvs FrankaCubeStack reward terms — SEPARATE versions
#
# Same math as `compute_franka_reward` but split into 4 independent RewTerms
# so each is individually visible in the training tracker. The original `where`
# composition is approximated by summing all four; when stacked, all four fire
# but stack (×16) dominates dist (×0.1) + lift (×1.5) + align (×2.0).
# ---------------------------------------------------------------------------


def _fcs_lifted_mask(cube_a_pos: torch.Tensor, table_height: float) -> torch.Tensor:
    """Helper: cubeA.bottom > 4cm above table top, IsaacGymEnvs convention."""
    cube_a_height = cube_a_pos[:, 2] - table_height
    return (cube_a_height - CUBE_SIZE) > 0.02


def _fcs_align_value(cube_a_pos, cube_b_pos, lifted_mask) -> torch.Tensor:
    """Helper: (1 - tanh(10·d_ab)) * lifted_mask, with d_ab to cubeB + stack offset."""
    cube_a_to_b = cube_b_pos - cube_a_pos
    offset = torch.zeros_like(cube_a_to_b)
    offset[:, 2] = CUBE_SIZE  # (cubeA_size + cubeB_size) / 2 == CUBE_SIZE for equal cubes
    d_ab = torch.norm(cube_a_to_b + offset, dim=-1)
    return (1.0 - torch.tanh(10.0 * d_ab)) * lifted_mask.float()


def fcs_dist_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IGenvs `dist_reward` (after `max(dist, align)` step).

    raw_dist = 1 − tanh(10·(d_hand + d_lf + d_rf) / 3)
    return max(raw_dist, align_reward)
    """
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    lf_idx = robot.find_bodies("panda_leftfinger")[0][0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0][0]
    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    lf_pos = robot.data.body_state_w[:, lf_idx, :3]
    rf_pos = robot.data.body_state_w[:, rf_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    d_lf = torch.norm(cube_a_pos - lf_pos, dim=-1)
    d_rf = torch.norm(cube_a_pos - rf_pos, dim=-1)
    raw_dist = 1.0 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3.0)
    lifted = _fcs_lifted_mask(cube_a_pos, table_height)
    align = _fcs_align_value(cube_a_pos, cube_b_pos, lifted)
    return torch.max(raw_dist, align)


def fcs_lift_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """IGenvs `lift_reward` = 1.0 if (cubeA.z − CUBE_SIZE − table_h) > 0.04 else 0."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    return _fcs_lifted_mask(cube_a_pos, table_height).float()


def fcs_align_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """IGenvs `align_reward = (1 − tanh(10·d_ab)) · lifted`."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    lifted = _fcs_lifted_mask(cube_a_pos, table_height)
    return _fcs_align_value(cube_a_pos, cube_b_pos, lifted)


def fcs_stack_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IGenvs `stack_reward` = xy_align AND |Δheight|<2cm AND ||hand−cubeA||>4cm."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    cube_a_to_b = cube_b_pos - cube_a_pos
    cube_a_height = cube_a_pos[:, 2] - table_height
    target_height = CUBE_SIZE + CUBE_SIZE / 2.0
    aligned_xy = torch.norm(cube_a_to_b[:, :2], dim=-1) < 0.02
    on_top = torch.abs(cube_a_height - target_height) < 0.02
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    gripper_away = d > 0.04
    return (aligned_xy & on_top & gripper_away).float()
