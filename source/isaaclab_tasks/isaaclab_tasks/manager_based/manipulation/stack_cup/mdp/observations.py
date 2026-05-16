# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the stack_cup task.

Edit_mode_012 (3-tier tower): the policy sees joint_pos / joint_vel / per-cube
positions (cube_0, cube_1, cube_2 all in robot root frame) / per-pair stack
targets (cube_0 goal = top of cube_1; cube_1 goal = top of cube_2) /
last_action.

Edit_mode_013 (§5 re-author): the policy obs is collapsed to a 4-term layout
(joint_pos + grasping_cube_position + grasping_target_position + actions = 23).
joint_vel and the per-cube absolute positions are dropped from the obs; a
stateless per-step mux switches the "currently relevant" cube + target based
on whether cube_0 is already on cube_1 (predicate mirror of
`mdp.terminations.cube_0_stacked_on_cube_1`). The dropped obs helpers
(`cube_*_position_in_robot_root_frame`, `stack_target_*`) are KEPT in this
file but no longer wired into ObservationsCfg — they may still be referenced
by §6 reward code or future edits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """7-D end-effector pose in the robot root frame: `[x, y, z, qw, qx, qy, qz]`.

    Reads world-frame EE pose from the `ee_frame` FrameTransformer (target 0;
    LiftCube convention: `panda_hand` with [0,0,0.1034] offset) and transforms
    it into the robot's root frame via `subtract_frame_transforms`.
    """
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


# Stacking-tier Δz — for cup geometry this is the EMPIRICAL nested-stack
# spacing (≠ cup geometric height), measured at 0.077 m for the 9 cm
# scaled cup. Must stay in sync with `CUBE_SIZE` in terminations.py,
# rewards.py, and config/franka/joint_pos_env_cfg.py.
CUBE_SIZE = 0.077


def _cube_pos_in_robot_root_frame(env: "ManagerBasedRLEnv", cube_key: str) -> torch.Tensor:
    """Shared helper: cube xyz in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[cube_key]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    cube_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_pos_w)
    return cube_pos_b


def cube_0_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_0 xyz in the robot root frame (LiftCube `object_position_in_robot_root_frame` mirror)."""
    return _cube_pos_in_robot_root_frame(env, "cube_0")


def cube_1_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_1 xyz in the robot root frame."""
    return _cube_pos_in_robot_root_frame(env, "cube_1")


def cube_2_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_2 xyz in the robot root frame."""
    return _cube_pos_in_robot_root_frame(env, "cube_2")


def _stack_target_in_root_frame(env: "ManagerBasedRLEnv", base_cube_key: str) -> torch.Tensor:
    """Shared helper: `base_cube.pos_w + [0, 0, CUBE_SIZE]` in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    base: RigidObject = env.scene[base_cube_key]
    target_w = base.data.root_pos_w[:, :3].clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_w)
    return target_b


def stack_target_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_0 goal = `cube_1.pos_w + [0, 0, CUBE_SIZE]` in the robot root frame.

    The goal is implicit (top of cube_1) rather than command-manager-driven.
    Kept under the historical name so the §6 reward helpers that read this
    in observation-space stay stable.
    """
    return _stack_target_in_root_frame(env, "cube_1")


def cube_1_stack_target_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_1 goal = `cube_2.pos_w + [0, 0, CUBE_SIZE]` in the robot root frame."""
    return _stack_target_in_root_frame(env, "cube_2")


def _cube_0_on_cube_1_predicate(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True per env when cube_0 is stacked on cube_1 (position-only).

    Mirrors `mdp.terminations.cube_0_stacked_on_cube_1` but lives here so the
    observation mux doesn't take a cross-module dependency on the termination
    module. Same thresholds.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    top = cube_0.data.root_pos_w[:, :3]
    bot = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(top[:, :2] - bot[:, :2], dim=-1)
    z_gap = top[:, 2] - bot[:, 2]
    return (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)


def grasping_cube_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """xyz of the cube the policy currently needs to grasp (robot-root frame).

    Stateless per-step mux:
        cube_0_on_cube_1 -> cube_2   (second-stage target object)
        otherwise        -> cube_0   (initial state OR after a drop reverts here)
    """
    cube_0_pos = _cube_pos_in_robot_root_frame(env, "cube_0")
    cube_2_pos = _cube_pos_in_robot_root_frame(env, "cube_2")
    use_cube_2 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)  # (N, 1)
    return torch.where(use_cube_2, cube_2_pos, cube_0_pos)


def grasping_target_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """xyz target for the current grasping cube (robot-root frame).

    Stateless per-step mux:
        cube_0_on_cube_1 -> cube_0.xyz + [0,0,CUBE_SIZE]   (cube_2 stacks on cube_0)
        otherwise        -> cube_1.xyz + [0,0,CUBE_SIZE]   (cube_0 stacks on cube_1)
    """
    target_on_cube_1 = _stack_target_in_root_frame(env, "cube_1")
    target_on_cube_0 = _stack_target_in_root_frame(env, "cube_0")
    use_target_on_cube_0 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)
    return torch.where(use_target_on_cube_0, target_on_cube_0, target_on_cube_1)
