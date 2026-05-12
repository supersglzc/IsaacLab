# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the stack_cube task.

Edit_mode_012 (3-tier tower): the policy sees joint_pos / joint_vel / per-cube
positions (cube_0, cube_1, cube_2 all in robot root frame) / per-pair stack
targets (cube_0 goal = top of cube_1; cube_1 goal = top of cube_2) /
last_action.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Cube edge length — must stay in sync with the constant in rewards.py /
# terminations.py and the USD scale in config/franka/joint_pos_env_cfg.py.
CUBE_SIZE = 0.043


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
