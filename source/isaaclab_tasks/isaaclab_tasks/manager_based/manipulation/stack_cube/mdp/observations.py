# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the stack_cube task.

LiftCube-style obs layout: the policy sees joint_pos / joint_vel / cube_0 pos /
target pos / last_action. `cube_0_position_in_robot_root_frame` mirrors
LiftCube's `object_position_in_robot_root_frame`; the target position is
derived from cube_1 (the stacking base) plus a vertical CUBE_SIZE offset.
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


def cube_0_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cube_0 xyz in the robot root frame (LiftCube `object_position_in_robot_root_frame` mirror)."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene["cube_0"]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    cube_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_pos_w)
    return cube_pos_b


def stack_target_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stack target (cube_1.pos_w + [0, 0, CUBE_SIZE]) in the robot root frame.

    Replaces LiftCube's `generated_commands(command_name="object_pose")`. The
    goal is implicit (top of cube_1) rather than command-manager-driven.
    """
    robot: RigidObject = env.scene["robot"]
    cube_1: RigidObject = env.scene["cube_1"]
    target_w = cube_1.data.root_pos_w[:, :3].clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_w)
    return target_b
