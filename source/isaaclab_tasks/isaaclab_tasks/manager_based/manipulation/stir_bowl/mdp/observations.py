# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the stir_bowl task.

PolicyCfg term order (concatenated):
    1.  ee_pose                       (7,)  — ee_pose_in_robot_root_frame
    2.  spoon_position_in_world       (3,)  — spoon root_pos in env-local world frame
    3.  spoon_quat_in_world           (4,)  — spoon root_quat (wxyz) in world frame
    4.  bowl_position_in_world        (3,)  — bowl root_pos in env-local world frame
    5.  ball_1_position_in_world      (3,)
    6.  ball_1_lin_vel_in_world       (3,)
    7.  ball_2_position_in_world      (3,)
    8.  ball_2_lin_vel_in_world       (3,)
    9.  ball_3_position_in_world      (3,)
    10. ball_3_lin_vel_in_world       (3,)
    11. gripper_joint_pos             (2,)
    12. last_action                   (4,)
Total = 7 + 3 + 4 + 3 + (3+3)*3 + 2 + 4 = 41.

`env.scene.env_origins` is subtracted from world positions so positions are in
the env-local world frame (matches the bidex obs convention).
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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """7-D end-effector pose `[x, y, z, qw, qx, qy, qz]` in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def _root_pos_env_local(env: "ManagerBasedRLEnv", asset_name: str) -> torch.Tensor:
    """Asset root xyz in env-local world frame (root_pos_w − env_origin)."""
    asset: RigidObject = env.scene[asset_name]
    return asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def _root_quat_world(env: "ManagerBasedRLEnv", asset_name: str) -> torch.Tensor:
    """Asset root quaternion (wxyz) in world frame."""
    asset: RigidObject = env.scene[asset_name]
    return asset.data.root_quat_w[:, :4]


def _root_lin_vel_env_local(env: "ManagerBasedRLEnv", asset_name: str) -> torch.Tensor:
    """Asset root linear velocity in world frame (env_origins are static, so
    `root_lin_vel_w` already equals the env-local linear velocity)."""
    asset: RigidObject = env.scene[asset_name]
    return asset.data.root_lin_vel_w[:, :3]


def spoon_position_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_pos_env_local(env, "spoon")


def spoon_quat_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_quat_world(env, "spoon")


def bowl_position_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_pos_env_local(env, "bowl")


def ball_1_position_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_pos_env_local(env, "ball_1")


def ball_1_lin_vel_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_lin_vel_env_local(env, "ball_1")


def ball_2_position_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_pos_env_local(env, "ball_2")


def ball_2_lin_vel_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_lin_vel_env_local(env, "ball_2")


def ball_3_position_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_pos_env_local(env, "ball_3")


def ball_3_lin_vel_in_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _root_lin_vel_env_local(env, "ball_3")
