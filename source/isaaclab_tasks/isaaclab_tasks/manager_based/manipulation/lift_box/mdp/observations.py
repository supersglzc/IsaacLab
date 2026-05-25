# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the lift_box task.

PolicyCfg term order (concatenated):
    1. ee_pose_0              (7,)  — per-robot end-effector pose in robot_0's root frame
    2. ee_pose_1              (7,)  — per-robot end-effector pose in robot_1's root frame
    3. box_position_in_world  (3,)  — box xyz in env-local world frame (root_pos_w - env_origin)
    4. box_quat_in_world      (4,)  — box quaternion (wxyz) in world frame
    5. gripper_joint_pos_0    (2,)  — robot_0 fr3 finger joint positions
    6. gripper_joint_pos_1    (2,)  — robot_1 fr3 finger joint positions
    7. last_action            (8,)  — last action (3 xyz + 1 gripper) x 2 robots
Total = 7 + 7 + 3 + 4 + 2 + 2 + 8 = 33.

The per-robot `ee_pose_in_robot_root_frame` is a parameterized variant of
insert_drawer's helper that accepts both a `robot_cfg` and an `ee_frame_cfg`.
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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_0"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_0"),
) -> torch.Tensor:
    """7-D end-effector pose `[x, y, z, qw, qx, qy, qz]` in the given robot's root frame.

    Parameterized variant of insert_drawer's helper: takes BOTH the robot
    SceneEntityCfg and the ee_frame SceneEntityCfg so the same function works
    for either of the two arms (just bind `robot_cfg="robot_0"` +
    `ee_frame_cfg="ee_frame_0"` or the `_1` pair as ObsTerm params).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def box_position_in_world(
    env: "ManagerBasedRLEnv",
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box xyz in env-local world coordinates (`root_pos_w - env_origin`).

    The lift target is also expressed in env-local world coordinates (see
    `terminations.lift_box_success`), so the policy sees the box's
    position in the same frame as the implicit goal.
    """
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def box_quat_in_world(
    env: "ManagerBasedRLEnv",
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box orientation quaternion (wxyz) in world frame. Quaternion is
    rotation-frame-invariant up to the env's root rotation, which is identity
    in IsaacLab's per-env scene, so no subtract_frame_transforms is needed."""
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_quat_w[:, :4]
