# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the insert_drawer task.

PolicyCfg term order (concatenated):
    1. ee_pose                  (7,) -- mdp.ee_pose_in_robot_root_frame
    2. cube_position            (3,) -- mdp.cube_position_in_robot_root_frame
    3. drawer_body_position     (3,) -- mdp.drawer_body_position_in_robot_root_frame
    4. gripper_joint_pos        (2,) -- two fr3 finger joints
    5. last_action              (4,) -- mdp.last_action
Total = 19.

All position helpers transform world-frame positions into the robot root frame
via `subtract_frame_transforms` (StackCube convention).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

from .rewards import _cube_inside_latch_active

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """7-D end-effector pose in the robot root frame: `[x, y, z, qw, qx, qy, qz]`.

    Mirror of `manipulation/stack_cube/mdp/observations.ee_pose_in_robot_root_frame`.
    """
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def cube_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """`cube_0` xyz in the robot root frame, zero-masked once the
    `cube_inside_bonus` latch has fired this episode.

    Rationale: after the cube is inserted, the policy's downstream phase
    (retract → close drawer) doesn't depend on cube position, so masking
    it to zero removes a distracting input and signals the phase switch.
    """
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    cube_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_pos_w)
    return cube_pos_b * (1.0 - _cube_inside_latch_active(env)).unsqueeze(-1)


def drawer_body_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_body_name: str = "drawer",
) -> torch.Tensor:
    """Sliding drawer body xyz in the robot root frame.

    Iter 10: replaces `drawer_handle_position_in_robot_root_frame`. The new
    drawer asset has no handle — only a fixed cabinet (`base_link` + walls)
    and the sliding `drawer` body. We now expose the SLIDING body's position
    (this is what moves when the joint opens/closes); this is also what the
    `cube_inside_drawer_geometric` predicate is computed relative to, so the
    policy sees the same reference frame the reward + success use.
    """
    robot: RigidObject = env.scene["robot"]
    drawer: Articulation = env.scene[drawer_cfg.name]
    body_idx = drawer.find_bodies(drawer_body_name)[0][0]
    body_pos_w = drawer.data.body_pos_w[:, body_idx, :]
    body_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w
    )
    return body_pos_b


