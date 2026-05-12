# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the push-block environments.

Modeled on `manipulation/lift/mdp/rewards.py` but adapted for a planar push task:
  * No `minimal_height` lift gate on goal-tracking (block stays on the table).
  * EE -> object reaching uses `body_pos_w[panda_hand]` directly instead of an
    `ee_frame` FrameTransformer (Push's scene does not register one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance_body(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward the agent for reaching the block using a tanh kernel.

    Distance is measured between the robot body referenced by `robot_cfg.body_names[0]`
    (resolved to `body_ids[0]` by SceneEntityCfg) and the object's CoM, both in world frame.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_to_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the block-to-goal position using a tanh kernel.

    Mirrors `manipulation/lift/mdp/rewards.py:object_goal_distance` but DROPS the
    `minimal_height` gate -- Push is planar and the block never leaves the table.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success indicator: 1.0 when block-to-goal distance < `threshold`, else 0.0.

    Logging-only term (used at weight=0.0). Wired into `RewardsCfg.success` so
    `info["detailed_reward"]["success"]` exposes per-step success without
    contributing to the optimization signal.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return (distance < threshold).float()
