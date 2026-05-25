"""Observation helpers for the dex_grasp task.

PolicyCfg term order (concatenated):
    1. joint_pos_right_normalized   (22,)  -- robot joint pos, normalized to [-1, 1] using
                                              JOINT_LOWER_LIMIT / JOINT_UPPER_LIMIT
    2. dog_position_in_world         (3,)  -- dog xyz in env-local world frame
    3. last_action                  (22,)  -- mdp.last_action
Total = 22 + 3 + 22 = 47.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import scale_transform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_right_normalized(
    env: "ManagerBasedRLEnv",
    joint_lower_limit: list[float] | None = None,
    joint_upper_limit: list[float] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot joint positions normalized to [-1, 1] using the explicit per-joint
    `joint_lower_limit` / `joint_upper_limit` lists (vendored from bidex).

    Falls back to the asset's `soft_joint_pos_limits` when either list is None.
    Mirrors bidex's `joint_pos_limit_normalized` helper (right-hand variant).
    """
    asset: Articulation = env.scene[robot_cfg.name]
    joint_ids = asset_cfg_to_joint_ids(asset, robot_cfg)
    if joint_lower_limit is None:
        lower = asset.data.soft_joint_pos_limits[:, joint_ids, 0]
    else:
        lower = torch.tensor(joint_lower_limit, device=env.device)
    if joint_upper_limit is None:
        upper = asset.data.soft_joint_pos_limits[:, joint_ids, 1]
    else:
        upper = torch.tensor(joint_upper_limit, device=env.device)
    return scale_transform(asset.data.joint_pos[:, joint_ids], lower, upper)


def dog_position_in_world(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """Dog xyz in env-local world coordinates (`root_pos_w - env_origin`).

    Mirrors bidex `object_pos`. The reset event teleports the dog to env-local
    `(0.05, -0.35, 0.0)`, so this signal is referenced to that same origin.
    """
    dog: RigidObject = env.scene[object_cfg.name]
    return dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def asset_cfg_to_joint_ids(asset: Articulation, cfg: SceneEntityCfg):
    """Resolve `cfg.joint_ids` to a Python slice / list usable as a tensor index.

    Robots whose `SceneEntityCfg` was constructed with no `joint_names` filter
    surface `joint_ids` as the literal `slice(None)`, which is fine to index
    directly. With a regex filter it's a list of ints.
    """
    if cfg.joint_ids is None or cfg.joint_ids == slice(None):
        return slice(None)
    return cfg.joint_ids
