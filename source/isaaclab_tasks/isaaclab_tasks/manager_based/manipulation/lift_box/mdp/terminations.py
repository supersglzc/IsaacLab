# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the lift_box task.

Goal: lift the box COM (xy) to the target xy AND z = `BOX_INIT_Z + lift_height`
AND box linear velocity is small (i.e. the lift has stabilized). The
`lift_box_success` termination fires when all three conditions hold.

The goal is HARD-CODED in this module (no CommandsCfg). `BOX_INIT_Z` matches
the constant used in `config/franka/joint_pos_env_cfg.py` for the box spawn
height (= half_z_extent of the recentered eurobox USD).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Half-z extent of the eurobox after MeshConverter recentering. Must match the
# `BOX_INIT_Z` constant used in `joint_pos_env_cfg.py`.
BOX_INIT_Z: float = 0.11025


def lift_box_success(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.25,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """True when the box is centered (xy) within `xy_pos_tol` of `target_xy`,
    its z is within `z_pos_tol` of `BOX_INIT_Z + lift_height`, AND its linear
    velocity magnitude is below `vel_tol`. All quantities are in env-local
    world coordinates (root_pos_w - env_origins).
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]

    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    xy_err = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    z_err = torch.abs(box_z - (BOX_INIT_Z + lift_height))

    lin_vel_w = box.data.root_lin_vel_w[:, :3]
    vel_norm = torch.norm(lin_vel_w, dim=-1)

    return (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol)
