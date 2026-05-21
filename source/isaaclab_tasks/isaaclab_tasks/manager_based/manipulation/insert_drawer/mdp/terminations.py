# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the insert_drawer task."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .rewards import _cube_inside_latch_active

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_termination(
    env: "ManagerBasedRLEnv",
    drawer_closed_threshold: float = 0.05,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    joint_name: str = "base_drawer_joint",
) -> torch.Tensor:
    """True when the `cube_inside_bonus` latch is active AND the drawer joint
    position is below `drawer_closed_threshold` — i.e. cube inserted + drawer
    fully closed."""
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    drawer_closed = joint_pos < drawer_closed_threshold
    latch_active = _cube_inside_latch_active(env) > 0.5
    return latch_active & drawer_closed
