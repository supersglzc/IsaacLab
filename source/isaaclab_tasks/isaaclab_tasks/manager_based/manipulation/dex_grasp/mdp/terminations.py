# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the dex_grasp task."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dog_reached_target(
    env: "ManagerBasedRLEnv",
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    threshold: float = 0.10,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """True when the dog's env-local position is within `threshold` of `target_local`.

    `target_local` is in the same env-local world frame as `dog.data.root_pos_w -
    env_origins`. The default `(0.05, -0.35, 0.30)` lifts the dog 30 cm above
    its spawn xy.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_local.dtype)
    err = torch.norm(dog_local - target.unsqueeze(0), dim=-1)
    return err < threshold
