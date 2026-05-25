# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward placeholder for the stir_bowl task.

`task-generator` leaves §6 as a constant-zero reward (weight=0). The user
runs `/nautilus:reward-tune` to fill in real shaping terms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def placeholder_zero(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant-zero per-env reward; placeholder for §6."""
    return torch.zeros(env.num_envs, device=env.device)
