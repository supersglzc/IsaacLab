# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the dual-arm `lift_box` task.

Five-phase manipulation shaping ladder (one per phase below). Mirrors the
converged insert_drawer recipe at
`source/.../manipulation/insert_drawer/mdp/rewards.py` plus the per-robot
duplication required for bimanual cooperation:

  1. dense EE -> grasp-point attractors (one per robot)
  2. binary contact gates (one per robot, both fingers in contact with box)
  3. linear lift ramp (gated on BOTH robots in contact)
  4. tanh attractor on box xy -> world (0, 0) (gated on box lifted)
  5. one-shot terminal success bonus (mirrors the success termination predicate)

dt-scaling: This fork of IsaacLab has REMOVED the per-weight `* dt` multiplier
inside `RewardManager.compute` (see `isaaclab/managers/reward_manager.py:149`).
Weights below are therefore the raw per-step magnitudes the user sees at runtime.

Per-stage saturated per-step magnitude budget (composer = sum, raw weights):

    reach (per-robot)    0.0125 each  ->  0.025 total / step
    grasp contact (per-r) 0.025 each  ->  0.05  total / step (only when held)
    lift_height (gated)  0.1875       ->  0.19  / step (only after dual contact)
    box_xy_align (gated) 0.125        ->  0.125 / step (only after box lifted >5 cm)
    success_bonus      100.0          -> +100   one-shot at success predicate

200-step episode (10 s @ 20 Hz) ceiling check:
    dense ceiling (reach + contact + lift + align)  ~ 78
    success bonus one-shot                           +100
    => sparse strictly dominates dense (100 > 78).

Sign convention: positive = good (no penalties).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Half-z extent of the eurobox after MeshConverter recentering. Must match the
# `BOX_INIT_Z` constant used in `joint_pos_env_cfg.py` and `terminations.py`.
BOX_INIT_Z: float = 0.11025


# ---------------------------------------------------------------------------
# Module-level per-(env, key) latch buffers (success bonus one-shot).
# ---------------------------------------------------------------------------

_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


# ---------------------------------------------------------------------------
# Phase 1 — dense EE -> grasp-point attractors (one per robot).
# ---------------------------------------------------------------------------


def ee_to_grasp_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_0"),
    grasp_frame_cfg: SceneEntityCfg = SceneEntityCfg("grasp_frame_0"),
) -> torch.Tensor:
    """`1 - tanh(||ee - grasp|| / std)` — dense EE -> box-grasp-point attractor.

    Per-robot reach term. `ee_frame_cfg` is the robot's fingertip TCP and
    `grasp_frame_cfg` is the box-local grasp marker (top of the box's short
    y-end face). Both frames carry shape `(num_envs, 1, 3)` in `target_pos_w`.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasp_frame: FrameTransformer = env.scene[grasp_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    grasp_w = grasp_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(ee_w - grasp_w, dim=-1)
    return 1.0 - torch.tanh(d / max(std, 1e-6))


# ---------------------------------------------------------------------------
# Phase 2 — per-robot binary "both fingers in contact with box" gate.
# ---------------------------------------------------------------------------


def _both_fingers_in_contact(
    env: "ManagerBasedRLEnv",
    left_sensor_name: str,
    right_sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    """Boolean (num_envs,) — True iff both finger contact sensors report force
    above `threshold` against the box (filter index 0)."""
    left: ContactSensor = env.scene[left_sensor_name]
    right: ContactSensor = env.scene[right_sensor_name]
    # force_matrix_w shape: (num_envs, n_bodies=1, n_filters=1, 3) per finger.
    left_f = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    return (left_f > threshold) & (right_f > threshold)


def grasp_contact(
    env: "ManagerBasedRLEnv",
    robot_idx: int = 0,
    contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Binary `1.0 if BOTH fingers of robot_<robot_idx> are touching the box`,
    else 0.0. `robot_idx` is 0 or 1 (selects `finger_left_contact_<idx>` and
    `finger_right_contact_<idx>`).
    """
    gate = _both_fingers_in_contact(
        env,
        left_sensor_name=f"finger_left_contact_{robot_idx}",
        right_sensor_name=f"finger_right_contact_{robot_idx}",
        threshold=contact_force_threshold,
    )
    return gate.float()


# ---------------------------------------------------------------------------
# Phase 3 — linear lift ramp gated on BOTH robots in dual contact.
# ---------------------------------------------------------------------------


def lift_height(
    env: "ManagerBasedRLEnv",
    init_z: float = BOX_INIT_Z,
    target_lift: float = 0.25,
    contact_force_threshold: float = 1e-3,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """`clamp((box.z - init_z) / target_lift, 0, 1) * (both arms in dual contact)`.

    Linear ramp on the box's z position above its init height, gated on BOTH
    robots simultaneously having both fingers in contact with the box. Without
    the dual-contact gate the policy could earn lift reward by knocking the
    box up with the back of a hand.

    `box.z` is the box COM in env-local world frame (after subtracting env
    origin). `init_z = BOX_INIT_Z` is the bottom-of-table resting COM height.
    Ramp saturates at `init_z + target_lift` so the policy doesn't keep
    pushing higher forever.
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_z = box_pos_local[:, 2]
    progress = ((box_z - init_z) / max(target_lift, 1e-6)).clamp(0.0, 1.0)
    gate_0 = _both_fingers_in_contact(env, "finger_left_contact_0", "finger_right_contact_0", contact_force_threshold)
    gate_1 = _both_fingers_in_contact(env, "finger_left_contact_1", "finger_right_contact_1", contact_force_threshold)
    dual_contact = (gate_0 & gate_1).float()
    return progress * dual_contact


# ---------------------------------------------------------------------------
# Phase 4 — tanh attractor on box xy -> target xy, gated on box lifted.
# ---------------------------------------------------------------------------


def box_xy_align(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_threshold: float = 0.05,
    init_z: float = BOX_INIT_Z,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """`(box.z > init_z + lift_threshold) * (1 - tanh(||box.xy - target_xy|| / std))`.

    Dense attractor on the box's xy position toward `target_xy`, gated on the
    box being lifted at least `lift_threshold` above its init height. The lift
    gate prevents the policy from earning align reward by sliding the box
    horizontally across the table.

    `target_xy = (0, 0)` matches the success predicate target in
    `terminations.lift_box_success`.
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]
    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    d = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    base = 1.0 - torch.tanh(d / max(std, 1e-6))
    lifted = (box_z > (init_z + lift_threshold)).float()
    return lifted * base


# ---------------------------------------------------------------------------
# Phase 5 — one-shot success bonus mirroring the termination predicate.
# ---------------------------------------------------------------------------


def success_bonus(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.25,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    init_z: float = BOX_INIT_Z,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """+1.0 the FIRST frame the box meets the success predicate this episode;
    0.0 thereafter. Per-env latch resets at `env.episode_length_buf <= 1`.

    The predicate exactly mirrors `terminations.lift_box_success`:
        - `|box.xy - target_xy| < xy_pos_tol`
        - `|box.z - (init_z + lift_height)| < z_pos_tol`
        - `|box.lin_vel_w| < vel_tol`
    """
    latch = _get_latch_buffer(env, "success_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]
    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    xy_err = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    z_err = torch.abs(box_z - (init_z + lift_height))
    lin_vel_w = box.data.root_lin_vel_w[:, :3]
    vel_norm = torch.norm(lin_vel_w, dim=-1)
    now_success = (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol)

    fire = now_success & (~latch)
    latch = latch | now_success
    _LATCH_BUFFERS[(id(env), "success_once")] = latch
    return fire.float()
