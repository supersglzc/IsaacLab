# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the stack_cube task — grasping-cube mux (edit_mode_015).

Dense shaping + sparse bonuses, all wired through the §5 obs `predicate`
`_cube_0_on_cube_1_predicate` so the reward and observation switch the
"currently-relevant cube" on the same step:

  Stateless per-step grasping-cube mux:
      not-yet-stacked (cube_0 NOT on cube_1):
          grasping_cube   <- cube_0
          target          <- cube_1.xyz + [0, 0, CUBE_SIZE]
      stacked (cube_0 ON cube_1):
          grasping_cube   <- cube_2
          target          <- cube_0.xyz + [0, 0, CUBE_SIZE]

Dense shaping (mux on grasping cube):
    grasping_cube_ee_distance     — 1 - tanh(||ee_w - grasping_cube||/std)
    grasping_cube_is_lifted       — 1.0 if grasping_cube.z_w > min_h else 0.0
    grasping_cube_goal_distance   — lifted * (1 - tanh(d_to_target/std))
    linear_lift_grasping_cube     — contact-gated linear lift ramp

Sparse / once-per-episode (cube_0 stacking + full tower):
    cube_0_stacked_bonus_once_per_episode      — +1 first step cube_0 latched
    cube_0_stack_broken_penalty_once_per_episode — −1 first step the latched
                                                     stack breaks again
    three_tier_tower_bonus_once_per_episode    — +1 first step BOTH pairs hold

EE pose is read from the FrameTransformer scene entity `ee_frame` (LiftCube
convention). The Franka per-robot cfg installs this sensor pointing at
`panda_hand` with an offset of [0, 0, 0.1034] (fingertip).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Cube edge length — the expected z-gap between cube_0 (top) and cube_1 (bottom)
# when stacked. Mirrors the constant in `mdp/terminations.py`.
CUBE_SIZE = 0.043


# ---------------------------------------------------------------------------
# edit_mode_014 — grasping-cube predicate (private clone of
# `mdp.observations._cube_0_on_cube_1_predicate`). Kept here to avoid the
# cross-module dependency `rewards.py -> observations.py`. Identical thresholds
# (xy<0.02 AND |Δz - CUBE_SIZE|<0.01). Used by the three grasping-cube reward
# terms so the reward and observation mux flip on the same condition.
# ---------------------------------------------------------------------------


def _cube_0_on_cube_1_predicate(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True per env when cube_0 is geometrically stacked on cube_1.

    Same thresholds as `mdp.observations._cube_0_on_cube_1_predicate` and
    `mdp.terminations.cube_0_stacked_on_cube_1`. Mirrors the §5 obs mux so the
    reward and observation switch at the same instant.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    top = cube_0.data.root_pos_w[:, :3]
    bot = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(top[:, :2] - bot[:, :2], dim=-1)
    z_gap = top[:, 2] - bot[:, 2]
    return (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)


def grasping_cube_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    std_state_b: float | None = None,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||grasping_cube - ee|| / std)`.

    Per-state std: `std` applies when predicate is False (state A — chasing
    cube_0); `std_state_b` (defaults to `std`) applies when predicate is True
    (state B — chasing cube_2). A sharper state-B std rewards precision near
    cube_2; a wider state-A std attracts the EE from far away.

    grasping_cube = cube_0 when `_cube_0_on_cube_1_predicate` False, cube_2 when True.
    """
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    on_stack = _cube_0_on_cube_1_predicate(env)            # bool (N,)
    grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(grasping_pos - ee_w, dim=1)
    std_b = std if std_state_b is None else float(std_state_b)
    effective_std = torch.where(on_stack, torch.full_like(d, std_b), torch.full_like(d, std))
    base = 1.0 - torch.tanh(d / effective_std)
    # State B: scale base by 100× and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 30.0 * base + 1.0, base)


def grasping_cube_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """`1.0 if grasping_cube.z_w > minimal_height else 0.0`.

    Same grasping-cube mux as `grasping_cube_ee_distance`.
    """
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    on_stack = _cube_0_on_cube_1_predicate(env)
    z_0 = cube_0.data.root_pos_w[:, 2]
    z_2 = cube_2.data.root_pos_w[:, 2]
    z = torch.where(on_stack, z_2, z_0)
    lifted = torch.where(z > minimal_height, 1.0, 0.0)
    # State B: scale base by 10× and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 10.0 * lifted + 1.0, lifted)


def grasping_cube_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    minimal_height: float = 0.04,
    minimal_height_b: float | None = None,
) -> torch.Tensor:
    """`(grasping_cube lifted) * (1 - tanh(||grasping_cube - target|| / std))`.

    target = cube_1.xyz + [0,0,CUBE_SIZE] when predicate False,
             cube_0.xyz + [0,0,CUBE_SIZE] when True.

    Per-state lift gate:
      state A (predicate False): `minimal_height` applies (default 0.04)
      state B (predicate True ): `minimal_height_b` applies — set this to the
        target z (= cube_0.z + CUBE_SIZE ≈ 0.1075 for a cube on cube_1 on
        the table) to keep `align` from firing until cube_2 is lifted above
        the existing stack. Default = `minimal_height` (state A behaviour).
    """
    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    on_stack = _cube_0_on_cube_1_predicate(env)            # bool (N,)
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
    base_pos     = torch.where(on_stack.unsqueeze(-1), pos_0, pos_1)
    target_pos   = base_pos.clone()
    target_pos[:, 2] = target_pos[:, 2] + CUBE_SIZE
    d = torch.norm(grasping_pos - target_pos, dim=1)
    h_b = minimal_height if minimal_height_b is None else float(minimal_height_b)
    effective_min_h = torch.where(
        on_stack,
        torch.full_like(grasping_pos[:, 2], h_b),
        torch.full_like(grasping_pos[:, 2], minimal_height),
    )
    lifted = (grasping_pos[:, 2] > effective_min_h).float()
    base = lifted * (1.0 - torch.tanh(d / std))
    # State B: scale base by 50× and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 50.0 * base + 1.0, base)


def linear_lift_grasping_cube(
    env: "ManagerBasedRLEnv",
    init_z: float = 0.0215,
    target_z_a: float = 0.06,
    target_z_b: float = 0.1075,
    contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Dense linear lift reward for the grasping cube, contact-sensor-gated.

    Per-state grasping-cube mux (same predicate as §5 obs / other §6
    grasping-cube terms):
      state A (`_cube_0_on_cube_1_predicate` False) — grasping cube = cube_0;
        base_a = clamp((cube_0.z - init_z) / (target_z_a - init_z), 0, 1)
                 with default ramp init_z=0.0215 → target_z_a=0.06
                 (denominator = 0.06 − 0.0215 = 0.0385).
        Contact gate: both fingers in contact with cube_0 (filter idx 0).
      state B (`_cube_0_on_cube_1_predicate` True)  — grasping cube = cube_2;
        base_b = clamp((cube_2.z - init_z) / (target_z_b - init_z), 0, 1)
                 (target_z_b=0.1075 = cube_1.z + 2·CUBE_SIZE — above the
                 existing two-cube stack).
        Contact gate: both fingers in contact with cube_2 (filter idx 2).

    Contact gating mirrors `insert_drawer.mdp.rewards.lift_distance`: the ramp
    fires only when BOTH fingertip contact sensors report force > threshold
    against the relevant cube, so the policy can't earn lift reward by
    knocking the cube up with the body of the hand or by single-finger flicks.

    Composition (matches the +1.0 state-B offset used by other §6
    grasping-cube terms — `grasping_cube_ee_distance`, etc. — so transitioning
    into state B never lowers reward):
        torch.where(on_stack, 10 * base_b * gate_b + 1.0, base_a * gate_a)
    """
    on_stack = _cube_0_on_cube_1_predicate(env)
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    z_0 = cube_0.data.root_pos_w[:, 2]
    z_2 = cube_2.data.root_pos_w[:, 2]
    base_a = ((z_0 - init_z) / max(target_z_a - init_z, 1e-6)).clamp(0.0, 1.0)
    base_b = ((z_2 - init_z) / max(target_z_b - init_z, 1e-6)).clamp(0.0, 1.0)

    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    # filter_prim_paths_expr order: [Cube_0 (idx 0), Cube_1 (idx 1), Cube_2 (idx 2)]
    left_f0 = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f0 = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    gate_a = ((left_f0 > contact_force_threshold) & (right_f0 > contact_force_threshold)).float()
    left_f2 = torch.norm(left.data.force_matrix_w[:, 0, 2, :], dim=-1)
    right_f2 = torch.norm(right.data.force_matrix_w[:, 0, 2, :], dim=-1)
    gate_b = ((left_f2 > contact_force_threshold) & (right_f2 > contact_force_threshold)).float()

    return torch.where(on_stack, 10.0 * base_b * gate_b + 1.0, base_a * gate_a)


# iter 33 — module-level per-env latch buffers (keyed by id(env), key_str).
# Used by `cube_1_was_stacked_latched_indicator` to gate cube_0 shaping on
# 'cube_1 has been stacked on cube_2 AT LEAST ONCE in this episode'. Once the
# latch is set it stays True for the rest of the episode (giving the policy
# ~150-200 frames of cube_0 shaping signal per stage-1-success episode vs the
# ~21 frames of the strict instantaneous gate). The latch resets on episode
# reset via the episode_length_buf <= 1 check. Single train-job safety:
# /reward-tune runs one train at a time; reassigns happen each call.
_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    """Get or lazily-create a per-env boolean latch tensor of shape (num_envs,).

    Stored on the module-level `_LATCH_BUFFERS` dict keyed by `(id(env), key)`.
    """
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def _no_contact_between_cube_0_and_gripper_or_ee(
    env: "ManagerBasedRLEnv",
    eps: float = 1e-3,
) -> torch.Tensor:
    """True per env if cube_0 has NO contact force vs either fingertip OR the hand body.

    Reads `force_matrix_w[:, 0, 0, :]` (filter index 0 = Cube_0) on
    finger_left_contact, finger_right_contact, and hand_contact sensors. All
    three must have force magnitude ≤ eps for this to return True.
    """
    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    hand = env.scene["hand_contact"]
    left_f0 = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f0 = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    hand_f0 = torch.norm(hand.data.force_matrix_w[:, 0, 0, :], dim=-1)
    in_contact = (left_f0 > eps) | (right_f0 > eps) | (hand_f0 > eps)
    return ~in_contact


def cube_0_stacked_bonus_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """+1.0 the FIRST step cube_0 is stacked on cube_1 AND not in contact with the EE, else 0.0.

    Success criteria (BOTH must hold):
      - geometric: |cube_0.xy − cube_1.xy| < xy_threshold AND |Δz − CUBE_SIZE| < z_threshold
      - no contact: neither fingertip nor the panda_hand body has any contact force on cube_0

    Per-env latch reset when `env.episode_length_buf <= 1` and set the first
    time the combined criteria hold. Once latched the bonus stops firing —
    encourages the policy to MOVE ON to stacking cube_2 on cube_0.
    """
    latch = _get_latch_buffer(env, "cube_0_stacked_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    geometric = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)
    no_contact = _no_contact_between_cube_0_and_gripper_or_ee(env)
    now_stacked = geometric & no_contact

    fire = now_stacked & (~latch)
    latch = latch | now_stacked
    _LATCH_BUFFERS[(id(env), "cube_0_stacked_once")] = latch
    return fire.float()


def three_tier_tower_bonus_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """+1.0 the FIRST step the full 3-tier tower is assembled, else 0.0.

    Tower pairing now mirrors the 2-cube success contract
    (`cube_0_stacked_bonus_once_per_episode`):
        cube_0 on cube_1:  geometric (|Δxy|<xy_thr AND |Δz−CUBE_SIZE|<z_thr)
                           AND no contact between cube_0 and gripper/ee
        cube_2 on cube_0:  geometric (same thresholds)
                           AND no contact between cube_2 and gripper/ee

    Per-env latch resets at `env.episode_length_buf <= 1` and locks once fired
    so the bonus counts at most once per episode.
    """
    from .terminations import _no_contact_between_cube_and_gripper_or_ee
    latch = _get_latch_buffer(env, "three_tier_tower_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    # Upper pair: cube_0 on cube_1 — geometric AND no contact between cube_0 and gripper/ee.
    xy_01 = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_01 = pos_0[:, 2] - pos_1[:, 2]
    upper_geom = (xy_01 < xy_threshold) & (torch.abs(z_01 - CUBE_SIZE) < z_threshold)
    upper = upper_geom & _no_contact_between_cube_and_gripper_or_ee(env, cube_idx=0)
    # Top pair: cube_2 on cube_0 — geometric AND no contact between cube_2 and gripper/ee.
    xy_20 = torch.norm(pos_2[:, :2] - pos_0[:, :2], dim=-1)
    z_20 = pos_2[:, 2] - pos_0[:, 2]
    top_geom = (xy_20 < xy_threshold) & (torch.abs(z_20 - CUBE_SIZE) < z_threshold)
    top = top_geom & _no_contact_between_cube_and_gripper_or_ee(env, cube_idx=2)
    tower_built = upper & top

    fire = tower_built & (~latch)
    latch = latch | tower_built
    _LATCH_BUFFERS[(id(env), "three_tier_tower_once")] = latch
    return fire.float()


def cube_0_stack_broken_penalty_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """+1.0 ONCE per episode when cube_0 had been successfully stacked (success_bonus
    latch is set) AND the stack has subsequently broken (cube_0 no longer
    geometrically on cube_1). Otherwise 0.0.

    Intended to be weighted negatively in RewardsCfg (e.g. weight=-100) — the
    policy is penalized once when it disturbs a previously successful stack.

    Uses two latches:
      - `cube_0_stacked_once` (read-only here, written by `cube_0_stacked_bonus_once_per_episode`):
        records "this episode had a successful cube_0 stack at some point".
      - `stack_broke_penalty_fired` (managed here): records "this episode
        already paid the broken-stack penalty", preventing repeated firing.

    Declaration order in `RewardsCfg` matters — place this AFTER `success_bonus`
    so the `cube_0_stacked_once` latch reads the fresh value.
    """
    stacked_once = _get_latch_buffer(env, "cube_0_stacked_once")

    penalty_fired = _get_latch_buffer(env, "stack_broke_penalty_fired")
    just_reset = env.episode_length_buf <= 1
    penalty_fired = torch.where(just_reset, torch.zeros_like(penalty_fired), penalty_fired)

    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    currently_stacked = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)

    fire = stacked_once & (~currently_stacked) & (~penalty_fired)

    penalty_fired = penalty_fired | fire
    _LATCH_BUFFERS[(id(env), "stack_broke_penalty_fired")] = penalty_fired
    return fire.float()


