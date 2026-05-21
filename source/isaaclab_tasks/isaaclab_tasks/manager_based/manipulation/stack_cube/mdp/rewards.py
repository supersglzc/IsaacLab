# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the stack_cube task — 3-tier tower extension (iter 23).

Two parallel LiftCube-style stages (cube_1 → cube_2 first, then cube_0 → cube_1)
plus a sparse full-tower bonus that aligns with the success termination:

  Lower pair (stage 1, must complete first):
    cube_1_ee_distance              — 1 - tanh(||ee_w - cube_1||/std)
    cube_1_is_lifted                — 1.0 if cube_1.z_w > minimal_height else 0.0
    cube_1_goal_distance            — lifted_1 * (1 - tanh(d_1_to_top_of_2 / std))
    cube_1_stacked_on_cube_2_bonus  — sparse stage-1 indicator

  Upper pair (stage 2, gated on lower pair being stacked):
    cube_0_ee_distance              — 1 - tanh(||ee_w - cube_0||/std)   (always on)
    cube_0_is_lifted                — 1.0 if cube_0.z_w > minimal_height else 0.0
    cube_0_goal_distance_staged     — (lifted_0 AND cube_1-on-cube_2) *
                                       (1 - tanh(d_0_to_top_of_1 / std))
    cube_0_stacked_bonus            — sparse stage-2 indicator

  Full tower (alignment with success termination):
    three_tier_tower_bonus          — 1.0 iff BOTH pairs stacked simultaneously

Plus two regularizers from the shared `isaaclab.envs.mdp` namespace
(`action_rate_l2`, `joint_vel_l2`).

Goals are implicit: `cube_1`'s goal = `cube_2.pos_w + [0, 0, CUBE_SIZE]`,
`cube_0`'s goal = `cube_1.pos_w + [0, 0, CUBE_SIZE]` (dynamic — updates as
cube_1 settles into its stacked pose).

EE pose is read from the FrameTransformer scene entity `ee_frame` (LiftCube
convention). The Franka per-robot cfg installs this sensor pointing at
`panda_hand` with an offset of [0, 0, 0.1034] (fingertip).

The legacy 2-cube helpers (`cube_0_ee_distance`, `cube_0_is_lifted`,
`cube_0_goal_distance`, `cube_0_stacked_bonus`) are PRESERVED — the new
RewardsCfg reuses them directly for the upper pair; only the goal-tracking
term is replaced by the staged variant so it gates on lower-pair stacking.
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


# ---------------------------------------------------------------------------
# s1 — reach cube_0 (LiftCube `object_ee_distance` mirror)
# ---------------------------------------------------------------------------


def cube_0_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||cube_0 - ee_w|| / std)` — LiftCube `object_ee_distance` mirror, cube_0 retarget."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(d / std)


# ---------------------------------------------------------------------------
# s2 — lift cube_0 (LiftCube `object_is_lifted` mirror, binary)
# ---------------------------------------------------------------------------


def cube_0_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """`1.0 if cube_0.z_w > minimal_height else 0.0`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return torch.where(cube.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# ---------------------------------------------------------------------------
# s3 / s4 — coarse / fine goal tracking (LiftCube `object_goal_distance` mirror)
# ---------------------------------------------------------------------------


def cube_0_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    goal_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """`(cube_0 lifted) * (1 - tanh(d / std))` — LiftCube `object_goal_distance` mirror.

    Target is `cube_1.pos_w + [0, 0, CUBE_SIZE]` (the stack top). No
    command_manager dependency: the goal is implicit in the scene.
    """
    cube_0: RigidObject = env.scene[cube_cfg.name]
    cube_1: RigidObject = env.scene[goal_cube_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    target_w = pos_1.clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    distance = torch.norm(target_w - pos_0, dim=1)
    lifted = pos_0[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# s5 — success bonus: sparse +1 when cube_0 is stacked on cube_1
# ---------------------------------------------------------------------------


def cube_0_stacked_bonus(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """`1.0 if (|xy_0 - xy_1|<xy_threshold AND |Δz - CUBE_SIZE|<z_threshold) else 0.0`.

    Same geometric condition as `cube_0_stacked_on_cube_1` in
    `mdp/terminations.py` — this is the dense per-step success indicator and
    the termination wraps the very same check.
    """
    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    aligned_xy = xy_dist < xy_threshold
    on_top_z = torch.abs(z_gap - CUBE_SIZE) < z_threshold
    return (aligned_xy & on_top_z).float()


def _gripper_far_from_cube_0(
    env: "ManagerBasedRLEnv",
    min_distance: float = 0.04,
) -> torch.Tensor:
    """True per env if EE (ee_frame target 0) is at least `min_distance` away from cube_0.

    Uses ee_frame's `target_pos_w` (panda_hand + [0,0,0.1034] offset) and
    cube_0 root pos, L2 distance in world frame. 4 cm default ≈ one cube edge.
    """
    cube_0 = env.scene["cube_0"]
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube_0.data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    return distance > min_distance


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
    gripper_away_min_distance: float = 0.04,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """+1.0 the FIRST step cube_0 stacked on cube_1 + EE moved away ≥ 4 cm + no contact, else 0.0.

    Success criteria (ALL THREE must hold):
      - geometric: |cube_0.xy − cube_1.xy| < xy_threshold AND |Δz − CUBE_SIZE| < z_threshold
      - distance: ||EE_pos − cube_0_pos|| ≥ gripper_away_min_distance (default 4 cm)
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
    far_enough = _gripper_far_from_cube_0(env, min_distance=gripper_away_min_distance)
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


def cube_0_currently_stacked_on_cube_1_indicator(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Float (num_envs,): 1.0 iff cube_0 is geometrically stacked on cube_1 right now.

    Same geometric condition as the success termination's cube_0-on-cube_1 leg
    (xy < xy_threshold AND |Δz - CUBE_SIZE| < z_threshold) but evaluated each
    frame. Used to gate cube_2 shaping — the cube_2 reward only fires while
    cube_0 IS currently on cube_1 (not "has been at any point earlier").
    Forces the policy to MAINTAIN cube_0's stack while engaging cube_2.

    No release check here — the policy may be holding cube_0 against cube_1.
    The success termination still requires release.
    """
    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    stacked = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)
    return stacked.float()


def cube_2_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||cube_2 - ee_w|| / std)` — mirror of `cube_0_ee_distance` retargeted to cube_2."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(d / std)


def cube_2_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """`1.0 if cube_2.z_w > minimal_height else 0.0`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return torch.where(cube.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def cube_2_ee_distance_gated_on_cube_0_stacked(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
) -> torch.Tensor:
    """Reach cube_2 — only fires while cube_0 IS CURRENTLY stacked on cube_1.

    Iter-4 change: gate switched from a sticky "stacked at any point this
    episode" latch to the current-frame geometric check. Forces the policy to
    keep cube_0 on cube_1 while engaging cube_2 — if cube_0 falls off, the
    cube_2 reward shuts off.
    """
    return cube_2_ee_distance(env, std=std) * cube_0_currently_stacked_on_cube_1_indicator(env)


def cube_2_is_lifted_gated_on_cube_0_stacked(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Lift cube_2 — only fires while cube_0 IS CURRENTLY stacked on cube_1."""
    return cube_2_is_lifted(env, minimal_height=minimal_height) * cube_0_currently_stacked_on_cube_1_indicator(env)


# ---------------------------------------------------------------------------
# 3-tier tower extension (iter 23): cube_1 → cube_2 stage + full-tower bonus.
#
# Design mirror of cube_0 helpers, retargeted to (cube_1 top, cube_2 base).
# The goal-distance helper for cube_0 is REPLACED below by a staged variant
# that additionally requires cube_1 to be already stacked on cube_2 before
# the dense goal-tracking signal fires — so the policy is pulled through the
# curriculum: stack cube_1 first, then cube_0.
# ---------------------------------------------------------------------------


def cube_1_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||cube_1 - ee_w|| / std)` — mirror of `cube_0_ee_distance` for cube_1."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(d / std)


def cube_1_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """`1.0 if cube_1.z_w > minimal_height else 0.0`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return torch.where(cube.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def cube_1_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    goal_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """`(cube_1 lifted) * (1 - tanh(d / std))`.

    Target is `cube_2.pos_w + [0, 0, CUBE_SIZE]` (cube_1's stacked-on-cube_2
    pose). Mirror of `cube_0_goal_distance` retargeted to the lower pair.
    """
    cube_1: RigidObject = env.scene[cube_cfg.name]
    cube_2: RigidObject = env.scene[goal_cube_cfg.name]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    target_w = pos_2.clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    distance = torch.norm(target_w - pos_1, dim=1)
    lifted = pos_1[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(distance / std))


def cube_1_stacked_on_cube_2_bonus(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """`1.0 if cube_1 is stacked on cube_2 else 0.0` — sparse stage-1 bonus."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_1[:, :2] - pos_2[:, :2], dim=-1)
    z_gap = pos_1[:, 2] - pos_2[:, 2]
    aligned_xy = xy_dist < xy_threshold
    on_top_z = torch.abs(z_gap - CUBE_SIZE) < z_threshold
    return (aligned_xy & on_top_z).float()


# ---------------------------------------------------------------------------
# iter 25 — strict curriculum helpers: gate ALL cube_0 shaping on cube_1-on-cube_2
# ---------------------------------------------------------------------------


def cube_1_on_cube_2_indicator(
    env, xy_threshold: float = 0.02, z_threshold: float = 0.01
) -> torch.Tensor:
    """Float (num_envs,) indicator: 1.0 iff cube_1 is stacked on cube_2 right now."""
    cube_1 = env.scene["cube_1"]; cube_2 = env.scene["cube_2"]
    p1 = cube_1.data.root_pos_w[:, :3]; p2 = cube_2.data.root_pos_w[:, :3]
    xy_ok = torch.norm(p1[:, :2] - p2[:, :2], dim=-1) < xy_threshold
    z_ok  = torch.abs((p1[:, 2] - p2[:, 2]) - CUBE_SIZE) < z_threshold
    return (xy_ok & z_ok).float()


def cube_0_ee_distance_gated(env, std: float = 0.1) -> torch.Tensor:
    """Reaching cube_0 — multiplied by the cube_1_on_cube_2 indicator.

    Zero reward for reaching cube_0 until the base pair (cube_1 on cube_2)
    is already in place. Closes the iter-24 loophole that let the policy
    collect cube_0 reaching/lifting reward while ignoring cube_1.
    """
    gate = cube_1_on_cube_2_indicator(env)
    return cube_0_ee_distance(env, std=std) * gate


def cube_0_is_lifted_gated(env, minimal_height: float = 0.04) -> torch.Tensor:
    """Lifting cube_0 — multiplied by the cube_1_on_cube_2 indicator."""
    gate = cube_1_on_cube_2_indicator(env)
    return cube_0_is_lifted(env, minimal_height=minimal_height) * gate


def cube_0_goal_distance_staged(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float = 0.04,
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Staged version of `cube_0_goal_distance` for the 3-tier tower.

    `(cube_0 lifted) * (cube_1 stacked on cube_2) * (1 - tanh(d / std))`,
    where the target is the CURRENT `cube_1.pos_w + [0, 0, CUBE_SIZE]`.

    Two gates: cube_0 must be lifted AND cube_1 must already be stacked on
    cube_2. The second gate prevents the policy from being rewarded for
    placing cube_0 on a free-floating cube_1, which would dis-incentivize
    completing the lower pair first.
    """
    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    target_w = pos_1.clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    distance = torch.norm(target_w - pos_0, dim=1)
    lifted = pos_0[:, 2] > minimal_height
    # Lower pair already stacked? (uses the same geometric check as the bonus)
    xy_dist_12 = torch.norm(pos_1[:, :2] - pos_2[:, :2], dim=-1)
    z_gap_12 = pos_1[:, 2] - pos_2[:, 2]
    lower_stacked = (xy_dist_12 < xy_threshold) & (torch.abs(z_gap_12 - CUBE_SIZE) < z_threshold)
    gate = lifted & lower_stacked
    return gate.float() * (1.0 - torch.tanh(distance / std))


def cube_0_stacked_bonus_gated(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """+1.0 iff cube_0 is stacked on cube_1 AND cube_1 is already stacked on cube_2.

    Closes the iter-23 curriculum loophole: the legacy `mdp.cube_0_stacked_bonus`
    fires geometrically regardless of cube_1 placement, allowing the policy to
    short-circuit by dropping cube_0 on a free-floating cube_1. This gated
    variant requires the base pair (cube_1 on cube_2) to hold first so the
    +200 sparse bonus can only be claimed AFTER stage-1 completes.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    p0 = cube_0.data.root_pos_w[:, :3]
    p1 = cube_1.data.root_pos_w[:, :3]
    p2 = cube_2.data.root_pos_w[:, :3]
    cube_0_on_1 = (
        (torch.norm(p0[:, :2] - p1[:, :2], dim=-1) < xy_threshold)
        & (torch.abs((p0[:, 2] - p1[:, 2]) - CUBE_SIZE) < z_threshold)
    )
    cube_1_on_2 = (
        (torch.norm(p1[:, :2] - p2[:, :2], dim=-1) < xy_threshold)
        & (torch.abs((p1[:, 2] - p2[:, 2]) - CUBE_SIZE) < z_threshold)
    )
    return (cube_0_on_1 & cube_1_on_2).float()


# ---------------------------------------------------------------------------
# iter 32 — LOOSE-shaping helpers: cube_0 shaping gated on cube_1 LIFTED
# (not strictly stacked on cube_2). Opens the gate ~31% of the time instead of
# ~8% (iter-26 policy stats), giving PPO 4× more learning windows to bootstrap
# cube_0 manipulation while the sparse bonuses and full_tower stay on the
# strict stacked gate so the "drop cube_0 on a free-floating cube_1" shortcut
# still pays zero.
# ---------------------------------------------------------------------------


def cube_1_is_lifted_indicator(env, minimal_height: float = 0.04) -> torch.Tensor:
    """Float (num_envs,) indicator: 1.0 iff cube_1 is currently above `minimal_height`.

    Looser gate than `cube_1_on_cube_2_indicator` — fires whenever cube_1 is in
    the air, not just when it's stacked. Used by iter-32's loose-shaping pattern:
    cube_0 shaping reward becomes available as soon as cube_1 is lifted (gives the
    policy more learning windows to bootstrap cube_0 manipulation) while the cube_0
    sparse bonus and full_tower bonus remain on the strict stacked gate so the
    'just stack cube_0 on a free-floating cube_1' shortcut still pays zero.
    """
    cube_1 = env.scene["cube_1"]
    return (cube_1.data.root_pos_w[:, 2] > minimal_height).float()


def cube_0_ee_distance_loose_gated(env, std: float = 0.1, minimal_height: float = 0.04) -> torch.Tensor:
    """Reaching cube_0, gated on cube_1 being LIFTED (not strictly stacked).

    iter-32 loose-gated variant of `cube_0_ee_distance_gated`. Opens whenever
    cube_1 is in the air (~31% of frames in the iter-26 policy) instead of
    only when cube_1 is geometrically stacked on cube_2 (~8% of frames).
    """
    return cube_0_ee_distance(env, std=std) * cube_1_is_lifted_indicator(env, minimal_height=minimal_height)


def cube_0_is_lifted_loose_gated(env, minimal_height: float = 0.04) -> torch.Tensor:
    """Lifting cube_0, gated on cube_1 being LIFTED (not strictly stacked).

    iter-32 loose-gated variant of `cube_0_is_lifted_gated`. Pairs with
    `cube_0_ee_distance_loose_gated` to expose cube_0 manipulation shaping to
    the policy as soon as cube_1 is airborne.
    """
    return cube_0_is_lifted(env, minimal_height=minimal_height) * cube_1_is_lifted_indicator(env, minimal_height=minimal_height)


def cube_0_goal_distance_loose_staged(env, std: float, minimal_height: float = 0.04) -> torch.Tensor:
    """Two-part loose gate (cube_0 lifted AND cube_1 lifted) * (1 - tanh(d_to_goal/std)).

    Goal = current `cube_1.pos_w + [0, 0, CUBE_SIZE]` (same as the strict
    variant `cube_0_goal_distance_staged`, just a looser gate on cube_1's
    state — requires cube_1 LIFTED, not strictly stacked on cube_2). This
    keeps the cube_0 dense goal-tracking signal alive across the larger
    set of frames where cube_1 is airborne.
    """
    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    p0 = cube_0.data.root_pos_w[:, :3]
    p1 = cube_1.data.root_pos_w[:, :3]
    target = p1.clone()
    target[:, 2] = target[:, 2] + CUBE_SIZE
    distance = torch.norm(target - p0, dim=1)
    cube_0_lifted = p0[:, 2] > minimal_height
    cube_1_lifted = p1[:, 2] > minimal_height
    gate = (cube_0_lifted & cube_1_lifted).float()
    return gate * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# iter 33 — LATCHED cube_0 shaping helpers
#
# Per-episode per-env latch: once cube_1 has been stacked on cube_2 AT LEAST
# ONCE in this episode the latch stays True for the rest of the episode. Uses
# the same module-level `_LATCH_BUFFERS` dict declared near the top of this
# file. Resets via `env.episode_length_buf <= 1` (post-`_reset_idx` the first
# step's episode_length_buf is 1, so this is a safe edge).
#
# iter 32 (loose-lifted gate) gave the policy a "lift cube_1 then collect
# cube_0 shaping" shortcut. iter 33 closes that by REQUIRING a true stack
# (latched) to open the shaping gate. Strict sparse bonuses unchanged.
# ---------------------------------------------------------------------------


def cube_1_was_stacked_latched_indicator(
    env, xy_threshold: float = 0.02, z_threshold: float = 0.01
) -> torch.Tensor:
    """Float (num_envs,) indicator: 1.0 iff cube_1 has been stacked on cube_2 AT LEAST ONCE this episode.

    Per-env latch:
      - Reset to False when `env.episode_length_buf <= 1` (the very first step
        after `_reset_idx` sets the counter to 0 then `step` increments to 1).
      - Set to True whenever the instantaneous `cube_1_on_cube_2` geometric
        check fires.
      - Stays True until the next episode resets the latch.

    Gives the policy ~150-200 frames of cube_0 shaping signal per stage-1-
    success episode (instead of the ~21 transient frames the strict
    instantaneous gate yields), while blocking the iter-32 "fake the lift then
    chase cube_0" shortcut (the gate now requires a true stack, not just a
    lift).

    Caveat: state lives in a module-level dict keyed by id(env); safe for the
    reward-tune loop (one train at a time). Resets across Python process
    restarts because the dict is in-process state.
    """
    latch = _get_latch_buffer(env, "cube_1_stacked")

    # Reset latch for envs that just started a new episode.
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    # Instantaneous cube_1 on cube_2 check (same geometry as the sparse bonus).
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    p1 = cube_1.data.root_pos_w[:, :3]
    p2 = cube_2.data.root_pos_w[:, :3]
    xy_ok = torch.norm(p1[:, :2] - p2[:, :2], dim=-1) < xy_threshold
    z_ok = torch.abs((p1[:, 2] - p2[:, 2]) - CUBE_SIZE) < z_threshold
    now_stacked = xy_ok & z_ok

    # Sticky update — once True, stays True for the rest of the episode.
    latch = latch | now_stacked

    # Save back. (Cannot use in-place ops because `torch.where` returned a
    # fresh tensor on the reset branch; we rebind the dict entry instead.)
    _LATCH_BUFFERS[(id(env), "cube_1_stacked")] = latch

    return latch.float()


def cube_0_ee_distance_latched_gated(env, std: float = 0.1) -> torch.Tensor:
    """Reaching cube_0 — LATCHED gate (fires for rest of episode once cube_1 was stacked)."""
    return cube_0_ee_distance(env, std=std) * cube_1_was_stacked_latched_indicator(env)


def cube_0_is_lifted_latched_gated(env, minimal_height: float = 0.04) -> torch.Tensor:
    """Lifting cube_0 — LATCHED gate (fires for rest of episode once cube_1 was stacked)."""
    return cube_0_is_lifted(env, minimal_height=minimal_height) * cube_1_was_stacked_latched_indicator(env)


def cube_0_goal_distance_latched_staged(env, std: float, minimal_height: float = 0.04) -> torch.Tensor:
    """Two-part gate (cube_0 lifted AND cube_1-was-stacked-latched) * (1 - tanh(d/std)).

    Goal = current `cube_1.pos_w + [0, 0, CUBE_SIZE]` (same as the existing
    strict variant `cube_0_goal_distance_staged`). The cube_1 check is the
    LATCHED version so the gate stays open even if cube_1 has fallen off
    cube_2 again — preserving the cube_0 shaping signal across the full
    bootstrap window once stage 1 has succeeded at any point this episode.
    """
    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    p0 = cube_0.data.root_pos_w[:, :3]
    p1 = cube_1.data.root_pos_w[:, :3]
    target = p1.clone()
    target[:, 2] = target[:, 2] + CUBE_SIZE
    distance = torch.norm(target - p0, dim=1)
    cube_0_lifted = p0[:, 2] > minimal_height
    latch = cube_1_was_stacked_latched_indicator(env) > 0.5
    gate = (cube_0_lifted & latch).float()
    return gate * (1.0 - torch.tanh(distance / std))


def three_tier_tower_bonus(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """`1.0 if BOTH (cube_0 on cube_1) AND (cube_1 on cube_2) else 0.0`.

    Matches the geometric condition of `mdp.three_tier_tower_stacked` (the
    success termination). Without this term the policy could earn the sum of
    `cube_0_stacked_bonus` + `cube_1_stacked_on_cube_2_bonus` even when the
    two pairs are NOT simultaneously aligned (e.g. cube_0 lands on cube_1
    after cube_1 has already wobbled off cube_2). The full-tower bonus closes
    that loophole and aligns the dense reward with the success termination.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    # Upper pair: cube_0 on cube_1
    xy_01 = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_01 = pos_0[:, 2] - pos_1[:, 2]
    upper = (xy_01 < xy_threshold) & (torch.abs(z_01 - CUBE_SIZE) < z_threshold)
    # Lower pair: cube_1 on cube_2
    xy_12 = torch.norm(pos_1[:, :2] - pos_2[:, :2], dim=-1)
    z_12 = pos_1[:, 2] - pos_2[:, 2]
    lower = (xy_12 < xy_threshold) & (torch.abs(z_12 - CUBE_SIZE) < z_threshold)
    return (upper & lower).float()


# ---------------------------------------------------------------------------
# IsaacGymEnvs FrankaCubeStack `compute_franka_reward` re-implementation
# ---------------------------------------------------------------------------


def franka_cube_stack_reward(
    env: "ManagerBasedRLEnv",
    r_dist_scale: float = 0.1,
    r_lift_scale: float = 1.5,
    r_align_scale: float = 2.0,
    r_stack_scale: float = 16.0,
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    ),
) -> torch.Tensor:
    """Verbatim re-implementation of IsaacGymEnvs
    `franka_cube_stack.compute_franka_reward` with FrankaCubeStack.yaml scales:

        r_dist_scale = 0.1, r_lift_scale = 1.5, r_align_scale = 2.0, r_stack_scale = 16.0

    Mapping:
      cubeA → `cube_0` (the cube being moved)
      cubeB → `cube_1` (the destination cube)
      eef / eef_lf / eef_rf → robot bodies `panda_hand`, `panda_leftfinger`,
                              `panda_rightfinger`

    Composition (per IsaacGymEnvs):

        dist  = 1 − tanh(10 · (||cubeA−eef|| + ||cubeA−lf|| + ||cubeA−rf||) / 3)
        lift  = (cubeA.z − cubeA_size − table_height) > 0.04
        d_ab  = ||(cubeB − cubeA) + [0, 0, (sA+sB)/2]||
        align = (1 − tanh(10 · d_ab)) · lift
        dist  = max(dist, align)
        stack = (xy_dist < 2cm) AND (|height − target_h| < 2cm) AND (||eef−cubeA|| > 4cm)
        reward = where(stack, r_stack_scale · 1,
                              r_dist_scale·dist + r_lift_scale·lift + r_align_scale·align)
    """
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Body indices for hand + fingertips
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    lf_idx = robot.find_bodies("panda_leftfinger")[0][0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0][0]

    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    lf_pos = robot.data.body_state_w[:, lf_idx, :3]
    rf_pos = robot.data.body_state_w[:, rf_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]

    cube_a_size = CUBE_SIZE
    cube_b_size = CUBE_SIZE

    # distance: hand + both fingers to cubeA (sharp tanh slope=10)
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    d_lf = torch.norm(cube_a_pos - lf_pos, dim=-1)
    d_rf = torch.norm(cube_a_pos - rf_pos, dim=-1)
    dist_reward = 1.0 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3.0)

    # lift indicator (cubeA bottom > 4 cm above table top)
    cube_a_height = cube_a_pos[:, 2] - table_height
    cube_a_lifted = (cube_a_height - cube_a_size) > 0.04
    lift_reward = cube_a_lifted.float()

    # align: distance from cubeA to (cubeB + stack-offset) — only when lifted
    cube_a_to_b = cube_b_pos - cube_a_pos
    offset = torch.zeros_like(cube_a_to_b)
    offset[:, 2] = (cube_a_size + cube_b_size) / 2.0
    d_ab = torch.norm(cube_a_to_b + offset, dim=-1)
    align_reward = (1.0 - torch.tanh(10.0 * d_ab)) * lift_reward

    # IsaacGymEnvs: dist takes the max of (hand-to-cube) and align
    dist_reward = torch.max(dist_reward, align_reward)

    # stack indicator — xy aligned AND on top AND gripper RELEASED (>4cm away)
    target_height = cube_b_size + cube_a_size / 2.0
    aligned_xy = torch.norm(cube_a_to_b[:, :2], dim=-1) < 0.02
    on_top = torch.abs(cube_a_height - target_height) < 0.02
    gripper_away = d > 0.04
    stack_indicator = aligned_xy & on_top & gripper_away

    # compose: stack overrides; otherwise dist + lift + align (scaled)
    rewards = torch.where(
        stack_indicator,
        r_stack_scale * stack_indicator.float(),
        r_dist_scale * dist_reward
        + r_lift_scale * lift_reward
        + r_align_scale * align_reward,
    )
    return rewards


# ---------------------------------------------------------------------------
# IsaacGymEnvs FrankaCubeStack reward terms — SEPARATE versions
#
# Same math as `compute_franka_reward` but split into 4 independent RewTerms
# so each is individually visible in the training tracker. The original `where`
# composition is approximated by summing all four; when stacked, all four fire
# but stack (×16) dominates dist (×0.1) + lift (×1.5) + align (×2.0).
# ---------------------------------------------------------------------------


def _fcs_lifted_mask(cube_a_pos: torch.Tensor, table_height: float) -> torch.Tensor:
    """Helper: cubeA.bottom > 4cm above table top, IsaacGymEnvs convention."""
    cube_a_height = cube_a_pos[:, 2] - table_height
    return (cube_a_height - CUBE_SIZE) > 0.02


def _fcs_align_value(cube_a_pos, cube_b_pos, lifted_mask) -> torch.Tensor:
    """Helper: (1 - tanh(10·d_ab)) * lifted_mask, with d_ab to cubeB + stack offset."""
    cube_a_to_b = cube_b_pos - cube_a_pos
    offset = torch.zeros_like(cube_a_to_b)
    offset[:, 2] = CUBE_SIZE  # (cubeA_size + cubeB_size) / 2 == CUBE_SIZE for equal cubes
    d_ab = torch.norm(cube_a_to_b + offset, dim=-1)
    return (1.0 - torch.tanh(10.0 * d_ab)) * lifted_mask.float()


def fcs_dist_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IGenvs `dist_reward` (after `max(dist, align)` step).

    raw_dist = 1 − tanh(10·(d_hand + d_lf + d_rf) / 3)
    return max(raw_dist, align_reward)
    """
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    lf_idx = robot.find_bodies("panda_leftfinger")[0][0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0][0]
    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    lf_pos = robot.data.body_state_w[:, lf_idx, :3]
    rf_pos = robot.data.body_state_w[:, rf_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    d_lf = torch.norm(cube_a_pos - lf_pos, dim=-1)
    d_rf = torch.norm(cube_a_pos - rf_pos, dim=-1)
    raw_dist = 1.0 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3.0)
    lifted = _fcs_lifted_mask(cube_a_pos, table_height)
    align = _fcs_align_value(cube_a_pos, cube_b_pos, lifted)
    return torch.max(raw_dist, align)


def fcs_lift_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """IGenvs `lift_reward` = 1.0 if (cubeA.z − CUBE_SIZE − table_h) > 0.04 else 0."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    return _fcs_lifted_mask(cube_a_pos, table_height).float()


def fcs_align_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """IGenvs `align_reward = (1 − tanh(10·d_ab)) · lifted`."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    lifted = _fcs_lifted_mask(cube_a_pos, table_height)
    return _fcs_align_value(cube_a_pos, cube_b_pos, lifted)


def fcs_stack_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.0,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """IGenvs `stack_reward` = xy_align AND |Δheight|<2cm AND ||hand−cubeA||>4cm."""
    cube_a = env.scene[cube_a_cfg.name]
    cube_b = env.scene[cube_b_cfg.name]
    robot = env.scene[robot_cfg.name]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    hand_pos = robot.data.body_state_w[:, hand_idx, :3]
    cube_a_pos = cube_a.data.root_pos_w[:, :3]
    cube_b_pos = cube_b.data.root_pos_w[:, :3]
    cube_a_to_b = cube_b_pos - cube_a_pos
    cube_a_height = cube_a_pos[:, 2] - table_height
    target_height = CUBE_SIZE + CUBE_SIZE / 2.0
    aligned_xy = torch.norm(cube_a_to_b[:, :2], dim=-1) < 0.02
    on_top = torch.abs(cube_a_height - target_height) < 0.02
    d = torch.norm(cube_a_pos - hand_pos, dim=-1)
    gripper_away = d > 0.04
    return (aligned_xy & on_top & gripper_away).float()


# --------------------------------------------------------------------------- #
# Release-in-drop-zone bonus                                                  #
# --------------------------------------------------------------------------- #
#
# Per-step bonus when the *grasping cube* (same mux used by §5 obs and §6
# align reward) is within `xy_threshold` and `z_threshold` of the stack target
# AND the policy outputs an "open gripper" action this step. Encourages the
# policy to release the cube once it's hovering over the goal position.
#
# Gripper action convention (`BinaryJointPositionAction` in IsaacLab):
#   `action[:, -1] < 0`  → close
#   `action[:, -1] >= 0` → open
#
# Action index `-1` is the gripper because `ActionsCfg` declares
# `arm_action` (3-D) then `gripper_action` (1-D); concatenated layout is
# `[ee_dx, ee_dy, ee_dz, gripper]`.
def release_bonus_in_drop_zone(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.03,
) -> torch.Tensor:
    """`1.0` if (xy_to_target < xy_threshold AND |z_to_target| < z_threshold
    AND gripper_action >= 0) else `0.0`.

    Target follows the same mux as `grasping_cube_goal_distance`:
        not-yet-stacked: grasping_cube=cube_0, target=cube_1.xyz+[0,0,CUBE_SIZE]
        stacked        : grasping_cube=cube_2, target=cube_0.xyz+[0,0,CUBE_SIZE]
    """
    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    on_stack = _cube_0_on_cube_1_predicate(env)              # (N,) bool
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
    base_pos     = torch.where(on_stack.unsqueeze(-1), pos_0, pos_1)
    target_pos   = base_pos.clone()
    target_pos[:, 2] = target_pos[:, 2] + CUBE_SIZE

    delta = grasping_pos - target_pos                        # (N, 3)
    xy_dist = torch.norm(delta[:, :2], dim=-1)
    z_dist  = torch.abs(delta[:, 2])
    in_zone = (xy_dist < xy_threshold) & (z_dist < z_threshold)

    # action_manager.action carries the latest policy output, raw, in the
    # concatenated [arm(3), gripper(1)] layout. >= 0 → open per the
    # BinaryJointAction.process_actions threshold.
    gripper_open = env.action_manager.action[:, -1] >= 0.0   # (N,) bool

    return (in_zone & gripper_open).float()
