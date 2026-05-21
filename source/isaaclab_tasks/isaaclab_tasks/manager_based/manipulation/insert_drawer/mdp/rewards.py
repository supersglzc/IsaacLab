# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the insert_drawer task.

Active terms (referenced by `RewardsCfg`):

  Phase 2 — pick + place (all zero-masked once `cube_inside_bonus_latch` fires)
    reach_cube       `1 - tanh(||ee - cube|| / std)`
    is_lifted        binary `cube.z > minimal_height`
    lift_distance    ramp `clamp((cube.z - init_z) / (target_z - init_z))`,
                     gated on BOTH finger contact sensors reporting force > thr
    align            `(1 - tanh(||cube - drop_frame|| / std))` gated on
                     `cube.z > minimal_height_b`

  Phase 3 — post-latch
    cube_inside_bonus_once_per_episode   one-shot +1.0 the first frame the cube
                                         is geometrically inside the drawer AND
                                         the EE is far from the cube
    ee_retract_to_front_face             `(1 - tanh(||ee - front_face|| / std))`
                                         only after the latch fires

  Phase 4 — close drawer
    close_drawer     `cube_inside * (ee.y > front_face.y) *
                     clamp((max_open - joint_pos) / max_open)`

The `_cube_inside_latch_active(env)` helper reads the latch buffer key that
`cube_inside_bonus_once_per_episode` writes, applying its own episode-boundary
reset so it's safe to call before that term runs in the per-step reward sweep.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Module-level per-(env, key) latch buffers.
# ---------------------------------------------------------------------------

_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    """Get or lazily-create a per-env boolean latch tensor of shape (num_envs,)."""
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def _cube_inside_latch_active(env) -> torch.Tensor:
    """Returns float (num_envs,): 1.0 if the cube_inside_bonus latch has fired
    earlier in the current episode, else 0.0.

    Reads the SAME buffer key (`"cube_inside_once"`) that
    `cube_inside_bonus_once_per_episode` writes to. Applies the same
    episode-boundary reset locally so this helper is safe to call BEFORE
    `cube_inside_bonus_once_per_episode` runs in the per-step reward sweep.
    """
    latch = _get_latch_buffer(env, "cube_inside_once")
    just_reset = env.episode_length_buf <= 1
    return (latch & ~just_reset).float()


# ---------------------------------------------------------------------------
# Phase 2 — pick + place (zero-masked once the cube_inside latch fires).
# ---------------------------------------------------------------------------


def reach_cube(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||ee - cube|| / std)` — dense EE→cube attractor."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return (1.0 - torch.tanh(d / std)) * (1.0 - _cube_inside_latch_active(env))


def is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """Binary `1.0 if cube.z > minimal_height else 0.0`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    return (cube_z > minimal_height).float() * (1.0 - _cube_inside_latch_active(env))


def lift_distance(
    env: "ManagerBasedRLEnv",
    init_z: float = 0.0215,
    target_z: float = 0.20,
    contact_force_threshold: float = 1e-3,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """Linear ramp `clamp((cube.z - init_z) / (target_z - init_z), 0, 1)`,
    gated on BOTH fingertips being in contact with the cube.

    Reads `force_matrix_w[:, 0, 0, :]` (filter index 0 = Cube_0) on the
    `finger_left_contact` and `finger_right_contact` sensors. The ramp only
    fires when ‖left_force‖ > thr AND ‖right_force‖ > thr — preventing the
    policy from earning lift reward by knocking the cube up with the body
    of the hand or by single-finger flicks.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    base = ((cube_z - init_z) / max(target_z - init_z, 1e-6)).clamp(0.0, 1.0)
    left: ContactSensor = env.scene["finger_left_contact"]
    right: ContactSensor = env.scene["finger_right_contact"]
    left_f = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    grasp_gate = ((left_f > contact_force_threshold) & (right_f > contact_force_threshold)).float()
    return base * grasp_gate * (1.0 - _cube_inside_latch_active(env))


def align(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    minimal_height_b: float = 0.15,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    drawer_drop_frame_cfg: SceneEntityCfg = SceneEntityCfg("drawer_drop_frame"),
) -> torch.Tensor:
    """Dense cube→drop_frame attractor, gated on `cube.z > minimal_height_b`."""
    cube: RigidObject = env.scene[cube_cfg.name]
    drop_frame: FrameTransformer = env.scene[drawer_drop_frame_cfg.name]
    cube_pos = cube.data.root_pos_w[:, :3]
    drop_pos = drop_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos - drop_pos, dim=-1)
    base = 1.0 - torch.tanh(d / std)
    high_enough = (cube_pos[:, 2] > minimal_height_b).float()
    return high_enough * base * (1.0 - _cube_inside_latch_active(env))


def ee_retract_to_front_face(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    drawer_front_face_frame_cfg: SceneEntityCfg = SceneEntityCfg("drawer_front_face_frame"),
) -> torch.Tensor:
    """`(1 - tanh(|ee.y - front_face.y| / std)) * latch_active`.

    Y-axis-only attractor: encourages the gripper to retract along Y toward
    the drawer's front face once the cube has been inserted, so the policy
    can transition into the close_drawer phase from a sensible pose.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    front_face: FrameTransformer = env.scene[drawer_front_face_frame_cfg.name]
    ee_y = ee_frame.data.target_pos_w[..., 0, 1]
    front_y = front_face.data.target_pos_w[..., 0, 1]
    dy = torch.abs(ee_y - front_y)
    base = 1.0 - torch.tanh(dy / std)
    return base * _cube_inside_latch_active(env)


# ---------------------------------------------------------------------------
# Phase 3 — release latch + geometric helper.
# ---------------------------------------------------------------------------


def _cube_inside_drawer_geometric(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.20,
    z_rel_floor: float = -0.02,
    z_rel_ceiling: float = 0.07,
) -> torch.Tensor:
    """Geometric-only 'cube inside drawer interior' predicate (no EE clause).

    Used to gate `close_drawer` and `cube_inside_bonus_once_per_episode`.
    """
    cube: RigidObject = env.scene["cube_0"]
    drawer: Articulation = env.scene["drawer"]
    body_idx = drawer.find_bodies("drawer")[0][0]
    cube_pos = cube.data.root_pos_w[:, :3]
    drawer_pos = drawer.data.body_pos_w[:, body_idx, :]
    rel = cube_pos - drawer_pos
    xy_in = torch.norm(rel[:, :2], dim=-1) < xy_threshold
    z_in = (rel[:, 2] > z_rel_floor) & (rel[:, 2] < z_rel_ceiling)
    return xy_in & z_in


def cube_inside_bonus_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.20,
    z_rel_floor: float = -0.02,
    z_rel_ceiling: float = 0.07,
    ee_cube_no_contact_threshold: float = 0.10,
) -> torch.Tensor:
    """+1.0 the FIRST frame the cube is geometrically inside the drawer AND
    the EE is at least `ee_cube_no_contact_threshold` away from the cube.

    Per-env latch resets at `env.episode_length_buf <= 1`.
    """
    latch = _get_latch_buffer(env, "cube_inside_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)
    inside = _cube_inside_drawer_geometric(
        env, xy_threshold=xy_threshold,
        z_rel_floor=z_rel_floor, z_rel_ceiling=z_rel_ceiling,
    )
    cube_pos = env.scene["cube_0"].data.root_pos_w[:, :3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    ee_far = torch.norm(ee_pos - cube_pos, dim=-1) > ee_cube_no_contact_threshold
    now_qualifying = inside & ee_far
    fire = now_qualifying & (~latch)
    latch = latch | now_qualifying
    _LATCH_BUFFERS[(id(env), "cube_inside_once")] = latch
    return fire.float()


# ---------------------------------------------------------------------------
# Phase 4 — close drawer.
# ---------------------------------------------------------------------------


def close_drawer(
    env: "ManagerBasedRLEnv",
    max_open: float = 0.30,
    alpha: float = 0.5,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    joint_name: str = "base_drawer_joint",
    drawer_front_face_frame_cfg: SceneEntityCfg = SceneEntityCfg("drawer_front_face_frame"),
) -> torch.Tensor:
    """`latch_active * gate_ee_outside * closeness ** alpha`,
    where `closeness = clamp((max_open - joint_pos) / max_open, 0, 1)`.

    `alpha < 1` makes the reward concave in closeness — large dr/dx near 0
    (motivates the first push) and small near 1 (so total reward stays bounded
    while not over-rewarding fully-closed dwell).

    Gates on the `cube_inside_bonus` LATCH (sticky once the cube was inserted
    this episode) rather than the active geometric `cube inside drawer` check —
    so close-drawer reward keeps firing even if the cube shifts inside the
    drawer as it slides.

    The EE-outside gate compares the EE's world Y to the
    `drawer_front_face_frame`:

        gate_ee_outside = (ee.y > drawer_front_face.y)
    """
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    closeness = torch.clamp((max_open - joint_pos) / max(max_open, 1e-6), min=0.0, max=1.0)
    shaped = closeness.pow(alpha)
    gate_cube = _cube_inside_latch_active(env)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    front_face: FrameTransformer = env.scene[drawer_front_face_frame_cfg.name]
    ee_y = ee_frame.data.target_pos_w[..., 0, 1]
    front_y = front_face.data.target_pos_w[..., 0, 1]
    gate_ee_outside = (ee_y > front_y).float()
    return gate_cube * gate_ee_outside * shaped


# ---------------------------------------------------------------------------
# Phase 5 — success bonus.
# ---------------------------------------------------------------------------


def success_bonus(
    env: "ManagerBasedRLEnv",
    drawer_closed_threshold: float = 0.05,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    joint_name: str = "base_drawer_joint",
) -> torch.Tensor:
    """`latch_active * (joint_pos < drawer_closed_threshold)`.

    +1.0 the frame the cube has been inserted (latch sticky) AND the drawer
    joint position is below `drawer_closed_threshold`. Paired with
    `success_termination` (which kills the episode the same frame), this is
    effectively a one-shot +weight bonus at task success.
    """
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    drawer_closed = (joint_pos < drawer_closed_threshold).float()
    return _cube_inside_latch_active(env) * drawer_closed
