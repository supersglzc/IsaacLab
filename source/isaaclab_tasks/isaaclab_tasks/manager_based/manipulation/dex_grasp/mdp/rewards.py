# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the single-arm Allegro `dex_grasp` task.

Five-stage manipulation shaping ladder (composer = sum):

  1. palm_to_dog       dense palm -> dog attractor (Allegro palm via FrameTransformer)
  2. fingertip_to_dog  weighted-mean per-fingertip attractor (thumb 1.5x others)
  3. grasp_contact     binary "thumb + at least one of {idx, mid, pinky} in
                       contact with the dog" (bidex Allegro convention)
  4. lift_height       linear ramp on dog.z, gated on grasp_contact
  5. dog_to_target     tanh attractor on ||dog - DOG_TARGET||, gated on
                       dog.z > init_z + 0.05 (dog actually lifted)
  6. success_bonus     one-shot latch the first frame the dog is within 10 cm
                       of the target (mirrors the `success` termination)

dt-scaling: This fork of IsaacLab has REMOVED the per-weight `* dt` multiplier
inside `RewardManager.compute` (see `isaaclab/managers/reward_manager.py:149`).
Weights below are therefore RAW per-step magnitudes the user sees at runtime.

Per-stage saturated per-step magnitude budget (composer = sum, raw weights):

    palm_to_dog          0.05         -> 0.05 / step
    fingertip_to_dog     0.05         -> 0.05 / step
    grasp_contact        0.10         -> 0.10 / step (only when held)
    lift_height          0.50         -> 0.50 / step (only when held)
    dog_to_target        0.50         -> 0.50 / step (only when lifted >5 cm)
    success_bonus      100.0          -> +100 one-shot at success predicate

166-step episode (8.33 s @ 20 Hz) ceiling check (all stages saturated):
    dense ceiling (palm + fingertip + contact + lift + dog->target)
        ~ (0.05 + 0.05 + 0.10 + 0.50 + 0.50) * 166 = ~199
    success bonus one-shot                          +100
The dense ceiling exceeds the sparse landmark, but the dense terms only
saturate at the success state (lift_height saturates only when dog.z hits
target_z; dog_to_target saturates only when dog xyz hits target xyz). The
policy still has a strict reason to terminate the episode rather than dwell.

Sign convention: positive = good (no penalties iter 0).
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
# Module-level per-(env, key) latch buffers (success bonus one-shot).
# ---------------------------------------------------------------------------

_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    """Get-or-lazily-create a per-env boolean latch tensor of shape (num_envs,)."""
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


# ---------------------------------------------------------------------------
# Stage 1 -- dense palm -> dog attractor.
# ---------------------------------------------------------------------------


def palm_to_dog(
    env: "ManagerBasedRLEnv",
    std: float = 0.20,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`1 - tanh(||palm - dog|| / std)` -- dense palm -> dog attractor.

    `palm` is the Allegro `palm_link` exposed via the `ee_frame` FrameTransformer
    (zero offset). `dog` is the rigid object's COM. Standard reach attractor;
    wider std (0.20 m) than `fingertip_to_dog` since the palm is the
    big-motion mover and we want a gradient over the whole ~30 cm gap.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dog: RigidObject = env.scene[dog_cfg.name]
    palm_w = ee_frame.data.target_pos_w[..., 0, :]
    dog_w = dog.data.root_pos_w[:, :3]
    d = torch.norm(palm_w - dog_w, dim=-1)
    return 1.0 - torch.tanh(d / max(std, 1e-6))


# ---------------------------------------------------------------------------
# Stage 2 -- per-fingertip distance attractor (Allegro thumb-1.5x convention).
# ---------------------------------------------------------------------------


def fingertip_to_dog(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    fingertip_links: tuple[str, ...] = ("if5", "mf5", "pf5", "th5"),
    fingertip_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.5),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """Weighted-mean fingertip-to-dog attractor over the 4 Allegro fingertips.

    For each fingertip `i`:
        per_finger_i = 1 - tanh(||fingertip_i - dog|| / std)
    Return `sum(w_i * per_finger_i) / sum(w_i)` -- a weighted MEAN so the
    saturation is still 1.0 per step (regardless of how many fingers / what
    the weights sum to). Thumb is 1.5x others (bidex `object_robot_distance`
    convention for the Allegro right hand).

    Tighter std (0.10 m) than `palm_to_dog` (0.20 m): the fingertips matter
    in the close-range "wrap around the dog" regime, not the gross-motion
    early reach.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dog: RigidObject = env.scene[dog_cfg.name]
    # Resolve the 4 fingertip body indices (order: if5, mf5, pf5, th5).
    body_ids = robot.find_bodies(list(fingertip_links))[0]
    # (num_envs, 4, 3) -- positions of the 4 fingertips in world frame.
    fingertip_w = robot.data.body_pos_w[:, body_ids, :3]
    # (num_envs, 1, 3) -- dog COM, broadcasted to subtract from all fingertips.
    dog_w = dog.data.root_pos_w[:, None, :3]
    # (num_envs, 4) -- per-finger distance.
    d = torch.norm(fingertip_w - dog_w, dim=-1)
    # (num_envs, 4) -- per-finger attractor (saturates at 1.0 each).
    per_finger = 1.0 - torch.tanh(d / max(std, 1e-6))
    # Weighted mean over fingers.
    w = torch.tensor(fingertip_weights, device=env.device, dtype=per_finger.dtype)
    w_norm = w / w.sum().clamp_min(1e-6)
    return (per_finger * w_norm.unsqueeze(0)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Stage 3 -- bidex Allegro grasp predicate (thumb + any of {idx, mid, pinky}).
# ---------------------------------------------------------------------------


def _allegro_grasp_predicate(
    env: "ManagerBasedRLEnv",
    contact_force_threshold: float,
) -> torch.Tensor:
    """Boolean (num_envs,) -- True iff `thumb` AND at least one of {index,
    middle, pinky} fingertips are in contact with the dog above threshold.

    Mirrors bidex `get_allegro_contact` (for 4 sensors): "thumb opposable, any
    other finger gripping" -- the right predicate for a 4-finger Allegro hand
    grasping a small object like the 0.11 kg dog.

    Sensor name order (set by `DexGraspSceneCfg`):
        contact_sensors_0 -> if5 (index)
        contact_sensors_1 -> mf5 (middle)
        contact_sensors_2 -> pf5 (pinky)
        contact_sensors_3 -> th5 (thumb)
    Each sensor's `data.force_matrix_w` has shape (num_envs, n_bodies=1,
    n_filters=1, 3) -- one body, one filter (the dog).
    """
    forces = []
    for name in ("contact_sensors_0", "contact_sensors_1",
                 "contact_sensors_2", "contact_sensors_3"):
        sensor: ContactSensor = env.scene[name]
        # (num_envs,) -- scalar contact force magnitude on the dog filter.
        f = torch.norm(sensor.data.force_matrix_w[:, 0, 0, :], dim=-1)
        forces.append(f > contact_force_threshold)
    in_contact_idx, in_contact_mid, in_contact_pf, in_contact_thumb = forces
    any_non_thumb = in_contact_idx | in_contact_mid | in_contact_pf
    return in_contact_thumb & any_non_thumb


def grasp_contact(
    env: "ManagerBasedRLEnv",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 when the Allegro hand is grasping the dog (thumb opposing any
    non-thumb finger above `contact_force_threshold` newtons), else 0.0.

    Default threshold = 1.0 N matches bidex `get_allegro_contact` exactly. The
    dog is light (0.11 kg) so this is loose enough to fire on gentle grasps
    but tight enough that brushing against the dog with one finger does not
    spuriously count.
    """
    return _allegro_grasp_predicate(env, contact_force_threshold).float()


# ---------------------------------------------------------------------------
# Stage 4 -- linear lift ramp gated on grasp_contact.
# ---------------------------------------------------------------------------


def lift_height(
    env: "ManagerBasedRLEnv",
    init_z: float = 0.0,
    target_lift: float = 0.30,
    contact_force_threshold: float = 1.0,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`clamp((dog.z - init_z) / target_lift, 0, 1) * grasp_contact`.

    Linear ramp on the dog's env-local z position above its init height,
    multiplied by the Allegro grasp predicate. Without the grasp gate the
    policy can earn "lift" reward by knocking the dog up with the back of
    the hand or by table contact-force jitter.

    `dog.z` is computed in env-local frame (subtract env origin). The reset
    event teleports the dog to env-local (0.05, -0.35, 0.0), so init_z=0.0
    is the canonical resting height. `target_lift=0.30` matches
    `DOG_TARGET_LOCAL[2] = 0.30` exactly -- saturate when dog reaches target z.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    dog_z = dog_pos_local[:, 2]
    progress = ((dog_z - init_z) / max(target_lift, 1e-6)).clamp(0.0, 1.0)
    gate = _allegro_grasp_predicate(env, contact_force_threshold).float()
    return progress * gate


# ---------------------------------------------------------------------------
# Stage 5 -- tanh attractor on dog -> DOG_TARGET, gated on dog lifted.
# ---------------------------------------------------------------------------


def dog_to_target(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    lift_threshold: float = 0.05,
    init_z: float = 0.0,
    contact_force_threshold: float = 1.0,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`(grasp_contact AND dog.z > init_z + lift_threshold) * (1 - tanh(||dog - target|| / std))`.

    Dense full-3D attractor on the dog's env-local position toward
    `target_local`, gated on BOTH (a) the Allegro grasp predicate firing
    (thumb opposing any non-thumb finger >= `contact_force_threshold` N) AND
    (b) the dog lifted at least `lift_threshold` (5 cm) above its init height.

    Why the AND-gate: a dense xyz attractor at fingertip range from init is
    big enough that PPO will park the dog near the target xy without ever
    grasping it (knocking the dog up with the back of the hand satisfies the
    z-only lift gate). Requiring grasp_contact as well forces the policy to
    actually grip-then-lift before this reward fires.

    `target_local = (0.05, -0.35, 0.30)` matches `DOG_TARGET_LOCAL` in the
    env_cfg and `dog_reached_target` termination predicate.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_pos_local.dtype)
    d = torch.norm(dog_pos_local - target.unsqueeze(0), dim=-1)
    base = 1.0 - torch.tanh(d / max(std, 1e-6))
    lifted = (dog_pos_local[:, 2] > (init_z + lift_threshold)).float()
    held = _allegro_grasp_predicate(env, contact_force_threshold).float()
    return held * lifted * base


# ---------------------------------------------------------------------------
# Stage 6 -- one-shot success bonus mirroring the termination predicate.
# ---------------------------------------------------------------------------


def success_bonus(
    env: "ManagerBasedRLEnv",
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    threshold: float = 0.10,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """+1.0 the FIRST frame the dog meets the success predicate this episode;
    0.0 thereafter. Per-env latch resets at `env.episode_length_buf <= 1`.

    The predicate EXACTLY mirrors `terminations.dog_reached_target`:
        ||dog_local - target_local|| < threshold
    so the latched signal fires the same step the success termination kills
    the episode. Effectively a one-shot +weight bonus at task success.
    """
    latch = _get_latch_buffer(env, "success_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_pos_local.dtype)
    err = torch.norm(dog_pos_local - target.unsqueeze(0), dim=-1)
    now_success = err < threshold

    fire = now_success & (~latch)
    latch = latch | now_success
    _LATCH_BUFFERS[(id(env), "success_once")] = latch
    return fire.float()


# ---------------------------------------------------------------------------
# Legacy shim kept so older `placeholder` weight=0.0 RewardsCfg blocks (and
# the §6 task-generator scaffolding) still import without breaking.
# ---------------------------------------------------------------------------


def placeholder_zero(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant-zero per-env reward; kept as a legacy shim."""
    return torch.zeros(env.num_envs, device=env.device)
