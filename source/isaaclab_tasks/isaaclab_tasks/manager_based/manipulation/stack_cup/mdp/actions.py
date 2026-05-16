"""EMA cumulative-delta task-space (EE position) action — fixed RPY variant.

Edit_mode_014 (§2): swap stock `mdp.JointPositionActionCfg` for a custom
position-only EMA EE-delta term. This is the 3-D-position-only variant of
the plugin's `EMACumulativeDeltaPoseAction` template — the policy outputs
ONLY a 3-D position delta `(dx, dy, dz)`. The EE quaternion is FIXED at the
post-reset value (no rotation channel, no axis-angle compose).

Behavior:
    1. Accumulates the 3-D position delta over the episode:
           delta_t = delta_{t-1} + s · a_t
    2. Anchors on the EE pose captured lazily at the first
       `process_actions` after `env.reset()`:
           abs_pos_t  = init_ee_pos + delta_t
           abs_quat_t = init_ee_quat                       (RPY locked)
    3. EMA-smooths against the previously applied target on the position
       channel only (orientation stays exactly at `init_ee_quat`):
           target_t.pos  = α · abs_pos_t + (1 - α) · prev_applied_pos
           target_t.quat = init_ee_quat
    4. Forwards the absolute 7-D pose target to the IK controller (forced
       into `use_relative_mode=False`) and reuses the parent's
       `apply_actions()` for jacobian + IK numerics.

The policy interface is 3-D (`action_dim == 3`); the IK command is 7-D
(handled internally by composing `(pos, init_ee_quat)`).

State across steps: `del_action` (3-D), `_prev_applied_pos` (3-D).
On `env.reset()`: `del_action := 0`, re-anchor flag set so the NEXT
`process_actions` captures fresh `init_ee_{pos,quat}` from `body_pose_w`.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class EMACumulativeDeltaPositionAction(DifferentialInverseKinematicsAction):
    cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg"

    def __init__(
        self,
        cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg",
        env: "ManagerBasedEnv",
    ) -> None:
        # Force the IK controller into absolute pose mode — we manage relative deltas ourselves.
        if cfg.controller.use_relative_mode:
            raise ValueError(
                "EMACumulativeDeltaPositionAction handles relative deltas itself; "
                "set controller.use_relative_mode=False"
            )
        if cfg.controller.command_type != "pose":
            raise ValueError(
                "EMACumulativeDeltaPositionAction requires controller.command_type='pose'; "
                f"got '{cfg.controller.command_type}'"
            )
        super().__init__(cfg, env)

        # Override the policy-input action shape to 3 (position delta only).
        # The parent allocated _raw_actions / _processed_actions / _scale at
        # the IK controller's action_dim (=7 for pose+abs). We need:
        #   raw_actions / scale : 3-D (policy interface)
        #   _processed_actions  : 7-D (IK controller command in abs pose mode)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._scale = torch.zeros((env.num_envs, 3), device=env.device)
        self._scale[:] = torch.tensor(cfg.scale, device=env.device)

        # Alpha (EMA weight)
        if not 0.0 <= cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {cfg.alpha}.")
        self._alpha = cfg.alpha

        # Cumulative position delta state (3-D)
        self.del_action = torch.zeros((env.num_envs, 3), device=env.device)

        # Init pose buffers — populated lazily on the next process_actions for
        # any env in `_needs_reanchor`. We CANNOT capture init pose here (or in
        # reset()) because at those call sites the articulation's body data is
        # stale — IsaacLab's ManagerBasedRLEnv runs the post-reset sim step
        # AFTER all term.reset() calls. Deferring to process_actions means we
        # read body_pose_w once it's been refreshed by the env's internal
        # post-reset step.
        self.init_ee_pos        = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat       = torch.zeros((env.num_envs, 4), device=env.device)
        self.init_ee_quat[:, 0] = 1.0                                                  # identity quat (wxyz)
        self._prev_applied_pos  = torch.zeros((env.num_envs, 3), device=env.device)
        # Mark all envs for re-anchoring on first process_actions.
        self._needs_reanchor    = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        # Optional position clamp (per-axis lower/upper)
        self.pos_lower_limit = (
            torch.tensor(cfg.pos_lower_limit, device=self.device)
            if cfg.pos_lower_limit is not None
            else None
        )
        self.pos_upper_limit = (
            torch.tensor(cfg.pos_upper_limit, device=self.device)
            if cfg.pos_upper_limit is not None
            else None
        )

    # Policy sees a 3-D action; IK command is 7-D (handled internally).
    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # Defer init_ee_pos refresh — articulation body_pose_w is stale here.
        if env_ids is None:
            self._needs_reanchor[:] = True
            self.del_action[:]      = 0.0
        else:
            self._needs_reanchor[env_ids] = True
            self.del_action[env_ids]      = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        # Re-anchor init pose for any env that just reset. body_pose_w is now
        # fresh (the env's post-reset sim step has run since reset()).
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask]      = ee_pos_curr[mask]
            self.init_ee_quat[mask]     = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]
            self._needs_reanchor[:]     = False

        # Clamp the policy's raw action to [-1, 1] before any further compute.
        # With scale=0.02 this caps the per-step Cartesian delta at 2 cm/axis,
        # regardless of how aggressive the (unbounded) policy logit happens to be.
        actions = torch.clamp(actions, -1.0, 1.0)

        # Store raw (already clamped) 3-D action and apply per-axis scale.
        self._raw_actions[:] = actions
        scaled = actions * self._scale  # (N, 3)

        # Cumulative position delta over the episode.
        self.del_action += scaled

        # Absolute position target (RPY is locked at init).
        abs_pos = self.init_ee_pos + self.del_action  # (N, 3)

        # EMA on the position channel only.
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos

        # Optional per-axis position clamp.
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            ema_pos = torch.clamp(ema_pos, self.pos_lower_limit, self.pos_upper_limit)

        # Store the 7-D absolute pose target: (pos, init_ee_quat).
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos

        # Hand off to the IK controller in absolute mode.
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
