"""EMA cumulative-delta task-space (EE position) action — fixed RPY variant.

Copied verbatim from `manipulation/stack_cube/mdp/actions.py`. Position-only
3-D EE-delta action; the policy outputs ONLY `(dx, dy, dz)` and the EE
quaternion is FIXED at the post-reset value.

Behavior:
    1. Accumulates the 3-D position delta over the episode:
           delta_t = delta_{t-1} + s . a_t
    2. Anchors on the EE pose captured lazily at the first
       `process_actions` after `env.reset()`:
           abs_pos_t  = init_ee_pos + delta_t
           abs_quat_t = init_ee_quat                       (RPY locked)
    3. EMA-smooths against the previously applied target on the position
       channel only (orientation stays exactly at `init_ee_quat`):
           target_t.pos  = alpha . abs_pos_t + (1 - alpha) . prev_applied_pos
           target_t.quat = init_ee_quat
    4. Forwards the absolute 7-D pose target to the IK controller (forced
       into `use_relative_mode=False`) and reuses the parent's
       `apply_actions()` for jacobian + IK numerics.
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
        # Force the IK controller into absolute pose mode -- we manage relative deltas ourselves.
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

        # Init pose buffers -- populated lazily on the next process_actions for
        # any env in `_needs_reanchor`.
        self.init_ee_pos        = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat       = torch.zeros((env.num_envs, 4), device=env.device)
        self.init_ee_quat[:, 0] = 1.0                                                  # identity quat (wxyz)
        self._prev_applied_pos  = torch.zeros((env.num_envs, 3), device=env.device)
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
        # Iter 11: optional 'no-go' xy square around robot root.
        self.forbidden_xy_half: float | None = getattr(cfg, "forbidden_xy_half", None)

    # Policy sees a 3-D action; IK command is 7-D (handled internally).
    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_reanchor[:] = True
            self.del_action[:]      = 0.0
        else:
            self._needs_reanchor[env_ids] = True
            self.del_action[env_ids]      = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        # Re-anchor init pose for any env that just reset.
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask]      = ee_pos_curr[mask]
            self.init_ee_quat[mask]     = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]
            self._needs_reanchor[:]     = False

        # Clamp the policy's raw action to [-1, 1] before any further compute.
        actions = torch.clamp(actions, -1.0, 1.0)

        # Store raw (already clamped) 3-D action and apply per-axis scale.
        self._raw_actions[:] = actions
        scaled = actions * self._scale  # (N, 3)

        # Cumulative position delta over the episode.
        self.del_action += scaled

        # Iter 16: clamp `del_action` directly so `abs_pos = init + del_action`
        # is always inside the workspace box. Without this, del_action grows
        # unbounded when the policy commands past the clamp boundary — the
        # absolute target gets clipped each step (good for IK), but the
        # internal del_action remembers the saturated direction, so when the
        # policy reverses it has to burn through the accumulated delta before
        # any motion resumes. Clamping del_action at the source removes that
        # latency: a reversal action takes effect on the next step.
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            del_lower = self.pos_lower_limit - self.init_ee_pos
            del_upper = self.pos_upper_limit - self.init_ee_pos
            self.del_action = torch.clamp(self.del_action, del_lower, del_upper)

        # Absolute position target (RPY is locked at init).
        abs_pos = self.init_ee_pos + self.del_action  # (N, 3) — now guaranteed inside the box

        # EMA on the position channel only.
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos

        # Defensive: the box clamp on ema_pos is now redundant when del_action
        # is clamped (both abs_pos and prev_applied are inside the box, so any
        # convex combination is also inside). Kept for safety in case
        # init_ee_pos itself sits outside the workspace clamp at reset (the
        # policy's first step might then need the ema clamp to project the
        # init pose into the box).
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            ema_pos = torch.clamp(ema_pos, self.pos_lower_limit, self.pos_upper_limit)

        # Iter 11: optional 'no-go' xy square around the robot root.
        # If the EE target lands in |x| <= h ∧ |y| <= h, push it to the
        # nearest edge of the square. Z is unconstrained.
        if self.forbidden_xy_half is not None:
            h = float(self.forbidden_xy_half)
            in_x = (ema_pos[:, 0] > -h) & (ema_pos[:, 0] < h)
            in_y = (ema_pos[:, 1] > -h) & (ema_pos[:, 1] < h)
            in_forb = in_x & in_y
            if in_forb.any():
                dx_pos = h - ema_pos[:, 0]       # distance to edge x = +h
                dx_neg = ema_pos[:, 0] + h       # distance to edge x = -h
                dy_pos = h - ema_pos[:, 1]       # distance to edge y = +h
                dy_neg = ema_pos[:, 1] + h       # distance to edge y = -h
                dists = torch.stack([dx_pos, dx_neg, dy_pos, dy_neg], dim=-1)
                choice = torch.argmin(dists, dim=-1)
                new_x = torch.where((choice == 0) & in_forb,  torch.full_like(ema_pos[:, 0],  h), ema_pos[:, 0])
                new_x = torch.where((choice == 1) & in_forb,  torch.full_like(ema_pos[:, 0], -h), new_x)
                new_y = torch.where((choice == 2) & in_forb,  torch.full_like(ema_pos[:, 1],  h), ema_pos[:, 1])
                new_y = torch.where((choice == 3) & in_forb,  torch.full_like(ema_pos[:, 1], -h), new_y)
                ema_pos = torch.stack([new_x, new_y, ema_pos[:, 2]], dim=-1)

        # Store the 7-D absolute pose target: (pos, init_ee_quat).
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos
        # print(scaled, self._processed_actions[:, :3])
        # Hand off to the IK controller in absolute mode.
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)


