"""EMA cumulative-relative joint position action.

Vendored verbatim from `bidex/env/action_managers/actions.py:EMACumulativeRelativeJointPositionAction`.
We do NOT import from `bidex` at runtime; the class lives here so the task has
no cross-repo dependency.

Per-step processing (raw action `a_t`, scale `s`, offset `o`, alpha `α`,
init joint pose captured at reset `q_init`):

    1. processed_t = scale * a_t + offset            # JointPositionAction
    2. processed_t = processed_t + del_{t-1}         # accumulate delta
    3. del_t       = processed_t                     # remember cumulative delta
    4. processed_t = processed_t + q_init            # anchor on init pose
    5. ema_t       = α * processed_t + (1 - α) * prev_applied_{t-1}
    6. processed_t = clamp(ema_t, joint_lower_limit, joint_upper_limit)
    7. prev_applied_t = processed_t

At t=0 (post-reset, prev_applied == q_init, del_{-1} == 0, offset == 0):
    target_0 = clamp(q_init + α * scale * a_0, lower, upper)
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from . import actions_cfg


class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    cfg: "actions_cfg.EMACumulativeRelativeJointPositionActionCfg"
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(
        self,
        cfg: "actions_cfg.EMACumulativeRelativeJointPositionActionCfg",
        env: "ManagerBasedRLEnv",
    ) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        # initialize the cumulative del action
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = (
            torch.tensor(cfg.joint_lower_limit, device=self.device)
            if cfg.joint_lower_limit is not None
            else None
        )
        self.joint_upper_limit = (
            torch.tensor(cfg.joint_upper_limit, device=self.device)
            if cfg.joint_upper_limit is not None
            else None
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # reset history to current joint positions
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        # reset the del action
        if isinstance(env_ids, slice):
            self.del_action[:] = 0.0
        else:
            self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations (scale * raw + offset)
        super().process_actions(actions)
        # compute the del action
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        # add the initial position
        self._processed_actions += self.init_joint_pos.clone()
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(
                ema_actions,
                self.joint_lower_limit,
                self.joint_upper_limit,
            )
        else:
            self._processed_actions[:] = ema_actions
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
