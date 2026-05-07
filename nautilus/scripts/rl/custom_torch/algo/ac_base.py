"""ActorCriticBase — shared dataclass for SAC / TD3 / PPO.

Adapted from PQL/DDiffPG ac_base.py with the following stripped:
  - bidex.utils.symmetry import (no equivariant policies in this generated tree)
  - cfg.algo.multi_agent path
  - info_track_keys plumbing
  - info['detailed_reward'] and info['success'] are now OPTIONAL — most envs
    don't ship them. update_tracker() guards each access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.utils import clip_grad_norm_

from utils.common import Tracker, load_class_from_path
from utils.torch_util import RunningMeanStd
from models import model_name_to_path


@dataclass
class ActorCriticBase:
    env: Any
    cfg: DictConfig
    obs_dim: int = None
    action_dim: int = None

    def __post_init__(self):
        self.device = torch.device(self.cfg.device)
        self.obs = None
        if self.obs_dim is None:
            shape = self.env.observation_space.shape
            self.obs_dim = shape if len(shape) > 1 else int(shape[0])
        if self.action_dim is None:
            self.action_dim = int(self.env.action_space.shape[-1])

        act_cls = load_class_from_path(self.cfg.algo.act_class,
                                       model_name_to_path[self.cfg.algo.act_class])
        cri_cls = load_class_from_path(self.cfg.algo.cri_class,
                                       model_name_to_path[self.cfg.algo.cri_class])
        self.actor = act_cls(self.obs_dim, self.action_dim).to(self.device)
        self.critic = cri_cls(self.obs_dim, self.action_dim).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=float(self.cfg.algo.actor_lr))
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=float(self.cfg.algo.critic_lr))

        n = int(self.cfg.num_envs)
        self.current_returns = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(n, dtype=torch.float32, device=self.device)
        tlen = int(self.cfg.algo.get("tracker_len", 100))
        self.return_tracker = Tracker(tlen)
        self.success_tracker = Tracker(tlen)
        self.step_tracker = Tracker(tlen)

        if bool(self.cfg.algo.get("obs_norm", False)):
            self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.device)
        else:
            self.obs_rms = None

    def reset_agent(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            self.obs, _extras = out[0], out[1] if len(out) > 1 else {}
        else:
            self.obs = out
        n = int(self.cfg.num_envs)
        self.dones = torch.zeros(n, device=self.device)
        self.current_returns = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(n, dtype=torch.float32, device=self.device)
        return self.obs

    def update_tracker(self, reward, done, info):
        # reward / done expected as torch tensors (env_wrapper bridges numpy → torch).
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, device=self.device)
        self.current_returns += reward.float()
        self.current_lengths += 1
        done_idx = torch.where(done.bool())[0]
        if len(done_idx) > 0:
            self.return_tracker.update(self.current_returns[done_idx])
            self.step_tracker.update(self.current_lengths[done_idx])
            if isinstance(info, dict) and "success" in info:
                succ = info["success"]
                if isinstance(succ, torch.Tensor):
                    self.success_tracker.update(succ[done_idx])
            self.current_returns[done_idx] = 0
            self.current_lengths[done_idx] = 0

        # Per-reward-term tracking — only when the env emits info["detailed_reward"].
        # Lazy-init: discover term keys on the first non-empty observation.
        # Filter out gymnasium-1.x VectorEnv mask keys ("_<term>") that get auto-added
        # alongside the real per-term arrays during info aggregation.
        if isinstance(info, dict) and "detailed_reward" in info:
            detailed = info["detailed_reward"]
            if isinstance(detailed, dict) and detailed:
                term_items = [(k, v) for k, v in detailed.items() if not k.startswith("_")]
                if term_items:
                    if not hasattr(self, "_term_returns") or self._term_returns is None:
                        n = int(self.cfg.num_envs)
                        tlen = int(self.cfg.algo.get("tracker_len", 100))
                        self._term_returns = {
                            k: torch.zeros(n, dtype=torch.float32, device=self.device)
                            for k, _ in term_items
                        }
                        self._term_trackers = {k: Tracker(tlen) for k, _ in term_items}
                    for k, v in term_items:
                        if k not in self._term_returns:
                            n = int(self.cfg.num_envs)
                            tlen = int(self.cfg.algo.get("tracker_len", 100))
                            self._term_returns[k] = torch.zeros(n, dtype=torch.float32, device=self.device)
                            self._term_trackers[k] = Tracker(tlen)
                        if not isinstance(v, torch.Tensor):
                            v = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                        self._term_returns[k] += v.float()
                        if len(done_idx) > 0:
                            self._term_trackers[k].update(self._term_returns[k][done_idx])
                            self._term_returns[k][done_idx] = 0

    def reward_log_info(self) -> dict:
        """Canonical reward metrics: total + per-term cumulative episodic-return mean.

        Always emits `reward/total/episodic_return_mean`. Per-term keys
        (`reward/<term>/episodic_return_mean`) are populated only when the env
        provides `info["detailed_reward"]` as a dict — otherwise just `reward/total/...`.
        """
        out = {"reward/total/episodic_return_mean": float(self.return_tracker.mean())}
        if hasattr(self, "_term_trackers") and self._term_trackers:
            for k, t in self._term_trackers.items():
                out[f"reward/{k}/episodic_return_mean"] = float(t.mean())
        return out

    def optimizer_update(self, optimizer, loss, retain_graph: bool = False):
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        max_grad_norm = self.cfg.algo.get("max_grad_norm", None)
        grad_norm = None
        if max_grad_norm is not None:
            grad_norm = clip_grad_norm_(optimizer.param_groups[0]["params"],
                                        max_norm=float(max_grad_norm))
        optimizer.step()
        return grad_norm

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
