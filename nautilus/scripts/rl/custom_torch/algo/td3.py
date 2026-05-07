"""TD3 — twin-delayed DDPG. Adapted from PQL/DDiffPG SAC + TD3 patterns."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from algo.ac_base import ActorCriticBase
from replay.nstep_replay import NStepReplay
from utils.torch_util import soft_update
from utils.common import handle_timeout


@dataclass
class AgentTD3(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self._update_count = 0
        self.n_step_buffer = NStepReplay(
            self.obs_dim, self.action_dim, int(self.cfg.num_envs),
            int(self.cfg.algo.get("nstep", 1)), device=self.device,
            gamma=float(self.cfg.algo.get("gamma", 0.99)),
        )

    def get_actions(self, obs, sample: bool = True):
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        action = self.actor(obs)
        if sample:
            noise = torch.randn_like(action) * float(self.cfg.algo.get("explore_noise", 0.1))
            action = (action + noise).clamp(-1.0, 1.0)
        return action

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool):
        n = int(self.cfg.num_envs)
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_obs = torch.empty((n, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((n, timesteps, self.action_dim), device=self.device)
        traj_rewards = torch.empty((n, timesteps), device=self.device)
        traj_next_obs = torch.empty((n, timesteps) + (*obs_dim,), device=self.device)
        traj_dones = torch.empty((n, timesteps), device=self.device)

        obs = self.obs
        for i in range(timesteps):
            if self.obs_rms is not None:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((n, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            self.update_tracker(reward, done, info)
            if bool(self.cfg.algo.get("handle_timeout", False)):
                done = handle_timeout(done, info)

            traj_obs[:, i] = obs
            traj_actions[:, i] = action
            traj_dones[:, i] = done.float() if isinstance(done, torch.Tensor) else torch.as_tensor(done, device=self.device, dtype=torch.float32)
            traj_rewards[:, i] = reward.float() if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            traj_next_obs[:, i] = next_obs
            obs = next_obs
        self.obs = obs

        scale = float(self.cfg.algo.get("reward_scale", 1.0))
        traj_rewards = scale * traj_rewards.reshape(n, timesteps, 1)
        traj_dones = traj_dones.reshape(n, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards,
                                                traj_next_obs, traj_dones)
        return data, timesteps * n

    def update_net(self, memory):
        critic_losses, actor_losses, q_values = [], [], []
        update_times = int(self.cfg.algo.get("update_times", 1))
        bs = int(self.cfg.algo.get("batch_size", 256))
        tau = float(self.cfg.algo.get("tau", 0.005))
        policy_delay = int(self.cfg.algo.get("policy_delay", 2))

        for _ in range(update_times):
            obs, action, reward, next_obs, done = memory.sample_batch(bs)
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            critic_loss, q_mean = self._update_critic(obs, action, reward, next_obs, done)
            critic_losses.append(critic_loss)
            q_values.append(q_mean)

            self._update_count += 1
            if self._update_count % policy_delay == 0:
                actor_loss = self._update_actor(obs)
                actor_losses.append(actor_loss)
                soft_update(self.actor_target, self.actor, tau)
                soft_update(self.critic_target, self.critic, tau)

        log_info = {
            "train/critic_loss": float(np.mean(critic_losses)),
            "train/actor_loss":  float(np.mean(actor_losses)) if actor_losses else 0.0,
            "train/q_value":     float(np.mean(q_values)),
            "train/episode_length": float(self.step_tracker.mean()),
        }
        log_info.update(self.reward_log_info())
        return log_info

    def _update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            policy_noise = float(self.cfg.algo.get("policy_noise", 0.2))
            noise_clip = float(self.cfg.algo.get("noise_clip", 0.5))
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            target_Q = self.critic_target.get_q_min(next_obs, next_action)
            gamma = float(self.cfg.algo.get("gamma", 0.99))
            nstep = int(self.cfg.algo.get("nstep", 1))
            target_Q = reward + (1 - done) * (gamma ** nstep) * target_Q
        q1, q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.optimizer_update(self.critic_optimizer, critic_loss)
        with torch.no_grad():
            q_mean = float(torch.min(q1, q2).mean().item())
        return critic_loss.item(), q_mean

    def _update_actor(self, obs) -> float:
        self.critic.requires_grad_(False)
        action = self.actor(obs)
        actor_loss = -self.critic.get_q1(obs, action).mean()
        self.optimizer_update(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)
        return actor_loss.item()
