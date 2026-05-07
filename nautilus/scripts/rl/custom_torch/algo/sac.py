"""SAC — soft actor-critic. Adapted from PQL/DDiffPG; bidex / equivariant deps stripped."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algo.ac_base import ActorCriticBase
from replay.nstep_replay import NStepReplay
from utils.torch_util import soft_update
from utils.common import handle_timeout


@dataclass
class AgentSAC(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.critic_target = deepcopy(self.critic)
        self.actor_target = (deepcopy(self.actor)
                             if not bool(self.cfg.algo.get("no_tgt_actor", True))
                             else self.actor)

        if self.cfg.algo.get("alpha") is None:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_optim = torch.optim.AdamW(
                [self.log_alpha], lr=float(self.cfg.algo.get("alpha_lr", 3.0e-4)))
        self.target_entropy = -self.action_dim

        self.n_step_buffer = NStepReplay(
            self.obs_dim, self.action_dim, int(self.cfg.num_envs),
            int(self.cfg.algo.get("nstep", 1)), device=self.device,
            gamma=float(self.cfg.algo.get("gamma", 0.99)),
        )

    def get_alpha(self, detach: bool = True, scalar: bool = False):
        if self.cfg.algo.get("alpha") is None:
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
            return alpha
        return float(self.cfg.algo.alpha)

    def get_actions(self, obs, sample: bool = True):
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        return self.actor.get_actions(obs, sample=sample)

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
        critic_losses, actor_losses, entropies, q_values = [], [], [], []
        update_times = int(self.cfg.algo.get("update_times", 1))
        bs = int(self.cfg.algo.get("batch_size", 256))
        tau = float(self.cfg.algo.get("tau", 0.005))
        for _ in range(update_times):
            obs, action, reward, next_obs, done = memory.sample_batch(bs)
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            critic_loss, _, q_mean = self.update_critic(obs, action, reward, next_obs, done)
            critic_losses.append(critic_loss)
            q_values.append(q_mean)
            actor_loss, _, ent_est = self.update_actor(obs)
            actor_losses.append(actor_loss)
            entropies.append(ent_est)
            soft_update(self.critic_target, self.critic, tau)
            if not bool(self.cfg.algo.get("no_tgt_actor", True)):
                soft_update(self.actor_target, self.actor, tau)
        log_info = {
            "train/critic_loss": float(np.mean(critic_losses)),
            "train/actor_loss":  float(np.mean(actor_losses)),
            "train/entropy":     float(np.mean(entropies)),
            "train/q_value":     float(np.mean(q_values)),
            "train/episode_length": float(self.step_tracker.mean()),
            "train/alpha":       self.get_alpha(scalar=True),
        }
        log_info.update(self.reward_log_info())
        return log_info

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions, _, log_prob = self.actor.get_actions_logprob(next_obs)
            target_Q = self.critic_target.get_q_min(next_obs, next_actions) - self.get_alpha() * log_prob
            gamma = float(self.cfg.algo.get("gamma", 0.99))
            nstep = int(self.cfg.algo.get("nstep", 1))
            target_Q = reward + (1 - done) * (gamma ** nstep) * target_Q
        q1, q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        gn = self.optimizer_update(self.critic_optimizer, critic_loss)
        # Diagnostics: mean of min(Q1, Q2) on the sampled batch.
        with torch.no_grad():
            q_mean = float(torch.min(q1, q2).mean().item())
        return critic_loss.item(), gn, q_mean

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        actions, _, log_prob = self.actor.get_actions_logprob(obs)
        Q = self.critic.get_q_min(obs, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()
        gn = self.optimizer_update(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)
        # Entropy estimate for tanh-squashed Gaussian: H ≈ -E[log_prob].
        with torch.no_grad():
            ent_est = float(-log_prob.mean().item())

        if self.cfg.algo.get("alpha") is None:
            alpha_loss = (self.get_alpha(detach=False) * (-log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss.item(), gn, ent_est
