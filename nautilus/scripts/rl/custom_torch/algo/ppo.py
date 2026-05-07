"""PPO — proximal policy optimization. On-policy, single-process. Adapted from PQL."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from algo.ac_base import ActorCriticBase
from utils.torch_util import RunningMeanStd
from utils.common import handle_timeout, aggregate_traj_info


@dataclass
class AgentPPO(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.timeout_info = None
        if bool(self.cfg.algo.get("value_norm", False)):
            self.value_rms = RunningMeanStd(shape=(1,), device=self.device)
        else:
            self.value_rms = None

    def get_actions(self, obs):
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        actions, _, log_prob, _ = self.actor.get_actions_logprob_entropy(obs)
        value = self.critic(obs)
        if self.value_rms is not None:
            self.value_rms.update(value)
            value = self.value_rms.unnormalize(value)
        return actions, log_prob, value.flatten()

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False):
        n = int(self.cfg.num_envs)
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_obs = torch.zeros((timesteps, n) + (*obs_dim,), device=self.device)
        traj_actions = torch.zeros((timesteps, n, self.action_dim), device=self.device)
        traj_logprobs = torch.zeros((timesteps, n), device=self.device)
        traj_rewards = torch.zeros((timesteps, n), device=self.device)
        traj_dones = torch.zeros((timesteps, n), device=self.device)
        traj_values = torch.zeros((timesteps, n), device=self.device)
        infos: list = []

        ob = self.obs
        dones = self.dones
        for step in range(timesteps):
            if self.obs_rms is not None:
                self.obs_rms.update(ob)
            traj_obs[step] = deepcopy(ob)
            traj_dones[step] = dones
            action, logprob, val = self.get_actions(ob)
            next_ob, reward, done, info = env.step(action)
            self.update_tracker(reward, done, info)
            traj_actions[step] = action
            traj_logprobs[step] = logprob
            traj_rewards[step] = (reward.float() if isinstance(reward, torch.Tensor)
                                  else torch.as_tensor(reward, device=self.device, dtype=torch.float32))
            traj_values[step] = val
            infos.append(info if isinstance(info, dict) else {})
            ob = next_ob
            dones = done.float() if isinstance(done, torch.Tensor) else torch.as_tensor(done, device=self.device, dtype=torch.float32)

        if bool(self.cfg.algo.get("handle_timeout", False)):
            for key in ("TimeLimit.truncated", "time_outs"):
                if infos and key in infos[0]:
                    self.timeout_info = aggregate_traj_info(infos, key)
                    break

        self.obs = ob
        self.dones = dones
        return self._compute_adv(traj_obs, traj_actions, traj_logprobs, traj_rewards,
                                 traj_dones, traj_values, ob, dones), timesteps * n

    def _compute_adv(self, obs, actions, logprobs, rewards, dones, values, next_obs, next_done):
        timesteps = obs.shape[0]
        gamma = float(self.cfg.algo.get("gamma", 0.99))
        gae_lambda = float(self.cfg.algo.get("gae_lambda", 0.95))
        with torch.no_grad():
            n_obs = self.obs_rms.normalize(next_obs) if self.obs_rms is not None else next_obs
            next_value = self.critic(n_obs).flatten()
            if self.value_rms is not None:
                self.value_rms.update(next_value)
                next_value = self.value_rms.unnormalize(next_value)
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(timesteps)):
                if t == timesteps - 1:
                    next_non_term = 1.0 - next_done.float()
                    next_v = next_value
                else:
                    next_non_term = 1.0 - dones[t + 1]
                    next_v = values[t + 1]
                delta = rewards[t] + gamma * next_v * next_non_term - values[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            returns = advantages + values
            if self.value_rms is not None:
                self.value_rms.update(returns.reshape(-1))
                returns = self.value_rms.normalize(returns)
                self.value_rms.update(values.reshape(-1))
                values = self.value_rms.normalize(values)
        return obs, actions, logprobs, advantages, returns, values

    def update_net(self, data):
        obs, actions, old_logprobs, advantages, returns, old_values = data
        bs = int(self.cfg.algo.get("batch_size", 64))
        n_epochs = int(self.cfg.algo.get("n_epochs", 10))
        clip = float(self.cfg.algo.get("clip_range", 0.2))
        ent_coef = float(self.cfg.algo.get("ent_coef", 0.0))
        vf_coef = float(self.cfg.algo.get("vf_coef", 0.5))

        obs = obs.reshape(-1, *obs.shape[2:])
        actions = actions.reshape(-1, actions.shape[-1])
        old_logprobs = old_logprobs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        old_values = old_values.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        idx = torch.randperm(n, device=self.device)
        policy_losses, value_losses, entropies, kls, clip_fracs = [], [], [], [], []
        for _ in range(n_epochs):
            for start in range(0, n, bs):
                b = idx[start:start + bs]
                _, _, logprob, entropy = self.actor.logprob_entropy(
                    self.obs_rms.normalize(obs[b]) if self.obs_rms is not None else obs[b],
                    actions[b],
                )
                log_ratio = logprob - old_logprobs[b]
                ratio = torch.exp(log_ratio)
                pg_loss1 = -advantages[b] * ratio
                pg_loss2 = -advantages[b] * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v = self.critic(
                    self.obs_rms.normalize(obs[b]) if self.obs_rms is not None else obs[b]
                ).flatten()
                if bool(self.cfg.algo.get("value_clip", False)):
                    v_loss_unclipped = (v - returns[b]) ** 2
                    v_clipped = old_values[b] + torch.clamp(
                        v - old_values[b], -clip, clip,
                    )
                    v_loss_clipped = (v_clipped - returns[b]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((v - returns[b]) ** 2).mean()

                ent_loss = -entropy.mean()
                loss = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_grad_norm = self.cfg.algo.get("max_grad_norm", None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(max_grad_norm))
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(max_grad_norm))
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                with torch.no_grad():
                    # Schulman 2020 unbiased low-variance KL estimator: ((ratio - 1) - log_ratio).
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip).float().mean().item()

                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(-ent_loss.item())
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        log_info = {
            "train/loss/policy":   float(np.mean(policy_losses)),
            "train/loss/value":    float(np.mean(value_losses)),
            "train/entropy":       float(np.mean(entropies)),
            "train/approx_kl":     float(np.mean(kls)),
            "train/clip_fraction": float(np.mean(clip_fracs)),
            "train/episode_length": float(self.step_tracker.mean()),
        }
        log_info.update(self.reward_log_info())
        return log_info
