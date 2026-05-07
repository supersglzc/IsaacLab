"""Cyclic replay buffer for off-policy algorithms (SAC / TD3 / DDPG).

Adapted from the reference custom-torch RL pattern. Uses torch tensors throughout.
Compatible with both gpu_sim=true (env.step returns torch tensors) and gpu_sim=false
(env_wrapper bridges numpy → torch before reaching this buffer).
"""
from __future__ import annotations

import torch


def create_buffer(capacity, obs_dim, action_dim, device: str = "cuda"):
    if isinstance(capacity, int):
        capacity = (capacity,)
    obs_size = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.empty(obs_size, dtype=torch.float32, device=device)
    buf_action = torch.empty((*capacity, int(action_dim)), dtype=torch.float32, device=device)
    buf_reward = torch.empty((*capacity, 1), dtype=torch.float32, device=device)
    buf_next_obs = torch.empty(obs_size, dtype=torch.float32, device=device)
    buf_done = torch.empty((*capacity, 1), dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim, action_dim: int, device: str = "cuda"):
        self.obs_dim = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
        self.action_dim = action_dim
        self.device = device
        self.capacity = int(capacity)
        self.next_p = 0
        self.if_full = False
        self.cur_capacity = 0
        ret = create_buffer(self.capacity, obs_dim, action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = obs.reshape(-1, *self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, *self.obs_dim)
        dones = dones.reshape(-1, 1).bool()
        n = rewards.shape[0]
        p = self.next_p + n

        if p > self.capacity:
            self.if_full = True
            head = self.capacity - self.next_p
            self.buf_obs[self.next_p:self.capacity] = obs[:head]
            self.buf_action[self.next_p:self.capacity] = actions[:head]
            self.buf_reward[self.next_p:self.capacity] = rewards[:head]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:head]
            self.buf_done[self.next_p:self.capacity] = dones[:head]
            tail = p - self.capacity
            self.buf_obs[:tail] = obs[head:head + tail]
            self.buf_action[:tail] = actions[head:head + tail]
            self.buf_reward[:tail] = rewards[head:head + tail]
            self.buf_next_obs[:tail] = next_obs[head:head + tail]
            self.buf_done[:tail] = dones[head:head + tail]
            p = tail
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    @torch.no_grad()
    def sample_batch(self, batch_size: int):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=self.device)
        return (
            self.buf_obs[indices],
            self.buf_action[indices],
            self.buf_reward[indices],
            self.buf_next_obs[indices],
            self.buf_done[indices].float(),
        )
