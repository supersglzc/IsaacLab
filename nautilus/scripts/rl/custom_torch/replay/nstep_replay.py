"""N-step return buffer for vec envs (per-env nstep window, used by SAC/TD3).

Adapted from the reference custom-torch RL pattern. Pure-torch; works with both
gpu_sim modes once env_wrapper has bridged numpy → torch.
"""
from __future__ import annotations

import torch

from replay.simple_replay import create_buffer


class NStepReplay:
    def __init__(self, obs_dim, action_dim: int, num_envs: int = 1,
                 nstep: int = 3, device: str = "cuda", gamma: float = 0.99):
        self.num_envs = num_envs
        self.nstep = nstep
        self.gamma = gamma
        ret = create_buffer(capacity=(self.num_envs, self.nstep),
                            obs_dim=obs_dim, action_dim=action_dim, device=device)
        (self.nstep_buf_obs, self.nstep_buf_action, self.nstep_buf_next_obs,
         self.nstep_buf_reward, self.nstep_buf_done) = ret
        self.nstep_count = 0
        self.gamma_array = torch.tensor(
            [self.gamma ** i for i in range(self.nstep)], device=device
        ).view(-1, 1)

    @torch.no_grad()
    def add_to_buffer(self, obs, actions, rewards, next_obs, dones):
        if self.nstep <= 1:
            return obs, actions, rewards, next_obs, dones

        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []
        for i in range(obs.shape[1]):
            self.nstep_buf_obs = self._fifo(self.nstep_buf_obs, obs[:, i])
            self.nstep_buf_next_obs = self._fifo(self.nstep_buf_next_obs, next_obs[:, i])
            self.nstep_buf_done = self._fifo(self.nstep_buf_done, dones[:, i])
            self.nstep_buf_action = self._fifo(self.nstep_buf_action, actions[:, i])
            self.nstep_buf_reward = self._fifo(self.nstep_buf_reward, rewards[:, i])
            self.nstep_count += 1
            if self.nstep_count < self.nstep:
                continue
            obs_list.append(self.nstep_buf_obs[:, 0])
            action_list.append(self.nstep_buf_action[:, 0])
            r, no, d = _compute_nstep_return(self.nstep_buf_next_obs,
                                             self.nstep_buf_done,
                                             self.nstep_buf_reward,
                                             self.gamma_array)
            reward_list.append(r)
            next_obs_list.append(no)
            done_list.append(d)
        if not obs_list:
            return None
        return (torch.cat(obs_list), torch.cat(action_list), torch.cat(reward_list),
                torch.cat(next_obs_list), torch.cat(done_list))

    @staticmethod
    def _fifo(queue, new_tensor):
        return torch.cat((queue[:, 1:], new_tensor.unsqueeze(1)), dim=1)


def _compute_nstep_return(buf_next_obs, buf_done, buf_reward, gamma_array):
    buf_done_flat = buf_done.squeeze(-1)
    done_idx = torch.where(buf_done_flat)
    done_envs = torch.unique_consecutive(done_idx[0])
    done_steps = buf_done_flat.argmax(dim=1)

    done = buf_done[:, -1].clone()
    done[done_envs] = True
    next_obs = buf_next_obs[:, -1].clone()
    next_obs[done_envs] = buf_next_obs[done_envs, done_steps[done_envs]].clone()

    mask = torch.ones(buf_done_flat.shape, device=buf_done_flat.device, dtype=torch.bool)
    mask[done_envs] = (torch.arange(mask.shape[1], device=buf_done_flat.device)
                       <= done_steps[done_envs][:, None])
    discounted = buf_reward * gamma_array
    discounted = (discounted * mask.unsqueeze(-1)).sum(1)
    return discounted, next_obs, done
