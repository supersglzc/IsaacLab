"""
MLP actors / critics for SAC, PPO, TD3.

Self-contained — no escnn, no equivariant variants. Includes:
    MLPNet                       — generic MLP backbone
    DiagGaussianMLPPolicy        — PPO policy (diagonal Gaussian, NOT squashed)
    TanhDiagGaussianMLPPolicy    — SAC policy (tanh-squashed Gaussian)
    TanhMLPPolicy                — TD3 policy (tanh-bounded deterministic)
    DoubleQ                      — twin Q networks (SAC / TD3)
    MLPCritic                    — single value head (PPO)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal

from utils.torch_util import SquashedNormal


def create_simple_mlp(in_dim: int, out_dim: int, hidden_layers, act=nn.ELU,
                      use_batchnorm: bool = False) -> nn.Sequential:
    layer_nums = [in_dim, *hidden_layers, out_dim]
    layers = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        layers.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_f))
            layers.append(act())
    return nn.Sequential(*layers)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None, use_batchnorm: bool = False):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [256, 256]
        self.net = create_simple_mlp(in_dim, out_dim, hidden_layers, use_batchnorm=use_batchnorm)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DiagGaussianMLPPolicy(MLPNet):
    """PPO actor — state-independent log-std parameter, Gaussian (no tanh)."""

    def __init__(self, state_dim, act_dim: int, hidden_layers=None, init_log_std: float = 0.0):
        super().__init__(in_dim=state_dim, out_dim=act_dim, hidden_layers=hidden_layers)
        self.logstd = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        return self.get_actions(x, sample=sample)[0]

    def get_actions(self, x: Tensor, sample: bool = True):
        mean = self.net(x)
        std = torch.exp(self.logstd.expand_as(mean))
        dist = Independent(Normal(mean, std), 1)
        actions = dist.rsample() if sample else mean
        return actions, dist

    def get_actions_logprob_entropy(self, state: Tensor, sample: bool = True):
        actions, dist = self.get_actions(state, sample=sample)
        return actions, dist, dist.log_prob(actions), dist.entropy()

    def logprob_entropy(self, state: Tensor, actions: Tensor):
        _, dist = self.get_actions(state)
        return actions, dist, dist.log_prob(actions), dist.entropy()


class TanhDiagGaussianMLPPolicy(MLPNet):
    """SAC actor — tanh-squashed Gaussian with state-dependent log-std."""

    LOG_STD_MIN = -5
    LOG_STD_MAX = 5

    def __init__(self, state_dim, act_dim: int, hidden_layers=None):
        super().__init__(in_dim=state_dim, out_dim=act_dim * 2, hidden_layers=hidden_layers)
        self.log_sqrt_2pi = float(np.log(np.sqrt(2 * np.pi)))

    def forward(self, state: Tensor, sample: bool = False) -> Tensor:
        return self.get_actions(state, sample=sample)

    def _dist(self, state: Tensor):
        mu, log_std = self.net(state).chunk(2, dim=-1)
        std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return SquashedNormal(mu, std)

    def get_actions(self, state: Tensor, sample: bool = True) -> Tensor:
        dist = self._dist(state)
        return dist.rsample() if sample else dist.mean

    def get_actions_logprob(self, state: Tensor):
        dist = self._dist(state)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, dist, log_prob


class TanhMLPPolicy(MLPNet):
    """TD3 / DDPG actor — deterministic, tanh-bounded."""

    def forward(self, state: Tensor) -> Tensor:
        return super().forward(state).tanh()

    def get_actions(self, state: Tensor, sample: bool = True) -> Tensor:
        # TD3 explore noise is added by the agent, not the policy.
        return self.forward(state)


class DoubleQ(nn.Module):
    """Twin Q networks (SAC / TD3)."""

    def __init__(self, state_dim, act_dim: int):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))

    def get_q1_q2(self, state: Tensor, action: Tensor):
        x = torch.cat((state, action), dim=1)
        return self.net_q1(x), self.net_q2(x)

    def get_q1(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat((state, action), dim=1)
        return self.net_q1(x)


class MLPCritic(nn.Module):
    """PPO critic — single scalar value head."""

    def __init__(self, state_dim, action_dim: int = 0):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.critic = MLPNet(in_dim=state_dim, out_dim=1)

    def forward(self, state: Tensor) -> Tensor:
        return self.critic(state)
