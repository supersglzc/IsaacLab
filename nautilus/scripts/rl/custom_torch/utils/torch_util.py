"""
Torch utilities — RunningMeanStd, soft_update, SquashedNormal (tanh-Gaussian).

Adapted from the reference custom-torch RL pattern (PQL / DDiffPG style),
self-contained — no escnn / bidex / equivariant deps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class RunningMeanStd:
    """Welford-style running mean / variance, on a torch device."""

    def __init__(self, epsilon: float = 1e-4, shape=(), device: str = "cuda"):
        self.device = device
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = m_2 / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def unnormalize(self, x):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean

    def get_states(self, device=None):
        if device is not None:
            return self.mean.to(device), self.var.to(device), self.epsilon
        return self.mean, self.var, self.epsilon

    def load_state_dict(self, info):
        self.mean = info[0]
        self.var = info[1]
        self.count = info[2]
