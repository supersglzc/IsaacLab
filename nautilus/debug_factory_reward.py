"""Probe Factory-GearMesh reward magnitudes with random actions.

Goal: find out whether the reported `return=0.000` from train.py reflects
genuinely-zero rewards or just %.3f display rounding. Also confirm the
`logs_rew_*` keys appear in info["extras"]/info as the env source promises.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=4)
p.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(p)
args_cli = p.parse_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _isaaclab_env  # noqa: E402

env = _isaaclab_env.make_isaaclab_env(
    "Isaac-Factory-GearMesh-Direct-v0",
    num_envs=args_cli.num_envs, device=args_cli.device or "cuda")
try:
    env.reset(seed=0)
except TypeError:
    env.reset()

action_shape = env.action_space.shape
device = env.unwrapped.device
print(f"action_shape={action_shape} device={device}")

ep_return = torch.zeros(args_cli.num_envs, device=device)
all_rewards = []
all_extras_keys: set[str] = set()

for step in range(args_cli.steps):
    action = 2 * torch.rand(action_shape, device=device) - 1
    with torch.inference_mode():
        obs, reward, term, trunc, info = env.step(action)
    ep_return += reward.float()
    all_rewards.append(float(reward.float().mean().cpu()))
    if isinstance(info, dict):
        all_extras_keys.update(info.keys())
    if step in (0, 1, 2, 5, 20, 40, 79):
        det = info.get("detailed_reward") if isinstance(info, dict) else None
        det_keys = list(det.keys()) if isinstance(det, dict) else None
        rew_logs = {k: float(info[k].mean()) if hasattr(info.get(k), "mean") else info.get(k)
                    for k in info if isinstance(k, str) and k.startswith("logs_rew_")}
        print(f"step {step:>3d}  reward.mean={float(reward.mean()):+.6e}  "
              f"min/max=[{float(reward.min()):+.3e},{float(reward.max()):+.3e}]  "
              f"detailed_reward.keys={det_keys}")
        if rew_logs:
            print(f"           logs_rew_*: {rew_logs}")

print(f"\nsummed return / env (after {args_cli.steps} steps): "
      f"mean={float(ep_return.mean()):.6f}  max={float(ep_return.max()):.6f}  "
      f"min={float(ep_return.min()):.6f}")
print(f"per-step reward mean range: [{min(all_rewards):.4e}, {max(all_rewards):.4e}]")
print(f"info keys seen across run: {sorted(all_extras_keys)}")

env.close()
simulation_app.close()
