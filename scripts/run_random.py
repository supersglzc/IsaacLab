"""IsaacLab — random-action rollout (env sanity check).

Purpose
-------
Builds an Isaac Lab GPU-vectorized env, runs N random-action steps, and verifies
that ``env.step(...)`` returns finite numeric rewards on every step. This is the
fastest end-to-end check that env-generator + benchmark-generator wired
the simulator correctly. NO trained policy is needed; NO video is produced.

Used as the L1 smoke tier by benchmark-generator.

Important
---------
This script MUST be invoked through ``isaaclab.sh -p`` (or with the same
env-sourcing shim) so ``isaacsim`` / ``pxr`` / ``omni.kit`` resolve. Direct
invocation of ``.venv/bin/python`` will fail with ``ModuleNotFoundError: No
module named 'isaacsim'``.

Example
-------
    # via the wrapper (recommended)
    ./isaaclab.sh -p scripts/run_random.py --task Isaac-Cartpole-Direct-v0 --n-steps 10

    # via the manual shim (what benchmark-generator uses)
    source .venv/bin/activate
    source _isaac_sim/setup_conda_env.sh
    OMNI_KIT_ALLOW_ROOT=1 python scripts/run_random.py --task Isaac-Cartpole-Direct-v0 --n-steps 10
"""
from __future__ import annotations

import argparse
import sys

# AppLauncher boilerplate — MUST run before importing torch/gym/isaaclab.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab random-action smoke (L1).")
parser.add_argument("--task", default="Isaac-Cartpole-Direct-v0",
                    help="Isaac Lab gym ID (default: %(default)s)")
parser.add_argument("--n-steps", type=int, default=10,
                    help="Number of env.step() calls (default: %(default)s)")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Parallel envs (default: 1 for smoke)")
parser.add_argument("--seed", type=int, default=0,
                    help="env.reset(seed=...) when supported (default: %(default)s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless for L1 — we are not rendering.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports follow only after the kit app is up.
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  — registers Isaac-* gym IDs
from isaaclab_tasks.utils import parse_env_cfg


def build_env(task: str, num_envs: int, device: str):
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=True)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    return env


def sample_action(env):
    # Isaac Lab actions are torch tensors of shape (num_envs, action_dim) on the env device.
    return 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1


def main():
    env = build_env(args_cli.task, args_cli.num_envs, args_cli.device)
    try:
        env.reset(seed=args_cli.seed)
    except TypeError:
        env.reset()

    bad = 0
    for i in range(args_cli.n_steps):
        with torch.inference_mode():
            out = env.step(sample_action(env))
        # Isaac Lab follows the gymnasium 5-tuple API.
        reward = out[1] if isinstance(out, tuple) else getattr(out, "reward", None)
        # reward is a torch.Tensor (num_envs,) — collapse to a python float for the check.
        if isinstance(reward, torch.Tensor):
            r_arr = reward.detach().cpu().float().numpy()
            r_repr = float(r_arr.mean())
            ok = bool(np.all(np.isfinite(r_arr)))
        else:
            r_arr = np.asarray(reward, dtype=float).ravel()
            r_repr = float(r_arr.mean()) if r_arr.size else float("nan")
            ok = bool(np.all(np.isfinite(r_arr)))
        if not ok or reward is None:
            print(f"step {i}: reward={reward!r} (NOT finite)", file=sys.stderr)
            bad += 1
        else:
            print(f"step {i}: reward[mean]={r_repr:.4f}")

    env.close()

    if bad:
        print(f"FAIL: {bad}/{args_cli.n_steps} steps returned non-finite reward", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    print(f"L1 OK: {args_cli.n_steps} steps, all rewards finite")
    simulation_app.close()


if __name__ == "__main__":
    main()
