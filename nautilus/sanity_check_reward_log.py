"""Sanity check for /add-reward-log on IsaacLab — verifies the composer-sum
invariant ``Σ info['detailed_reward'].values() == env_reward`` per step,
batched-aware.

Mirrors scripts/run_random.py's AppLauncher boilerplate so it can be invoked
the same way:

    source .venv/bin/activate
    source _isaac_sim/setup_conda_env.sh
    OMNI_KIT_ALLOW_ROOT=1 python -u nautilus/sanity_check_reward_log.py \\
        --task Isaac-Open-Drawer-Franka-v0 --num-envs 4 --steps 200
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab /add-reward-log sanity check.")
parser.add_argument("--task", default="Isaac-Open-Drawer-Franka-v0",
                    help="Isaac Lab gym ID (default: %(default)s)")
parser.add_argument("--num-envs", type=int, default=4,
                    help="Parallel envs (default: %(default)s)")
parser.add_argument("--steps", type=int, default=200,
                    help="Random-action steps (default: %(default)s)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

# Imports that depend on Kit being up.
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _isaaclab_env  # noqa: E402


def main() -> int:
    device = args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = _isaaclab_env.make_isaaclab_env(
        args_cli.task, num_envs=args_cli.num_envs, device=device)
    try:
        env.reset(seed=args_cli.seed)
    except TypeError:
        env.reset()

    action_shape = env.action_space.shape
    env_device = env.unwrapped.device
    composer_seen = None
    bad = 0

    for step in range(args_cli.steps):
        action = 2 * torch.rand(action_shape, device=env_device) - 1
        with torch.inference_mode():
            obs, reward, term_, trunc_, info = env.step(action)

        detailed = info.get("detailed_reward")
        composer = info.get("reward_composer", "sum")
        if not isinstance(detailed, dict) or not detailed:
            print(f"[FAIL] step {step}: missing/empty detailed_reward (got {detailed!r})",
                  file=sys.stderr)
            bad += 1
            break
        if composer_seen is None:
            composer_seen = composer
        elif composer != composer_seen:
            print(f"[FAIL] step {step}: composer changed mid-episode "
                  f"({composer_seen!r} -> {composer!r})", file=sys.stderr)
            bad += 1
            break

        # Batched invariant: sum of per-term tensors must match env_reward per env.
        composed = sum(v for v in detailed.values())
        if not torch.allclose(composed.float(), reward.float(),
                              atol=1e-4, rtol=1e-3):
            diff = (composed - reward).abs().max().item()
            print(f"[FAIL] step {step}: max |Σterms - reward| = {diff:.3e}",
                  file=sys.stderr)
            print(f"       reward (per env) = {reward.detach().cpu().tolist()}",
                  file=sys.stderr)
            print(f"       composed        = {composed.detach().cpu().tolist()}",
                  file=sys.stderr)
            bad += 1
            break

        # Show breakdown for the first 3 steps for visual sanity.
        if step < 3:
            terms_mean = {k: float(v.float().mean()) for k, v in detailed.items()}
            print(f"step {step}: reward.mean={float(reward.float().mean()):+.4f}  "
                  f"composer={composer}  num_terms={len(terms_mean)}")
            for k, v in terms_mean.items():
                print(f"    {k:<32s} {v:+.6f}")

    if bad == 0:
        print(f"[OK] composer-{composer_seen} invariant holds for {args_cli.task}: "
              f"{args_cli.steps}/{args_cli.steps} steps "
              f"(num_envs={args_cli.num_envs})")
        rc = 0
    else:
        rc = 1

    try:
        env.close()
    except Exception:
        pass
    simulation_app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
