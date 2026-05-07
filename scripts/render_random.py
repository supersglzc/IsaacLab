"""IsaacLab — random-action rollout that writes an MP4 (render sanity check).

Purpose
-------
Builds an Isaac Lab GPU-vectorized env with cameras enabled, runs N random-action
steps, captures one RGB frame per step via ``env.render()``, and writes them
to an MP4. Proves that the offscreen render pipeline (RTX renderer + EGL/X11
+ ffmpeg) is wired without requiring a trained policy.

Used as the L2 smoke tier by benchmark-generator.

Important
---------
This script MUST be invoked through ``isaaclab.sh -p`` (or with the same
env-sourcing shim) so ``isaacsim`` / ``pxr`` / ``omni.kit`` resolve. Cameras
are enabled programmatically so the right Kit experience file is picked.

Example
-------
    # via the wrapper
    ./isaaclab.sh -p scripts/render_random.py \\
        --task Isaac-Cartpole-Direct-v0 --n-steps 30 --output /tmp/random.mp4

    # via the manual shim
    source .venv/bin/activate
    source _isaac_sim/setup_conda_env.sh
    OMNI_KIT_ALLOW_ROOT=1 python scripts/render_random.py \\
        --task Isaac-Cartpole-Direct-v0 --n-steps 30 --output /tmp/random.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab render-to-MP4 smoke (L2).")
parser.add_argument("--task", default="Isaac-Cartpole-Direct-v0",
                    help="Isaac Lab gym ID (default: %(default)s)")
parser.add_argument("--n-steps", type=int, default=30,
                    help="Frames to capture (default: %(default)s)")
parser.add_argument("--output", required=True,
                    help="Output MP4 path (e.g. /tmp/random.mp4)")
parser.add_argument("--fps", type=int, default=30,
                    help="MP4 frame rate (default: %(default)s)")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Parallel envs (default: 1 for smoke)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless + cameras-on. enable_cameras switches the Kit experience file
# to one that runs the RTX renderer offscreen and exposes env.render('rgb_array').
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def build_env(task: str, num_envs: int, device: str):
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=True)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    return env


def sample_action(env):
    return 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1


def extract_frame(env):
    """Render the current frame as an (H, W, 3) uint8 ndarray."""
    frame = env.render()
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def main():
    out_path = Path(args_cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args_cli.task, args_cli.num_envs, args_cli.device)
    try:
        env.reset(seed=args_cli.seed)
    except TypeError:
        env.reset()

    frames: list[np.ndarray] = []
    for i in range(args_cli.n_steps):
        with torch.inference_mode():
            env.step(sample_action(env))
        f = extract_frame(env)
        if f is not None:
            frames.append(f)

    env.close()

    if not frames:
        print("FAIL: no RGB frames captured (env.render() returned None — check enable_cameras)",
              file=sys.stderr)
        simulation_app.close()
        sys.exit(2)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]

    imageio.mimsave(str(out_path), frames, fps=args_cli.fps, codec="libx264", quality=8)
    size_kb = out_path.stat().st_size / 1024
    print(f"L2 OK: wrote {len(frames)} frames to {out_path} ({size_kb:.1f} KB)")
    simulation_app.close()


if __name__ == "__main__":
    main()
