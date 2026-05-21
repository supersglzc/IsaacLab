"""Headed live viewer for a trained custom_torch checkpoint on IsaacLab.

Boots Kit in non-headless mode (opens a GLFW window via the IsaacLab/Kit viewer),
loads the checkpoint, runs the policy in a loop. Requires $DISPLAY (Vulkan
windowing); will not work over SSH without X forwarding.

Usage:
    cd <repo>
    source .venv/bin/activate
    OMNI_KIT_ALLOW_ROOT=1 python -u nautilus/scripts/rl/custom_torch/visualize_headed.py \
        --checkpoint <path-to-checkpoint.pth> \
        [--num-envs 4] [--steps 100000]

Closes when the Kit window is closed by the user, or after --steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure we can import siblings (algo, utils, env_wrapper) regardless of cwd.
HERE = Path(__file__).resolve().parent
NAUTILUS = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NAUTILUS))

# 1) BOOT KIT NON-HEADLESS BEFORE ANY isaaclab IMPORT.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True, help="path to checkpoint.pth")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100_000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headed mode regardless of what AppLauncher's add_app_launcher_args defaults to.
args.headless = False
args.enable_cameras = False

app = AppLauncher(args).app  # opens GLFW window

# 2) NOW safe to import isaaclab / torch / wrapper.
import torch
from omegaconf import OmegaConf

from env_wrapper import create_env
from algo import alg_name_to_path
from utils.common import load_class_from_path


ckpt_path = Path(args.checkpoint).resolve()
if not ckpt_path.is_file():
    print(f"[viz] checkpoint not found: {ckpt_path}", file=sys.stderr)
    sys.exit(2)
cfg_path = ckpt_path.parent / "resolved_config.yaml"
if not cfg_path.is_file():
    print(f"[viz] resolved_config.yaml not found next to checkpoint: {cfg_path}", file=sys.stderr)
    sys.exit(2)

cfg = OmegaConf.load(cfg_path)
cfg.num_envs = int(args.num_envs)

# IMPORTANT: _ensure_isaac_app in env_wrapper insists on headless=True. Since we
# already booted headed, the helper's idempotent path picks up the existing app
# and skips the re-launch — but it will overwrite the global state only if it
# tries to launch. The current env_wrapper does NOT re-launch if Kit is up, so
# we're fine. (See _ensure_isaac_app in env_wrapper.py.)
env = create_env(cfg)

state = torch.load(str(ckpt_path), map_location=cfg.device, weights_only=False)
if "obs_rms" in state:
    OmegaConf.update(cfg, "algo.obs_norm", True, merge=True)
if "value_rms" in state:
    OmegaConf.update(cfg, "algo.value_norm", True, merge=True)

agent_cls = load_class_from_path("AgentPPO", alg_name_to_path["AgentPPO"])
agent = agent_cls(env=env, cfg=cfg)
agent.actor.load_state_dict(state["actor"])
if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
    agent.obs_rms.load_state_dict(state["obs_rms"])
    print("[viz] restored obs_rms from checkpoint", flush=True)
if "value_rms" in state and getattr(agent, "value_rms", None) is not None:
    agent.value_rms.load_state_dict(state["value_rms"])
agent.actor.eval()

obs, _ = env.reset()
print(f"[viz] running policy live for up to {args.steps} steps "
      f"({cfg.num_envs} envs). Close the Kit window or Ctrl-C to exit.", flush=True)
try:
    for step in range(int(args.steps)):
        with torch.no_grad():
            nobs = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
            action = agent.actor.get_actions(nobs, sample=False)
            if isinstance(action, tuple):
                action = action[0]
        obs, _, done, _ = env.step(action)
        # Kit's renderer drives off env.step() under the hood; nothing else to do here.
except KeyboardInterrupt:
    print("[viz] interrupted", flush=True)
finally:
    env.close()
    app.close()
