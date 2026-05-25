"""IsaacLab — trained-policy rollout (headed live viewer).

Mirrors the structure of ``scripts/run_random.py``: bootstraps AppLauncher,
builds the env via ``gym.make + parse_env_cfg`` (no nautilus wrappers),
and runs in a loop. Random actions are replaced by a checkpoint-loaded policy.

Pass ``--headless False`` (default) to get a live Kit viewer window. Requires
``$DISPLAY`` (Vulkan windowing); will not work over plain SSH without X
forwarding.

Example
-------
    source .venv/bin/activate
    OMNI_KIT_ALLOW_ROOT=1 python scripts/run_policy.py \
        --task Triton-Franka-StackCup \
        --checkpoint nautilus/outputs/ppo_Triton-Franka-StackCup_<ts>/checkpoint.pth \
        --num-envs 4 --n-steps 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# AppLauncher boilerplate — MUST run before importing torch/gym/isaaclab.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Triton-Franka-StackCup",
                    help="Isaac Lab gym ID (default: %(default)s)")
parser.add_argument("--checkpoint", required=True,
                    help="path to checkpoint.pth produced by nautilus train.py")
parser.add_argument("--num-envs", type=int, default=4,
                    help="parallel envs (default: 4)")
parser.add_argument("--n-steps", type=int, default=100_000,
                    help="env.step() calls to run; default 100k = until user closes window")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sample", action="store_true",
                    help="sample stochastic actions instead of the mean (default: deterministic mean)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default to HEADED. The user can still pass `--headless` to override.
if not getattr(args_cli, "headless", False):
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports below must follow the launcher.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402 — registers gym IDs
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Sibling imports from the nautilus tree (algo / utils / IsaacLabVecAdapter).
REPO = Path(__file__).resolve().parents[1]
NAUTILUS_TREE = REPO / "nautilus" / "scripts" / "rl" / "custom_torch"
NAUTILUS = REPO / "nautilus"
sys.path.insert(0, str(NAUTILUS_TREE))
sys.path.insert(0, str(NAUTILUS))

from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path  # noqa: E402
from env_wrapper import IsaacLabVecAdapter  # noqa: E402


def main():
    # 1) Build the raw IsaacLab env (mirror run_random.py).
    device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device,
                            num_envs=int(args_cli.num_envs), use_fabric=True)
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env.reset(seed=args_cli.seed)

    # 2) Wrap with the adapter the agent expects.
    episode_len = int(getattr(raw_env.unwrapped, "max_episode_length", 0)) or None
    env = IsaacLabVecAdapter(raw_env, device=device, max_episode_length=episode_len)

    # 3) Resolve the algorithm + load the checkpoint.
    ckpt = Path(args_cli.checkpoint).resolve()
    if not ckpt.is_file():
        print(f"[run_policy] checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(2)
    cfg_path = ckpt.parent / "resolved_config.yaml"
    if not cfg_path.is_file():
        print(f"[run_policy] resolved_config.yaml not found next to checkpoint: {cfg_path}",
              file=sys.stderr)
        sys.exit(2)
    cfg = OmegaConf.load(cfg_path)
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = device
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    if "obs_rms" in state:
        OmegaConf.update(cfg, "algo.obs_norm", True, merge=True)
    if "value_rms" in state:
        OmegaConf.update(cfg, "algo.value_norm", True, merge=True)
    algo_name = str(cfg.algo.name).lower()
    cls_key = {"ppo": "AgentPPO", "sac": "AgentSAC", "td3": "AgentTD3"}.get(algo_name, "AgentPPO")
    agent_cls = load_class_from_path(cls_key, alg_name_to_path[cls_key])
    agent = agent_cls(env=env, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
        print("[run_policy] restored obs_rms from checkpoint", flush=True)
    if "value_rms" in state and getattr(agent, "value_rms", None) is not None:
        agent.value_rms.load_state_dict(state["value_rms"])
    agent.actor.eval()

    # 4) Run the loop.
    obs, _ = env.reset()
    print(f"[run_policy] {args_cli.task}: running policy live up to {args_cli.n_steps} steps "
          f"({cfg.num_envs} envs). Close the Kit window or Ctrl-C to exit.", flush=True)
    sample = bool(args_cli.sample)
    try:
        for step in range(int(args_cli.n_steps)):
            if not simulation_app.is_running():
                break
            with torch.no_grad():
                nobs = (agent.obs_rms.normalize(obs)
                        if getattr(agent, "obs_rms", None) is not None else obs)
                action = agent.actor.get_actions(nobs, sample=sample)
                if isinstance(action, tuple):
                    action = action[0]
            obs, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        print("[run_policy] interrupted", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
