"""
isaaclab — RL evaluation entry point (custom_torch).

Loads a checkpoint, runs `n_episodes` deterministic rollouts on the SAME vec env
type used by train.py, prints aggregate metrics + writes metrics.json.

Always headless. Always vec/parallel env (matches train.py shapes).

Example:
    python nautilus/scripts/rl/custom_torch/eval.py algo=ppo task=Isaac-Cartpole-Direct-v0 \\
        checkpoint=nautilus/rl_experiments/runs/<trial_id>/checkpoint.pth \\
        n_episodes=10
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path, set_random_seed  # noqa: E402
from env_wrapper import create_env  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found in registry.")


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Same defaults synthesis as train.py — eval reuses the same algo class.
    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "act_class": cfg.get("act_class") or _default_act_class(algo_name),
            "cri_class": cfg.get("cri_class") or _default_cri_class(algo_name),
            "actor_lr": cfg.get("learning_rate", 3.0e-4),
            "critic_lr": cfg.get("learning_rate", 3.0e-4),
            "tracker_len": cfg.get("tracker_len", 100),
            "obs_norm": cfg.get("obs_norm", False),
            "alpha": cfg.get("alpha", None),
            "alpha_lr": cfg.get("alpha_lr", 3.0e-4),
            "no_tgt_actor": cfg.get("no_tgt_actor", True),
            "nstep": cfg.get("nstep", 1),
            "explore_noise": 0.0,    # no exploration noise during eval
        })
    cfg.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.num_envs = int(cfg.get("n_envs", cfg.get("num_envs", 1)))
    cfg.gpu_sim = bool(cfg.get("gpu_sim", False))

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    n_episodes = int(cfg.get("n_episodes", 10))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)
    set_random_seed(int(cfg.get("seed") or 42))  # tolerate seed=null

    env = create_env(cfg)
    # Pre-load checkpoint to enable obs/value normalization if training used it,
    # then build agent so it allocates obs_rms / value_rms.
    state = torch.load(str(ckpt), map_location=cfg.device, weights_only=False)
    if state.get("obs_norm") or "obs_rms" in state:
        OmegaConf.update(cfg, "algo.obs_norm", True, merge=True)
    if state.get("value_norm") or "value_rms" in state:
        OmegaConf.update(cfg, "algo.value_norm", True, merge=True)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
        print("[eval] restored obs_rms from checkpoint", flush=True)
    if "value_rms" in state and getattr(agent, "value_rms", None) is not None:
        agent.value_rms.load_state_dict(state["value_rms"])
        print("[eval] restored value_rms from checkpoint", flush=True)
    agent.actor.eval()
    agent.critic.eval()

    n = int(cfg.num_envs)
    returns = torch.zeros(n, device=cfg.device)
    lengths = torch.zeros(n, device=cfg.device)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    obs, _ = env.reset()
    start = time.time()
    target = n_episodes
    safety_steps = 0
    max_safety = 100_000
    while len(completed_returns) < target and safety_steps < max_safety:
        with torch.no_grad():
            # Normalize obs via obs_rms BEFORE calling the actor — training's
            # rollout path (agent.get_actions in ppo.py) normalizes first, so
            # the actor was trained on normalized inputs. Skipping this step
            # silently produces near-zero performance even with the correct
            # checkpoint loaded.
            obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
            if hasattr(agent.actor, "get_actions"):
                action = agent.actor.get_actions(obs_in, sample=False)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = agent.actor(obs_in)
        next_obs, reward, done, info = env.step(action)
        returns += reward.float()
        lengths += 1
        done_idx = torch.where(done.bool())[0]
        for i in done_idx.tolist():
            completed_returns.append(float(returns[i].item()))
            completed_lengths.append(int(lengths[i].item()))
            returns[i] = 0
            lengths[i] = 0
        obs = next_obs
        safety_steps += 1

    metrics = {
        "algo": str(cfg.algo.name),
        "task": str(task),
        "checkpoint": str(ckpt),
        "n_episodes_completed": len(completed_returns),
        "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "std_return":  float(np.std(completed_returns)) if completed_returns else 0.0,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "wall_time_sec": time.time() - start,
    }
    print(f"[eval] mean_return={metrics['mean_return']:.4f} ± {metrics['std_return']:.4f} "
          f"(n={metrics['n_episodes_completed']})", flush=True)
    (ckpt.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo":
        return "DiagGaussianMLPPolicy"
    if a == "sac":
        return "TanhDiagGaussianMLPPolicy"
    return "TanhMLPPolicy"


def _default_cri_class(algo: str) -> str:
    return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


if __name__ == "__main__":
    main()
