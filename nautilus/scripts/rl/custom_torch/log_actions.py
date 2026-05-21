"""Run a trained policy for one episode and write per-step action data to JSON.

Mirrors render.py's hydra-based agent loading. Output goes to
  <trial_dir>/policy_action_log.json
right next to the checkpoint, with one record per step containing the policy
action, the env's processed joint targets, and cube/EE positions.

Usage:
    python nautilus/scripts/rl/custom_torch/log_actions.py \
        --config-name=ppo.parallel task=Triton-Franka-StackCup \
        checkpoint=<trial_dir>/checkpoint.pth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

REPO     = Path(__file__).resolve().parents[4]
NAUTILUS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path  # noqa: E402
from env_wrapper import create_render_env  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found in registry.")


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a in ("ppo", "ppoparallel"):
        return "GaussianActor"
    return "DeterministicActor"


def _default_cri_class(algo: str) -> str:
    return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


@hydra.main(config_path="../../../configs/rl", config_name="ppo.parallel", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    # Promote the string `algo: ppo` into the DictConfig render.py expects.
    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "act_class": cfg.get("act_class") or _default_act_class(algo_name),
            "cri_class": cfg.get("cri_class") or _default_cri_class(algo_name),
            "actor_lr": 3.0e-4, "critic_lr": 3.0e-4,
            "tracker_len": 100, "obs_norm": False,
            "alpha": cfg.get("alpha", None), "alpha_lr": 3.0e-4,
            "no_tgt_actor": True, "nstep": 1,
        })
    cfg.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.num_envs = 1

    ckpt = Path(cfg.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", flush=True)
        sys.exit(1)
    trial_dir = ckpt.parent

    env = create_render_env(cfg)
    raw_env = getattr(env, "_raw_env", env)
    n_steps = int(cfg.get("render_max_steps", 250))
    env_max_len = getattr(getattr(raw_env, "unwrapped", raw_env), "max_episode_length", None)
    if env_max_len is not None and int(env_max_len) > 0:
        n_steps = min(n_steps, int(env_max_len) + 1)

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
        print("[log_actions] restored obs_rms")
    agent.actor.eval()

    unw = raw_env.unwrapped
    arm_term = unw.action_manager.get_term("arm_action")
    log = {
        "task": str(cfg.task),
        "trial_dir": str(trial_dir),
        "n_steps": n_steps,
        "action_dim": int(env.action_space.shape[-1]),
        "obs_dim": int(env.observation_space.shape[-1]),
        "steps": [],
    }

    obs, _ = env.reset()
    print(f"[log_actions] starting rollout, n_steps={n_steps}")
    for t in range(n_steps):
        if isinstance(obs, dict):
            o = obs["policy"]
        else:
            o = obs
        if getattr(agent, "obs_rms", None) is not None:
            o = agent.obs_rms.normalize(o)
        with torch.no_grad():
            a, *_ = agent.actor.get_actions(o, sample=False)
        step_out = env.step(a)
        if len(step_out) == 5:
            next_obs, rew, term, trunc, info = step_out
        else:
            next_obs, rew, term, info = step_out
            trunc = torch.zeros_like(term) if term is not None else None
        cube_0 = unw.scene["cube_0"].data.root_pos_w[0, :3].tolist()
        cube_1 = unw.scene["cube_1"].data.root_pos_w[0, :3].tolist()
        cube_2 = unw.scene["cube_2"].data.root_pos_w[0, :3].tolist()
        ee = unw.scene["ee_frame"].data.target_pos_w[0, 0, :].tolist()
        proc = arm_term._processed_actions[0].tolist()
        log["steps"].append({
            "t": t,
            "policy_action": [float(x) for x in a[0].tolist()],
            "joint_targets": [float(x) for x in proc],
            "cube_0": [round(v, 4) for v in cube_0],
            "cube_1": [round(v, 4) for v in cube_1],
            "cube_2": [round(v, 4) for v in cube_2],
            "ee": [round(v, 4) for v in ee],
            "reward": float(rew[0].item()) if rew is not None else 0.0,
        })
        obs = next_obs
        if (term is not None and bool(term[0].item())) or (trunc is not None and bool(trunc[0].item())):
            print(f"[log_actions] episode end at step {t} (term={bool(term[0])} trunc={bool(trunc[0])})")
            break

    out = trial_dir / "policy_action_log.json"
    with open(out, "w") as f:
        json.dump(log, f)
    print(f"[log_actions] wrote {out} with {len(log['steps'])} steps")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
