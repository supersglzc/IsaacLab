"""Deployment probe: load a checkpoint, run for one episode, and at each step
record (a) raw policy action, (b) the action-term's internal state BEFORE and
AFTER the workspace clamp, (c) actual fingertip pose + quaternion drift vs the
locked init_ee_quat. Writes JSONL to <trial_dir>/probe_clamp.jsonl.

Usage:
  ./.venv/bin/python nautilus/scripts/rl/custom_torch/probe_clamp.py \
      --config-name=ppo.parallel task=Triton-Insert-Drawer \
      checkpoint=<trial_dir>/checkpoint.pth
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import hydra, torch
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
    raise KeyError(name)


def _default_act_class(algo): return "GaussianActor" if algo.lower() in ("ppo","ppoparallel") else "DeterministicActor"
def _default_cri_class(algo): return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


@hydra.main(config_path="../../../configs/rl", config_name="ppo.parallel", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "act_class": cfg.get("act_class") or _default_act_class(algo_name),
            "cri_class": cfg.get("cri_class") or _default_cri_class(algo_name),
            "actor_lr": 3.0e-4, "critic_lr": 3.0e-4, "tracker_len": 100,
            "obs_norm": False, "alpha": None, "alpha_lr": 3.0e-4,
            "no_tgt_actor": True, "nstep": 1,
        })
    cfg.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.num_envs = 1

    ckpt = Path(cfg.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}"); sys.exit(1)
    trial_dir = ckpt.parent

    env = create_render_env(cfg)
    raw_env = getattr(env, "_raw_env", env)
    unw = raw_env.unwrapped
    n_steps = 200
    env_max_len = getattr(unw, "max_episode_length", None)
    if env_max_len is not None and int(env_max_len) > 0:
        n_steps = min(n_steps, int(env_max_len) + 1)

    state = torch.load(str(ckpt), map_location=cfg.device, weights_only=False)
    if state.get("obs_norm") or "obs_rms" in state: OmegaConf.update(cfg, "algo.obs_norm", True, merge=True)
    if state.get("value_norm") or "value_rms" in state: OmegaConf.update(cfg, "algo.value_norm", True, merge=True)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
    agent.actor.eval()

    arm_term = unw.action_manager.get_term("arm_action")
    ee_frame = unw.scene["ee_frame"]
    robot = unw.scene["robot"]
    env_origin = unw.scene.env_origins[0].cpu().numpy().tolist()

    print(f"[probe] pos_lower={arm_term.pos_lower_limit.tolist() if arm_term.pos_lower_limit is not None else None}")
    print(f"[probe] pos_upper={arm_term.pos_upper_limit.tolist() if arm_term.pos_upper_limit is not None else None}")
    print(f"[probe] forbidden_xy_half={getattr(arm_term, 'forbidden_xy_half', None)}")
    print(f"[probe] scale={arm_term._scale[0].tolist()}  alpha={arm_term._alpha}")

    obs, _ = env.reset()
    # Hand-step a few zeros to capture init_ee_pos
    for _ in range(2):
        env.step(torch.zeros((1, env.action_space.shape[-1]), device=cfg.device))

    init_ee_pos_root = arm_term.init_ee_pos[0].detach().cpu().tolist()
    init_ee_quat = arm_term.init_ee_quat[0].detach().cpu().tolist()
    print(f"[probe] init_ee_pos (root frame, after re-anchor): {init_ee_pos_root}")
    print(f"[probe] init_ee_quat (wxyz, locked target):       {init_ee_quat}")

    out_path = trial_dir / "probe_clamp.jsonl"
    print(f"[probe] writing to {out_path}")
    out = open(out_path, "w")

    # reset again to start fresh
    obs, _ = env.reset()

    for t in range(n_steps):
        o = obs["policy"] if isinstance(obs, dict) else obs
        if getattr(agent, "obs_rms", None) is not None:
            o = agent.obs_rms.normalize(o)
        with torch.no_grad():
            a, *_ = agent.actor.get_actions(o, sample=False)

        # Inspect action term state BEFORE step (init_ee_pos + del_action accumulated so far)
        del_action_pre = arm_term.del_action[0].detach().cpu().tolist()
        init_pos = arm_term.init_ee_pos[0].detach().cpu().tolist()
        init_quat = arm_term.init_ee_quat[0].detach().cpu().tolist()
        prev_applied = arm_term._prev_applied_pos[0].detach().cpu().tolist()

        # The action term itself will produce the target on next env.step. We can
        # compute what abs_pos and ema_pos WOULD be using the raw policy action.
        # The arm action term only sees the first 3 dims (xyz); the 4th dim is
        # gripper handled by a separate term.
        a_arm = a[0, :3].detach().cpu().clamp(-1.0, 1.0)
        scaled = (a_arm * arm_term._scale[0].detach().cpu()).tolist()
        del_action_next = [del_action_pre[i] + scaled[i] for i in range(3)]
        abs_pos_pre_clamp = [init_pos[i] + del_action_next[i] for i in range(3)]
        ema_pos_pre_clamp = [arm_term._alpha * abs_pos_pre_clamp[i] + (1.0 - arm_term._alpha) * prev_applied[i] for i in range(3)]

        # Step the env (action term applies clamp inside).
        step_out = env.step(a)
        if len(step_out) == 5:
            next_obs, rew, term, trunc, info = step_out
        else:
            next_obs, rew, term, info = step_out
            trunc = torch.zeros_like(term)

        # AFTER step: read the actual processed action target (post-clamp).
        proc = arm_term._processed_actions[0].detach().cpu().tolist()
        ema_pos_post_clamp = proc[:3]
        cmd_quat = proc[3:7]

        # Actual EE TCP world pose
        ee_pos_w = ee_frame.data.target_pos_w[0, 0].detach().cpu().tolist()
        ee_quat_w = ee_frame.data.target_quat_w[0, 0].detach().cpu().tolist()
        # Convert to root frame
        root_pos_w = robot.data.root_pos_w[0].detach().cpu().tolist()
        root_quat_w = robot.data.root_quat_w[0].detach().cpu().tolist()
        ee_pos_root = [ee_pos_w[i] - root_pos_w[i] for i in range(3)]  # robot is identity-rotated, so just translate

        # Quat deviation from init_ee_quat: 1 - |q1 . q2|
        q_target = torch.tensor(init_quat); q_actual = torch.tensor(ee_quat_w)
        cos_half = float(abs(torch.dot(q_target, q_actual).item()))
        quat_dev_rad = 2.0 * float((1.0 - min(cos_half, 1.0)) ** 0.5)  # approx angle in radians

        # Was clamp triggered?
        eps = 1e-4
        x_clamped = (abs(ema_pos_post_clamp[0] - ema_pos_pre_clamp[0]) > eps)
        y_clamped = (abs(ema_pos_post_clamp[1] - ema_pos_pre_clamp[1]) > eps)
        z_clamped = (abs(ema_pos_post_clamp[2] - ema_pos_pre_clamp[2]) > eps)

        rec = {
            "step": t,
            "raw_action": a[0].detach().cpu().tolist(),
            "del_action": del_action_next,
            "abs_pos_pre_clamp_root": abs_pos_pre_clamp,
            "ema_pos_pre_clamp_root": ema_pos_pre_clamp,
            "ema_pos_post_clamp_root": ema_pos_post_clamp,
            "cmd_quat": cmd_quat,
            "ee_pos_world": ee_pos_w,
            "ee_pos_root": ee_pos_root,
            "ee_quat_world": ee_quat_w,
            "init_quat_locked": init_quat,
            "quat_dev_rad": quat_dev_rad,
            "clamp_triggered": [x_clamped, y_clamped, z_clamped],
            "reward_total": float(rew[0].item()) if rew is not None else 0.0,
        }
        out.write(json.dumps(rec) + "\n"); out.flush()

        if t % 20 == 0 or t == n_steps - 1:
            print(f"step={t:3d}  ema_pre={[f'{v:+.3f}' for v in ema_pos_pre_clamp]}  "
                  f"ema_post={[f'{v:+.3f}' for v in ema_pos_post_clamp]}  "
                  f"ee_root={[f'{v:+.3f}' for v in ee_pos_root]}  "
                  f"clamp={[int(x) for x in (x_clamped, y_clamped, z_clamped)]}  "
                  f"q_dev_rad={quat_dev_rad:.4f}")

        if (term.any() or trunc.any()).item():
            print(f"[probe] episode ended at step {t}; restarting")
            obs, _ = env.reset()
        else:
            obs = next_obs

    out.close()
    print(f"[probe] DONE. Wrote {out_path}")
    import os; os._exit(0)


if __name__ == "__main__":
    main()
