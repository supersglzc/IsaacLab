"""
isaaclab — RL training entry point (custom_torch).

Single-process Hydra-driven train loop modeled on the PQL/DDiffPG `train_baselines.py`
pattern. Picks an algorithm from algo/ via the cfg.algo.name field, builds env via
env_wrapper.create_env(), and loops:

    explore_env(env, horizon_len) → trajectory
    update_net(memory_or_trajectory) → log_info

Writes:
    outputs/<algo>_<task>_<YYYYMMDD-HHMMSS>/
        checkpoint.pth               — torch.save({'actor': ..., 'critic': ...})
        resolved_config.yaml         — hydra config snapshot
        tb/                          — TensorBoard event files
        metrics.jsonl                — per-record (step, key, value) log
        curves/                      — PNG plots of training curves
        train.log                    — stdout/stderr mirror

W&B is wired automatically when cfg.wandb is set (use `wandb=null` to disable).

Example:
    python nautilus/scripts/rl/custom_torch/train.py algo=ppo task=Isaac-Cartpole-Direct-v0 max_step=1_000_000
    python nautilus/scripts/rl/custom_torch/train.py algo=sac task=Isaac-Cartpole-Direct-v0 max_step=5000 wandb=null
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from itertools import count
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]   # nautilus/scripts/rl/custom_torch
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import set_random_seed, load_class_from_path  # noqa: E402
from utils.model_util import save_model  # noqa: E402
from replay.simple_replay import ReplayBuffer  # noqa: E402

# DataLogger lives at <repo>/nautilus/utils/data_logger.py. The custom_torch tree has
# its own nautilus/scripts/rl/custom_torch/utils/ package (common, model_util,
# torch_util) which shadows <repo>/utils/ once `import utils.common` runs above.
# Load DataLogger by file path to side-step the name collision.
import importlib.util as _ilu  # noqa: E402
_dl_spec = _ilu.spec_from_file_location("_repo_data_logger", NAUTILUS / "utils" / "data_logger.py")
_dl_mod = _ilu.module_from_spec(_dl_spec)
_dl_spec.loader.exec_module(_dl_mod)
DataLogger = _dl_mod.DataLogger  # noqa: E402

from env_wrapper import create_env  # noqa: E402  (sibling import)


def _resolve_algo_class(name: str):
    """Find AgentXXX class by case-insensitive match in alg_name_to_path."""
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    # Try case-insensitive
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found. Available: {sorted(alg_name_to_path)}")


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Synthesize cfg.algo namespace if the unified yaml put hyperparams at top level.
    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        algo_cfg = OmegaConf.create({
            "name": algo_name,
            "act_class": cfg.get("act_class") or _default_act_class(algo_name),
            "cri_class": cfg.get("cri_class") or _default_cri_class(algo_name),
            "actor_lr": cfg.get("learning_rate", 3.0e-4),
            "critic_lr": cfg.get("critic_learning_rate", cfg.get("learning_rate", 3.0e-4)),
            "gamma": cfg.get("gamma", 0.99),
            "tau": cfg.get("tau", 0.005),
            "batch_size": cfg.get("batch_size", 256),
            "update_times": cfg.get("update_times", 1),
            "warm_up": cfg.get("warm_up", 1000),
            "horizon_len": cfg.get("n_steps", 16),
            "memory_size": cfg.get("buffer_size", 1_000_000),
            "tracker_len": cfg.get("tracker_len", 100),
            "max_grad_norm": cfg.get("max_grad_norm", 0.5),
            "obs_norm": cfg.get("obs_norm", False),
            "value_norm": cfg.get("value_norm", False),
            "reward_scale": cfg.get("reward_scale", 1.0),
            "nstep": cfg.get("nstep", 1),
            "alpha": cfg.get("alpha", None),                # SAC
            "alpha_lr": cfg.get("alpha_lr", 3.0e-4),
            "no_tgt_actor": cfg.get("no_tgt_actor", True),
            "explore_noise": cfg.get("explore_noise", 0.1),  # TD3
            "policy_noise": cfg.get("policy_noise", 0.2),
            "noise_clip": cfg.get("noise_clip", 0.5),
            "policy_delay": cfg.get("policy_delay", 2),
            "n_epochs": cfg.get("n_epochs", 10),              # PPO
            "clip_range": cfg.get("clip_range", 0.2),
            "ent_coef": cfg.get("ent_coef", 0.0),
            "vf_coef": cfg.get("vf_coef", 0.5),
            "gae_lambda": cfg.get("gae_lambda", 0.95),
            "use_gae": cfg.get("use_gae", True),
            "handle_timeout": cfg.get("handle_timeout", False),
        })
        cfg.algo = algo_cfg

    # Common runtime fields the algorithms expect.
    cfg.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.num_envs = int(cfg.get("n_envs", cfg.get("num_envs", 8)))
    cfg.gpu_sim = bool(cfg.get("gpu_sim", False))

    task = cfg.get("task")
    if task is None:
        print("[error] task=<id> is required (Hydra override).", file=sys.stderr)
        sys.exit(2)
    set_random_seed(int(cfg.get("seed", 42)))

    env = create_env(cfg)

    # Trial dir + logger. outputs/<algo>_<task>_<YYYYMMDD-HHMMSS>/
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = str(task).replace('/', '_')
    trial_id = f"{str(cfg.algo.name).lower()}_{task_slug}_{ts}"
    log_dir = NAUTILUS / "outputs" / trial_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "curves").mkdir(parents=True, exist_ok=True)

    use_wandb = bool(cfg.get("wandb")) and cfg.get("wandb") not in ("null", "None", "")
    dl = DataLogger(
        log_dir=str(log_dir),
        log_tb=True,
        log_wandb=use_wandb,
        project=str(cfg.wandb) if use_wandb else None,
        run_name=trial_id,
        config=OmegaConf.to_container(cfg, resolve=True) if use_wandb else None,
    )
    dl.log_text("config", OmegaConf.to_yaml(cfg, resolve=True), step=0)

    # Local metrics buffer — written to metrics.jsonl + plotted at end.
    metrics_log: list[dict] = []
    metrics_jsonl = log_dir / "metrics.jsonl"
    _jsonl_fp = metrics_jsonl.open("w", buffering=1)

    def _record(key: str, value: float, step: int):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(f):
            return
        dl.record(key, f, step=step)
        rec = {"step": int(step), "key": key, "value": f}
        metrics_log.append(rec)
        _jsonl_fp.write(json.dumps(rec) + "\n")

    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)

    is_off_policy = str(cfg.algo.name).lower() != "ppo"
    memory = None
    if is_off_policy:
        memory = ReplayBuffer(
            capacity=int(cfg.algo.memory_size),
            obs_dim=agent.obs_dim, action_dim=agent.action_dim, device=cfg.device,
        )

    agent.reset_agent()
    global_steps = 0
    if is_off_policy:
        traj, used = agent.explore_env(env, int(cfg.algo.warm_up), random=True)
        if traj is not None:
            memory.add_to_buffer(traj)
        global_steps += int(used)

    max_step = int(cfg.get("total_timesteps", 1_000_000))
    log_interval = int(cfg.get("log_interval", int(cfg.algo.horizon_len) * int(cfg.num_envs)))
    print(f"[train] algo={cfg.algo.name} task={task} num_envs={cfg.num_envs} max_step={max_step} "
          f"log_interval={log_interval} wandb={'on' if use_wandb else 'off'}", flush=True)

    next_log_at = log_interval
    try:
        for it in count():
            traj, used = agent.explore_env(env, int(cfg.algo.horizon_len), random=False)
            global_steps += int(used)
            if is_off_policy:
                if traj is not None:
                    memory.add_to_buffer(traj)
                log_info = agent.update_net(memory)
            else:
                log_info = agent.update_net(traj)

            if global_steps >= next_log_at:
                for k, v in (log_info or {}).items():
                    _record(k, v, global_steps)
                print(f"[train] iter={it} steps={global_steps}/{max_step} "
                      f"return={log_info.get('reward/total/episodic_return_mean', 0.0):.3f}", flush=True)
                next_log_at = ((global_steps // log_interval) + 1) * log_interval

            if global_steps >= max_step:
                break

        ckpt = log_dir / "checkpoint.pth"
        save_model(ckpt, actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                   algo=str(cfg.algo.name), task=str(task))
        dl.log_text("final/checkpoint_path", str(ckpt), step=global_steps)
        print(f"[train] saved checkpoint to {ckpt}", flush=True)
        if bool(cfg.get("record_video", True)):
            _render_and_log_video(agent, cfg, log_dir, global_steps, use_wandb,
                                  fps=int(cfg.get("render_fps", 30)),
                                  max_steps=int(cfg.get("render_max_steps", 1000)))
    finally:
        _jsonl_fp.close()
        dl.close()
        (log_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        # Plot training curves to outputs/<run>/curves/<key>.png
        _plot_curves(metrics_log, log_dir / "curves", title_prefix=trial_id)


def _render_and_log_video(agent, cfg: DictConfig, log_dir: Path, step: int,
                          use_wandb: bool, fps: int = 30, max_steps: int = 1000) -> Path | None:
    """Run a deterministic rollout with the trained custom_torch agent and write
    `<log_dir>/render.mp4`. If `use_wandb`, also log the video to W&B as
    `render/video`. Reference: humanoid_juggling/envs/train.py:record_video block.
    """
    import numpy as np
    from omegaconf import OmegaConf as _OC
    render_cfg = _OC.create(_OC.to_container(cfg, resolve=True))
    render_cfg.num_envs = 1
    # Keep cfg.gpu_sim as-is — GPU-batched simulators (IsaacLab, IsaacGym, ManiSkill GPU)
    # cannot build a render env via the CPU gymnasium.make path. Their create_render_env
    # is responsible for re-launching with cameras enabled (or raising cleanly if the
    # underlying engine doesn't allow mid-process kit-experience swaps, in which case
    # this wrapper's try/except logs the message and skips gracefully — render.py is
    # the canonical entry point for video).
    try:
        from env_wrapper import create_render_env
        env = create_render_env(render_cfg)
    except Exception as e:
        print(f"[render] could not build render env: {e}", file=sys.stderr)
        return None
    try:
        obs, _ = env.reset()
        frames: list[np.ndarray] = []
        ep_return = 0.0

        def _grab():
            raw = getattr(env, "_raw_env", None)
            target = raw if raw is not None else env
            try:
                f = target.render()
            except Exception:
                return None
            if isinstance(f, np.ndarray):
                return f.astype(np.uint8)
            if isinstance(f, list) and f and isinstance(f[0], np.ndarray):
                return f[0].astype(np.uint8)
            return None

        agent.actor.eval()
        for _ in range(max_steps):
            f = _grab()
            if f is not None:
                frames.append(f)
            with torch.no_grad():
                if hasattr(agent.actor, "get_actions"):
                    action = agent.actor.get_actions(obs, sample=False)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    action = agent.actor(obs)
            next_obs, reward, done, _info = env.step(action)
            ep_return += float(reward.item() if isinstance(reward, torch.Tensor) else reward)
            if bool(done.item() if isinstance(done, torch.Tensor) else done):
                f = _grab()
                if f is not None:
                    frames.append(f)
                break
            obs = next_obs

        if not frames:
            print("[render] no frames captured — skipping mp4 write", file=sys.stderr)
            return None
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore[no-redef]
        mp4 = log_dir / "render.mp4"
        imageio.mimsave(str(mp4), frames, fps=fps, codec="libx264", quality=8)
        print(f"[render] wrote {mp4} ({len(frames)} frames, return={ep_return:.4f})", flush=True)
        if use_wandb:
            try:
                import wandb
                wandb.log({"render/video": wandb.Video(str(mp4), fps=fps, format="mp4")}, step=step)
                print("[render] logged render/video to W&B", flush=True)
            except Exception as e:
                print(f"[render] wandb video log failed: {e}", file=sys.stderr)
        return mp4
    finally:
        try:
            env.env.close() if hasattr(env, "env") else None
        except Exception:
            pass


def _plot_curves(metrics: list[dict], out_dir: Path, title_prefix: str = "") -> None:
    """One PNG per metric key, X=step, Y=value. Skips silently if matplotlib missing."""
    if not metrics:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping curve plots", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key: dict[str, list[tuple[int, float]]] = {}
    for r in metrics:
        by_key.setdefault(r["key"], []).append((r["step"], r["value"]))
    written = 0
    for key, pts in by_key.items():
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig, ax = plt.subplots(figsize=(6, 4))
        # Use marker so 1-point series (smoke-budget runs) still produces a visible dot.
        ax.plot(xs, ys, linewidth=1.2, marker="o" if len(pts) <= 3 else None, markersize=4)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.set_title(f"{title_prefix}\n{key}" if title_prefix else key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = key.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=110)
        plt.close(fig)
        written += 1
    print(f"[plot] wrote {written}/{len(by_key)} curves to {out_dir}", flush=True)


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo":
        return "DiagGaussianMLPPolicy"
    if a == "sac":
        return "TanhDiagGaussianMLPPolicy"
    if a in ("td3", "ddpg"):
        return "TanhMLPPolicy"
    return "DiagGaussianMLPPolicy"


def _default_cri_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo":
        return "MLPCritic"
    return "DoubleQ"


if __name__ == "__main__":
    main()
