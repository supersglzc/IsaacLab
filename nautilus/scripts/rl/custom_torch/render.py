"""
isaaclab — render a trained policy to video.mp4 (custom_torch).

Single env (n=1), offscreen rendering enabled, runs ONE episode with the loaded
checkpoint and writes:

    <checkpoint-dir>/render.mp4
    <checkpoint-dir>/render_contact_sheet.jpg

Always produces a visible artifact — works on headless hosts because it uses
offscreen rendering. To open a live window, use the env's native viewer.

Example:
    python nautilus/scripts/rl/custom_torch/render.py algo=ppo task=Isaac-Cartpole-Direct-v0 \\
        checkpoint=nautilus/rl_experiments/runs/<trial_id>/checkpoint.pth
"""
from __future__ import annotations

import sys
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


def _grab_frame(env) -> np.ndarray | None:
    raw = getattr(env, "_raw_env", None)
    target = raw if raw is not None else env
    try:
        frame = target.render()
    except Exception:
        return None
    if isinstance(frame, np.ndarray):
        return frame.astype(np.uint8)
    if isinstance(frame, list) and frame and isinstance(frame[0], np.ndarray):
        return frame[0].astype(np.uint8)
    return None


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    if not frames:
        print("[render] no frames captured — skipping mp4 write", file=sys.stderr); return
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)


def _write_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    n = len(frames)
    indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    sample = [Image.fromarray(frames[i]) for i in indices]
    h = max(im.height for im in sample)
    w = sum(im.width for im in sample)
    sheet = Image.new("RGB", (w, h))
    x = 0
    for im in sample:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(str(path), quality=85)


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

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
    # Honor cfg.gpu_sim — GPU-batched simulators (IsaacLab / IsaacGym / ManiSkill GPU)
    # cannot be replaced with a CPU gymnasium.make for render. create_render_env knows
    # how to launch with cameras enabled when gpu_sim=True.

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    fps = int(cfg.get("fps", 30))
    max_steps = int(cfg.get("max_steps", 1000))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)

    env = create_render_env(cfg)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)
    state = torch.load(str(ckpt), map_location=cfg.device, weights_only=False)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    agent.actor.eval()

    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    ep_return = 0.0
    for step in range(max_steps):
        f = _grab_frame(env)
        if f is not None:
            frames.append(f)
        with torch.no_grad():
            if hasattr(agent.actor, "get_actions"):
                action = agent.actor.get_actions(obs, sample=False)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = agent.actor(obs)
        next_obs, reward, done, info = env.step(action)
        ep_return += float(reward.item() if isinstance(reward, torch.Tensor) else reward)
        if bool(done.item() if isinstance(done, torch.Tensor) else done):
            f = _grab_frame(env)
            if f is not None:
                frames.append(f)
            break
        obs = next_obs

    out_dir = ckpt.parent
    mp4 = out_dir / "render.mp4"
    sheet = out_dir / "render_contact_sheet.jpg"
    _write_mp4(frames, mp4, fps=fps)
    _write_contact_sheet(frames, sheet)
    print(f"[render] wrote {mp4} ({len(frames)} frames, return={ep_return:.4f})", flush=True)


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo": return "DiagGaussianMLPPolicy"
    if a == "sac": return "TanhDiagGaussianMLPPolicy"
    return "TanhMLPPolicy"


def _default_cri_class(algo: str) -> str:
    return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


if __name__ == "__main__":
    main()
