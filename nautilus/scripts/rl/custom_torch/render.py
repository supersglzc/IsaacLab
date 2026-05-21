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


def _grab_frame_via(target) -> np.ndarray | None:
    """Direct render bypass — call ``target.render()`` and normalize to ndarray."""
    if not hasattr(target, "render"):
        return None
    frame = target.render()
    if isinstance(frame, np.ndarray):
        return frame.astype(np.uint8)
    if isinstance(frame, list) and frame and isinstance(frame[0], np.ndarray):
        return frame[0].astype(np.uint8)
    return None


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
    # Cap at the env's max_episode_length when reachable so we never overshoot
    # one episode. Falls back to the cfg value (default 1000) for envs that
    # don't expose max_episode_length.
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
    # Tighten max_steps to the env's episode horizon when available — saves up
    # to 700 steps of pointless rendering on short-horizon envs (StackCup uses
    # max_episode_length=300; default cfg.max_steps=1000 would run 3.3× longer
    # than needed for a single rollout).
    raw_env = getattr(env, "_raw_env", env)
    env_max_len = getattr(getattr(raw_env, "unwrapped", raw_env), "max_episode_length", None)
    if env_max_len is not None and int(env_max_len) > 0:
        max_steps = min(max_steps, int(env_max_len) + 1)
        print(f"[render] capping max_steps at env.max_episode_length+1 = {max_steps}", flush=True)
    # Pre-load checkpoint to detect whether obs/value normalization was used
    # during training. If yes, force-enable on cfg.algo BEFORE building the
    # agent so the agent allocates obs_rms / value_rms ready to be populated.
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
        print("[render] restored obs_rms from checkpoint", flush=True)
    if "value_rms" in state and getattr(agent, "value_rms", None) is not None:
        agent.value_rms.load_state_dict(state["value_rms"])
        print("[render] restored value_rms from checkpoint", flush=True)
    agent.actor.eval()

    frames: list[np.ndarray] = []
    obs, _ = env.reset()

    # Position the IsaacLab viewer camera close to the action — the default
    # camera sits far above the env grid, so manipulation-scale motion (5–10 cm
    # of EE travel + 8 cm cups) is visually imperceptible. Pin a tabletop-eye
    # vantage of env 0 (the canonical render env). Override via cfg.viewer_eye
    # / cfg.viewer_target if the task has different geometry.
    # Wider isometric view — shows the full Franka (base at root, arm up to
    # ~1.2 m), the table (centered at ~0.5 m forward), and the cups together.
    # Eye is ~3 m diagonal from workspace center, lifted to 1.6 m so we look
    # SLIGHTLY DOWN onto the table — captures the robot base and the cup
    # tops in one frame.
    viewer_eye    = list(cfg.get("viewer_eye",    [2.5, 2.5, 1.6]))
    viewer_target = list(cfg.get("viewer_target", [0.30, 0.0, 0.4]))
    env_origin = raw_env.unwrapped.scene.env_origins[0].cpu().numpy().tolist()
    eye    = [viewer_eye[i]    + env_origin[i] for i in range(3)]
    target = [viewer_target[i] + env_origin[i] for i in range(3)]

    def _set_viewer_camera():
        try:
            raw_env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        except Exception:
            pass

    def _refresh_render():
        """Force a render pass so env.render() returns an up-to-date frame.
        ManagerBasedEnv.step() runs sim.step(render=False) — it does NOT update
        the offscreen render buffer. Without an explicit sim.render() call
        between step and env.render(), env.render() returns the buffer from
        the LAST render pass (i.e. a stale frame from when the renderer was
        first initialized). Calling sim.render() forces a fresh capture."""
        try:
            raw_env.unwrapped.sim.render()
        except Exception:
            pass

    _set_viewer_camera()
    print(f"[render] viewer camera   eye={eye}  target={target} (re-pinned every step)", flush=True)

    ep_return = 0.0
    action_norms: list[float] = []

    # Capture the post-reset frame ONCE before the step loop. Subsequent frames
    # are captured AFTER each env.step — this matches the IsaacLab gym render
    # contract: env.render() returns the buffer rendered during the most recent
    # sim.step(). Capturing BEFORE env.step gave stale frames (the buffer hadn't
    # been refreshed since the previous loop iteration's render).
    _set_viewer_camera()
    f0 = _grab_frame(env)
    if f0 is not None:
        frames.append(f0)

    for step in range(max_steps):
        with torch.no_grad():
            if cfg.get("random_action", False):
                # Bypass the policy entirely — uniform sample in [-1, 1] per dim.
                # Use this to verify the renderer captures motion: a random policy
                # SHOULD produce a clearly-moving robot. If the video still looks
                # static under random actions, the issue is in the renderer; if
                # only the trained policy looks static, the issue is the policy.
                act_dim = env.single_action_space.shape[-1] if hasattr(env, "single_action_space") else (
                    env.action_space.shape[-1] if hasattr(env.action_space, "shape") else
                    env.action_space.shape[0]
                )
                action = (torch.rand((env.num_envs, act_dim), device=cfg.device) * 2 - 1) * 1.5
            elif hasattr(agent.actor, "get_actions"):
                # Normalize obs via obs_rms BEFORE calling the actor — training's
                # rollout path (agent.get_actions in ppo.py) normalizes first, so
                # the actor was trained on normalized inputs. Skipping this step
                # silently produces near-zero performance even with the correct
                # checkpoint loaded.
                obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
                action = agent.actor.get_actions(obs_in, sample=True)  # stochastic eval
                if isinstance(action, tuple):
                    action = action[0]
            else:
                obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
                action = agent.actor(obs_in)
        # Debug: track action magnitude AND EE pose — if EE moves but video looks
        # static, the render-buffer is broken; if EE stays put despite non-zero
        # actions, the action pipeline / policy is the issue.
        if isinstance(action, torch.Tensor):
            action_norms.append(float(action.abs().mean().item()))
            if step in (0, 5, 20, 50, 100, 200, max_steps - 1):
                a = action[0] if action.dim() > 1 else action
                try:
                    _r = raw_env.unwrapped.scene["robot"]
                    _h = _r.body_names.index("panda_hand")
                    _ee_pos = _r.data.body_pos_w[0, _h].detach().cpu().numpy().round(3).tolist()
                except Exception:
                    _ee_pos = "n/a"
                print(f"[render] step={step:3d}  action.mean|abs|={action_norms[-1]:.4f}  "
                      f"action[0]={a.detach().cpu().numpy().round(3).tolist()}  ee={_ee_pos}", flush=True)
        next_obs, reward, done, info = env.step(action)
        # Capture frame AFTER env.step — env.step internally calls sim.step
        # which updates the render buffer; env.render() then reads that buffer.
        # Capturing BEFORE env.step would return the previous step's stale buffer.
        _set_viewer_camera()
        _refresh_render()
        # Try the FULLY UNWRAPPED env's render first — bypasses the gym.Wrapper
        # chain (DetailedRewardWrapper / IsaacLabVecAdapter). Some IsaacLab
        # versions cache the render buffer at the wrapper level and don't
        # invalidate on step. Falling back to the wrapper if unwrapped fails.
        unwrapped_target = getattr(raw_env, "unwrapped", raw_env)
        f = _grab_frame_via(unwrapped_target)
        if f is None:
            f = _grab_frame(env)
        if f is not None:
            frames.append(f)
        if isinstance(reward, torch.Tensor):
            ep_return += float(reward.mean().item())   # multi-env: mean across envs
        else:
            ep_return += float(reward)
        # IsaacLab auto-resets terminated envs internally on multi-env render;
        # don't break on first done — keep going for max_steps so the user sees
        # the full rollout horizon. (For single-env CPU fallback, done IS the
        # episode terminator and we DO break.)
        if isinstance(done, torch.Tensor):
            if done.numel() == 1 and bool(done.item()):
                break
        elif done:
            break
        obs = next_obs

    out_dir = ckpt.parent
    mp4 = out_dir / "render.mp4"
    sheet = out_dir / "render_contact_sheet.jpg"
    _write_mp4(frames, mp4, fps=fps)
    _write_contact_sheet(frames, sheet)
    a_arr = np.asarray(action_norms) if action_norms else np.zeros(1)
    print(
        f"[render] wrote {mp4} ({len(frames)} frames, return={ep_return:.4f}, "
        f"action|abs|.mean={a_arr.mean():.4f} max={a_arr.max():.4f} min={a_arr.min():.4f})",
        flush=True,
    )

    # Hard-exit BEFORE Kit shutdown. With Isaac Sim 5.1 + IsaacLab,
    # `sim_app.close()` (called via atexit hooks) hangs on USD stage detach —
    # the render hangs indefinitely after the MP4 is on disk. The MP4 + contact
    # sheet are flushed by `_write_*` above, so we can skip the cleanup safely.
    # This avoids the need for an external watchdog that pkills the python.
    import os as _os
    _os.sync()
    _os._exit(0)


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo": return "DiagGaussianMLPPolicy"
    if a == "sac": return "TanhDiagGaussianMLPPolicy"
    return "TanhMLPPolicy"


def _default_cri_class(algo: str) -> str:
    return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


if __name__ == "__main__":
    main()
