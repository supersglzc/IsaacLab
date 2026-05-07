"""
isaaclab — env factory for custom_torch train/eval/render.

GPU-sim branch (cfg.gpu_sim=True): builds an IsaacLab GPU-batched env via
``isaaclab_tasks.utils.parse_env_cfg`` + ``gymnasium.make``. Critical
ordering quirks IsaacLab imposes:

  1. ``isaaclab.app.AppLauncher`` MUST run before any other ``isaaclab`` /
     ``isaaclab_tasks`` / ``gymnasium.make`` import. We do this lazily inside
     ``create_env`` / ``create_render_env`` so importing this file at the top of
     train/eval/render does not trigger the launcher prematurely.
  2. ``_isaac_sim/setup_conda_env.sh`` MUST be sourced in the calling shell — it
     sets ``CARB_APP_PATH`` / ``EXP_PATH`` / ``ISAAC_PATH``. ``./isaaclab.sh -p``
     is NOT sufficient on its own. The smoke shim sources it; see
     <repo>/nautilus/rl-integration.md for the recipe.
  3. For rendering, ``args_cli.enable_cameras = True`` MUST be set BEFORE
     ``AppLauncher`` so Kit picks the RTX-rendering experience file.

CPU branch (cfg.gpu_sim=False): legacy gymnasium.vector path — kept for the
no-GPU regression case.

Exposes:
    create_env(cfg)         — vec/parallel env for train + eval (GPU-batched)
    create_render_env(cfg)  — single env for render.py (1 env, render_mode='rgb_array')
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))


# -----------------------------------------------------------------------------
# Per-benchmark single-env factory — auto-detects scripts/_<family>_env.py
# -----------------------------------------------------------------------------
def _make_single_env(task_id: str, render_mode=None, seed: int = 0):
    import gymnasium as gym
    env = None
    scripts_dir = REPO / "scripts"
    if scripts_dir.is_dir():
        for helper in scripts_dir.glob("_*_env.py"):
            # Some benchmarks (e.g. dm_control) check a mounted source dir into
            # the repo root that shadows the installed wheel. The helper module
            # is responsible for normalizing cwd, but we ALSO drop the repo root
            # from sys.path here so `from <pkg> import ...` inside the helper
            # finds the wheel, not the mount.
            _saved_path = list(sys.path)
            sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO]
            sys.path.insert(0, str(scripts_dir))
            try:
                module = __import__(helper.stem)
            except Exception:
                sys.path[:] = _saved_path
                continue
            for name in dir(module):
                if name.startswith("make_") and name.endswith("_env"):
                    fn = getattr(module, name)
                    try:
                        env = fn(task_id, render_mode=render_mode)
                    except TypeError:
                        env = fn(task_id)
                    break
            sys.path[:] = _saved_path
            break
    if env is None:
        env = gym.make(task_id, render_mode=render_mode)
    # The custom_torch algorithms (PPO/SAC/TD3 with MLP policies) require a
    # Box observation space. Wrap Dict / Tuple observations with FlattenObservation.
    if isinstance(env.observation_space, (gym.spaces.Dict, gym.spaces.Tuple)):
        env = gym.wrappers.FlattenObservation(env)
    return env


# -----------------------------------------------------------------------------
# CPU vec-env wrapper: numpy ↔ torch bridge over gymnasium.vector
# -----------------------------------------------------------------------------
class GymVecEnvWrapper:
    """Wraps a gymnasium.vector.VectorEnv so step/reset return torch tensors on `device`.

    Algorithm code (algo/sac.py etc.) assumes:
      env.step(action_torch)              -> (obs_torch, reward_torch, done_torch, info)
      env.reset()                         -> (obs_torch, extras) OR obs_torch
      env.observation_space.shape         -> shape of single env's obs (NOT including num_envs)
      env.action_space.shape[-1]          -> action dim
    """
    def __init__(self, vec_env, device: str | torch.device, episode_len: int | None = None):
        self.env = vec_env
        self.device = torch.device(device)
        self.num_envs = vec_env.num_envs
        # Expose single-env shapes (the algo code expects shape[0] == obs_dim, not num_envs).
        single_obs = vec_env.single_observation_space
        single_act = vec_env.single_action_space
        self._obs_shape = tuple(single_obs.shape)
        self._act_shape = tuple(single_act.shape)
        self.observation_space = type("_Spc", (), {"shape": self._obs_shape})()
        self.action_space = type("_Spc", (), {"shape": self._act_shape})()
        self.max_episode_length = episode_len

    def _to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.asarray(x)).float().to(self.device)
        if isinstance(x, (list, tuple)):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out[0], out[1]
        else:
            obs, info = out, {}
        return self._to_torch(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        # Clip to bounds for envs with bounded action_space (most continuous envs).
        try:
            low = self.env.single_action_space.low
            high = self.env.single_action_space.high
            action_np = np.clip(action_np, low, high)
        except Exception:
            pass
        step_out = self.env.step(action_np)
        if len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = np.logical_or(term, trunc)
            if isinstance(info, dict):
                info = {**info, "TimeLimit.truncated": self._to_torch(trunc).bool()}
        else:
            obs, reward, done, info = step_out
        return (self._to_torch(obs),
                self._to_torch(reward),
                self._to_torch(done).long(),
                info if isinstance(info, dict) else {})


def _make_cpu_vec_env(task_id: str, num_envs: int, seed: int):
    import gymnasium as gym
    factories = []
    for i in range(num_envs):
        s = seed + i
        def _f(task_id=task_id, s=s):
            env = _make_single_env(task_id, render_mode=None, seed=s)
            return env
        factories.append(_f)
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(factories)
    return gym.vector.AsyncVectorEnv(factories)


# -----------------------------------------------------------------------------
# IsaacLab GPU-batched adapter
# -----------------------------------------------------------------------------
# Module-level handles so the AppLauncher / SimulationApp survive past
# create_env() and stay alive for the duration of training. We launch lazily
# (only when the user actually asks for a GPU env) so unit tests on hosts
# without IsaacSim don't pay the cost.
_ISAAC_APP = None      # SimulationApp returned by AppLauncher.app
_ISAAC_LAUNCHED_WITH_CAMERAS = False  # remember the kit experience that was selected


def _ensure_isaac_app(enable_cameras: bool, headless: bool = True):
    """Idempotent: launch ``isaaclab.app.AppLauncher`` once with the requested kit
    experience. The Kit experience file is selected at launch time and CANNOT be
    swapped without restarting the process — so if the user already launched
    with ``enable_cameras=False`` and now asks for True, we raise loudly rather
    than silently lying about render support.
    """
    global _ISAAC_APP, _ISAAC_LAUNCHED_WITH_CAMERAS
    if _ISAAC_APP is not None:
        if enable_cameras and not _ISAAC_LAUNCHED_WITH_CAMERAS:
            raise RuntimeError(
                "AppLauncher was previously started with enable_cameras=False; "
                "Kit cannot swap experience files mid-process. Restart Python and "
                "set cfg.enable_cameras=True (or call create_render_env first)."
            )
        return _ISAAC_APP

    # Mirror scripts/run_random.py + scripts/render_random.py — the
    # known-good incantation on this host (per benchmark-generator's run-log).
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    # Use parse_known_args with empty argv so user CLI flags going to Hydra
    # do NOT leak into AppLauncher. Defaults: headless, no cameras.
    args_cli, _ = parser.parse_known_args([])
    args_cli.headless = bool(headless)
    args_cli.enable_cameras = bool(enable_cameras)

    launcher = AppLauncher(args_cli)
    _ISAAC_APP = launcher.app
    _ISAAC_LAUNCHED_WITH_CAMERAS = bool(enable_cameras)
    return _ISAAC_APP


class IsaacLabVecAdapter:
    """Bridge IsaacLab's gymnasium env → the (obs, reward, done, info) torch
    interface the custom_torch algos expect.

    Surface used by ac_base.py / explore_env:
      - observation_space.shape       — single-env shape (obs_dim,)
      - action_space.shape[-1]        — action_dim
      - num_envs                      — batch size
      - reset()                       → (obs_torch [N, obs_dim], info_dict)
      - step(action_torch)            → (obs, reward, done, info)
      - render()                      → np.uint8 frame (cameras-enabled mode only)
      - close()                       — close underlying env

    IsaacLab's observation_space is a Dict with 'policy' (and optionally
    'critic') key; we extract the 'policy' branch and expose its single-env
    shape so the MLP policy gets the right input dim.
    """

    def __init__(self, env, device: str | torch.device, max_episode_length: int | None = None):
        self.env = env
        self.device = torch.device(device)
        # IsaacLab exposes num_envs on the unwrapped env.
        self.num_envs = int(getattr(env.unwrapped, "num_envs", 1))

        single_obs = env.unwrapped.single_observation_space
        if hasattr(single_obs, "spaces") and "policy" in single_obs.spaces:
            policy_space = single_obs["policy"]
        else:
            policy_space = single_obs
        # Some IsaacLab envs nest the policy obs as another Dict. We don't
        # support that today — surface it loudly.
        if hasattr(policy_space, "spaces"):
            raise RuntimeError(
                f"Unsupported nested observation space {type(policy_space).__name__}; "
                "expected a Box at single_observation_space['policy']. "
                "File a bug or wrap with FlattenObservation upstream."
            )
        self._obs_shape = tuple(policy_space.shape)
        single_act = env.unwrapped.single_action_space
        self._act_shape = tuple(single_act.shape)
        # Lightweight stand-in spaces — ac_base.py only reads .shape.
        self.observation_space = type("_Spc", (), {"shape": self._obs_shape})()
        self.action_space = type("_Spc", (), {"shape": self._act_shape})()
        self.max_episode_length = max_episode_length
        # Action bounds for clipping (IsaacLab single_action_space is typically Box).
        self._act_low = torch.as_tensor(getattr(single_act, "low", -1.0),
                                        dtype=torch.float32, device=self.device)
        self._act_high = torch.as_tensor(getattr(single_act, "high", 1.0),
                                         dtype=torch.float32, device=self.device)

    @staticmethod
    def _extract_policy_obs(obs):
        """IsaacLab returns obs as a dict with 'policy' (and optionally 'critic')."""
        if isinstance(obs, dict):
            if "policy" in obs:
                return obs["policy"]
            # Fallback: take the first value if the only key isn't 'policy'.
            return next(iter(obs.values()))
        return obs

    def _to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out[0], out[1] if len(out) > 1 else {}
        else:
            obs, info = out, {}
        obs_t = self._to_torch(self._extract_policy_obs(obs)).float()
        return obs_t, info if isinstance(info, dict) else {}

    def step(self, action):
        # IsaacLab expects a (num_envs, action_dim) torch tensor on env device.
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device).float()
        action = torch.clamp(action, self._act_low, self._act_high)
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = torch.logical_or(self._to_torch(term).bool(),
                                    self._to_torch(trunc).bool())
            if isinstance(info, dict):
                info = {**info, "TimeLimit.truncated": self._to_torch(trunc).bool()}
        else:
            obs, reward, done, info = step_out
            done = self._to_torch(done).bool()
        obs_t = self._to_torch(self._extract_policy_obs(obs)).float()
        reward_t = self._to_torch(reward).float()
        return obs_t, reward_t, done.long(), info if isinstance(info, dict) else {}

    def render(self):
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def _build_isaaclab_env(task: str, num_envs: int, device: str, render_mode=None):
    """Common path for create_env / create_render_env. Routes through
    ``scripts/_isaaclab_env.py`` when present (per-term reward visibility
    via /add-reward-log); otherwise builds the gym env directly."""
    helper = REPO / "scripts" / "_isaaclab_env.py"
    if helper.is_file():
        scripts_dir = str(REPO / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import _isaaclab_env
        return _isaaclab_env.make_isaaclab_env(
            task, num_envs=num_envs, device=device, render_mode=render_mode)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  — registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=True)
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    return env


# -----------------------------------------------------------------------------
# Public factories
# -----------------------------------------------------------------------------
def create_env(cfg):
    task = str(cfg.get("task"))
    n = int(cfg.get("num_envs", cfg.get("n_envs", 1)))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))

    if gpu_sim:
        # IsaacLab path: launch Kit (no cameras for train), then build the env.
        _ensure_isaac_app(enable_cameras=False, headless=True)
        env = _build_isaaclab_env(task, num_envs=n, device=device, render_mode=None)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        # Episode-length hint for the agent's tracker (best-effort).
        episode_len = int(getattr(env.unwrapped, "max_episode_length", 0)) or None
        return IsaacLabVecAdapter(env, device=device, max_episode_length=episode_len)

    vec = _make_cpu_vec_env(task, n, seed)
    return GymVecEnvWrapper(vec, device=device)


def create_render_env(cfg):
    """Single env (n=1) with rendering enabled — used by render.py and the
    in-train video logger. For IsaacLab this requires AppLauncher to be
    started with ``enable_cameras=True`` BEFORE any sim build."""
    task = str(cfg.get("task"))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))

    if gpu_sim:
        _ensure_isaac_app(enable_cameras=True, headless=True)
        env = _build_isaaclab_env(task, num_envs=1, device=device, render_mode="rgb_array")
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        adapter = IsaacLabVecAdapter(env, device=device)
        adapter._raw_env = env  # for frame-extraction helpers in render.py / train.py
        return adapter

    # CPU fallback (unchanged) — gymnasium single-env wrapped in a 1-env vec.
    env = _make_single_env(task, render_mode="rgb_array", seed=seed)
    import gymnasium as gym

    class _SingleVec:
        num_envs = 1
        single_observation_space = env.observation_space
        single_action_space = env.action_space

        def reset(self, **kw):
            o = env.reset(**kw)
            obs = o[0] if isinstance(o, tuple) else o
            return np.expand_dims(np.asarray(obs), 0), (o[1] if isinstance(o, tuple) else {})

        def step(self, action_np):
            a = action_np[0] if action_np.ndim > 1 else action_np
            out = env.step(a)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                return (np.expand_dims(obs, 0), np.array([r]),
                        np.array([term]), np.array([trunc]), info)
            obs, r, done, info = out
            return np.expand_dims(obs, 0), np.array([r]), np.array([done]), info

        def render(self):
            return env.render()

    wrapped = GymVecEnvWrapper(_SingleVec(), device=device)
    wrapped._raw_env = env  # render.py uses this for frame extraction
    return wrapped
