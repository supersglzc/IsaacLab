"""IsaacLab — env factory with per-term reward visibility wrapper.

Provides ``make_isaaclab_env(task_id, num_envs, device, render_mode)`` that
builds the gym-registered IsaacLab env and wraps it with
``_DetailedRewardWrapper``. The wrapper picks one of three coverage modes
per step depending on what the underlying env exposes:

1. **Manager-based** (``env.unwrapped.reward_manager`` exists). Reads
   ``manager._step_reward`` (shape ``(num_envs, num_terms)``) plus
   ``manager.active_terms``. Per-term value = ``step_reward[:, i] * step_dt``;
   the composer-sum invariant ``Σ terms == env_reward`` holds by
   construction and is asserted each step.
   ``info["reward_composer"] = "sum"``.

2. **Direct-with-logs_rew** (no ``reward_manager`` but env writes
   ``logs_rew_<term>`` scalars onto ``info`` — Factory / Forge /
   Industreal pattern via ``_log_*_metrics``). Surfaces those scalars
   broadcast to ``(num_envs,)`` as per-term diagnostic curves alongside
   ``total``. The values are pre-scale means, so they DO NOT sum to
   ``env_reward``; ``info["reward_composer"] = "diagnostic"`` advertises
   this and downstream sanity checks should skip the equality assertion
   for this composer.

3. **Direct passthrough** (no manager, no ``logs_rew_*``). Emits only
   ``info["detailed_reward"] = {"total": env_reward}``;
   ``info["reward_composer"] = "sum"``.

PhysX override: ``gpu_collision_stack_size`` defaults to 2**26 (64 MiB)
in IsaacLab's ``PhysxCfg``. Contact-heavy Direct envs (Factory, Forge,
Industreal) at num_envs ≥ 1024 overflow this; the factory bumps to 2**30
when below.

Prerequisite: ``isaaclab.app.AppLauncher`` MUST already be running before
this module is used. Importing this module does not launch Kit.

Rendered by /add-reward-log from
${CLAUDE_PLUGIN_ROOT}/skills/add-reward-log/templates/isaaclab_env_helper.py.template
"""
from __future__ import annotations

import torch


def _make_detailed_reward_wrapper():
    """Build the gym.Wrapper class lazily — gymnasium isn't imported at module load."""
    import gymnasium as gym

    class _DetailedRewardWrapper(gym.Wrapper):
        """Surface per-term reward values from IsaacLab's RewardManager.

        Each step:
          1. Take env's native (obs, reward, term, trunc, info) — UNCHANGED.
          2. Read ``reward_manager._step_reward`` (shape [num_envs, num_terms])
             and ``active_terms``; per-term per-step value
             = step_reward[:, i] * step_dt.
          3. Verify ``torch.allclose(Σ terms, env_reward, atol=1e-4, rtol=1e-3)``.
          4. Stash {term_name: per-env tensor} on ``info["detailed_reward"]``.

        Tasks without a RewardManager (Direct envs) → passthrough with
        ``info["detailed_reward"] = {"total": env_reward}``.
        """

        def __init__(self, env, task_id: str):
            super().__init__(env)
            self._task_id = task_id
            unwrapped = env.unwrapped
            self._manager = getattr(unwrapped, "reward_manager", None)
            self._step_dt = float(getattr(unwrapped, "step_dt", 0.0))
            self._composer = "sum"

        def step(self, action):
            obs, env_rew, term, trunc, info = self.env.step(action)
            info = dict(info) if isinstance(info, dict) else {}

            # Path 1 — manager-based env: exact per-term decomposition.
            if self._manager is not None and self._step_dt > 0.0:
                step_reward = self._manager._step_reward  # (num_envs, num_terms)
                names = list(self._manager.active_terms)
                terms = {
                    names[i]: step_reward[:, i] * self._step_dt
                    for i in range(len(names))
                }
                composed = sum(terms.values())
                if not torch.allclose(composed.float(), env_rew.float(),
                                      atol=1e-4, rtol=1e-3):
                    diff = (composed - env_rew).abs().max().item()
                    raise AssertionError(
                        f"detailed_reward sum mismatch for {self._task_id}: "
                        f"max |Σ terms - env_reward| = {diff:.3e} "
                        f"(num_envs={env_rew.shape[0]}, num_terms={len(names)})"
                    )
                info["detailed_reward"] = terms
                info["reward_composer"] = "sum"
                return obs, env_rew, term, trunc, info

            # Path 2 — Direct env that writes `logs_rew_<term>` scalars onto info
            # (Factory / Forge / Industreal _log_*_metrics convention). Surface
            # those as diagnostic per-term curves; do NOT enforce composer-sum
            # (the values are pre-scale means; their sum ≠ env_reward).
            log_keys = [k for k in info if isinstance(k, str) and k.startswith("logs_rew_")]
            if log_keys:
                if isinstance(env_rew, torch.Tensor) and env_rew.ndim >= 1:
                    n = env_rew.shape[0]
                    dev = env_rew.device
                else:
                    n = 1
                    dev = "cpu"
                terms: dict = {}
                for k in log_keys:
                    name = k[len("logs_rew_"):]
                    v = info[k]
                    if isinstance(v, torch.Tensor):
                        if v.ndim == 0:
                            terms[name] = v.expand(n).clone()
                        elif v.shape[0] == n:
                            terms[name] = v
                        else:
                            continue
                    else:
                        try:
                            terms[name] = torch.full((n,), float(v), device=dev)
                        except (TypeError, ValueError):
                            continue
                terms["total"] = env_rew
                info["detailed_reward"] = terms
                info["reward_composer"] = "diagnostic"
                return obs, env_rew, term, trunc, info

            # Path 3 — passthrough (no decomposition source).
            info["detailed_reward"] = {"total": env_rew}
            info["reward_composer"] = "sum"
            return obs, env_rew, term, trunc, info

    return _DetailedRewardWrapper


def make_isaaclab_env(task_id: str, num_envs: int = 1, device: str = "cpu",
                      render_mode=None):
    """Build a gym-registered IsaacLab env with per-term reward visibility.

    AppLauncher must already be running (caller's responsibility).

    PhysX override: `gpu_collision_stack_size` defaults to 2**26 (64 MiB)
    in IsaacLab's `PhysxCfg`. Contact-heavy Direct envs (Factory, Forge,
    Industreal) at num_envs ≥ 1024 overflow this — symptom is a flood of
    ``[Error] PhysX error: PxGpuDynamicsMemoryConfig::collisionStackSize
    buffer overflow detected ... Contacts have been dropped``. The sim
    keeps running but silently drops contacts, which corrupts assembly
    tasks. We bump to 2**30 (1 GiB) when the cfg's value is below that —
    no-op when the user already set a larger value.
    """
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  — registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg

    # `use_fabric=False`: the pip-installed isaacsim 5.1 cloner errors with
    # `[isaacsim.core.cloner.impl.cloner] Failed to clone in Fabric` and
    # silently aborts scene init when this is True against the supersglzc
    # IsaacLab fork (2026-05-06). Falling back to non-Fabric cloning works
    # cleanly with a small perf cost. Flip back to True if/when the cloner
    # is fixed upstream.
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, use_fabric=False)
    sim = getattr(env_cfg, "sim", None)
    physx = getattr(sim, "physx", None) if sim is not None else None
    if physx is not None and getattr(physx, "gpu_collision_stack_size", 0) < 2**30:
        physx.gpu_collision_stack_size = 2**30
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    Wrapper = _make_detailed_reward_wrapper()
    return Wrapper(env, task_id)
