# 01 — RL integration

- **Generator:** rl-integration-generator (Layer 6a)
- **Date:** 2026-05-05T22:35:00+02:00
- **Algorithm source:** custom_torch (vendored PQL/DDiffPG-style PyTorch tree under `nautilus/scripts/rl/custom_torch/`)
- **Algorithm slug:** custom_torch
- **gpu_sim (from benchmark-spec):** true → parallel configs (`{ppo,sac,td3}.parallel.yaml`) used for smoke
- **Tasks wired:** Isaac-Cartpole-Direct-v0, Isaac-Cartpole-v0, Isaac-Velocity-Flat-Anymal-C-v0, Isaac-Reach-Franka-v0, Isaac-Lift-Cube-Franka-v0 (the five `L1 pass` / `L1+L2 pass` tasks; the three Nucleus-asset failures from benchmark-spec.json are excluded)
- **Smoke task:** Isaac-Cartpole-Direct-v0 (smallest, L1+L2 pass)

## Tools invoked

1. `scripts/rl-integration-generator/render_rl_suite.py` — rendered scaffold
2. `scripts/rl-integration-generator/validate_rl_suite.py` — T1/T2/T3/T4 all pass
3. Per-algorithm smoke (T1 train, T2 eval, T3 render, T4 plot, T5 log) on Isaac-Cartpole-Direct-v0 with budget `num_envs=4`, `total_timesteps=200`, `batch_size=128`, `warm_up=16` (off-policy)

## Smoke verdict

| Algo | T1 train | T2 eval | T3 render | T4 plot | T5 log | Trial dir |
|---|---|---|---|---|---|---|
| ppo | PASS (ckpt 544KB, exit 0) | PASS (mean_return=5.36, n=2) | PASS (render.mp4 = 2.9MB) | PASS (7 PNG) | PASS (TB=1, jsonl=49 lines) | `outputs/ppo_Isaac-Cartpole-Direct-v0_20260505-220444/` |
| sac | PASS (ckpt 817KB, exit 0) | PASS (mean_return=-5.57, n=2) | PASS (render.mp4 = 1.5MB) | PASS (7 PNG) | PASS (TB=1, jsonl=238 lines) | `outputs/sac_Isaac-Cartpole-Direct-v0_20260505-221754/` |
| td3 | PASS (ckpt 816KB, exit 0) | PASS (mean_return=-3.12, n=3) | PASS (render.mp4 = 1.6MB) | PASS (5 PNG) | PASS (TB=1, jsonl=170 lines) | `outputs/td3_Isaac-Cartpole-Direct-v0_20260505-223416/` |

## IsaacLab-specific surgery (rendered files only — NOT shipped back to template)

`nautilus/scripts/rl/custom_torch/env_wrapper.py` was patched in-place to fill in the
`gpu_sim=True` branch with the IsaacLab incantation that already works in
`scripts/run_random.py` / `scripts/render_random.py`. Concretely:

- Added `_ensure_isaac_app(enable_cameras, headless)` — idempotent module-level launcher of `isaaclab.app.AppLauncher`. The Kit experience file is selected at launch time and CANNOT be swapped (e.g. cameras-on vs. cameras-off) without restarting Python; the helper raises a clear error in that case.
- Added `IsaacLabVecAdapter` — bridges IsaacLab's `Dict({"policy": ..., "critic": ...})` observation space + 5-tuple gymnasium API to the `(obs_torch, reward_torch, done_torch, info)` interface the custom_torch algos expect. Extracts `obs["policy"]`, exposes single-env `observation_space.shape == (obs_dim,)` and `action_space.shape[-1]`, clips actions to `single_action_space` bounds.
- `create_env(cfg)` and `create_render_env(cfg)` honor `cfg.gpu_sim`: GPU branch uses `parse_env_cfg(...)` + `gym.make(task, cfg=env_cfg)` with `use_fabric=True`, render branch additionally sets `args_cli.enable_cameras = True` BEFORE the launcher.

This is benchmark-specific glue, not a generic renderer fix — left out of the `custom_torch` template intentionally. The template's `NotImplementedError` placeholder still surfaces clearly when a NEW GPU-sim benchmark is being wired.

## Plugin template patches (auto-update memory directive)

Three template-level bugs were fixed in-place (not benchmark-specific):

1. `scripts/rl-integration-generator/render_rl_suite.py` — guard the `docker-compose.rl.yaml.template` read with `compose_tpl.exists()`. The template was removed in commit `8d9ecfd "remove docker"` but the renderer still expected it. Also dropped the now-dead `compose` field from the JSON receipt.

2. `scripts/rl-integration-generator/validate_rl_suite.py` — dropped `docker/docker-compose.rl.yaml` from the T1 expected-files list (same root cause as #1).

3. `templates/rl-integration-generator/custom_torch/scripts/{train,render}.py.template` — both files used to force `cfg.gpu_sim = False` before calling `create_render_env(...)`. That's wrong for any GPU-batched simulator (IsaacLab/IsaacGym/ManiSkill GPU): a CPU `gym.make(task)` with no `cfg=` argument fails with `__init__() missing 1 required positional argument: 'cfg'`. Patched to keep `cfg.gpu_sim` as-is and let `create_render_env` decide; the `try/except` already in place handles the case where the render-mode env can't be built (e.g. AppLauncher kit-experience mismatch in IsaacLab — the standalone `render.py` entry point handles that case correctly because it starts a fresh Python process).

The rendered files at `<repo>/nautilus/scripts/rl/custom_torch/{train,render}.py` were patched with the same fix so this run is unblocked.

## Operational notes for the user

The `setup_conda_env.sh` requirement (documented in `nautilus/benchmark.md`) carries forward to all RL invocations. Wrap any train/eval/render CLI in:

```bash
cd /home/steven/code/agentic/IsaacLab
source _isaac_sim/setup_conda_env.sh   # MUST come first; sets EXP_PATH / CARB_APP_PATH / ISAAC_PATH
export OMNI_KIT_ALLOW_ROOT=1            # only matters when running as root
.venv/bin/python -u nautilus/scripts/rl/custom_torch/train.py \
    --config-name=ppo.parallel task=Isaac-Cartpole-Direct-v0 ...
```

`./isaaclab.sh -p` is NOT a substitute on its own — verified again during this run.

Hydra config-name selection: pass `--config-name=ppo.parallel` (or `sac.parallel` / `td3.parallel`) for the GPU-batched defaults. The plain `{ppo,sac,td3}.yaml` configs are SB3-CPU-flavored and will hit the `gpu_sim=False` CPU branch (which IsaacLab does NOT support — the env requires `parse_env_cfg`).

Override the smoke-bait keys at CLI: `num_envs=2048`, `batch_size=8192`, `warm_up=2048`, `total_timesteps=10_000_000`. The defaults in the parallel configs are reasonable starting points.

Per-run `wandb` opt-in: append `wandb=<project_name>` to enable W&B logging (credentials managed by `/nautilus:wandb-setup`); default is `wandb=null` (off).

Render: `nautilus/scripts/rl/custom_torch/render.py --config-name=<algo>.parallel checkpoint=<path> +max_steps=N`. Note `max_steps` requires the `+` Hydra prefix (the parallel configs only declare `render_max_steps`, not `max_steps`). Render must be a fresh Python process — Kit's experience file (cameras vs. no-cameras) is locked at AppLauncher start time; you can't switch mid-run.

Shutdown hangs: the IsaacLab Kit takes ~30s on shutdown; `OMNI_KIT_FAST_SHUTDOWN=1` reduces but does not eliminate this. The MP4 is written before shutdown, so artifact-level checks pass even when the process appears to hang.
