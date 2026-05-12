# Reward Generator — `Isaac-Push-Block-Franka-v0`

| Field | Value |
|---|---|
| task_id | Isaac-Push-Block-Franka-v0 |
| slug | isaac-push-block-franka-v0 |
| benchmark_family | isaaclab-manager-based |
| reward_placeholder_path | `push_env_cfg.py:RewardsCfg` (lines 140–149) + `mdp/rewards.py` (whole file) |
| dry_run | false |
| started_at | 2026-05-09T00:00:00Z |
| finished_at | 2026-05-09T20:01:30Z |
| status | pass |

## Pre-flight

- `task-implementation.md` exists — confirmed.
- `handoff-task-generator.md` exists — confirmed.
- `spec.json` exists — confirmed.
- Build check: deferred until first smoke (would import IsaacLab + boot sim — expensive). Source files exist on disk per task-generator's pass; smoke will fail fast if registration is broken.

## Inputs read

| Source | Purpose |
|---|---|
| `task-implementation.md` §6 (lines 544–650) | family idiom, code template, decisions, smoke command |
| `handoff-task-generator.md` | §6 placeholder paths + suggested term layout |
| `dry-run-reward.md` | starting blueprint (decisions + diff) |
| `push/push_env_cfg.py` | live placeholder body |
| `push/mdp/rewards.py` | live placeholder file |
| `manipulation/lift/mdp/rewards.py` | source idiom for `object_ee_distance` + `object_goal_distance` |

## Decisions resolved

| Decision | Value | Source |
|---|---|---|
| primary_term | `block_to_goal_tracking` (custom `block_to_goal_distance`, std=0.3, weight=16.0) | canonical-mirror (Lift) |
| secondary_term | `block_to_goal_tracking_fine_grained` (std=0.05, weight=5.0) | canonical-mirror (Lift) |
| reaching_term | `reaching_block` (custom `object_ee_distance_body` via `body_pos_w[panda_hand]`, std=0.1, weight=1.0) | Lift adapted (Push scene has no `ee_frame` FrameTransformer) |
| success_indicator | `success` (custom `block_at_goal`, threshold=0.05, weight=0.0 logging-only) | user description ("<5cm") |
| action_rate | `mdp.action_rate_l2`, weight=-1e-4 | canonical-mirror |
| joint_vel | `mdp.joint_vel_l2`, weight=-1e-4 | canonical-mirror |
| shaping_vs_sparse | shaping (dense) | default |
| composer | `sum` | family-default |
| curriculum | none | default (short 200-step horizon) |
| per_term_logging | yes | family-default (manager-based) |

No `AskUserQuestion` fired.

## Files written

| Path | Action | Notes |
|---|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` | overwrite | replaced placeholder with 3 helpers: `object_ee_distance_body`, `block_to_goal_distance`, `block_at_goal` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | edit `RewardsCfg` body | replaced single `placeholder` term with 6 real terms (3 positive shaping + 1 zero-weight success + 2 negative regularizers) |

No §1–§5 file touches required. `mdp/__init__.py` re-exports via `from .rewards import *`, so all three new helpers (`object_ee_distance_body`, `block_to_goal_distance`, `block_at_goal`) become available as `mdp.<name>` in `push_env_cfg.py` automatically. `placeholder_zero` was removed from `rewards.py` and `RewardsCfg` no longer references it — consistent.

## §6 smoke

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -u -c "
import argparse, torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=4, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
env.reset()
rewards = []
for _ in range(30):
    a = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
    obs, rew, term, trunc, info = env.step(a)
    assert torch.isfinite(rew).all(), f'non-finite reward: {rew}'
    rewards.append(rew.detach().cpu().clone())
    if 'detailed_reward' in info:
        per_term = sum(v for k, v in info['detailed_reward'].items() if k != 'total')
        assert torch.allclose(per_term, rew, atol=1e-5), 'detailed_reward terms do not sum to total'
all_rew = torch.stack(rewards)
print('REWARD_OK: mean=%.4f std=%.4f finite_steps=%d/%d' % (
    all_rew.mean().item(), all_rew.std().item(), int(torch.isfinite(all_rew).all(dim=-1).sum()), len(rewards)))
assert all_rew.std().item() > 0.0, 'reward variance is zero'
env.close()
simulation_app.close()
"
```

**Last 50 lines of stdout (filtered to relevant manager output + result):**

```
[INFO] Reward Manager:  <RewardManager> contains 6 active terms.
+-------------------------------------------------------+
|                  Active Reward Terms                  |
+-------+-------------------------------------+---------+
| Index | Name                                |  Weight |
+-------+-------------------------------------+---------+
|   0   | reaching_block                      |     1.0 |
|   1   | block_to_goal_tracking              |    16.0 |
|   2   | block_to_goal_tracking_fine_grained |     5.0 |
|   3   | success                             |     0.0 |
|   4   | action_rate                         | -0.0001 |
|   5   | joint_vel                           | -0.0001 |
+-------+-------------------------------------+---------+

[INFO]: Completed setting up the environment...
AFTER_GYM_MAKE
AFTER_RESET
REWARD_OK: mean=0.2854 std=0.0942 finite_steps=30/30
SMOKE_DONE
```

**Verdict:** pass

- 30/30 steps produced finite rewards.
- Reward std=0.0942 > 0 (variance assertion satisfied — random actions generate state variance, the tanh kernels amplify it).
- Composer assertion: `info["detailed_reward"]` was either absent (passthrough mode — no `_DetailedRewardWrapper` active) or summed to `rew` (no AssertionError fired). Either case is consistent with `composer="sum"` and the family default; once `/add-reward-log` wraps the env, the per-term decomposition will be exact.
- Reward Manager confirmed all 6 terms wired with the intended weights.

## Iteration log

| Attempt | Diagnosis | Patch | Result |
|---|---|---|---|
| 1 | n/a — first run passed | n/a | pass |

---

## Verdict

| Field | Value |
|---|---|
| status | pass |
| composer | sum |
| per_term_logging | yes |
| iterations | 1 |
| files_written | 2 (source) + 2 (workspace docs) |
| finished_at | 2026-05-09T20:01:30Z |
| handoff | `nautilus/task-creation/isaac-push-block-franka-v0/handoff-reward-generator.md` |
