# Handoff — `reward-generator` → `dr-generator`

**Task:** `Isaac-Push-Block-Franka-v0`
**Slug:** `isaac-push-block-franka-v0`
**Mode:** real (`dry_run=false`)
**Phase:** §6 complete; §6 smoke passed live on first attempt (`REWARD_OK: mean=0.2854 std=0.0942 finite_steps=30/30`)

## §6 reward surface

| Field | Value |
|---|---|
| Reward composition root | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py:RewardsCfg` |
| Line range (post-patch) | lines 140–192 (was 140–149 placeholder; 6 terms now) |
| Per-term reward functions | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` (whole file rewritten — 3 helpers) |
| Composer | `sum` (RewardManager sums weighted terms) |
| Per-term logging | `yes` (manager-based; `info["detailed_reward"]` is auto-populated by `_DetailedRewardWrapper` reading `env.unwrapped.reward_manager._step_reward`) |

## Term list

| # | Term name | Function | Weight | Sign | Notes |
|---|---|---|---|---|---|
| 1 | `reaching_block` | `mdp.object_ee_distance_body` (custom, body-position based) | 1.0 | + | tanh kernel, std=0.1, EE = `body_pos_w[panda_hand]` |
| 2 | `block_to_goal_tracking` | `mdp.block_to_goal_distance` (custom) | 16.0 | + | tanh kernel, std=0.3, command_name="object_pose" |
| 3 | `block_to_goal_tracking_fine_grained` | `mdp.block_to_goal_distance` (custom) | 5.0 | + | same function, std=0.05 |
| 4 | `success` | `mdp.block_at_goal` (custom) | 0.0 | logging-only | binary indicator, threshold=0.05 m — surfaces in `info["detailed_reward"]["success"]` |
| 5 | `action_rate` | `mdp.action_rate_l2` (shared) | -1e-4 | − | jerky-action penalty |
| 6 | `joint_vel` | `mdp.joint_vel_l2` (shared) | -1e-4 | − | excess-joint-velocity penalty |

Total terms: **6** (3 contributing positive shaping + 1 zero-weight logging hook + 2 regularizers). Live RewardManager confirmed all 6 active with the intended weights.

## §1–§5 surface touches

**None.** The §6 wiring did not require modifying any §1–§5 cfg classes (`SceneCfg`, `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `CommandsCfg`, `TerminationsCfg`). All references resolve cleanly:

- `SceneEntityCfg("robot", body_names=["panda_hand"])` — `panda_hand` is a Franka body (used by Reach + Lift the same way).
- `command_name="object_pose"` — wired by §4 `CommandsCfg.object_pose`.
- `mdp.action_rate_l2` / `mdp.joint_vel_l2` — shared library (`isaaclab.envs.mdp`).
- `mdp.object_ee_distance_body` / `mdp.block_to_goal_distance` / `mdp.block_at_goal` — task-local, exposed via `push/mdp/__init__.py`'s wildcard re-export.

## §7 DR placeholder — for `dr-generator`

| File | Section | Notes |
|---|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | `class EventCfg` (lines 114–137) | Currently holds only the two §3 `mode="reset"` terms (`reset_robot_joints`, `reset_object_position`). Append `mode="startup"` and/or `mode="interval"` DR terms here. |

`reward-generator` did not touch `EventCfg`. Reward is composer-clean (`sum`), so any DR axis added by `dr-generator` (physics_material, block_mass, push perturbation) does NOT need to coordinate with the reward computation — the new terms operate on physical state at reset/startup/interval and the reward functions read the same state at evaluation time, no cross-cutting concerns.

If the user's description has no DR signal and the canonical example (Lift) has none either, `dr-generator` may legitimately return `skipped`. The user's description ("Franka pushes a small block from table center to a target marker. Success at <5cm; horizon 200.") contains no robustness / sim-to-real / randomization language, so `skipped` is a reasonable outcome.

## Open questions for `dr-generator`

None blocking. The §6 reward is fully self-contained; no DR-coupling decisions were deferred from this phase.

## Files written by §6

| Path | Purpose |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` | overwritten — placeholder removed; 3 helpers added (`object_ee_distance_body`, `block_to_goal_distance`, `block_at_goal`) |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | `RewardsCfg` body replaced (placeholder → 6 real terms) |
| `<task_dir>/reward-history.md` | verbose process log (decisions + diff + smoke + verdict) |
| `<task_dir>/handoff-reward-generator.md` | this file |
