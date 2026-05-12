# Handoff — `dr-generator` → user (final)

**Task:** `Isaac-Push-Block-Franka-v0`
**Slug:** `isaac-push-block-franka-v0`
**Mode:** dry-run (no source files modified — diff in `dry-run-dr.md`)
**Phase:** §7 complete (smoke skipped per dry-run contract)
**Pipeline status:** task-generator pass · reward-generator pass · dr-generator pass — chain complete.

> This is the LAST handoff in the `/nautilus:task-creation` chain. There is no downstream agent. Read this, then decide whether to re-run `/nautilus:task-creation` with `dry_run=false` to apply all three diffs (§1–§5, §6, §7).

## §7 DR surface

| Field | Value |
|---|---|
| DR file | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` |
| DR class | `EventCfg` (extended in place — not a new class) |
| Line range (post-patch) | the existing `EventCfg` (≈ lines 148–171 in `dry-run.md`) gains 2 startup terms, ending at ≈ line 200 in the patched file |

## Axes randomized

| Axis | Function | Range | Schedule |
|---|---|---|---|
| Friction (robot fingers) | `mdp.randomize_rigid_body_material` on `robot.body_names="panda_finger_.*"` | static_friction × U(0.8, 1.2), dynamic_friction × U(0.8, 1.2), restitution = 0 | `startup` |
| Block mass | `mdp.randomize_rigid_body_mass` on `object` | mass × U(0.8, 1.2) (multiplicative scale) | `startup` |

Schedule: **startup only** (per-env-instance, fixed for the env's lifetime). No interval pushes, no per-reset force/torque — the user's description has no robustness or disturbance language.

## §7 smoke

Skipped per dry-run contract (`dry_run=true`). At non-dry-run time the smoke from `task-implementation.md` §7 runs verbatim with `<NewTaskID>` → `Isaac-Push-Block-Franka-v0`. Expected: `DR_OK: max_obs_divergence > 1e-4` (friction randomization on contact bodies feeds the per-step `joint_pos_rel` / `joint_vel_rel` obs terms within 30 random-action steps).

## Files written

| Path | Purpose |
|---|---|
| `<task_dir>/dr-history.md` | verbose process log (skip-or-wire decision + axes resolved + verdict) |
| `<task_dir>/dry-run-dr.md` | full §7 diff (`EventCfg` body extension; before/after blocks) |
| `<task_dir>/handoff-dr-generator.md` | this file |

No `source/isaaclab_tasks/...` files were modified — dry-run.

## What to test next

1. **Apply the diffs.** Re-run `/nautilus:task-creation name=Isaac-Push-Block-Franka-v0 description="..."` with `dry_run=false` (or hand-apply the three files: `dry-run.md` then `dry-run-reward.md` then `dry-run-dr.md`). The patches are non-overlapping by class.
2. **Build sanity.** Confirm `Isaac-Push-Block-Franka-v0` registers and builds:

   ```bash
   /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "import gymnasium as gym, isaaclab_tasks; gym.make('Isaac-Push-Block-Franka-v0'); print('build ok')"
   ```

3. **Train smoke.** Quick PPO sanity run to verify reward + DR don't tank learning:

   ```bash
   /nautilus:rl-run task=Isaac-Push-Block-Franka-v0 algorithm=ppo total_timesteps=20000
   ```

   Look for `block_to_goal_tracking` reward term trending up and `success` (zero-weight indicator) climbing above zero in `info["detailed_reward"]`.

4. **DR active check.** During training, inspect a few env instances at startup and confirm finger friction values vary across `env_id` and block masses vary across `env_id`. (IsaacLab logs this if you set `cfg.events.physics_material.params['num_buckets']` lower; defaults are silent.)

## Open items / known limitations

- **No friction randomization on the table.** The table is loaded as a static `AssetBaseCfg` (USD), not a `RigidObjectCfg`, so `mdp.randomize_rigid_body_material` cannot target it. Robot-finger friction is the dominant push-contact axis anyway; revisit if sim-to-real reveals table-friction sensitivity.
- **No actuator-gain DR.** The §7 minimal preset doesn't include `randomize_actuator_gains`. Add it later if the user signals "stiffer/softer arm robustness" requirements.
- **No camera / visual DR.** The obs space has no camera terms (joint_pos, joint_vel, object_position, target_object_position, last_action — all state). Visual DR would no-op the §7 smoke and is therefore skipped.

## Pipeline summary (for the user)

| Phase | Status | Smoke | Diff file |
|---|---|---|---|
| §1–§5 (`task-generator`) | pass | skipped (dry_run) | `dry-run.md` |
| §6 (`reward-generator`) | pass | skipped (dry_run) | `dry-run-reward.md` |
| §7 (`dr-generator`) | pass | skipped (dry_run) | `dry-run-dr.md` |

All three phases produced clean dry-run diffs with no cross-phase conflicts. Ready for `dry_run=false` application.
