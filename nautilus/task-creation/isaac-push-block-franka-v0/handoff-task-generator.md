# Handoff — `task-generator` → `reward-generator` / `dr-generator`

**Task:** `Isaac-Push-Block-Franka-v0`
**Slug:** `isaac-push-block-franka-v0`
**Mode:** real (`dry_run=false`)
**Phase:** §1–§5 complete; all five smokes passed live on first attempt

## New task identity

| Field | Value |
|---|---|
| Task ID | `Isaac-Push-Block-Franka-v0` |
| Play variant | `Isaac-Push-Block-Franka-Play-v0` |
| Registration entry | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/__init__.py:14` (`gym.register(id="Isaac-Push-Block-Franka-v0", ...)`) |
| Abstract env-cfg class | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py:PushBlockEnvCfg` |
| Per-robot subclass | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/joint_pos_env_cfg.py:FrankaPushBlockEnvCfg` |

Smoke verification (live, on RTX 4090 + Isaac Sim 5.1):
- §1: `OBS_SPACE = Dict('policy': Box(-inf, inf, (1, 36), float32))`, `ACT_SPACE = Box(-inf, inf, (1, 8), float32)`
- §2: `ACTION_SHAPE_OK: (1, 8)`
- §3: `RESET_OK: torch.Size([5, 4, 36]) variance_across_resets=4.0365`
- §4: `TERM_OK: saw_done_within_horizon=True at_step=199/200`
- §5: `OBS_KEYS: ['policy']`, `policy: shape=(2, 36) finite=True`

## §1 — Scene

- Robot: `FRANKA_PANDA_CFG` (standard PD gains; not the HIGH_PD IK variant — joint-pos control doesn't need it).
- Object: `RigidObjectCfg` with USD `${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd`, scale `(0.8, 0.8, 0.8)`, spawned at `[0.5, 0.0, 0.055]`.
- Table + ground + dome light follow the canonical `lift_env_cfg.py` pattern.
- `num_envs = 4096` (training) / `50` (`_PLAY`).

## §2 — Action mode

- **Absolute joint position** on the 7 arm joints via `mdp.JointPositionActionCfg(joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)`.
- **Binary gripper** via `mdp.BinaryJointPositionActionCfg(joint_names=["panda_finger.*"], open_command_expr={"panda_finger_.*": 0.0}, close_command_expr={"panda_finger_.*": 0.0})` — both expressions set to 0.0, so the gripper stays *closed* and acts as a flat fingertip pusher. The action dim stays 8 for canonical compatibility.
- Total action dim: **8** (7 arm + 1 gripper-binary). Confirmed by §2 smoke.
- Rationale: matches the canonical Reach + Lift defaults. User said "push", which could go either joint-pos or delta-EE; joint-pos is the canonical default per `task-implementation.md` §2 and Lift (the closest sibling) uses joint-pos for an identical Franka+block setup.

Override location: `config/franka/joint_pos_env_cfg.py:FrankaPushBlockEnvCfg.__post_init__` (lines 22–43).

## §3 — Reset

`push_env_cfg.py:EventCfg` (lines 117–142).

- `reset_robot_joints`: `mdp.reset_joints_by_scale(position_range=(0.5, 1.5), velocity_range=(0.0, 0.0))` (Reach default).
- `reset_object_position`: `mdp.reset_root_state_uniform` on the `object` SceneEntity, with `pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}` (small jitter around table center) and empty `velocity_range`.

DR-extra terms (§7) attach to this same `EventCfg` class — see "§7 placeholder" below.

## §4 — Goal + termination

- Goal: `CommandsCfg.object_pose = mdp.UniformPoseCommandCfg(...)` with planar 3-D ranges (`pos_x=(0.4, 0.7)`, `pos_y=(-0.25, 0.25)`, `pos_z=(0.055, 0.055)` — fixed at block-rest height); resampled every 4 sim seconds. `body_name="panda_hand"` set in the per-robot subclass.
- Termination terms (`push_env_cfg.py:TerminationsCfg`, lines 156–164):
  - `time_out = DoneTerm(func=mdp.time_out, time_out=True)` — fires at exactly **200 control steps** because `episode_length_s = 200.0 / 30.0` (decimation=2 at 60 Hz → 30 Hz control rate). Confirmed live: §4 smoke saw the done flag at step 199.
  - `object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})` — block falls off table.
- **Success metric:** the user's "<5cm" threshold is NOT a termination — it lives as a logging-only zero-weight reward term that `reward-generator` will write into `RewardsCfg.success`. The episode runs the full 200-step horizon so the dense reward signal can shape the agent.

`episode_length_s` is set in `PushBlockEnvCfg.__post_init__` (`push_env_cfg.py:179`).

## §5 — Observation

`push_env_cfg.py:ObservationsCfg.PolicyCfg` (lines 92–110).

- 5 terms in declaration order:
  - `joint_pos = mdp.joint_pos_rel` with `Unoise(±0.01)` corruption.
  - `joint_vel = mdp.joint_vel_rel` with `Unoise(±0.01)`.
  - `object_position = mdp.object_position_in_robot_root_frame` (custom helper at `push/mdp/observations.py`).
  - `target_object_position = mdp.generated_commands(command_name="object_pose")` (full 7-D pose from CommandsCfg).
  - `actions = mdp.last_action`.
- `enable_corruption = True`, `concatenate_terms = True`.
- Confirmed obs dim: **36** (9 + 9 + 3 + 7 + 8). The 9-dim `joint_pos_rel` includes the 2 finger joints alongside the 7 arm joints.

## Placeholder locations (READ THIS FOR NEXT AGENTS)

### §6 reward placeholder — for `reward-generator`

| File | Section | Notes |
|---|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | `class RewardsCfg` (lines 145–154) | Currently has a single `placeholder = RewTerm(func=mdp.placeholder_zero, weight=0.0)`. Replace the entire class body with real terms. |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` | whole file | Currently has only a `placeholder_zero(env)` function returning `torch.zeros(env.num_envs, device=env.device)`. Replace with real reward functions (e.g. `block_to_goal_distance_tanh`, `object_ee_distance`, `block_at_goal_indicator`). Keep the file at this path so `mdp/__init__.py`'s `from .rewards import *` continues to expose them. |

Suggested term layout for §6 (drawn from Lift's pattern + the Push goal):
- `reaching_object` (positive tanh, ~weight 1.0): EE → block distance using `mdp.object_ee_distance` (already exists in shared `mdp` library).
- `block_to_goal_tracking` (positive tanh, ~weight 16.0): block → target distance using a custom helper modeled on Lift's `object_goal_distance` but without the `minimal_height` lift gate.
- `block_to_goal_tracking_fine_grained` (positive tanh, smaller std=0.05, ~weight 5.0): same helper, tighter kernel.
- `success` (positive indicator, weight=0.0 logging-only): `1.0` when block→goal distance < 0.05 m else `0.0`.
- `action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)`.
- `joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)`.

These are *suggestions* — `reward-generator` runs the §6 decision protocol and may diverge.

### §7 DR placeholder — for `dr-generator`

| File | Section | Notes |
|---|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | `class EventCfg` (lines 117–142) | Currently holds only the two §3 `mode="reset"` terms. Append `mode="startup"` and `mode="interval"` DR terms here. |

Suggested DR axes (drawn from `task-implementation.md` §7 defaults — `velocity_env_cfg.py` patterns):
- `physics_material` (`mode="startup"`): randomize friction on robot bodies + block (push contact dynamics depend heavily on it).
- `block_mass` (`mode="startup"`): `mdp.randomize_rigid_body_mass` on the block, `(-0.1, 0.1)` additive kg.
- (Optional) `apply_external_force_torque` (`mode="reset"`): small disturbance on the block at reset.

If the user's description has no DR signal and they're training pure-sim, `dr-generator` may legitimately return `skipped` per its agent contract.

## Files written by this phase

| Path | Purpose |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/__init__.py` | new package init |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | abstract `PushBlockEnvCfg` (Scene + Commands + Actions + Observations + Events + Rewards + Terminations) |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/__init__.py` | re-exports `isaaclab.envs.mdp` + local helpers |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/observations.py` | `object_position_in_robot_root_frame` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` | §6 PLACEHOLDER (`placeholder_zero`) |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/__init__.py` | new package init |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/__init__.py` | `gym.register` for the two task IDs |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/joint_pos_env_cfg.py` | `FrankaPushBlockEnvCfg` + `_PLAY`: Franka + actions + block USD + `panda_hand` body |
| `<task_dir>/task-history.md` | verbose process log (§1..§5 phase blocks) |
| `<task_dir>/handoff-task-generator.md` | this file |

Note: no parent-package `__init__.py` was edited — Isaac Sim auto-discovery picks up the registration via the `gym.register` calls in `config/franka/__init__.py`. This mirrors the canonical Lift pattern (which also has an empty `manipulation/lift/__init__.py`).

## Open questions for the next phases

None blocking. The user's description is unambiguous about the task semantics (push / Franka / small block / 5 cm success / 200 horizon). All §1–§5 decisions resolved by user-description + canonical-mirror without an `AskUserQuestion`. `reward-generator` and `dr-generator` will run their own decision protocols on their own scopes.
