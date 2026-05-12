# Environment Diff — Triton-Franka-StackCube: 2-cube → 3-cube

What changed in §1/§3/§4/§5 between the original two-cube task and the current
three-cube task. §2 (action) and §6 (reward) and §7 (DR) are unchanged by the
3-cube extension; §6 has just been reverted to the 2-cube design per user
direction.

All file paths are relative to:
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube/`

---

## §1 — Scene

`stack_cube_env_cfg.py:StackCubeSceneCfg` and `config/franka/joint_pos_env_cfg.py:FrankaStackCubeEnvCfg.__post_init__`

| element | 2-cube | 3-cube |
|---|---|---|
| Cube slots in scene cfg | `cube_0, cube_1` | `cube_0, cube_1, cube_2` |
| Fingertip contact-sensor filter list (left + right) | `[Cube_0, Cube_1]` | `[Cube_0, Cube_1, Cube_2]` |
| Cube_0 init position | `[0.45, -0.10, CUBE_INIT_Z]` | unchanged |
| Cube_1 init position | `[0.55, +0.10, CUBE_INIT_Z]` | unchanged |
| Cube_2 init position | — | `[0.50, 0.20, CUBE_INIT_Z]` (spread to minimize ±5 cm reset collision risk) |
| Cube asset | DexCube USD, scale 0.86, mass 55 g | identical for cube_2 (third instance of same spawn cfg) |
| Robot, table, ground plane, lights, ee_frame FrameTransformer | unchanged | unchanged |

Constants: `CUBE_SIZE = 0.043 m`, `CUBE_USD_SCALE = 0.86`, `CUBE_MASS = 0.055 kg`,
`CUBE_INIT_Z = CUBE_SIZE/2 = 0.0215 m` — all unchanged.

---

## §3 — Reset (EventCfg)

`stack_cube_env_cfg.py:EventCfg`

| EventTerm | 2-cube | 3-cube |
|---|---|---|
| `reset_robot_joints` | scale (1.0, 1.0), velocity 0 | unchanged |
| `reset_cube_0` | uniform xy ±0.05 m, z 0, no velocity | unchanged |
| `reset_cube_1` | uniform xy ±0.05 m, z 0, no velocity | unchanged |
| `reset_cube_2` | — | **NEW** — uniform xy ±0.05 m, z 0, no velocity (mirrors cube_0/1) |

`mdp/events.py` itself has no new helpers — `mdp.reset_root_state_uniform` is
the IsaacLab built-in reused for all three cubes.

---

## §4 — Terminations

`stack_cube_env_cfg.py:TerminationsCfg` and `mdp/terminations.py`

| DoneTerm | 2-cube | 3-cube |
|---|---|---|
| `time_out` | `mdp.time_out` (250-step episode cap) | unchanged |
| `cube_dropped` (failure) | `mdp.cube_0_dropping` (cube_0 below table − 0.05 m) | **`mdp.any_cube_dropping`** — fires if ANY of cube_0 / cube_1 / cube_2 drops below `table_height − drop_margin` |
| `success` | `mdp.cube_0_stacked_on_cube_1` — `|cube_0.xy − cube_1.xy| < 0.02 AND |Δz − CUBE_SIZE| < 0.01` | **`mdp.three_tier_tower_stacked`** — ANDs the two per-pair checks: (cube_0 on cube_1) AND (cube_1 on cube_2), same 0.02 m xy / 0.01 m z thresholds |

The legacy `mdp.cube_0_stacked_on_cube_1` and `mdp.cube_0_dropping` helpers
were **preserved** in `mdp/terminations.py` so that the §6 reward's
`mdp.cube_0_stacked_bonus` (which depends on the same cube_0-on-cube_1
geometric check) still imports cleanly. The new tower-stacked / any-dropping
helpers were added; the old ones are not used by `TerminationsCfg` anymore but
remain available.

---

## §5 — Observations (PolicyCfg)

`stack_cube_env_cfg.py:ObservationsCfg.PolicyCfg` and `mdp/observations.py`

| ObsTerm (order matters — concatenation) | 2-cube | 3-cube |
|---|---|---|
| `joint_pos` | `mdp.joint_pos_rel` (9 robot joints) | unchanged |
| `joint_vel` | `mdp.joint_vel_rel` (9 robot joints) | unchanged |
| `cube_0_position` | `mdp.cube_0_position_in_robot_root_frame` (3-D, robot frame) | unchanged |
| `cube_1_position` | — | **NEW** — `mdp.cube_1_position_in_robot_root_frame` (3-D) |
| `cube_2_position` | — | **NEW** — `mdp.cube_2_position_in_robot_root_frame` (3-D) |
| `cube_0_target_position` (was named `target_position` in 2-cube) | `mdp.stack_target_position_in_robot_root_frame` — top of cube_1 (cube_1.pos + [0,0,CUBE_SIZE]) in robot frame | unchanged function, same helper, just renamed `target_position` → `cube_0_target_position` for clarity |
| `cube_1_target_position` | — | **NEW** — `mdp.cube_1_stack_target_position_in_robot_root_frame` — top of cube_2 (cube_2.pos + [0,0,CUBE_SIZE]) in robot frame |
| `actions` | `mdp.last_action` (8-D: 7 arm joints + 1 binary gripper) | unchanged |

`enable_corruption=True` (LiftCube default) — unchanged.

**Total observation dimension**:
- 2-cube: 9 + 9 + 3 + 3 + 8 = **32**
- 3-cube: 9 + 9 + 3 + 3 + 3 + 3 + 3 + 8 = **41** (+9: cube_1_pos + cube_2_pos + cube_1_target_pos)

New helpers added in `mdp/observations.py`:
- `cube_1_position_in_robot_root_frame(env)` — cube_1 xyz in robot frame
- `cube_2_position_in_robot_root_frame(env)` — cube_2 xyz in robot frame
- `cube_1_stack_target_position_in_robot_root_frame(env)` — cube_2.pos + [0,0,CUBE_SIZE] in robot frame

The original `stack_target_position_in_robot_root_frame` is kept under its old
name and bound to the renamed `cube_0_target_position` obs slot.

---

## §2 — Action (UNCHANGED)

`stack_cube_env_cfg.py:ActionsCfg` and `config/franka/joint_pos_env_cfg.py`

- `arm_action`: `mdp.JointPositionActionCfg(joint_names=['panda_joint.*'], scale=0.5, use_default_offset=True)` — 7-D
- `gripper_action`: `mdp.BinaryJointPositionActionCfg(joint_names=['panda_finger.*'], open=0.04, close=0.0)` — 1-D

Total action dim = **8** in both 2-cube and 3-cube.

---

## §6 — Reward (RESTORED to 2-cube design per user)

`stack_cube_env_cfg.py:RewardsCfg`

Now identical to the 2-cube state. 7 terms:

| term | weight | function | gating |
|---|---:|---|---|
| reaching_object | 0.02 | `mdp.cube_0_ee_distance(std=0.1)` | always on |
| lifting_object | 0.1 | `mdp.cube_0_is_lifted(minimal_height=0.04)` | self |
| goal_tracking_coarse | 0.32 | `mdp.cube_0_goal_distance(std=0.3, minimal_height=0.04)` | cube_0 lifted |
| goal_tracking_fine | 0.0 (dormant) | `mdp.cube_0_goal_distance(std=0.05, minimal_height=0.04)` | cube_0 lifted |
| success_bonus | 200.0 | `mdp.cube_0_stacked_bonus(xy_threshold=0.02, z_threshold=0.01)` | self — fires when cube_0 sits on cube_1 anywhere |
| action_rate | -2e-6 | `mdp.action_rate_l2` | — |
| joint_vel | -2e-6 | `mdp.joint_vel_l2` | — |

Composer: sum. `mdp.cube_0_goal_distance` uses goal = `cube_1.pos_w + [0, 0, CUBE_SIZE]`.

Note: the 3-cube success termination requires BOTH pairs stacked, but this
reward only pushes for cube_0 on cube_1 → expect ~0 % full-tower success even
though `cube_0_stacked_bonus` will fire often (≈19 % of episodes per iter_023's
baseline measurement).

---

## §7 — Domain Randomization (UNCHANGED)

No DR terms in `EventCfg` beyond the per-cube position resets. `dr-generator`
has not been dispatched for this task.

---

## File summary

The 3-cube extension touched these files (vs the 2-cube state):

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube/
├── stack_cube_env_cfg.py          # +cube_2 slot, contact filters, +reset_cube_2,
│                                  # +cube_1/2 obs terms, +cube_1_target, success/dropped rewired
├── config/franka/joint_pos_env_cfg.py  # +cube_2 RigidObjectCfg instantiation
├── mdp/observations.py            # +cube_1_position_in_robot_root_frame,
│                                  # +cube_2_position_in_robot_root_frame,
│                                  # +cube_1_stack_target_position_in_robot_root_frame
└── mdp/terminations.py            # +three_tier_tower_stacked, +any_cube_dropping
                                   # (old single-pair helpers preserved)
```

§2 (`mdp/actions*.py`), §6 (`mdp/rewards.py` helpers), §7 (no DR) unchanged by
the 3-cube extension. The 3-cube-specific reward helpers in `mdp/rewards.py`
(`cube_1_*`, `*_gated`, `*_loose_*`, `*_latched_*`, `three_tier_tower_bonus`)
remain defined but no longer referenced by `RewardsCfg` after the §6 revert.
