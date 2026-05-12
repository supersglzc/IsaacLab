# Task Generator — `Isaac-Push-Block-Franka-v0`

| Field | Value |
|---|---|
| task_id | Isaac-Push-Block-Franka-v0 |
| slug | isaac-push-block-franka-v0 |
| repo_path | /home/steven/code/agentic/IsaacLab |
| description | Franka pushes a small block from table center to a target marker. Success at <5cm; horizon 200. |
| assets | (none) |
| started_at | 2026-05-09T19:48:00Z |
| dry_run | false |
| finished_at | 2026-05-09T19:55:00Z |
| status | pass |

## Mode

`dry_run=false`. Real-mode pass — files written to the repo and per-section smoke checks executed live with `<repo>/.venv/bin/python` against Isaac Sim 5.1 on an RTX 4090. The dry-run blueprint at `dry-run.md` was the starting plan; one deviation noted below.

## Canonical example

- Family: `isaaclab-manager-based`
- Canonical task (per `task-implementation.md`): `Isaac-Reach-Franka-v0`
- Closest sibling for object-on-table manipulation: `Isaac-Lift-Cube-Franka-v0` — same Franka + DexCube block + SeattleLabTable + binary gripper combo.

The new task lives at `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/`. Layout mirrors `lift/`.

**Deviation from blueprint.** The dry-run plan called for a one-line edit to `manipulation/__init__.py` (`from .push import *  # noqa`). On inspection, `lift/__init__.py` is empty — Isaac auto-discovery picks up the `gym.register` calls in `config/franka/__init__.py` without any parent-package re-export. We mirrored Lift's pattern and **did not** edit `manipulation/__init__.py`. The §1 smoke confirms `Isaac-Push-Block-Franka-v0` registers and instantiates cleanly without that edit.

---

## §1 — Register task and set up the scene

**Started:** 2026-05-09T19:48:30Z

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Task slug | `Isaac-Push-Block-Franka-v0` | user-description (verb=Push, object=Block, robot=Franka) |
| Robot articulation | `FRANKA_PANDA_CFG` (joint-position control — no IK in §2, so the standard PD-gain variant) | user-description ("Franka") + canonical-mirror (Reach uses `FRANKA_PANDA_CFG`) |
| Object asset | `RigidObjectCfg` pointing at `${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd` (small block) | canonical-mirror (Lift uses this exact USD; user said "small block") |
| `num_envs` | `4096` (training) + `_PLAY` variant with `50` envs and `enable_corruption=False` | canonical-mirror |
| Domain folder | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/` | mirrors sibling `lift/` |
| EE / `panda_hand` body | `panda_hand` (closed gripper acts as a flat fingertip pusher) | canonical-mirror |

No `AskUserQuestion` required.

### Files written

| Path | Action | Notes |
|---|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/__init__.py` | new | empty docstring; mirrors `lift/__init__.py` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py` | new | Scene + Commands + Actions + Observations + Events + Rewards + Terminations + EnvCfg |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/__init__.py` | new | re-exports `isaaclab.envs.mdp.*` + local `observations`, `rewards` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/observations.py` | new | `object_position_in_robot_root_frame` helper (mirror of Lift) |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/mdp/rewards.py` | new | §6 PLACEHOLDER — single `placeholder_zero` function |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/__init__.py` | new | empty docstring |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/__init__.py` | new | `gym.register` for `Isaac-Push-Block-Franka-v0` and `-Play-v0` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/config/franka/joint_pos_env_cfg.py` | new | `FrankaPushBlockEnvCfg` + `_PLAY`: Franka + actions + block USD + `panda_hand` body |

### Smoke check

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([]); args_cli.headless = True; args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=1, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
print('OBS_SPACE:', env.observation_space)
print('ACT_SPACE:', env.action_space)
env.close(); simulation_app.close()
"
```

**Last 50 lines of stdout/stderr (relevant excerpt):**

```
[INFO]: Completed setting up the environment...
OBS_SPACE: Dict('policy': Box(-inf, inf, (1, 36), float32))
ACT_SPACE: Box(-inf, inf, (1, 8), float32)
```

**Verdict:** pass

**Finished:** 2026-05-09T19:51:30Z · attempts: 1 · status: pass

---

## §2 — Action types

**Started:** 2026-05-09T19:51:30Z

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Action mode | Absolute joint position via `mdp.JointPositionActionCfg(scale=0.5, use_default_offset=True)` for the 7 arm joints | canonical-mirror (matches Reach + Lift) |
| `scale` / `use_default_offset` | `scale=0.5`, `use_default_offset=True` | canonical-mirror |
| Gripper | Second action term `mdp.BinaryJointPositionActionCfg` on `panda_finger.*`, both `open_command_expr` and `close_command_expr` set to `0.0` so the gripper stays closed and acts as a flat pusher | task-design (push tasks close the gripper); canonical-mirror for term shape |

Total action dim = 7 (arm) + 1 (gripper binary) = 8.

No `AskUserQuestion` required.

### Files written

The §2 changes live entirely inside the §1 file `config/franka/joint_pos_env_cfg.py` — no new files added.

### Smoke check

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "
import argparse, torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([]); args_cli.headless = True; args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=1, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
env.reset()
for i in range(5):
    a = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
    out = env.step(a)
    assert out[1].shape[0] == 1, out[1].shape
print('ACTION_SHAPE_OK:', env.action_space.shape)
env.close(); simulation_app.close()
"
```

**Last 50 lines of stdout/stderr (relevant excerpt):**

```
|   Active Action Terms (shape: 8)   |
+-------+----------------+-----------+
| Index | Name           | Dimension |
+-------+----------------+-----------+
|   0   | arm_action     |         7 |
|   1   | gripper_action |         1 |
+-------+----------------+-----------+
[INFO]: Completed setting up the environment...
ACTION_SHAPE_OK: (1, 8)
```

**Verdict:** pass

**Finished:** 2026-05-09T19:52:30Z · attempts: 1 · status: pass

---

## §3 — Initialize / reset the task

**Started:** 2026-05-09T19:52:30Z

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Robot joint reset | `mdp.reset_joints_by_scale(position_range=(0.5, 1.5), velocity_range=(0.0, 0.0))` | canonical-mirror (Reach default) |
| Object position reset | `mdp.reset_root_state_uniform` on `SceneEntityCfg("object")` with `pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}` and `velocity_range = {}` | canonical-mirror (Lift uses similar tight range); user-description "table center" + small jitter |
| Goal-marker sampling | Lives in `CommandsCfg.object_pose` (sampled every 4 sim seconds via `UniformPoseCommandCfg`); not an EventTerm | canonical-mirror |
| Velocity reset | `(0.0, 0.0)` arm; `{}` object | canonical-mirror |

No `AskUserQuestion` required.

### Files written

§3 changes are inside `push_env_cfg.py:EventCfg` (already in place from §1). No additional files.

### Smoke check

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([]); args_cli.headless = True; args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=4, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
obses = []
for _ in range(5):
    obs, _ = env.reset()
    obses.append(obs['policy'].clone())
stacked = torch.stack(obses)
spread = stacked.var(dim=0).sum().item()
print('RESET_OK:', stacked.shape, 'variance_across_resets=%.4f' % spread)
assert spread > 0.0
env.close(); simulation_app.close()
"
```

**Last 50 lines of stdout/stderr (relevant excerpt):**

```
[INFO]: Completed setting up the environment...
RESET_OK: torch.Size([5, 4, 36]) variance_across_resets=4.0365
```

**Verdict:** pass

**Finished:** 2026-05-09T19:53:00Z · attempts: 1 · status: pass

---

## §4 — Goal and termination

**Started:** 2026-05-09T19:53:00Z

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Goal representation | `mdp.UniformPoseCommandCfg` named `object_pose`, `body_name="panda_hand"`, `resampling_time_range=(4.0, 4.0)`, ranges `pos_x=(0.4, 0.7), pos_y=(-0.25, 0.25), pos_z=(0.055, 0.055)`, all roll/pitch/yaw=(0,0) (planar push, position-only target) | user-description ("target marker"); canonical-mirror (Lift) |
| Success threshold | `<0.05 m` (5 cm — verbatim from user). Recorded here as a decision; `reward-generator` wires the `success` indicator term in §6 (logging-only `RewTerm` with weight=0). NOT a TerminationTerm — episode runs full horizon. | user-description (5cm); canonical-mirror (Lift's success-as-zero-weight-reward pattern) |
| `episode_length_s` | `200 / 30 ≈ 6.6667 s` (control rate = `1 / (decimation * sim.dt) = 1 / (2 * (1/60)) = 30 Hz`; 200 control steps = 6.6667 s of sim) | user-description ("horizon 200") |
| Failure terminations | `object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})` | canonical-mirror (Lift) |
| Time-out | `time_out = DoneTerm(func=mdp.time_out, time_out=True)` | canonical-mirror |

No `AskUserQuestion` required.

### Files written

§4 changes live in `push_env_cfg.py:CommandsCfg`, `push_env_cfg.py:TerminationsCfg`, and `push_env_cfg.py:PushBlockEnvCfg.__post_init__` (sets `episode_length_s = 200.0 / 30.0`). All within §1's already-listed file.

### Smoke check

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([]); args_cli.headless = True; args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=4, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
env.reset()
ep_len = env.unwrapped.max_episode_length
saw_done = False
for t in range(ep_len + 5):
    a = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
    obs, rew, terminated, truncated, info = env.step(a)
    if bool((terminated | truncated).any()):
        saw_done = True
        break
print('TERM_OK: saw_done_within_horizon=%s at_step=%d/%d' % (saw_done, t, ep_len))
assert saw_done
env.close(); simulation_app.close()
"
```

**Last 50 lines of stdout/stderr (relevant excerpt):**

```
[INFO] Termination Manager:  <TerminationManager> contains 2 active terms.
+------------------------------------+
|      Active Termination Terms      |
+-------+-----------------+----------+
| Index | Name            | Time Out |
+-------+-----------------+----------+
|   0   | time_out        |   True   |
|   1   | object_dropping |  False   |
+-------+-----------------+----------+
[INFO]: Completed setting up the environment...
TERM_OK: saw_done_within_horizon=True at_step=199/200
```

**Verdict:** pass — done at step 199 of 200 horizon (time_out fires correctly).

**Finished:** 2026-05-09T19:54:00Z · attempts: 1 · status: pass

---

## §5 — Observation

**Started:** 2026-05-09T19:54:00Z

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Term selection | 5 PolicyCfg terms: `joint_pos = mdp.joint_pos_rel`, `joint_vel = mdp.joint_vel_rel`, `object_position = mdp.object_position_in_robot_root_frame`, `target_object_position = mdp.generated_commands(command_name="object_pose")`, `actions = mdp.last_action` | canonical-mirror (verbatim Lift PolicyCfg) |
| Noise model | `enable_corruption = True`; `Unoise(n_min=-0.01, n_max=0.01)` on `joint_pos` and `joint_vel`; no noise on object/target/actions | canonical-mirror (Reach §5 example) |
| `concatenate_terms` | `True` (flat Box) | canonical-mirror |
| Asymmetric critic | NOT included | canonical-mirror (Reach + Lift both omit) |

Actual obs dim from runtime: 36 = `joint_pos(9) + joint_vel(9) + object_position(3) + target_object_position(7) + last_action(8)` (Franka has 9 controllable joints — 7 arm + 2 finger; the binary gripper action term is 1-dim but `last_action` returns the full 8-dim action vector).

No `AskUserQuestion` required.

### Files written

§5 lives in `push_env_cfg.py:ObservationsCfg`. The `mdp.object_position_in_robot_root_frame` helper is a verbatim copy of `lift/mdp/observations.py` into `push/mdp/observations.py`. Resolution path: `push/mdp/__init__.py` does `from .observations import *`, so `mdp.object_position_in_robot_root_frame` resolves correctly.

### Smoke check

```bash
cd /home/steven/code/agentic/IsaacLab && /home/steven/code/agentic/IsaacLab/.venv/bin/python -c "
import argparse, torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([]); args_cli.headless = True; args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg('Isaac-Push-Block-Franka-v0', device='cuda:0', num_envs=2, use_fabric=True)
env = gym.make('Isaac-Push-Block-Franka-v0', cfg=env_cfg)
obs, _ = env.reset()
print('OBS_KEYS:', list(obs.keys()))
for k, v in obs.items():
    assert torch.isfinite(v).all(), f'{k} has non-finite values'
    print(f'  {k}: shape={tuple(v.shape)} finite={bool(torch.isfinite(v).all())}')
env.close(); simulation_app.close()
"
```

**Last 50 lines of stdout/stderr (relevant excerpt):**

```
+-----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (36,)) |
+----------+-------------------------------------+----------+
|  Index   | Name                                |  Shape   |
+----------+-------------------------------------+----------+
|    0     | joint_pos                           |   (9,)   |
|    1     | joint_vel                           |   (9,)   |
|    2     | object_position                     |   (3,)   |
|    3     | target_object_position              |   (7,)   |
|    4     | actions                             |   (8,)   |
+----------+-------------------------------------+----------+
[INFO]: Completed setting up the environment...
OBS_KEYS: ['policy']
  policy: shape=(2, 36) finite=True
```

**Verdict:** pass

**Finished:** 2026-05-09T19:55:00Z · attempts: 1 · status: pass

---

## Verdict

| Section | Status | Attempts |
|---|---|---|
| §1 register/scene | pass | 1 |
| §2 actions | pass | 1 |
| §3 reset | pass | 1 |
| §4 goal+termination | pass | 1 |
| §5 observation | pass | 1 |

**Files written total:** 8 (all new under `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/`)
**Finished:** 2026-05-09T19:55:00Z
**Status:** pass
**Handoff:** `<task_dir>/handoff-task-generator.md`
