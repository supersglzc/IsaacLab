# task-generator handoff — edit_mode_011 (latest)

| Field | Value |
|---|---|
| task_id | `Triton-Franka-StackCube` |
| mode | edit (sections=[2]) |
| status | pass |
| finished_at | 2026-05-12T01:46:00Z |

## Edit_mode_011 — revert §2 to LiftCube stock JointPositionActionCfg

Replace the custom `EMACumulativeRelativeJointPositionAction` (joint-space
EMA wrapper installed in edit_mode_010) with LiftCube's stock
`mdp.JointPositionActionCfg(scale=0.5, use_default_offset=True)`. Reward
(LiftCube 7-term recipe from edit_mode_008 + reward_generator_edit_mode_003)
and all other sections untouched. Action dim and obs dim unchanged (8 / 32).

## File-level diff

- `mdp/actions.py` + `mdp/actions_cfg.py` — REWRITE to docstring-only stubs
  (same pattern as edit_mode_008). Stock `mdp.JointPositionAction[Cfg]`
  is re-exported by `mdp/__init__.py` via `from isaaclab.envs.mdp import *`.
- `config/franka/joint_pos_env_cfg.py`: docstring rewrite for stock
  JointPositionAction; `arm_action` block replaced with
  `mdp.JointPositionActionCfg(asset_name="robot",
  joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)`.
- `stack_cube_env_cfg.py`: `ActionsCfg.arm_action` annotation
  `EMACumulativeRelativeJointPositionActionCfg` → `JointPositionActionCfg`;
  docstring rewritten.

Untouched: §1 scene, §3 events, §4 terminations, §5 obs structure, §6 reward,
§7 DR, curriculum (iter 15), `FRANKA_INIT_JOINT_POS`, `gripper_action`,
scene `ee_frame`.

## Surface dim deltas

|                       | before (edit_010) | after (edit_011) |
|---|---|---|
| arm action dim        | 7 (EMA cumulative delta) | 7 (stock JointPositionAction) |
| total action dim      | 8 | 8 |
| total obs dim         | 32 | 32 |

Dimensions are identical; only the action semantics changed (no EMA, no
cumulative delta — direct `target = 0.5 * action + default_joint_pos`).
RL policies trained against edit_010 will not transfer one-to-one.

## Smoke verdicts

| Smoke | Verdict | Notes |
|---|---|---|
| S2 | pass | dim 7+1=8; closed-form `0.5 * a_arm + default_joint_pos` matches `_processed_actions` exactly (max|diff|=0). |
| S5 | pass | order `joint_pos(9) / joint_vel(9) / cube_0_position(3) / target_position(3) / actions(8)`, total 32. |
| S6 | pass | 7 LiftCube terms all finite (128 envs × 30 steps); lift-gate verified. |
