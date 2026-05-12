"""S2 smoke — stock JointPositionActionCfg + BinaryGripper.

Edit_mode_011: revert §2 from joint-space EMA to LiftCube's stock
`mdp.JointPositionActionCfg(scale=0.5, use_default_offset=True)`.

arm_action is 7-D `mdp.JointPositionActionCfg`:
    joint_names=["panda_joint.*"]
    scale=0.5
    use_default_offset=True   (offset = default_joint_pos at __init__)
gripper is 1-D `mdp.BinaryJointPositionActionCfg` on panda_finger.*
Total action dim = 8.

Closed-form: `term._processed_actions == scale * raw_action + offset` for the
7 arm joints, where `offset == robot.data.default_joint_pos[arm_joints]` at
construction time (use_default_offset=True).
"""
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks    # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID   = "Triton-Franka-StackCube"
TERM_NAME = "arm_action"
NUM_ENVS  = 128
NUM_STEPS = 30

cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=NUM_ENVS, use_fabric=True)
env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

term = unw.action_manager._terms[TERM_NAME]
assert term.action_dim == 7, f"expected arm_action dim 7 (panda_joint.*), got {term.action_dim}"

# Total env action dim = arm (7) + gripper (1) = 8.
action_dim = env.action_space.shape[-1]
assert action_dim == 8, f"expected total action dim 8 (7 arm + 1 gripper), got {action_dim}"

# Verify cfg exposes raw scale = 0.5 (LiftCube convention).
cfg_scale = float(term.cfg.scale)
assert abs(cfg_scale - 0.5) < 1e-9, f"expected cfg.scale=0.5, got {cfg_scale}"

# Drive a known action: 0.2 across all dims.
known_action = 0.2 * torch.ones((unw.num_envs, action_dim), device=unw.device)
env.step(known_action)

# Closed-form for stock JointPositionAction:
#   _processed_actions = scale * raw_action + offset
# where offset == default_joint_pos[arm_joints] under use_default_offset=True.
s = term._scale
if isinstance(s, float) or (hasattr(s, "numel") and s.numel() == 1):
    s_scalar = float(s) if isinstance(s, float) else float(s.flatten()[0])
else:
    s_scalar = float(s[0, 0])
assert abs(s_scalar - 0.5) < 1e-9, f"expected internal _scale=0.5, got {s_scalar}"

offset = term._offset
a_arm = known_action[:, :7]
expected = s_scalar * a_arm + offset

actual = term._processed_actions
assert torch.allclose(actual, expected, atol=1e-5), (
    "JointPositionAction target mismatch at t=0:\n"
    f"  actual[0]={actual[0].tolist()}\n  expected[0]={expected[0].tolist()}"
)
print(f"S2 OK: arm target == 0.5 * action + default_joint_pos (max|diff|={float((actual-expected).abs().max()):.2e})")
print(f"S2 OK: ACTION_SHAPE_OK = {tuple(env.action_space.shape)}")

# Run NUM_STEPS more steps with bounded random actions; assert no NaN/Inf.
gen = torch.Generator(device=unw.device).manual_seed(0)
for step_i in range(NUM_STEPS):
    a = torch.empty((unw.num_envs, action_dim), device=unw.device).uniform_(-1.0, 1.0, generator=gen)
    obs, _, _, _, _ = env.step(a)
    flat = obs["policy"] if isinstance(obs, dict) else obs
    assert torch.isfinite(flat).all(), f"non-finite obs at step {step_i}"
print(f"S2 OK: {NUM_STEPS} env steps at {NUM_ENVS} envs produced finite obs throughout")

env.close()
sim_app.close()
