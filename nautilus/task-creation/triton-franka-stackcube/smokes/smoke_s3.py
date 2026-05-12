"""S3 smoke — reset values match simulation state (3-tier tower).

The cube poses follow `init_state.pos` (small uniform XY perturbation in
EventCfg; nominal mean equals init_state.pos). Verify:
  (a) cube_0, cube_1, cube_2 positions are within the expected XY perturbation
      window of their configured init_state.pos and z near CUBE_SIZE/2 (cube
      base on table top, z = 0.043/2 = 0.0215);
  (b) robot joint pos is finite across the full reset batch;
  (c) panda_hand local +z aligns with world -z (top-down grasp pose).
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

TASK_ID = "Triton-Franka-StackCube"
# 128 envs to verify the EE-down assertion and the XY perturbation window
# across the full reset batch.
cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=128, use_fabric=True)

env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

CUBE_SIZE = 0.043
CUBE_INIT_Z = CUBE_SIZE / 2.0  # ~ 0.0215
XY_RANGE = 0.05               # per EventCfg reset_cube_X pose_range
XY_TOL = XY_RANGE + 5e-3      # allow 5mm physics jitter
Z_TOL = 5e-3

# (a) Cube positions — init_state.pos for cube_0 = (0.45, -0.10, 0.0215),
# cube_1 = (0.55, 0.10, 0.0215), cube_2 = (0.50, 0.20, 0.0215). EventCfg
# uniformly perturbs XY by +/- 0.05; Z is pinned at zero offset so cube z
# must be near CUBE_INIT_Z.
env_origins = unw.scene.env_origins  # (N, 3) device tensor

cube_inits = {
    "cube_0": torch.tensor([0.45, -0.10, CUBE_INIT_Z], device=unw.device),
    "cube_1": torch.tensor([0.55, 0.10, CUBE_INIT_Z], device=unw.device),
    "cube_2": torch.tensor([0.50, 0.20, CUBE_INIT_Z], device=unw.device),
}
max_perturb = 0.0
for name, init in cube_inits.items():
    pos_w = unw.scene[name].data.root_pos_w  # (N, 3)
    pos_local = pos_w - env_origins
    dxy = (pos_local[:, :2] - init[:2]).abs()
    dz = (pos_local[:, 2] - init[2]).abs()
    assert (dxy <= XY_TOL).all(), f"{name} xy out of perturbation window: max={dxy.max().item()}"
    assert (dz <= Z_TOL).all(), f"{name} z drift > {Z_TOL}: max={dz.max().item()}"
    max_perturb = max(max_perturb, dxy.max().item())
    print(f"  {name}_local[0]={pos_local[0].tolist()}")

# Bonus: at least one env should be perturbed away from the nominal init pos
# (verifies the EventCfg ranges are actually active, not pinned to zero).
assert max_perturb > 1e-3, f"perturbation appears to be zero (max={max_perturb}); EventCfg reset_cube_X ranges may be no-op"

# (b) Joint reset must yield finite robot joint positions across the full batch.
robot = unw.scene["robot"]
robot_q = robot.data.joint_pos
assert torch.isfinite(robot_q).all(), f"non-finite joint pos after reset: {robot_q}"

# (c) EE direction — panda_hand local +z must align with world -z for every env.
panda_hand_idx = robot.body_names.index("panda_hand")
q = robot.data.body_quat_w[:, panda_hand_idx]   # (N, 4) wxyz
qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
ee_z_world = torch.stack([
    2 * (qx * qz + qw * qy),
    2 * (qy * qz - qw * qx),
    1 - 2 * (qx ** 2 + qy ** 2),
], dim=-1)  # (N, 3)
target_minus_z = torch.full((unw.num_envs,), -1.0, device=unw.device)
assert torch.allclose(ee_z_world[:, 2], target_minus_z, atol=1e-2), \
    f"EE z-axis not world-down: ee_z_world[0]={ee_z_world[0].tolist()} (want approx [0, 0, -1])"

print(f"S3 OK: max xy perturbation = {max_perturb:.4f} m (within +/- {XY_RANGE} m range)")
print(f"S3 OK: ee_z_world[0]={ee_z_world[0].tolist()} (z-component approx -1 across all {unw.num_envs} envs)")
print("S3 OK: cube_0/cube_1/cube_2 positions match init_state.pos within reset perturbation window; EE points world-down")
env.close()
sim_app.close()
