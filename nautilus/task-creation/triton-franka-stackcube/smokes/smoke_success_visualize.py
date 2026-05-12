"""Visualize the StackCube success scenario (3-tier tower).

Holds cube_0/cube_1/cube_2 stacked at a fixed XY on the table with the Franka
at its downward init pose. Loops forever until the user closes the GLFW window
or sends Ctrl+C.

Defaults: HEADED (`HEADLESS=0`), NUM_ENVS=1.
"""
import argparse
import os
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = os.environ.get("HEADLESS", "0") != "0"
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks    # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID   = "Triton-Franka-StackCube"
NUM_ENVS  = int(os.environ.get("NUM_ENVS", "1"))
CUBE_SIZE = 0.043

cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=NUM_ENVS, use_fabric=True)
env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

# === SUCCESS_HOLD_BLOCK pre-loop setup ===
cube_0 = unw.scene["cube_0"]
cube_1 = unw.scene["cube_1"]
cube_2 = unw.scene["cube_2"]

# Cache cube_2's INIT pose once and rebuild the tower on top of it every frame.
# We pin cube_2 (the base) so it never drifts away under gravity, and the
# stack column is rebuilt fresh each frame to keep gravity + integration drift
# from accumulating.
base_pos_init  = cube_2.data.root_pos_w[:, :3].clone()
base_quat_init = cube_2.data.root_quat_w.clone()

# Tower target poses (cube_2 at its init pos, cube_1 / cube_0 stacked above).
state_2 = torch.cat([base_pos_init, base_quat_init], dim=-1)        # (N, 7)
state_1 = state_2.clone(); state_1[:, 2] = state_1[:, 2] + CUBE_SIZE     # one cube above
state_0 = state_2.clone(); state_0[:, 2] = state_0[:, 2] + 2 * CUBE_SIZE  # two cubes above
zero_vel = torch.zeros((NUM_ENVS, 6), device=unw.device)

print(f"[visualize] holding success scenario  task={TASK_ID}  num_envs={NUM_ENVS}  headless={args.headless}")
print(f"[visualize] Press Ctrl+C (or close the GLFW window) to exit.")

frame = 0
try:
    while sim_app.is_running():
        # === SUCCESS_HOLD_BLOCK per-frame ===
        cube_2.write_root_pose_to_sim(state_2)
        cube_2.write_root_velocity_to_sim(zero_vel)
        cube_1.write_root_pose_to_sim(state_1)
        cube_1.write_root_velocity_to_sim(zero_vel)
        cube_0.write_root_pose_to_sim(state_0)
        cube_0.write_root_velocity_to_sim(zero_vel)
        # ====================================
        unw.sim.step(render=True)
        frame += 1
        if frame % 300 == 0:
            print(f"[visualize] {frame} frames rendered (Ctrl+C to exit)")
except KeyboardInterrupt:
    print("[visualize] interrupted by user")

env.close()
sim_app.close()
