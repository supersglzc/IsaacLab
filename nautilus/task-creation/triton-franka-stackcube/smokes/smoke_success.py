"""Success-replicate smoke for Triton-Franka-StackCube (3-tier tower).

Drives all envs into the 3-tier tower configuration:
  cube_2 (base on table): teleported to a fixed XY at z = CUBE_SIZE/2
  cube_1 (middle):        (cube_2.xy, cube_2.z + CUBE_SIZE)
  cube_0 (top):           (cube_2.xy, cube_2.z + 2 * CUBE_SIZE)

Then steps once with zero action and asserts the `success` termination
(`three_tier_tower_stacked`) specifically fires.
"""
import argparse
import os
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = os.environ.get("HEADLESS", "1") != "0"
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks    # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID      = "Triton-Franka-StackCube"
SUCCESS_TERM = "success"
NUM_ENVS     = int(os.environ.get("NUM_ENVS", "128"))
CUBE_SIZE    = 0.043

cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=NUM_ENVS, use_fabric=True)
env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

cube_0 = unw.scene["cube_0"]
cube_1 = unw.scene["cube_1"]
cube_2 = unw.scene["cube_2"]


def _teleport(asset, xyz):
    """Helper: teleport `asset` to per-env world-frame xyz (broadcast); zero velocity."""
    state = asset.data.default_root_state.clone()
    state[:, :2] = torch.tensor(xyz[:2], device=unw.device).expand(unw.num_envs, -1)
    state[:, 2] = xyz[2]
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=unw.device)
    state[:, 7:13] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:13])


# === SUCCESS_FORCE_BLOCK ===
# Tower base xy at (0.50, 0.0); z is centered half a cube above table top.
base_xy = (0.50, 0.0)
z_2 = CUBE_SIZE / 2.0           # cube_2 center on table top
z_1 = z_2 + CUBE_SIZE           # cube_1 center on top of cube_2
z_0 = z_1 + CUBE_SIZE           # cube_0 center on top of cube_1
_teleport(cube_2, (base_xy[0], base_xy[1], z_2))
_teleport(cube_1, (base_xy[0], base_xy[1], z_1))
_teleport(cube_0, (base_xy[0], base_xy[1], z_0))
# === /SUCCESS_FORCE_BLOCK ===

zero_action = torch.zeros((unw.num_envs, env.action_space.shape[-1]), device=unw.device)
obs, rew, terminated, truncated, info = env.step(zero_action)

tm          = unw.termination_manager
term_names  = tm.active_terms
last_dones  = tm._last_episode_dones                                   # (N, num_terms) bool
fired       = {name: int(last_dones[:, i].sum().item()) for i, name in enumerate(term_names)}
print(f"[success-smoke] termination terms fired (env count): {fired}")

assert SUCCESS_TERM in term_names, (
    f"success termination '{SUCCESS_TERM}' not in termination manager. Active: {term_names}"
)
n_succ = fired[SUCCESS_TERM]
assert n_succ > 0, (
    f"success termination '{SUCCESS_TERM}' did NOT fire across {unw.num_envs} envs "
    f"(terminated={int(terminated.sum().item())}, truncated={int(truncated.sum().item())}, "
    f"term_dones={fired})"
)

print(
    f"S-success OK: '{SUCCESS_TERM}' fired in {n_succ}/{unw.num_envs} envs after the "
    f"3-tier tower replicate-success scenario; total terminated={int(terminated.sum().item())}"
)
env.close()
sim_app.close()
