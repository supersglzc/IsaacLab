"""S1 smoke — env instantiation (3-tier tower, edit_mode_012).

The scene must contain cube_0, cube_1, AND cube_2 (all three 55 g DexCubes)
plus the ee_frame transformer.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks    # noqa: F401, E402  — registers all tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Triton-Franka-StackCube", device="cuda:0", num_envs=1, use_fabric=True)
env = gym.make("Triton-Franka-StackCube", cfg=cfg)
unw = env.unwrapped

print(f"action_space       = {env.action_space}")
print(f"observation_space  = {env.observation_space}")
print(f"max_episode_length = {unw.max_episode_length}")
print(f"num_envs           = {unw.num_envs}")
print(f"action_terms       = {list(unw.action_manager.active_terms)}")
print(f"obs_terms          = {dict(unw.observation_manager.active_terms)}")
print(f"termination_terms  = {list(unw.termination_manager.active_terms)}")
print(f"command_terms      = {list(unw.command_manager.active_terms) if unw.command_manager else []}")

assert env.action_space is not None and env.observation_space is not None
assert unw.max_episode_length > 0
assert unw.num_envs >= 1

# Scene must contain cube_0, cube_1, AND cube_2 plus ee_frame.
scene_keys = list(unw.scene.keys())
print(f"scene_keys         = {scene_keys}")
for name in ("cube_0", "cube_1", "cube_2"):
    assert name in scene_keys, f"{name} missing from scene: {scene_keys}"
assert "ee_frame" in scene_keys, f"ee_frame sensor missing from scene: {scene_keys}"

# All three cubes must be 4.3 cm cubes at 55 g (DexCube USD scaled by 0.86, mass override).
import torch  # noqa: E402
for name in ("cube_0", "cube_1", "cube_2"):
    body = unw.scene[name]
    masses = body.root_physx_view.get_masses()  # (num_envs, 1) tensor
    m0 = float(masses.flatten()[0].item())
    assert abs(m0 - 0.055) < 1e-4, f"{name}: expected mass 0.055 kg, got {m0}"
    print(f"  {name}: mass={m0:.4f} kg")

print("S1 OK: env instantiated; scene has cube_0 + cube_1 + cube_2 + ee_frame; all three cubes are 55 g DexCubes")
env.close()
sim_app.close()
