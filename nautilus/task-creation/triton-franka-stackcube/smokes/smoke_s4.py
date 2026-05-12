"""S4 smoke — termination fires when forced (3-tier tower variant).

Success: `three_tier_tower_stacked` ANDs the two per-pair stacking checks:
    |cube_0.xy - cube_1.xy| < xy_threshold (0.02 m) AND |Δz_0_1 - CUBE_SIZE| < z_threshold (0.01 m)
    |cube_1.xy - cube_2.xy| < xy_threshold AND       |Δz_1_2 - CUBE_SIZE| < z_threshold

Failure: `any_cube_dropping` — ANY of cube_0/cube_1/cube_2 falls below
    `table_height - drop_margin`. CUBE_SIZE = 0.043.

Verify:
  (a) teleporting cube_2 to a known XY on the table, cube_1 to (cube_2.xy, cube_2.z + CUBE_SIZE),
      and cube_0 to (cube_2.xy, cube_2.z + 2 * CUBE_SIZE) fires `success`;
  (b) teleporting cube_0 below the table fires `cube_dropped`;
  (c) teleporting cube_2 below the table also fires `cube_dropped` (verifies the
      cube_dropped function checks ALL cubes, not just cube_0).
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
cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1, use_fabric=True)

env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

CUBE_SIZE = 0.043

cube_0 = unw.scene["cube_0"]
cube_1 = unw.scene["cube_1"]
cube_2 = unw.scene["cube_2"]

zero_action = torch.zeros((unw.num_envs, env.action_space.shape[-1]), device=unw.device)


def _teleport(asset, xyz):
    """Helper: teleport `asset` to world-frame xyz; identity quat; zero velocity."""
    state = asset.data.default_root_state.clone()
    state[:, :2] = torch.tensor(xyz[:2], device=unw.device).expand(unw.num_envs, -1)
    state[:, 2] = xyz[2]
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=unw.device)
    state[:, 7:13] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:13])


# (a) Force success: build the 3-tier tower at a fixed world XY (cube_2 base).
# Strategy: write the three cube poses to the internal scene buffer, then
# call the termination manager directly. We do NOT advance sim_timestamp
# (no env.step / sim.step). The termination function reads `root_pos_w`,
# which reads from the timestamped buffer — write_root_pose_to_sim updates
# that buffer immediately, so the termination sees our injected stacked
# configuration without any physics overshoot.
base_xy = (0.50, 0.0)
z_2 = CUBE_SIZE / 2.0           # cube_2 center
z_1 = z_2 + CUBE_SIZE           # cube_1 center on top of cube_2
z_0 = z_1 + CUBE_SIZE           # cube_0 center on top of cube_1
_teleport(cube_2, (base_xy[0], base_xy[1], z_2))
_teleport(cube_1, (base_xy[0], base_xy[1], z_1))
_teleport(cube_0, (base_xy[0], base_xy[1], z_0))

print(
    f"[S4 debug] post-write (no sim.step) cube positions: "
    f"cube_0.z={cube_0.data.root_pos_w[0, 2].item():.5f}, "
    f"cube_1.z={cube_1.data.root_pos_w[0, 2].item():.5f}, "
    f"cube_2.z={cube_2.data.root_pos_w[0, 2].item():.5f}"
)

# Evaluate the termination manager directly.
combined = unw.termination_manager.compute()
success_fired = unw.termination_manager.get_term("success")
print(f"[S4 debug] combined termination = {combined.tolist()}")
print(f"[S4 debug] success term = {success_fired.tolist()}")
fired_per_term = {
    name: int(unw.termination_manager.get_term(name).sum().item())
    for name in unw.termination_manager.active_terms
}
print(f"[S4 debug] per-term dones: {fired_per_term}")
assert bool(success_fired.any()), (
    "success termination did not fire after teleporting all three cubes "
    "into the stacked configuration; check three_tier_tower_stacked against "
    "the cube positions printed above."
)
print(f"S4 OK: 3-tier tower success termination fires: success={success_fired.tolist()}")

# (b) Force cube_dropped via cube_0: write cube_0 root pose to z = -1 m.
obs, _ = env.reset(seed=0)
_teleport(cube_0, (0.50, 0.0, -1.0))
unw.termination_manager.compute()
fired_b = unw.termination_manager.get_term("cube_dropped")
assert bool(fired_b.any()), (
    f"any_cube_dropping did not fire after teleporting cube_0 below the table; "
    f"cube_dropped term = {fired_b.tolist()}"
)
print(f"S4 OK: cube_dropped fires when cube_0 is below the table: cube_dropped={fired_b.tolist()}")

# (c) Force cube_dropped via cube_2: verifies the failure term sweeps ALL three cubes.
obs, _ = env.reset(seed=0)
_teleport(cube_2, (0.50, 0.0, -1.0))
unw.termination_manager.compute()
fired_c = unw.termination_manager.get_term("cube_dropped")
assert bool(fired_c.any()), (
    f"any_cube_dropping did not fire after teleporting cube_2 below the table; "
    f"cube_dropped term = {fired_c.tolist()}"
)
print(f"S4 OK: cube_dropped fires when cube_2 is below the table: cube_dropped={fired_c.tolist()}")

print("S4 OK: three_tier_tower_stacked fires when forced; any_cube_dropping fires for cube_0 AND cube_2")
env.close()
sim_app.close()
