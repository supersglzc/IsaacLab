"""S5 smoke — observation shape, term order, and value match the §5 design (3-tier tower).

After task_generator_edit_mode_012: obs gains cube_1_position, cube_2_position,
and cube_1_target_position; the existing `target_position` is renamed
`cube_0_target_position`. Total obs width grows from 32 -> 41.

    joint_pos                (9,)   <- mdp.joint_pos_rel (all 9 robot joints)
    joint_vel                (9,)   <- mdp.joint_vel_rel (same 9 joints)
    cube_0_position          (3,)   <- mdp.cube_0_position_in_robot_root_frame
    cube_1_position          (3,)   <- mdp.cube_1_position_in_robot_root_frame  (NEW)
    cube_2_position          (3,)   <- mdp.cube_2_position_in_robot_root_frame  (NEW)
    cube_0_target_position   (3,)   <- mdp.stack_target_position_in_robot_root_frame
                                      (cube_1.pos_w + [0,0,CUBE_SIZE] in robot root frame)
    cube_1_target_position   (3,)   <- mdp.cube_1_stack_target_position_in_robot_root_frame
                                      (cube_2.pos_w + [0,0,CUBE_SIZE] in robot root frame, NEW)
    actions                  (8,)   <- mdp.last_action (7 arm + 1 gripper)
Total = 9 + 9 + 3 + 3 + 3 + 3 + 3 + 8 = 41 dims.
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
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

TASK_ID = "Triton-Franka-StackCube"
CUBE_SIZE = 0.043

cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1, use_fabric=True)
# Disable obs corruption so value checks are exact.
cfg.observations.policy.enable_corruption = False

env = gym.make(TASK_ID, cfg=cfg)
unw = env.unwrapped
obs, _ = env.reset(seed=0)

# Print the term order + shapes for the log
group_terms = dict(unw.observation_manager.active_terms)
group_dims  = unw.observation_manager.group_obs_term_dim
print("observation order (group -> term: shape):")
for group, terms in group_terms.items():
    for name, dim in zip(terms, group_dims[group]):
        print(f"  {group}.{name}: {tuple(dim)}")

# Assert policy-group order + shapes match the §5 design.
EXPECTED_TERMS = [
    ("joint_pos",              (9,)),
    ("joint_vel",              (9,)),
    ("cube_0_position",        (3,)),
    ("cube_1_position",        (3,)),
    ("cube_2_position",        (3,)),
    ("cube_0_target_position", (3,)),
    ("cube_1_target_position", (3,)),
    ("actions",                (8,)),
]
actual = [(name, tuple(dim)) for name, dim in zip(group_terms["policy"], group_dims["policy"])]
assert actual == EXPECTED_TERMS, (
    f"obs term order/shape mismatch:\n  actual={actual}\n  expected={EXPECTED_TERMS}"
)

# Total obs dim = sum of widths
total_dim = sum(d[0] for d in group_dims["policy"])
assert total_dim == 41, f"expected total obs dim 41 (9+9+3+3+3+3+3+8), got {total_dim}"

# Cumulative offsets per term — match EXPECTED_TERMS order.
offsets = [0]
for _, shape in EXPECTED_TERMS:
    offsets.append(offsets[-1] + shape[0])
slot = {name: (offsets[i], shape[0]) for i, (name, shape) in enumerate(EXPECTED_TERMS)}

robot = unw.scene["robot"]
cube_0 = unw.scene["cube_0"]
cube_1 = unw.scene["cube_1"]
cube_2 = unw.scene["cube_2"]


def _root_frame(world_pos):
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, world_pos)
    return pos_b


# Value check: each cube_<i>_position slot equals subtract_frame_transforms(robot_root, cube_<i>_world).
for name, cube in (("cube_0_position", cube_0), ("cube_1_position", cube_1), ("cube_2_position", cube_2)):
    off, w = slot[name]
    expected = _root_frame(cube.data.root_pos_w[:, :3])
    actual_slot = obs["policy"][:, off : off + w]
    assert torch.allclose(actual_slot, expected, atol=1e-4), (
        f"{name} obs slot mismatch:\n  obs[{off}:{off+w}]={actual_slot}\n  expected={expected}"
    )

# Value check: cube_0_target_position slot equals (cube_1.pos_w + [0,0,CUBE_SIZE]) in root frame.
target_0_w = cube_1.data.root_pos_w[:, :3].clone()
target_0_w[:, 2] = target_0_w[:, 2] + CUBE_SIZE
expected_target_0 = _root_frame(target_0_w)
off, w = slot["cube_0_target_position"]
target_0_in_obs = obs["policy"][:, off : off + w]
assert torch.allclose(target_0_in_obs, expected_target_0, atol=1e-4), (
    f"cube_0_target_position obs slot mismatch:\n  obs[{off}:{off+w}]={target_0_in_obs}\n"
    f"  expected={expected_target_0}"
)

# Value check: cube_1_target_position slot equals (cube_2.pos_w + [0,0,CUBE_SIZE]) in root frame.
target_1_w = cube_2.data.root_pos_w[:, :3].clone()
target_1_w[:, 2] = target_1_w[:, 2] + CUBE_SIZE
expected_target_1 = _root_frame(target_1_w)
off, w = slot["cube_1_target_position"]
target_1_in_obs = obs["policy"][:, off : off + w]
assert torch.allclose(target_1_in_obs, expected_target_1, atol=1e-4), (
    f"cube_1_target_position obs slot mismatch:\n  obs[{off}:{off+w}]={target_1_in_obs}\n"
    f"  expected={expected_target_1}"
)

# All obs values must be finite.
for k, v in obs.items():
    assert torch.isfinite(v).all(), f"non-finite values in obs[{k}]"

print(f"S5 OK: {len(EXPECTED_TERMS)} terms, order + shapes match (total_dim={total_dim})")
print(f"S5 OK: cube_0/cube_1/cube_2 position slots match root-frame transforms")
print(f"S5 OK: cube_0_target_position == (cube_1.pos_w + [0,0,{CUBE_SIZE}]) in root frame")
print(f"S5 OK: cube_1_target_position == (cube_2.pos_w + [0,0,{CUBE_SIZE}]) in root frame")
env.close()
sim_app.close()
