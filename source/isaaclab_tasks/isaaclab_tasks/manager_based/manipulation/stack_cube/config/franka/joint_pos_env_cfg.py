# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific cfg for the StackCube task.

Edit_mode_011: revert §2 action to LiftCube's stock
`mdp.JointPositionActionCfg` (scale=0.5, use_default_offset=True). The custom
EMA cumulative-delta wrapper from edit_mode_010 is removed; `mdp/actions.py`
and `mdp/actions_cfg.py` are docstring-only stubs again. Reward design
(LiftCube wholesale-copy, 7 terms) from edit_mode_008 retained.

ACTION: `mdp.JointPositionActionCfg` over the 7 Franka arm joints
(`panda_joint.*`). Per-step: `target = scale * action + offset`, where
`offset = default_joint_pos` because `use_default_offset=True`. Coupled with
a 1-D `mdp.BinaryJointPositionActionCfg` (open=0.04 / close=0.0). Total
action dim = 8.

Robot: `FRANKA_PANDA_HIGH_PD_CFG` retained for stable joint-position tracking
(LiftCube uses the low-PD variant; we keep HIGH_PD per the user instruction).

Init joint pose: the lowered-EE `FRANKA_INIT_JOINT_POS` (joint4 −0.3,
joint6 +0.3 from canonical) is preserved.

Cube asset: DexCube USD from `${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/`,
scaled by 0.86 to a 4.3 cm edge (the unscaled DexCube bounding box is 5 cm).
Mass overridden to 0.055 kg via `mass_props`.

End-effector sensor: `FrameTransformerCfg` rooted at `panda_link0` and tracking
`panda_hand` with a [0, 0, 0.1034] offset (LiftCube convention) — used by
`mdp.cube_0_ee_distance` reward via the `ee_frame` scene entity.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack_cube import mdp
from isaaclab_tasks.manager_based.manipulation.stack_cube.stack_cube_env_cfg import StackCubeEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# Cube geometry — DexCube USD scaled to a 4.3 cm edge length, 55 g mass.
# DexCube unscaled bbox is 5 cm; 0.043 / 0.05 = 0.86.
CUBE_SIZE = 0.043        # m — target edge length
CUBE_USD_SCALE = 0.86    # 0.043 / 0.05 = 0.86
CUBE_MASS = 0.055        # kg
CUBE_INIT_Z = CUBE_SIZE / 2.0  # center half a cube above table top → base on table


# Canonical Franka top-down-grasping init joint pose. The shipped
# FRANKA_PANDA_HIGH_PD_CFG default does NOT point the EE exactly straight down.
# These values DO — verified by smoke_s3 (rotated_local_z[..., 2] ≈ -1).
FRANKA_INIT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4,
    "panda_joint3": 0.0,
    "panda_joint4": -3 * math.pi / 4 - 0.3,   # bend elbow more to lower EE
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2 + 0.3,        # counter-rotate wrist by same Δ to keep EE pointing down
    "panda_joint7": math.pi / 4,
    "panda_finger_joint.*": 0.04,
}


@configclass
class FrankaStackCubeEnvCfg(StackCubeEnvCfg):
    """Franka + JointPositionAction (LiftCube layout) + BinaryGripper + two DexCubes."""

    def __post_init__(self):
        # Parent post-init first (sets decimation / episode_length_s / physics).
        super().__post_init__()

        # Robot — HIGH_PD variant for stable joint-position tracking. The init
        # joint pose puts the EE pointing straight down at reset.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=FRANKA_INIT_JOINT_POS),
        )
        # Enable contact-reporter API on the Franka USD so the ContactSensors
        # registered on panda_leftfinger / panda_rightfinger can read forces.
        self.scene.robot.spawn.activate_contact_sensors = True

        # Action — 7-D stock JointPositionAction over the Franka arm.
        # target = scale * action + offset, with offset = default_joint_pos
        # (since use_default_offset=True). LiftCube convention: scale=0.5.
        # Coupled with a 1-D binary gripper for a total action dim of 8.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # End-effector frame sensor — LiftCube convention. Used by
        # `mdp.cube_0_ee_distance` reward via the `ee_frame` scene entity.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        # Two DexCubes — Nucleus USD asset, scaled to 4.3 cm edge, 55 g mass.
        dex_cube_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        cube_spawn = UsdFileCfg(
            usd_path=dex_cube_usd,
            scale=(CUBE_USD_SCALE, CUBE_USD_SCALE, CUBE_USD_SCALE),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        )

        self.scene.cube_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_0",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, -0.10, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=cube_spawn,
        )
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.10, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=cube_spawn,
        )


@configclass
class FrankaStackCubeEnvCfg_PLAY(FrankaStackCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
