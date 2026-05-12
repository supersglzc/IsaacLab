# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.push import mdp
from isaaclab_tasks.manager_based.manipulation.push.push_env_cfg import PushBlockEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaPushBlockEnvCfg(PushBlockEnvCfg):
    def __post_init__(self):
        # Parent post-init first.
        super().__post_init__()

        # Robot — standard PD-gain Franka (no IK, so HIGH_PD variant not needed).
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions — absolute joint-position on the 7 arm joints.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # Gripper — keep binary action term for canonical compatibility, but force
        # both expressions to 0.0 so the gripper stays closed and acts as a flat pusher.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.0},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Goal command — bind the goal frame to panda_hand (debug-vis attaches to this body).
        self.commands.object_pose.body_name = "panda_hand"

        # Object — small DexCube block, spawned flat at table center.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


@configclass
class FrankaPushBlockEnvCfg_PLAY(FrankaPushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
