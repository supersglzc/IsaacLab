# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific cfg for the Lift-Box dual-arm task.

Two FR3 + Franka-hand robots cooperate to lift a 0.5 kg eurobox off the lab
table. Robot articulation + action stack mirror the insert_drawer cfg verbatim
(EMA cumulative-delta xyz EE-delta + binary gripper). The two robots use
symmetric init joint pose across the world y=0 plane.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

# Re-use insert_drawer's custom EMA cumulative-delta xyz action class verbatim.
from isaaclab_tasks.manager_based.manipulation.insert_drawer import mdp as insert_drawer_mdp
from isaaclab_tasks.manager_based.manipulation.insert_drawer.config.franka.joint_pos_env_cfg import (
    FR3_FRANKA_HAND_CFG,
)
from isaaclab_tasks.manager_based.manipulation.lift_box.lift_box_env_cfg import LiftBoxEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Box geometry — eurobox after MeshConverter recentering (geom xform offset by
# the centroid so the asset's local origin is at the box geometric center).
# Half-z extent probed from the STL via trimesh: 0.11025 m.
# Spawning at world (x, y, BOX_INIT_Z) places the bottom of the box on the
# table top (z ~ 0).
BOX_INIT_Z: float = 0.11025
BOX_MASS: float = 0.5  # kg

# Path to the converted eurobox USD. The file lives at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_box/config/franka/
# so `parents[8]` is the repo root, then `nautilus/assets/eurobox/eurobox.usd`.
_NAUTILUS_ASSETS = Path(__file__).resolve().parents[8] / "nautilus" / "assets"
_EUROBOX_USD_PATH = str(_NAUTILUS_ASSETS / "eurobox" / "eurobox.usd")


# Init joint poses (per robot). robot_0 = the existing insert_drawer FR3 pose.
# robot_1 is a symmetric flip across the world y=0 plane: flip the sign of
# joint1 (base yaw), joint3 (forearm yaw), joint5 (wrist yaw), joint7 (final
# wrist roll); keep joint2, joint4, joint6 identical. Finger joints open at
# 0.04 for both robots.
FRANKA_INIT_JOINT_POS_0 = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.87,
    "fr3_joint7": 0,
    "fr3_finger_joint.*": 0.04,
}
FRANKA_INIT_JOINT_POS_1 = {
    "fr3_joint1": 0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.87,
    "fr3_joint7": -1.57,
    "fr3_finger_joint.*": 0.04,
}


@configclass
class FrankaLiftBoxEnvCfg(LiftBoxEnvCfg):
    """Dual FR3 + EMA xyz EE-delta + BinaryGripper + eurobox."""

    def __post_init__(self):
        super().__post_init__()

        # Robot 0 — left side of the table (world y = +0.49).
        self.scene.robot_0 = FR3_FRANKA_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_0",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS_0,
                pos=(-0.274, 0.49, 0.01),
            ),
        )

        # Robot 1 — right side of the table (world y = -0.49). Same FR3 USD,
        # symmetric joint init across y=0; no world rotation (the joint flip
        # alone produces the mirrored arm posture).
        self.scene.robot_1 = FR3_FRANKA_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_1",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS_1,
                pos=(-0.274, -0.49, 0.01),
            ),
        )

        # Per-robot action stack — insert_drawer pattern (EMA cumulative-delta
        # xyz + RPY locked + DLS IK in absolute pose mode + binary gripper).
        # Workspace clamps are in EACH robot's ROOT frame after the body_offset
        # shift. The two robots sit at world y = +/- 0.49 facing each other; the
        # box sits centered at world (0, 0, BOX_INIT_Z). The clamps below let
        # each fingertip reach the box's near y-face (at world y = +/- 0.15)
        # from table level up to ~0.30 m above table.
        self.actions.arm_action_0 = insert_drawer_mdp.EMACumulativeDeltaPositionActionCfg(
            asset_name="robot_0",
            joint_names=["fr3_joint.*"],
            body_name="fr3_hand",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=(0.01, 0.01, 0.01),
            alpha=0.5,
            # robot_0 root is at world y=+0.49 -> "in front of" the robot in its
            # root frame is +x (toward world center) and -y (toward world y=0
            # side, i.e. toward y < +0.49). The clamps below let the fingertip
            # reach world y ~ -0.16 to +0.29 (= +0.49 + (-0.65 to -0.20)),
            # covering the box's +y face (world y ~ +0.15) and overlap with
            # the table center.
            pos_lower_limit=[0.20, -0.65, 0.005],
            pos_upper_limit=[0.55, -0.20, 0.40],
        )
        self.actions.gripper_action_0 = insert_drawer_mdp.BinaryJointPositionActionCfg(
            asset_name="robot_0",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )
        self.actions.arm_action_1 = insert_drawer_mdp.EMACumulativeDeltaPositionActionCfg(
            asset_name="robot_1",
            joint_names=["fr3_joint.*"],
            body_name="fr3_hand",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=(0.01, 0.01, 0.01),
            alpha=0.5,
            # robot_1 root is at world y=-0.49. Its root-frame +y points toward
            # +y world (since the robot has no world rotation), so the box's
            # -y face (world y ~ -0.15) is at root-frame y ~ +0.34 — i.e.
            # positive y in the clamp. The mirror of robot_0's clamps:
            pos_lower_limit=[0.20, 0.20, 0.005],
            pos_upper_limit=[0.55, 0.65, 0.40],
        )
        self.actions.gripper_action_1 = insert_drawer_mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )

        # Per-robot end-effector frame transformers — same pattern as
        # insert_drawer's ee_frame but one per robot.
        ee_marker_cfg_0 = FRAME_MARKER_CFG.copy()
        ee_marker_cfg_0.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker_cfg_0.prim_path = "/Visuals/FrameTransformer0"
        self.scene.ee_frame_0 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_0/fr3_link0",
            debug_vis=True,
            visualizer_cfg=ee_marker_cfg_0,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_0/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
                ),
            ],
        )
        ee_marker_cfg_1 = FRAME_MARKER_CFG.copy()
        ee_marker_cfg_1.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker_cfg_1.prim_path = "/Visuals/FrameTransformer1"
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/fr3_link0",
            debug_vis=True,
            visualizer_cfg=ee_marker_cfg_1,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
                ),
            ],
        )

        # Eurobox — converted from /home/steven/Downloads/eurobox.stl. The
        # MeshConverter recentered the geometry under the root Xform so the
        # asset origin sits at the box centroid. Spawning at z=BOX_INIT_Z
        # places the bottom on the table (z=0).
        #
        # 90° Z-rotation: the box's LOCAL geometry is 0.40 (x) x 0.30 (y) x
        # 0.22 (z). Without the rotation the long axis (0.40) runs along world
        # +x — the robots sit at world y=±0.49, so each robot would grasp
        # the LONG face (0.40 m wide). Rotating 90° around +Z swaps local x
        # and y in world coords: world-x extent becomes 0.30 (SHORT) and
        # world-y extent becomes 0.40 (LONG). Now each robot grasps the
        # SHORT y-end face (0.30 m wide x 0.22 m tall), which is the only
        # geometry a parallel-jaw gripper can wrap around at the rim.
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, BOX_INIT_Z],
                rot=[0.7071068, 0.0, 0.0, 0.7071068],  # 90° around +Z
            ),
            spawn=UsdFileCfg(
                usd_path=_EUROBOX_USD_PATH,
                mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
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

        # Grasp-point frame markers — visualize where each robot should
        # close its gripper. Offsets are in the box's BODY-LOCAL frame so
        # they track the box if it tilts / rotates further during the
        # rollout. With the 90° Z-rotation baked in, local +x maps to
        # world +y, so local (+0.20, 0, +0.11025) sits at the top of the
        # box's world +y short face (z = BOX_INIT_Z + half_z = 0.2205).
        grasp_marker_cfg_0 = FRAME_MARKER_CFG.copy()
        grasp_marker_cfg_0.markers["frame"].scale = (0.08, 0.08, 0.08)
        grasp_marker_cfg_0.prim_path = "/Visuals/BoxGraspFrame0"
        self.scene.grasp_frame_0 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            debug_vis=True,
            visualizer_cfg=grasp_marker_cfg_0,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Box",
                    name="box_grasp_0",
                    offset=OffsetCfg(pos=[0.20, 0.0, 0.11025]),
                ),
            ],
        )
        grasp_marker_cfg_1 = FRAME_MARKER_CFG.copy()
        grasp_marker_cfg_1.markers["frame"].scale = (0.08, 0.08, 0.08)
        grasp_marker_cfg_1.prim_path = "/Visuals/BoxGraspFrame1"
        self.scene.grasp_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            debug_vis=True,
            visualizer_cfg=grasp_marker_cfg_1,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Box",
                    name="box_grasp_1",
                    offset=OffsetCfg(pos=[-0.20, 0.0, 0.11025]),
                ),
            ],
        )


@configclass
class FrankaLiftBoxEnvCfg_PLAY(FrankaLiftBoxEnvCfg):
    """Smaller, no-noise variant for visualization / eval."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
