# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific cfg for the Stir-Bowl task.

Single FR3 + Franka-hand robot. EE controller = `EMACumulativeDeltaPositionAction`
(3-D xyz delta + locked RPY), same as stack_cube / lift_box. Gripper =
`BinaryJointPositionAction`. Scene wires the procedural spoon, stand, bowl
and three blue ball spheres.

Robot init joint pose = the stack_cube canonical pose (top-down gripper
orientation). Robot base at world (-0.274, +0.49, 0.01) — byte-for-byte
match with stack_cube. The bowl + stand sit in the reachable +x / -y
quadrant relative to the robot root frame (bowl/stand world y in
[0.10, 0.30] is south of robot base y=0.49, i.e. root-frame y < 0).
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

# Re-use insert_drawer's EMA cumulative-delta xyz action class verbatim (same
# pattern lift_box uses). FR3 robot config is sourced from the existing
# insert_drawer config (cross-task dependency mirrors the lift_box approach).
from isaaclab_tasks.manager_based.manipulation.insert_drawer import mdp as insert_drawer_mdp
from isaaclab_tasks.manager_based.manipulation.insert_drawer.config.franka.joint_pos_env_cfg import (
    FR3_FRANKA_HAND_CFG,
)
from isaaclab_tasks.manager_based.manipulation.stir_bowl.stir_bowl_env_cfg import StirBowlEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Asset directory — repo-relative. This file is at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stir_bowl/config/franka/
# so `parents[8]` is the repo root, then `nautilus/assets/isaac-stir-bowl/...`.
_NAUTILUS_ASSETS = Path(__file__).resolve().parents[8] / "nautilus" / "assets" / "isaac-stir-bowl"
_SPOON_USD_PATH = str(_NAUTILUS_ASSETS / "spoon" / "spoon.usd")
_STAND_USD_PATH = str(_NAUTILUS_ASSETS / "stand" / "stand.usd")
_BOWL_USD_PATH = str(_NAUTILUS_ASSETS / "bowl" / "bowl.usd")


# Init joint pose — stack_cube canonical pose (top-down gripper orientation).
FRANKA_INIT_JOINT_POS = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.87,
    "fr3_joint7": 0.0,
    "fr3_finger_joint.*": 0.04,
}


@configclass
class FrankaStirBowlEnvCfg(StirBowlEnvCfg):
    """Single FR3 + EMA xyz EE-delta + BinaryGripper + spoon + stand + bowl + 3 balls."""

    def __post_init__(self):
        super().__post_init__()

        # Robot — FR3 + Franka-hand. Base at world (-0.274, +0.49, 0.01) —
        # byte-for-byte match with stack_cube's canonical Franka init pose.
        # The bowl/stand sit at world y in [0.10, 0.30] (south of robot
        # base y=+0.49), so they're in the robot's root-frame -y forward
        # zone (root +y points along world +y since the robot has no world
        # rotation).
        self.scene.robot = FR3_FRANKA_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS,
                pos=(-0.274, 0.49, 0.01),
            ),
        )

        # Action — EMA cumulative-delta xyz EE-delta + binary gripper.
        # Workspace clamp is in ROBOT ROOT FRAME after the body_offset shift
        # (body_offset=(0,0,0.2) → TCP is 20 cm "forward" of the wrist along
        # the wrist's local +Z, which for the canonical top-down init pose
        # points along world -Z, i.e. TCP is the gripper-tip pose). Robot
        # base @ world (-0.274, +0.49, 0.01). Target the EE TCP to reach
        # world positions:
        #   x ∈ [0.05, 0.55]  — covers stand x=0.10 and bowl x=0.10 with margin
        #   y ∈ [0.05, 0.40]  — covers stand y=0.10 and bowl y=0.30 with margin;
        #                       stops BEFORE robot base at y=0.49
        #   z ∈ [0.005, 0.50] — table top up to ~50 cm above
        # Convert each bound to robot root frame (root coords = world coords
        # − robot.pos; root frame has no rotation):
        #   root x ∈ [0.05 - (-0.274), 0.55 - (-0.274)] = [0.324, 0.824]
        #   root y ∈ [0.05 - 0.49,     0.40 - 0.49    ] = [-0.44,  -0.09]
        #   root z ∈ [0.005 - 0.01,    0.50 - 0.01    ] = [-0.005,  0.49]
        self.actions.arm_action = insert_drawer_mdp.EMACumulativeDeltaPositionActionCfg(
            asset_name="robot",
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
            pos_lower_limit=[0.324, -0.44, -0.005],
            pos_upper_limit=[0.824, -0.09, 0.49],
        )
        self.actions.gripper_action = insert_drawer_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )

        # End-effector frame sensor — stack_cube convention.
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=True,
            visualizer_cfg=ee_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
                ),
            ],
        )

        # Spoon — procedural STL→USD. Mass 50 g, gravity ON, dynamic rigid body.
        # The spoon's local origin is at the geometric center (half-z = 0.065).
        # `init_state.pos` is zeroed so the env-local position is determined
        # entirely by `EventCfg.reset_spoon.params["pose_range"]` (the
        # IsaacLab reset event adds default_root + env_origins + pose_range_sample;
        # zeroing default_root makes the smoke / DR semantics easier to reason
        # about — pose_range is the entire env-local target).
        #
        # `init_state.rot` is the IDENTITY quaternion (1, 0, 0, 0) — the spoon
        # stands UPRIGHT with its long axis aligned with WORLD +Z. The spoon
        # STL is built with bowl-end at spoon-local -Z and handle-top at
        # spoon-local +Z (see `make_assets.py:_make_spoon`), so with identity
        # rotation:
        #   bowl-end (spoon -Z) → world -Z (POINTING DOWN)
        #   handle  (spoon +Z) → world +Z (POINTING UP)
        # i.e. the spoon stands vertically with its working end at the bottom,
        # supported laterally by the two upper pegs of the fork-shape stand
        # (see `_make_stand`). The gripper grasps the upper handle from above
        # and lifts straight up.
        self.scene.spoon = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spoon",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_SPOON_USD_PATH,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                activate_contact_sensors=True,
            ),
        )

        # Stand — procedural STL→USD. Kinematic-enabled rigid body so the
        # reset event manager can resample its root pose; gravity disabled so
        # it stays put once placed. init_state.pos zeroed (same convention as
        # spoon / bowl / balls) so the reset pose_range is the entire
        # env-local target.
        self.scene.stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Stand",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_STAND_USD_PATH,
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                ),
            ),
        )

        # Bowl — procedural STL→USD. Kinematic-enabled rigid body so the
        # reset event manager can resample its root pose; gravity disabled
        # so it stays pinned to the reset pose under contact forces from the
        # balls + spoon (Change 4 — the user wants the bowl FIXED to the
        # world, mirroring the stand). Mass is irrelevant for kinematic bodies
        # but is left set for completeness.
        self.scene.bowl = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_BOWL_USD_PATH,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                ),
            ),
        )

        # Balls — SphereCfg primitives (NOT meshes), radius 0.015 m (3 cm dia,
        # halved from the bidex 6 cm so the balls sit deeper inside the now
        # taller bowl), mass 0.1 kg, blue PreviewSurface, collision props
        # enabled.
        ball_spawn = sim_utils.SphereCfg(
            radius=0.015,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        )
        self.scene.ball_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=ball_spawn,
        )
        self.scene.ball_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=ball_spawn,
        )
        self.scene.ball_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=ball_spawn,
        )

        # Spoon handle frame — marker at the upper half of the handle, where
        # the gripper grasps. OffsetCfg in spoon body-local coords; the spoon's
        # geometric center is the local origin, so +0.06 in z lands ~5 mm
        # below the top of the handle.
        spoon_marker_cfg = FRAME_MARKER_CFG.copy()
        spoon_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        spoon_marker_cfg.prim_path = "/Visuals/SpoonHandleFrame"
        self.scene.spoon_handle_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Spoon",
            debug_vis=True,
            visualizer_cfg=spoon_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Spoon",
                    name="spoon_handle",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.06]),
                ),
            ],
        )

        # Bowl center frame — marker at the bowl's center-top (where stirring
        # should occur). Bowl origin is at its bottom; the bowl interior floor
        # is at z=0.005, the bowl is ~5 cm tall — so 0.03 m above the origin
        # lands ~2 cm above the bowl floor (inside the cavity).
        bowl_marker_cfg = FRAME_MARKER_CFG.copy()
        bowl_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        bowl_marker_cfg.prim_path = "/Visuals/BowlCenterFrame"
        self.scene.bowl_center_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            debug_vis=True,
            visualizer_cfg=bowl_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bowl",
                    name="bowl_center",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.03]),
                ),
            ],
        )


@configclass
class FrankaStirBowlEnvCfg_PLAY(FrankaStirBowlEnvCfg):
    """Smaller, no-noise variant for visualization / eval."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
