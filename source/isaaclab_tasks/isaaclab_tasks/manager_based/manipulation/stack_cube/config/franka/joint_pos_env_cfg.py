# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific cfg for the StackCube task.

Edit_mode_014 (§2): swap stock `mdp.JointPositionActionCfg` for a custom
position-only EMA EE-delta term (`EMACumulativeDeltaPositionActionCfg`).
The policy outputs a 3-D position delta `(dx, dy, dz)`; the EE quaternion
is FIXED at the post-reset value (no rotation channels). EMA-smooths
against the previously-applied position target then forwards an absolute
7-D pose command to the IK controller.

ACTION: `mdp.EMACumulativeDeltaPositionActionCfg` over the 7 Franka arm
joints (`panda_joint.*`) targeting body `panda_hand` with a 0.1034 m z
offset (matching the `ee_frame` convention). Per-step:
    delta_t  = delta_{t-1} + scale * a_t                       (3-D)
    abs_pos  = init_ee_pos + delta_t                            (3-D)
    target   = alpha * abs_pos + (1 - alpha) * prev_applied_pos
    cmd_quat = init_ee_quat                                    (locked)
Coupled with a 1-D `mdp.BinaryJointPositionActionCfg` (open=0.04 /
close=0.0). Total action dim = 4 (3 xyz + 1 gripper).

Robot: `FRANKA_PANDA_HIGH_PD_CFG` retained for stable joint-position tracking
(LiftCube uses the low-PD variant; we keep HIGH_PD per the user instruction).

Init joint pose: the lowered-EE `FRANKA_INIT_JOINT_POS` (joint4 −0.3,
joint6 +0.3 from canonical) is preserved.

Cube asset: DexCube USD from `${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/`,
scaled by 0.86 to a 4.3 cm edge (the unscaled DexCube bounding box is 5 cm).
Mass overridden to 0.055 kg via `mass_props`. Edit_mode_012 installs THREE
cubes (cube_0, cube_1, cube_2) spread in XY so the ±5 cm reset perturbation
does not collide them at reset.

End-effector sensor: `FrameTransformerCfg` rooted at `panda_link0` and tracking
`panda_hand` with a [0, 0, 0.1034] offset (LiftCube convention) — used by
`mdp.cube_0_ee_distance` reward via the `ee_frame` scene entity.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
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


# Cube geometry — DexCube USD scaled to a 4.3 cm edge length, 55 g mass.
# DexCube unscaled bbox is 5 cm; 0.043 / 0.05 = 0.86.
CUBE_SIZE = 0.043        # m — target edge length
CUBE_USD_SCALE = 0.86    # 0.043 / 0.05 = 0.86
CUBE_MASS = 0.055        # kg
CUBE_INIT_Z = CUBE_SIZE / 2.0  # center half a cube above table top → base on table

# Path to the FR3 + Franka-hand USD converted from
# `<agentic>/franka_description/urdfs/fr3_franka_hand.urdf`.
_FR3_USD_PATH = "/home/steven/code/agentic/IsaacLab/nautilus/assets/fr3/fr3_franka_hand.usd"

# FR3 + Franka-hand robot config — built fresh (the shipped FRANKA_PANDA_*
# cfgs assume the Isaac Sim Panda USD with `panda_*` joint/link names; the
# FR3 URDF prefixes everything with `fr3_`). Tuned with HIGH_PD stiffness so
# the joint-position tracker is stable enough for absolute-IK control.
FR3_FRANKA_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FR3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(),  # filled in __post_init__
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# Init joint pose mirrors bidex `bidex/env/tasks/StackCube/env_cfg.py`:
# panda_joint1 rotated 45° so the EE arcs over the table center given the
# robot's base position `pos=(-0.274, -0.49, 0.01)`. Joints 2–7 unchanged
# numerically from the prior Triton init; only renamed `panda_*` → `fr3_*`.
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
class FrankaStackCubeEnvCfg(StackCubeEnvCfg):
    """Franka + EMACumulativeDeltaPositionAction (xyz only, fixed RPY) + BinaryGripper + three DexCubes."""

    def __post_init__(self):
        # Parent post-init first (sets decimation / episode_length_s / physics).
        super().__post_init__()

        # Robot — FR3 + Franka-hand from the agentic `franka_description` URDF
        # (converted to USD via `scripts/tools/convert_urdf.py`). Joint/body
        # names are prefixed `fr3_*` (vs the prior Panda `panda_*`).
        # Base pos and joint1 angle mirror bidex StackCube so the EE arcs over
        # the table center.
        self.scene.robot = FR3_FRANKA_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS,
                pos=(-0.274, 0.49, 0.01),
            ),
        )

        # Action — Cartesian EE-delta with locked RPY. Policy outputs 3-D xyz
        # delta; IK target = fingertip TCP at body_offset=(0,0,0.2) — same
        # frame as ee_frame OffsetCfg. pos_lower_limit[2]=0.005 keeps the
        # fingertip from driving through the table (root-frame z; world floor
        # = root_z + 0.005 ≈ 0.015 given robot base at world z=0.01).
        self.actions.arm_action = mdp.EMACumulativeDeltaPositionActionCfg(
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
            # Workspace clamp in ROBOT ROOT FRAME (robot base @ world
            # (-0.274, 0.49, 0.01)). Tight bounds matching where the cubes
            # can spawn (per `reset_cube_*` events: world x ∈ [-0.20, 0.20],
            # world y ∈ [0.0, 0.4]) plus ~3 cm margin so the EE can swing
            # around a cube edge to grasp it. z-ceiling = target_z(world
            # 0.1075) + 5 cm headroom = 0.1575 world = 0.15 root. Keeps the
            # IK from being asked to drive outside the reachable workspace
            # and prevents lift-overshoot above the stacking height.
            pos_lower_limit=[ 0.05, -0.52, 0.005],
            pos_upper_limit=[ 0.50, -0.05, 0.16 ],
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )

        # End-effector frame sensor — LiftCube convention. Used by
        # `mdp.cube_0_ee_distance` reward via the `ee_frame` scene entity.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
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

        # Three DexCubes — tower base = cube_2 on the table, cube_1 on cube_2,
        # cube_0 on cube_1 at the goal pose. Initial XY positions are spread
        # so the random ±5 cm reset perturbation cannot collide them.
        self.scene.cube_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_0",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=cube_spawn,
        )
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=cube_spawn,
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
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
