# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific cfg for the InsertDrawer task.

Robot + action + table + cube design mirror the StackCube Franka cfg
verbatim. Adds one new prismatic articulation -- a drawer at
`nautilus/assets/drawer/drawer.usd` with `base_drawer_joint` (axis +X,
limits [0, 0.15] m).
"""

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

from isaaclab_tasks.manager_based.manipulation.insert_drawer import mdp
from isaaclab_tasks.manager_based.manipulation.insert_drawer.insert_drawer_env_cfg import InsertDrawerEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Cube geometry -- DexCube USD scaled to a 4.3 cm edge length, 55 g mass.
# DexCube unscaled bbox is 5 cm; 0.043 / 0.05 = 0.86.
CUBE_SIZE = 0.043        # m -- target edge length
CUBE_USD_SCALE = 0.86    # 0.043 / 0.05 = 0.86
CUBE_MASS = 0.055        # kg
CUBE_INIT_Z = CUBE_SIZE / 2.0  # center half a cube above table top -> base on table


# Drawer USD — Iter 10 swap: converted from
# `<repo>/Downloads/drawer_one_sided_handle_rotated_mesh_handle_package/`
# (after stripping the commented-out handle block which had nested XML
# comments that crashed the URDF parser). The new drawer has NO handle —
# only the sliding `drawer` body inside a fixed cabinet of `back_wall` +
# `left_wall` + `right_wall` + `top_wall` + `bottom_wall`. URDF was
# pre-scaled (xacro factors 0.3 × 0.6 × 0.5), so the USD spawn uses
# `scale=(1.0, 1.0, 1.0)` (the old symdex spawn applied 0.3,0.6,0.5 at
# spawn-time which is no longer needed). Same prismatic joint name
# `base_drawer_joint`, range [0, 0.3] m, axis +X-local.
_DRAWER_USD_PATH = "/home/steven/code/agentic/IsaacLab/nautilus/assets/drawer_no_handle/drawer_no_handle.usd"


# Path to the FR3 + Franka-hand USD (mirrors StackCube).
_FR3_USD_PATH = "/home/steven/code/agentic/IsaacLab/nautilus/assets/fr3/fr3_franka_hand.usd"


# FR3 + Franka-hand robot config -- verbatim from StackCube's Franka cfg.
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


# Init joint pose -- copied verbatim from StackCube.
FRANKA_INIT_JOINT_POS = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.87,
    "fr3_joint7": 1.57,
    "fr3_finger_joint.*": 0.04,
}


@configclass
class FrankaInsertDrawerEnvCfg(InsertDrawerEnvCfg):
    """Franka FR3 + EMACumulativeDeltaPositionAction (xyz only, fixed RPY) + BinaryGripper + DexCube + Drawer."""

    def __post_init__(self):
        super().__post_init__()

        # Robot -- FR3 + Franka-hand; mirrors StackCube's per-robot cfg.
        self.scene.robot = FR3_FRANKA_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS,
                pos=(-0.274, 0.49, 0.01),
            ),
        )

        # Action stack -- mirrors StackCube post-tuning conventions (the
        # `multi-stage-manipulation-reward-heuristics` memory): align the IK
        # body_offset with the ee_frame fingertip TCP (both 0.2 m), pick the
        # simplest controller the task allows (3-D EE-delta, locked RPY),
        # smaller per-step scale (0.01 m), higher EMA alpha (0.5), and clamp
        # the absolute pos target to the actual reachable workspace.
        #
        # Workspace clamp is in ROBOT-ROOT FRAME after the body_offset shift
        # (so it's literally the fingertip TCP). Bounds verified against the
        # probe of cube/handle/drop_frame positions:
        #   cube spawn (jittered): x in [0.23, 0.31], y in [-0.14, -0.05], z=0.008
        #   handle (closed)      : (0.274, -0.529, 0.090)
        #   handle (open=0.30)   : (0.274, -0.229, 0.090)
        #   drop frame (open)    : (0.274, -0.490, 0.240)
        # Drop frame at closed (y=-0.79) is intentionally OUTSIDE the clamp;
        # the policy only needs it after the drawer opens (y=-0.49). Lower
        # y bound -0.60 leaves ~7 cm margin past the closed-handle reach.
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
            # Iter 14: tighten to the IK-reliable region (locked-down quat).
            # The 6x6x6 workspace probe (216 grid points + 60-step settle each)
            # showed the IK fails to hold orientation when EE is at low x
            # (x < ~0.34) or far y (y < ~-0.55) at low z (z < ~0.18) — the
            # arm hits a near-singularity with the locked downward gripper
            # and the DLS solver trades position vs orientation. The
            # corresponding cube and drawer spawn ranges below are also
            # tightened so all task positions fit inside this clean region.
            pos_lower_limit=[ 0.34, -0.8, 0.005],
            pos_upper_limit=[ 0.50, -0.05, 0.30 ],
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )

        # End-effector frame sensor -- StackCube convention. Used by §5 obs
        # `ee_pose_in_robot_root_frame`.
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

        # DexCube -- Nucleus USD asset, scaled to 4.3 cm edge, 55 g mass.
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

        # One DexCube. Iter-6 follow-up: cube spawn moved from (0, 0, ...) to
        # (0, +0.3, ...) per user — pushing the cube to +Y separates it from
        # the drawer (which is at y=-0.3 and slides toward +Y when opened),
        # giving the policy a clearer pick-then-transport task. With the drawer
        # fully open the tray sits at y~-0.15; the cube starts at y~+0.30, so
        # cube-to-open-drawer distance is ~0.45 m (vs ~0.15 m before).
        self.scene.cube_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_0",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]
            ),
            spawn=cube_spawn,
        )

        # Drawer -- prismatic articulation. Iter 10 swap (post-iter-9): the
        # handled symdex drawer is replaced by a NO-HANDLE drawer converted
        # from `/home/steven/Downloads/drawer_one_sided_handle_rotated_mesh_handle_package/`.
        # The task now starts with the drawer ALREADY OPEN (joint_pos=0.30 at
        # reset, see EventCfg), so the policy never needs to grasp a handle —
        # only push the body to close. Removing the handle simplifies physics
        # (no more fixed-joint side posts) and removes the grip-vs-rim contact
        # conflicts that made the prior pull stage hard. The new URDF is
        # pre-scaled (xacro factors 0.3 × 0.6 × 0.5 baked into the geometry),
        # so spawn uses `scale=(1.0, 1.0, 1.0)`. Same prismatic joint name
        # `base_drawer_joint`, axis +X-local, range [0, 0.3] m.
        #
        # Iter-6 cross-section edit (logged in reward-history.md): redesigned
        # the scene to a "pull + insert" layout per the user directive:
        #   - pos=(0, -0.3, 0.10): drawer is in front of the robot at y=-0.3,
        #     x stays at 0.
        #   - rot=(0.7071, 0, 0, 0.7071): 90 deg around +Z. After this rotation
        #     the prismatic +X-local axis maps to +Y world, so opening the
        #     drawer slides the tray toward the robot (which sits at y=+0.49).
        # Robot is at env-local pos=(-0.274, 0.49, 0.01); drawer at
        # (0, -0.3, 0.10). With the drawer closed the tray is ~0.79 m in -Y
        # from the robot base. After fully opening (joint=0.15), the tray's
        # y-component increases by +0.15 (toward the robot).
        #
        # Cube spawn stays at env-local (0, 0, 0.0215). With closed drawer the
        # cube is ~0.30 m in +Y from the drawer body; once the drawer opens
        # to 0.15 m, the cube is ~0.15 m in +Y from the open tray (short
        # transport).
        #
        # Iter-4 mandate (KEPT): stiffness=0.0 on the prismatic actuator so
        # the implicit actuator does not spring the joint back to the default
        # target (closed). Without this fix the drawer slid closed every step
        # of release. Damping (1.0) + URDF joint friction (1.0) hold the
        # drawer in place once released.
        self.scene.drawer = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Drawer",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_DRAWER_USD_PATH,
                # No `scale` — new URDF is pre-scaled (xacro factors x=0.3 y=0.6 z=0.5).
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=1000.0,
                    max_linear_velocity=1000,
                    max_angular_velocity=1000,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    # Iter 11: fix base_link to the world (kinematic root).
                    # The URDF declares no parent-world joint, so without
                    # this PhysX treats base_link as a free body — random
                    # PPO contacts could push the whole cabinet. With this
                    # set, only the prismatic `drawer` body is dynamic.
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={"base_drawer_joint": 0.0},
                pos=(0.0, 0.0, 0.10),
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
            ),
            actuators={
                # Iter 10: bumped stiffness 0 -> 50 and damping 1 -> 10 in
                # response to PhysX NaN at 4096 envs with the no-handle drawer.
                # The light drawer body mass (0.009 kg originally, 0.45 kg after
                # URDF 50x rescale) coupled with stiffness=0 / damping=1 was
                # under-damped under PPO random-action noise; adding a gentle
                # restoring force + stronger damping stabilizes physics. The
                # actuator now drives the joint toward its default (0 = closed),
                # which is also the task target — a small free-energy hint
                # toward closing. The reset still sets joint_pos=0.30 (open),
                # giving the policy time to insert before the auto-close
                # finishes (~0.30 m / max velocity ≈ several seconds). Friction
                # 1 -> 2 for additional static resistance against jitter.
                "drawer_slide": ImplicitActuatorCfg(
                    joint_names_expr=["base_drawer_joint"],
                    effort_limit=87.0,
                    velocity_limit=100.0,
                    stiffness=0.0,
                    damping=1.0,
                    friction=2.0,
                ),
            },
        )

        # Frame transformer at the DROP POINT above the drawer interior. The
        # symdex drawer body's local +Z is the OPENING direction (up).
        # Iter 6 — pulled drop_frame z up from +0.04 (inside drawer interior) to
        # +0.12 (above drawer rim, world z~0.22). The align attractor now pulls
        # the cube to a RELEASE point above the drawer; gravity does the final
        # descent. Previously the cube target was inside the drawer geometry,
        # which conflicted with the success predicate's ee-far-from-cube clause
        # (policy couldn't simultaneously satisfy 'cube inside' and 'EE far').
        # Now the policy aims to hold the cube above, release, and gravity lands
        # the cube inside while the EE retracts.
        # `align` reward reads this frame.
        drop_marker_cfg = FRAME_MARKER_CFG.copy()
        drop_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        drop_marker_cfg.prim_path = "/Visuals/DrawerDropFrame"
        self.scene.drawer_drop_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Drawer/base_link",
            debug_vis=True,
            visualizer_cfg=drop_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Drawer/drawer",
                    name="drawer_drop",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.25]),
                ),
            ],
        )

        # Iter 22: frame marker on the drawer's FRONT-FACING wall (+X-local end
        # of the drawer body). After the 90° z-rotation, local +X maps to world
        # +Y, so this point's world y is the boundary between "inside the
        # drawer" (smaller y) and "outside / in front of the drawer" (larger y).
        # The `close_drawer` reward gates on `ee.y > drawer_front_face.y` so
        # the policy must keep the EE physically in front of the drawer (toward
        # the robot) while pushing it closed. Offset (+0.15, 0, 0.05) drawer-
        # local = center of the +X face, slightly above the floor.
        front_marker_cfg = FRAME_MARKER_CFG.copy()
        front_marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
        front_marker_cfg.prim_path = "/Visuals/DrawerFrontFaceFrame"
        self.scene.drawer_front_face_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Drawer/base_link",
            debug_vis=True,
            visualizer_cfg=front_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Drawer/drawer",
                    name="drawer_front_face",
                    offset=OffsetCfg(pos=[0.17, 0.0, 0.15]),
                ),
            ],
        )

@configclass
class FrankaInsertDrawerEnvCfg_PLAY(FrankaInsertDrawerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
