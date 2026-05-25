# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the Lift-Box dual-arm task.

Two FR3 + Franka-hand robots cooperate to lift a eurobox (0.4 x 0.3 x 0.22 m,
0.5 kg) off a lab table. Scene timing + table + ground + lighting + sim/physx
knobs mirror `manipulation/insert_drawer/` verbatim. Per-robot subclass adds
the two arms, the box, ee_frame_{0,1}, and the contact sensors.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp


# Repo-relative table asset path. This file is at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_box/
# so `parents[6]` is the repo root, then `nautilus/assets/table/...`.
_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "nautilus" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)


##
# Scene definition
##


@configclass
class LiftBoxSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + two FR3 robots (filled by subclass) + box +
    per-robot ee_frames + per-robot finger contact sensors + lights."""

    # Robots — filled by the per-robot subclass via __post_init__.
    robot_0: ArticulationCfg = MISSING
    robot_1: ArticulationCfg = MISSING

    # End-effector frame sensors — one per robot.
    ee_frame_0: FrameTransformerCfg = MISSING
    ee_frame_1: FrameTransformerCfg = MISSING

    # The box being lifted.
    box: RigidObjectCfg = MISSING

    # Box grasp-point frame markers (debug viz only — no rl-side reader).
    # After the box's 90° Z-rotation the box's local +x maps to world +y;
    # the two markers sit at the TOP of the box's two short y-end faces:
    #   grasp_frame_0  -> local (+x_extent, 0, +z_extent) = (+0.20, 0, +0.110)
    #                   -> world (0, +0.20, BOX_INIT_Z + 0.110) — for robot_0.
    #   grasp_frame_1  -> local (-x_extent, 0, +z_extent) = (-0.20, 0, +0.110)
    #                   -> world (0, -0.20, BOX_INIT_Z + 0.110) — for robot_1.
    grasp_frame_0: FrameTransformerCfg = MISSING
    grasp_frame_1: FrameTransformerCfg = MISSING

    # Contact sensors on each robot's fingertips, filtered against the box.
    finger_left_contact_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/fr3_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_right_contact_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/fr3_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_left_contact_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/fr3_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_right_contact_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/fr3_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )

    # Table — same as insert_drawer (lab table USD; surface at z ~ 0).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=_TABLE_USD_PATH),
    )

    # Ground plane — same convention as insert_drawer (below the table by 0.82 m).
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.82]),
        spawn=GroundPlaneCfg(),
    )

    # Lights — same as insert_drawer.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Per-robot action specs — filled by the per-robot subclass.

    Each robot exposes (arm_xyz_delta, gripper_binary) for a total per-robot
    action_dim of 4. The full vector concatenates declaration order:
        arm_action_0 (3) + gripper_action_0 (1) + arm_action_1 (3) + gripper_action_1 (1)
    -> total action_dim = 8.
    """

    arm_action_0: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_0: "mdp.BinaryJointPositionActionCfg" = MISSING
    arm_action_1: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_1: "mdp.BinaryJointPositionActionCfg" = MISSING


@configclass
class ObservationsCfg:
    """Observation specs — 33-D concatenated policy obs.

    Term order:
        ee_pose_0              (7,) — robot_0 end-effector pose in robot_0's root frame
        ee_pose_1              (7,) — robot_1 end-effector pose in robot_1's root frame
        box_position_in_world  (3,) — box xyz in env-local world frame
        box_quat_in_world      (4,) — box quaternion (wxyz) in world frame
        gripper_joint_pos_0    (2,) — robot_0 finger joint positions
        gripper_joint_pos_1    (2,) — robot_1 finger joint positions
        last_action            (8,) — full last action vector
    Total = 7 + 7 + 3 + 4 + 2 + 2 + 8 = 33.

    `enable_corruption=True` (per-term noise slots default to no-op; the
    dr-generator may widen later).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_0 = ObsTerm(
            func=mdp.ee_pose_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_0"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame_0"),
            },
        )
        ee_pose_1 = ObsTerm(
            func=mdp.ee_pose_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot_1"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame_1"),
            },
        )
        box_position_in_world = ObsTerm(func=mdp.box_position_in_world)
        box_quat_in_world = ObsTerm(func=mdp.box_quat_in_world)
        gripper_joint_pos_0 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=["fr3_finger.*"])},
        )
        gripper_joint_pos_1 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=["fr3_finger.*"])},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms only (§3). §7 DR is not in scope for this task-generator run;
    the dr-generator may add more terms later.

    Ranges are pinned to point intervals (or the small +/- 3 cm box jitter)
    so the §3 reset smoke can read deterministic values.
    """

    # Robot 0 joints: pinned to URDF home (scale 1.0). With no asset_cfg the
    # default is "robot" which does NOT exist in this dual-arm scene — we
    # therefore bind explicitly to "robot_0" / "robot_1".
    reset_robot_0_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_0"),
        },
    )
    reset_robot_1_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_1"),
        },
    )

    # Box pose: small +/- 3 cm xy perturbation around its init_state.pos; z pinned.
    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class RewardsCfg:
    """Reward ladder (composer = sum, sign = positive=good).

    All weights scaled down 80x from the iter-0 converged set so total
    episodic return lands around ~105 instead of ~8400 — same ratios,
    same convergence behaviour, more interpretable magnitudes.

    Per-stage saturated per-step magnitudes (raw weights; dt-multiplier
    removed from RewardManager.compute in this fork):

      ee_<i>_to_grasp_<i>          0.0125 * 2 robots  = 0.025 / step
      grasp_contact_<i>            0.025  * 2 robots  = 0.05  / step (only when held)
      lift_height                  0.1875             = 0.19  / step (dual contact)
      box_xy_align                 0.125              = 0.125 / step (box lifted)
      success_bonus               100.0               = +100  one-shot

    Episode ceiling (200 steps): dense ~ 78; sparse success ~ 100.
    Ladder strictly increasing: 0.025 -> 0.05 -> 0.19 -> 0.125 (gated) -> 100.
    """

    # Reach — per-robot dense EE -> grasp-point attractors.
    ee_0_to_grasp_0 = RewTerm(
        func=mdp.ee_to_grasp_distance,
        params={
            "std": 0.15,
            "ee_frame_cfg": SceneEntityCfg("ee_frame_0"),
            "grasp_frame_cfg": SceneEntityCfg("grasp_frame_0"),
        },
        weight=0.0125,
    )
    ee_1_to_grasp_1 = RewTerm(
        func=mdp.ee_to_grasp_distance,
        params={
            "std": 0.15,
            "ee_frame_cfg": SceneEntityCfg("ee_frame_1"),
            "grasp_frame_cfg": SceneEntityCfg("grasp_frame_1"),
        },
        weight=0.0125,
    )

    # Grasp — per-robot binary "both fingers in contact with box" indicator.
    grasp_contact_0 = RewTerm(
        func=mdp.grasp_contact,
        params={"robot_idx": 0, "contact_force_threshold": 1e-3},
        weight=0.025,
    )
    grasp_contact_1 = RewTerm(
        func=mdp.grasp_contact,
        params={"robot_idx": 1, "contact_force_threshold": 1e-3},
        weight=0.025,
    )

    # Lift — linear ramp on box z, gated on BOTH arms in dual contact.
    lift_height = RewTerm(
        func=mdp.lift_height,
        params={
            "init_z": 0.11025,
            "target_lift": 0.25,
            "contact_force_threshold": 1e-3,
        },
        weight=0.1875,
    )

    # Align — tanh attractor on box xy -> (0,0), gated on box.z > init+0.05.
    box_xy_align = RewTerm(
        func=mdp.box_xy_align,
        params={
            "std": 0.15,
            "target_xy": (0.0, 0.0),
            "lift_threshold": 0.05,
            "init_z": 0.11025,
        },
        weight=0.125,
    )

    # Success — one-shot terminal bonus mirroring the success termination.
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={
            "target_xy": (0.0, 0.0),
            "lift_height": 0.25,
            "xy_pos_tol": 0.05,
            "z_pos_tol": 0.05,
            "vel_tol": 0.10,
            "init_z": 0.11025,
        },
        weight=100.0,
    )


@configclass
class TerminationsCfg:
    """Terminations: `time_out` + `success` (box lifted to target xy/z with low
    velocity). Failure-mode terminations are intentionally not used."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.lift_box_success,
        time_out=False,
        params={
            "target_xy": (0.0, 0.0),
            "lift_height": 0.25,
            "xy_pos_tol": 0.05,
            "z_pos_tol": 0.05,
            "vel_tol": 0.10,
            "box_cfg": SceneEntityCfg("box"),
        },
    )


##
# Environment configuration
##


@configclass
class LiftBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dual-arm box-lift environment."""

    # Scene
    scene: LiftBoxSceneCfg = LiftBoxSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No CommandsCfg — goal is implicit (target xy/z baked into lift_box_success).
    commands = None

    def __post_init__(self):
        """Timing + physx — mirror insert_drawer."""
        self.decimation = 6
        self.episode_length_s = 10.0
        # Simulation — insert_drawer canonical
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Physics knobs — insert_drawer canonical
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
