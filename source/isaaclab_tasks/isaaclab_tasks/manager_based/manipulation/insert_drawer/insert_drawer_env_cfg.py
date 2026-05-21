# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the InsertDrawer task.

Franka FR3 (single arm) opens a prismatic drawer, drops one DexCube inside,
then closes the drawer. Mirrors `manipulation/stack_cube/` for the robot,
action, table, and cube; adds one new prismatic articulation (the drawer at
`nautilus/assets/drawer/drawer.usd`).
"""

from dataclasses import MISSING

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


##
# Scene definition
##


@configclass
class InsertDrawerSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + Franka (filled by subclass) + drawer + cube + ee_frame + lights."""

    # robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # ee sensor (StackCube convention): filled by subclass via __post_init__.
    ee_frame: FrameTransformerCfg = MISSING
    # cube: filled by the per-robot subclass via __post_init__.
    cube_0: RigidObjectCfg = MISSING
    # drawer (prismatic articulation): filled by the per-robot subclass via __post_init__.
    drawer: ArticulationCfg = MISSING
    # Frame transformer at the DROP POINT above the drawer interior. Tracks the
    # sliding `drawer` body with a +Z-local offset so the marker sits above
    # the open tray's rim. `align` reward reads this frame's `target_pos_w`
    # and attracts the LIFTED cube xyz toward it. `debug_vis=True` so the
    # marker is visible in render. Filled by per-robot subclass.
    drawer_drop_frame: FrameTransformerCfg = MISSING
    # Iter 22: frame marker on the drawer's +X-local face (front, after the
    # 90° z-rotation this maps to the +Y world face — the side facing the
    # robot). `close_drawer` reward gates on `ee.y > drawer_front_face.y`
    # to keep the gripper outside the drawer while it pushes the drawer
    # closed. Filled by the per-robot subclass.
    drawer_front_face_frame: FrameTransformerCfg = MISSING

    # Contact sensors on the two Franka fingertips, filtered against Cube_0.
    # Used by the `lift_distance` reward to require a valid two-fingered grasp
    # (both fingers in contact with the cube) before the lift signal fires.
    # Mirrors stack_cube's per-finger ContactSensorCfg pattern.
    finger_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_0"],
    )
    finger_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_0"],
    )

    # Table -- bidex StackCube convention (lab_table USD, instanceable + rotated +
    # colored, kinematic). pos=(0,0,0); the table surface sits at z approx 0.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="/home/steven/code/bidex/assets/Background/table/lab_table_instanceable_colored_rotated.usd",
        ),
    )

    # Ground plane -- bidex convention: below the table by 0.82 m.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.82]),
        spawn=GroundPlaneCfg(),
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specs -- filled by per-robot subclass.

    * arm_action -- `mdp.EMACumulativeDeltaPositionActionCfg` (3-D xyz EE-delta,
      RPY locked at the post-reset value). Copies StackCube's
      EMACumulativeDeltaPositionAction setup verbatim.
    * gripper_action -- `mdp.BinaryJointPositionActionCfg` over the 2 finger
      joints (open=0.04 m / close=0.0 m).

    Total action dim = 3 (xyz) + 1 (gripper) = 4.
    """

    arm_action: mdp.EMACumulativeDeltaPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specs — 19-D policy obs.

    Term order (concatenated):
        ee_pose                (7,) — mdp.ee_pose_in_robot_root_frame
        cube_position          (3,) — mdp.cube_position_in_robot_root_frame
        drawer_body_position   (3,) — mdp.drawer_body_position_in_robot_root_frame
        gripper_joint_pos      (2,) — two fr3 finger joint positions
        last_action            (4,) — mdp.last_action (3 xyz + 1 gripper)
    Total = 7+3+3+2+4 = 19.

    Iter 10: `drawer_handle_position` → `drawer_body_position` (sliding tray
    body, not the removed handle). The new drawer has no handle; the cube must
    end up INSIDE the sliding body, so the policy sees that body's pose
    directly.

    `enable_corruption=True` (StackCube convention). Per-term `noise=...`
    slots default to no-op (dr-generator widens later).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # 19-D layout: ee_pose(7) + cube_position(3) + drawer_body_position(3)
        #              + gripper_joint_pos(2) + last_action(4).
        ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame)
        cube_position = ObsTerm(func=mdp.cube_position_in_robot_root_frame)
        drawer_body_position = ObsTerm(func=mdp.drawer_body_position_in_robot_root_frame)
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger.*"])},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms (§3). DR terms are added by `dr-generator` in §7.

    All ranges are pinned to point intervals (or the StackCube +/- 0.05 cube
    perturbation) so the §3 reset smoke can read deterministic values.
    `dr-generator` widens them later.
    """

    # Robot joints: pinned to URDF home (scale 1.0).
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Cube: small +/- 5 cm xy perturbation around its `init_state.pos` (StackCube
    # default). z pinned (no offset).
    reset_cube_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Iter 14: tighten cube spawn x to (0.066, 0.20) world so cube root x
            # is in [0.34, 0.474] — fully inside the IK-reliable workspace region
            # (the locked-down gripper can't hold orientation at root x < 0.34).
            "pose_range": {"x": (0.1, 0.2), "y": (0.3, 0.4), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )

    # Drawer slide joint: pin to 0.30 m (FULLY OPEN) at reset, iter 3 pivot.
    # The task is now "insert cube into already-open drawer THEN close it",
    # so we start with the drawer at its joint-pos upper limit. NOTE:
    # `reset_joints_by_scale` is a multiplicative scale of the default joint
    # pos (which is 0 for the drawer), so a scale would always give 0. We
    # therefore use `reset_joints_by_offset` — additive bias on the default.
    # Range pinned to (0.30, 0.30) for deterministic reset (no jitter).
    reset_drawer_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.30, 0.30),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("drawer", joint_names=["base_drawer_joint"]),
        },
    )

    # Drawer world pose: pin to its `init_state.pos` (no jitter) for the first
    # pass. dr-generator may widen this in §7.
    reset_drawer_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Iter 14: tighten drawer spawn x to (0.066, 0.20) world to match the
            # IK-reliable workspace lower bound (root x ≥ 0.34). y stays at -0.3
            # so the drawer is in front of the robot.
            "pose_range": {"x": (0.1, 0.2), "y": (-0.3, -0.3), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("drawer"),
        },
    )


@configclass
class RewardsCfg:
    """Reward ladder — composer = sum, sign = positive=good, 7 active terms.

    Phase 2 — pick + place (all four zero-masked once `cube_inside_bonus_latch` fires):
        reach_cube                  EE→cube tanh attractor
        is_lifted                   binary `cube.z > minimal_height`
        lift_distance               linear ramp on `cube.z`, gated on both-finger contact
        align                       cube→drop_frame attractor, gated on `cube.z > minimal_height_b`

    Phase 3 — release:
        cube_inside_bonus_latch     +1.0 one-shot when cube is inside drawer AND EE is far from cube
        ee_retract_to_front_face    EE→front_face attractor; fires ONLY after the latch

    Phase 4 — close drawer:
        close_drawer                gated by (cube_inside ∧ ee.y > front_face.y) ×
                                    `clamp((max_open - joint_pos)/max_open)`

    Per-term decomposition is exposed via
    `env.unwrapped.reward_manager._step_reward` (the `/add-reward-log`
    integration point).
    """

    # Phase 2 — pick + place ----------------------------------------------------

    reach_cube = RewTerm(
        func=mdp.reach_cube,
        params={"std": 0.1},
        weight=0.02,
    )

    is_lifted = RewTerm(
        func=mdp.is_lifted,
        params={"minimal_height": 0.04},
        weight=0.2,
    )

    lift_distance = RewTerm(
        func=mdp.lift_distance,
        params={"init_z": 0.0215, "target_z": 0.25},
        weight=0.3,
    )

    align = RewTerm(
        func=mdp.align,
        params={
            "std": 0.20,
            "minimal_height_b": 0.25,
        },
        weight=2.0,
    )

    # Phase 3 — release ---------------------------------------------------------

    ee_retract_to_front_face = RewTerm(
        func=mdp.ee_retract_to_front_face,
        params={"std": 0.05},
        weight=2.0,
    )

    cube_inside_bonus_latch = RewTerm(
        func=mdp.cube_inside_bonus_once_per_episode,
        params={
            "xy_threshold": 0.15,
            "z_rel_floor": -0.02,
            "z_rel_ceiling": 0.07,
            "ee_cube_no_contact_threshold": 0.10,
        },
        weight=300.0,
    )

    # Phase 4 — close drawer ----------------------------------------------------

    close_drawer = RewTerm(
        func=mdp.close_drawer,
        params={"max_open": 0.30, "alpha": 1.0},
        weight=100.0,
    )

    # Phase 5 — success ---------------------------------------------------------

    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={"drawer_closed_threshold": 0.1},
        weight=2000.0,
    )


@configclass
class TerminationsCfg:
    """Terminations: `time_out` plus task-success.

    Failure-mode terminations are intentionally NOT used — letting the rollout
    run lets the policy recover from missed grasps / failed inserts. Success
    (cube inserted + drawer closed) ends the episode so the policy collects
    the `success_bonus` and resets cleanly.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.success_termination, time_out=False, params={"drawer_closed_threshold": 0.1})


##
# Environment configuration
##


@configclass
class InsertDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the insert-into-drawer environment."""

    # Scene
    scene: InsertDrawerSceneCfg = InsertDrawerSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No CommandsCfg -- the goal is implicit (cube inside the closed drawer).
    commands = None

    def __post_init__(self):
        """StackCube timing: 120 Hz physics / decimation 6 -> 20 Hz control / 9 s episode."""
        self.decimation = 6
        self.episode_length_s = 9.0
        # Simulation -- StackCube canonical
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Physics knobs -- StackCube canonical
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # Iter 10: bumped 16k -> 64k. The no-handle drawer adds 6 rigid bodies
        # (5 cabinet walls + sliding tray) vs the prior 1-body symdex drawer,
        # so at 4096 envs the aggregate-pair count exceeds the old 16k cap
        # (PhysX requested 28485). 64k gives ~2x headroom over the live demand.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
