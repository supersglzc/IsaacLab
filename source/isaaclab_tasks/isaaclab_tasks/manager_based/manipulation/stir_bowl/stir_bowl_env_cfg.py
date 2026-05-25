# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the Stir-Bowl task.

Single FR3 + Franka-hand robot stirs three balls inside a bowl using a spoon
that starts upright in a stand. Mirrors `manipulation/stack_cube/` and
`manipulation/lift_box/` for layout / sim timing / physx knobs.
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


# Repo-relative table asset path. This file is at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stir_bowl/
# so `parents[6]` is the repo root, then `nautilus/assets/table/...`.
_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "nautilus" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)


##
# Scene definition
##


@configclass
class StirBowlSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + FR3 robot (filled by subclass) + spoon + stand +
    bowl + 3 balls + ee_frame + spoon_handle_frame + bowl_center_frame +
    finger contact sensors + lights."""

    # Robot, frames, and rigid objects — filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    spoon_handle_frame: FrameTransformerCfg = MISSING
    bowl_center_frame: FrameTransformerCfg = MISSING
    spoon: RigidObjectCfg = MISSING
    stand: RigidObjectCfg = MISSING
    bowl: RigidObjectCfg = MISSING
    ball_1: RigidObjectCfg = MISSING
    ball_2: RigidObjectCfg = MISSING
    ball_3: RigidObjectCfg = MISSING

    # Contact sensors on the two FR3 fingertips, filtered against the spoon —
    # used by future reward terms to detect "fingers in contact with the spoon".
    finger_left_contact_spoon = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Spoon"],
    )
    finger_right_contact_spoon = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Spoon"],
    )

    # Table — stack_cube / lift_box convention (lab table USD; surface ~ z=0).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=_TABLE_USD_PATH),
    )

    # Ground plane — stack_cube / lift_box convention (below the table by 0.82 m).
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
    """Action specs — filled by per-robot subclass.

    * arm_action — `mdp.EMACumulativeDeltaPositionActionCfg` (3-D position-only
      EMA EE-delta action; xyz only, EE quaternion locked at post-reset value).
      Re-uses the same custom action class as stack_cube / lift_box (vendored
      under `insert_drawer.mdp`).
    * gripper_action — `mdp.BinaryJointPositionActionCfg` over the 2 finger
      joints (open=0.04 m / close=0.0 m).

    Total action dim = 3 (xyz EE-delta) + 1 (binary gripper) = 4.
    """

    arm_action: object = MISSING
    gripper_action: object = MISSING


@configclass
class ObservationsCfg:
    """Observation specs — 41-D concatenated policy obs.

    Term order:
        ee_pose                    (7,) — ee pose in robot root frame
        spoon_position_in_world    (3,)
        spoon_quat_in_world        (4,)
        bowl_position_in_world     (3,)
        ball_1_position_in_world   (3,)
        ball_1_lin_vel_in_world    (3,)
        ball_2_position_in_world   (3,)
        ball_2_lin_vel_in_world    (3,)
        ball_3_position_in_world   (3,)
        ball_3_lin_vel_in_world    (3,)
        gripper_joint_pos          (2,)
        last_action                (4,)
    Total = 7 + 3 + 4 + 3 + (3+3)*3 + 2 + 4 = 41.

    `enable_corruption=True` (per-term noise slots default to no-op via
    `Unoise(n_min=0, n_max=0)`; the dr-generator may widen later).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0))
        spoon_position_in_world = ObsTerm(func=mdp.spoon_position_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        spoon_quat_in_world = ObsTerm(func=mdp.spoon_quat_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        bowl_position_in_world = ObsTerm(func=mdp.bowl_position_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_1_position_in_world = ObsTerm(func=mdp.ball_1_position_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_1_lin_vel_in_world = ObsTerm(func=mdp.ball_1_lin_vel_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_2_position_in_world = ObsTerm(func=mdp.ball_2_position_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_2_lin_vel_in_world = ObsTerm(func=mdp.ball_2_lin_vel_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_3_position_in_world = ObsTerm(func=mdp.ball_3_position_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        ball_3_lin_vel_in_world = ObsTerm(func=mdp.ball_3_lin_vel_in_world, noise=Unoise(n_min=0.0, n_max=0.0))
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger.*"])},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms (§3). DR terms are added by `dr-generator` in §7.

    All ranges are pinned to point intervals so the §3 reset smoke can read
    deterministic values; `dr-generator` may widen.
    """

    # Robot joints — pinned to URDF home (scale 1.0).
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Stand — env-local nominal (0.10, 0.10, 0.0). Kinematic-enabled rigid
    # body so the reset event manager finds it; gravity disabled so it
    # behaves as a static prop.
    reset_stand = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.10, 0.10), "y": (0.10, 0.10), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("stand"),
        },
    )

    # Spoon — stands UPRIGHT (bowl-end pointing DOWN, handle pointing UP). The
    # spoon's `init_state.rot` (set in the per-robot subclass) is the IDENTITY
    # quaternion, and the spoon STL has its long axis along spoon-local +Z
    # (handle at +Z, bowl-end at -Z), so the spoon's world-Z half-extent equals
    # the spoon-local Z half-extent (= 0.065 m).
    #
    # The fork-shape stand has two upper pegs that span z in [0.060, 0.120]
    # with a 14 mm vertical gap between them along x. Place the spoon center
    # at z = 0.090 (stand mid-height between pegs) so the spoon's middle is
    # gripped laterally by the two pegs:
    #   bowl-end at world z = 0.090 - 0.065 = 0.025 (~2.5 cm above table)
    #   handle top at world z = 0.090 + 0.065 = 0.155 (gripper-approach height)
    # xy is pinned at the stand's env-local (0.10, 0.10) with ±5 mm jitter
    # range (dr-generator may widen; the smoke pins to (0.10, 0.10) exactly).
    reset_spoon = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.095, 0.105), "y": (0.095, 0.105), "z": (0.090, 0.090)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("spoon"),
        },
    )

    # Bowl — env-local nominal (0.10, 0.30, 0.0). Bowl origin is at its
    # bottom; spawning at z=0.005 sits the bowl floor on the table top.
    reset_bowl = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.10, 0.10), "y": (0.30, 0.30), "z": (0.005, 0.005)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bowl"),
        },
    )

    # Balls — bidex offsets in (x, y) translated +0.10 in y so they sit
    # inside the bowl positioned at (0.10, 0.30). Each ball z = 0.05 (above
    # the bowl floor).
    reset_ball_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.10, 0.10), "y": (0.30, 0.30), "z": (0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball_1"),
        },
    )
    reset_ball_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.14, 0.14), "y": (0.32, 0.32), "z": (0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball_2"),
        },
    )
    reset_ball_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.06, 0.06), "y": (0.28, 0.28), "z": (0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball_3"),
        },
    )


@configclass
class RewardsCfg:
    """Constant-zero placeholder. `/nautilus:reward-tune` fills in real shaping."""

    placeholder = RewTerm(func=mdp.placeholder_zero, weight=0.0)


@configclass
class TerminationsCfg:
    """Time-out only — first-pass design. Success criteria are out of scope
    for §4 (the user will add them via `/nautilus:reward-tune`)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class StirBowlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stir-bowl environment."""

    # Scene
    scene: StirBowlSceneCfg = StirBowlSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No CommandsCfg — goal is implicit (stir the balls).
    commands = None

    def __post_init__(self):
        """Timing: 120 Hz physics / decimation 6 → 20 Hz control / 10 s episode = 200 control steps."""
        self.decimation = 6
        self.episode_length_s = 10.0
        # Simulation
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Physics knobs — stack_cube / lift_box canonical
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
