# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


##
# Scene definition
##


@configclass
class PushBlockSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + Franka (filled by subclass) + small block + dome light."""

    # robot: filled by per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # target object: filled by per-robot subclass.
    object: RigidObjectCfg = MISSING

    # Table — same SeattleLabTable used by Reach / Lift.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
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
class CommandsCfg:
    """Goal: a 3-D position target for the block on the table surface."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,                       # filled by subclass (panda_hand)
        resampling_time_range=(4.0, 4.0),        # resample target every 4 sim seconds
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.7),
            pos_y=(-0.25, 0.25),
            pos_z=(0.055, 0.055),                # planar push: target z is fixed at block-rest height
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specs — filled by per-robot subclass."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specs — Lift's PolicyCfg with no asymmetric critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Observation terms — concatenation order is preserved.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms (§3) + startup DR terms (§7)."""

    # ---------------- §3 RESET TERMS ----------------

    # Robot joints: scale URDF home pose by uniform [0.5, 1.5] — Reach default.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Block position: small jitter around table center per episode.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # ---------------- §7 DR (STARTUP, MINIMAL) ----------------
    # Per-env-instance, fixed for the lifetime of the env. Mirrors the
    # task-implementation.md §7 minimal preset and the canonical
    # `velocity_env_cfg.py:EventCfg.physics_material` / `add_base_mass`
    # pattern, adapted to a manipulation+block setup (no legged base;
    # randomize the contact-relevant bodies instead — robot fingers + block).

    # Friction randomization on robot fingers. Push contact dynamics depend
    # heavily on this surface; restitution kept at 0 (no bouncing).
    # Body names: `panda_leftfinger`, `panda_rightfinger` (Franka USD).
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Block mass: ±20% multiplicative. Use `scale` (not `add`) because the
    # dex-cube nominal mass is small (~50 g); additive ±0.1 kg would distort.
    block_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the push-block MDP.

    Composer is `sum` -- RewardManager sums `weight_i * func_i(env)` per step.
    The `success` term is logging-only (weight=0.0) and surfaces the user's
    "<5cm" success criterion via `info["detailed_reward"]["success"]`.
    """

    # POSITIVE shaping -- pull EE toward the block (reaching).
    reaching_block = RewTerm(
        func=mdp.object_ee_distance_body,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=1.0,
    )

    # POSITIVE shaping -- pull the block toward the target marker (coarse kernel).
    block_to_goal_tracking = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=16.0,
    )

    # POSITIVE shaping -- sharper kernel for the last few centimetres.
    block_to_goal_tracking_fine_grained = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.05, "command_name": "object_pose"},
        weight=5.0,
    )

    # LOGGING-ONLY -- surfaces user's <5cm success criterion in info["detailed_reward"]["success"].
    success = RewTerm(
        func=mdp.block_at_goal,
        params={"threshold": 0.05, "command_name": "object_pose"},
        weight=0.0,
    )

    # REGULARIZER -- penalize jerky actions.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # REGULARIZER -- penalize excessive joint velocity.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Time-out + block-falls-off-table failure (mirrors Lift)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Environment configuration
##


@configclass
class PushBlockEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the push-block environment."""

    # Scene
    scene: PushBlockSceneCfg = PushBlockSceneCfg(num_envs=4096, env_spacing=2.5)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post-init: 30 Hz control rate * 6.667 s ≈ 200 control steps per episode."""
        self.decimation = 2
        # 200 control steps at decimation=2, sim.dt = 1/60s  →  episode_length_s = 200 / 30
        self.episode_length_s = 200.0 / 30.0
        # Simulation
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # Physics knobs (mirror Lift — needed for stable contact between fingertip and block).
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
