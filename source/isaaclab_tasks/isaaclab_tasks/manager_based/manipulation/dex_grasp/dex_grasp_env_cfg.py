# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the Dex-Grasp task.

Single-arm UFactory 850 + Allegro right hand (22-DoF: 6 arm + 16 hand) grasps
and lifts a "dog" object off the lab table. Scene + actuators + init pose mirror
bidex's `InsertDrawer/env_cfg.py` RIGHT-ROBOT half byte-for-byte; the left robot
and drawer are dropped. The action is the EMA cumulative-relative joint
position action vendored from bidex (`mdp/actions.py`).
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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import mdp
from .mdp.actions_cfg import JOINT_LOWER_LIMIT, JOINT_UPPER_LIMIT

# Small frame marker (1 cm scale) for visualizing the lift target -- matches
# the insert_drawer convention.
_TARGET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_TARGET_MARKER_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)
_TARGET_MARKER_CFG.prim_path = "/Visuals/DogTargetMarker"


# Repo-relative asset paths. This file is at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/
# so `parents[6]` is the repo root, then `nautilus/assets/...`.
_NAUTILUS_ASSETS = Path(__file__).resolve().parents[6] / "nautilus" / "assets"
_TABLE_USD_PATH = str(_NAUTILUS_ASSETS / "table" / "lab_table_instanceable_colored_rotated.usd")
_DOG_USD_PATH = str(_NAUTILUS_ASSETS / "grasp" / "dog.usd")

# Lift target the dog must reach for success (env-local world coords).
# Same xy as the dog spawn (0.05, -0.35); +0.30 m in z = lift by 30 cm.
DOG_TARGET_LOCAL: tuple[float, float, float] = (0.05, -0.35, 0.30)
# Success tolerance: episode terminates when |dog_pos - target| < 10 cm.
DOG_TARGET_TOL: float = 0.10


##
# Scene definition
##


@configclass
class DexGraspSceneCfg(InteractiveSceneCfg):
    """Scene: ground + lab table + UF850/Allegro robot + dog + 4 fingertip contact
    sensors (filtered against the dog) + palm ee_frame + lights."""

    # Robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # Palm EE frame transformer -- filled by per-robot subclass.
    ee_frame: FrameTransformerCfg = MISSING

    # Dog: the object to grasp+lift. Bidex verbatim: kinematic_enabled=False
    # (dynamic), mass=0.11 kg, scale=(1,1,1), activate_contact_sensors=True.
    # The reset event teleports it to env-local (0.05, -0.35, 0.0) with random
    # yaw on every episode reset.
    dog: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Dog",
        spawn=UsdFileCfg(
            usd_path=_DOG_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # Four fingertip contact sensors filtered against the dog
    # (bidex InsertDrawer right-robot variant: if5 / mf5 / pf5 / th5).
    contact_sensors_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/if5",  # index
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mf5",  # middle
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pf5",  # pinky
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/th5",  # thumb
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )

    # Lab table -- bidex convention: RigidObjectCfg with kinematic_enabled=True
    # so physx treats it as a fixed-base body (collision shapes participate in
    # contact, but no rigid-body dynamics — contact forces from the dog or the
    # robot fingers cannot push the table). Surface at z ~ 0.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=_TABLE_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Lift-target frame marker. Anchor on the (fixed) table whose root sits at
    # env_origin + (0, 0, 0); offset = DOG_TARGET_LOCAL puts the marker at the
    # env-local target position. debug_vis=True renders a 1 cm frame in viewer.
    target_marker = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        debug_vis=True,
        visualizer_cfg=_TARGET_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                name="dog_target",
                offset=OffsetCfg(pos=DOG_TARGET_LOCAL),
            ),
        ],
    )

    # Ground plane -- bidex convention: 0.82 m below the table top.
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
    """Action specs -- one term, 22-D EMA cumulative-relative joint position.

    Matches bidex `InsertDrawerActionsCfg.arm_hand_action` for the right robot.
    Joint regex `.*` picks up all 22 joints (6 arm + 16 hand) in the USD
    canonical order.
    """

    arm_hand_action = mdp.EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.03,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT,
        joint_upper_limit=JOINT_UPPER_LIMIT,
        alpha=0.2,
    )


@configclass
class ObservationsCfg:
    """Observation specs -- 47-D concatenated policy obs.

    Term order:
        joint_pos_right_normalized   (22,)
        dog_position_in_world         (3,)
        last_action                  (22,)
    Total = 22 + 3 + 22 = 47.

    `enable_corruption=True` activates the per-term noise slots; the default
    `Unoise(n_min=0, n_max=0)` is a no-op (dr-generator widens later).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_right_normalized = ObsTerm(
            func=mdp.joint_pos_right_normalized,
            params={
                "joint_lower_limit": JOINT_LOWER_LIMIT,
                "joint_upper_limit": JOINT_UPPER_LIMIT,
            },
        )
        dog_position_in_world = ObsTerm(func=mdp.dog_position_in_world)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms only (§3). §7 DR is not in scope; dr-generator may widen later.

    All ranges are pinned to point intervals so the §3 reset smoke can read
    deterministic values. (`dr-generator` widens later.)
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

    # Dog: bidex `reset_object_right` verbatim -- teleport to env-local
    # (0.05, -0.35, 0.0). Yaw range pinned to (0, 0) so §3 reset is
    # deterministic; dr-generator may widen to [-pi, pi] later (the bidex
    # default).
    reset_dog = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.05, 0.05),
                "y": (-0.35, -0.35),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dog"),
        },
    )


@configclass
class RewardsCfg:
    """Reward ladder (composer = sum, sign = positive=good, no penalties iter 0).

    All weights are RAW per-step magnitudes -- the dt-multiplier has been
    REMOVED inside `RewardManager.compute` in this fork (verified at
    `isaaclab/managers/reward_manager.py:149-153`).

    Per-stage saturated per-step magnitude budget (composer = sum):

      palm_to_dog          0.05         -> 0.05 / step
      fingertip_to_dog     0.05         -> 0.05 / step
      grasp_contact        0.10         -> 0.10 / step (only when held)
      lift_height          0.50         -> 0.50 / step (only when held)
      dog_to_target        0.50         -> 0.50 / step (only when lifted >5 cm)
      success_bonus      100.0          -> +100 one-shot at success predicate

    166-step episode (8.33 s @ 20 Hz):
      dense ceiling (sum_stages * 166) = 199
      sparse one-shot                  = 100
    """

    # Stage 1 -- dense palm -> dog attractor.
    palm_to_dog = RewTerm(
        func=mdp.palm_to_dog,
        params={
            "std": 0.20,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "dog_cfg": SceneEntityCfg("dog"),
        },
        weight=0.05,
    )

    # Stage 2 -- per-fingertip distance attractor (thumb 1.5x).
    fingertip_to_dog = RewTerm(
        func=mdp.fingertip_to_dog,
        params={
            "std": 0.10,
            "fingertip_links": ("if5", "mf5", "pf5", "th5"),
            "fingertip_weights": (1.0, 1.0, 1.0, 1.5),
        },
        weight=0.05,
    )

    # Stage 3 -- Allegro grasp predicate (thumb + any of {idx, mid, pinky}).
    grasp_contact = RewTerm(
        func=mdp.grasp_contact,
        params={"contact_force_threshold": 1.0},
        weight=0.10,
    )

    # Stage 4 -- linear lift ramp on dog.z, gated on grasp_contact.
    lift_height = RewTerm(
        func=mdp.lift_height,
        params={
            "init_z": 0.0,
            "target_lift": 0.30,
            "contact_force_threshold": 1.0,
        },
        weight=0.50,
    )

    # Stage 5 -- tanh attractor on dog -> DOG_TARGET, gated on dog lifted.
    dog_to_target = RewTerm(
        func=mdp.dog_to_target,
        params={
            "std": 0.15,
            "target_local": DOG_TARGET_LOCAL,
            "lift_threshold": 0.05,
            "init_z": 0.0,
        },
        weight=0.50,
    )

    # Stage 6 -- one-shot success bonus (mirrors `dog_reached_target` termination).
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={
            "target_local": DOG_TARGET_LOCAL,
            "threshold": DOG_TARGET_TOL,
        },
        weight=100.0,
    )


@configclass
class TerminationsCfg:
    """`time_out` + `success` (dog within 10 cm of the lift target)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.dog_reached_target,
        time_out=False,
        params={
            "target_local": DOG_TARGET_LOCAL,
            "threshold": DOG_TARGET_TOL,
            "dog_cfg": SceneEntityCfg("dog"),
        },
    )


##
# Environment configuration
##


@configclass
class DexGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dex_grasp environment."""

    # Scene
    scene: DexGraspSceneCfg = DexGraspSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No CommandsCfg -- the goal is implicit (grasp and lift the dog).
    commands = None

    def __post_init__(self):
        """Timing: bidex InsertDrawer canonical (120 Hz physics / decimation 6
        -> 20 Hz control / ~8.33 s episode)."""
        self.decimation = 6
        self.episode_length_s = 8.3333
        # Simulation -- bidex canonical
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Physics knobs -- bidex canonical (large rigid-contact / patch caps for
        # the 16-finger hand at 4096 envs).
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 24
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 24
        # Iter 0: match insert_drawer for aggregate-pair cap headroom.
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
