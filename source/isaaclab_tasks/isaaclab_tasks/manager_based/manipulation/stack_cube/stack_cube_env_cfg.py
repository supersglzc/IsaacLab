# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the StackCube task: Franka stacks cube_0 on top of cube_1.

Edit_mode_008: wholesale-copy LiftCube's design (action / obs / reward /
curriculum / timing). The success signal stays `cube_0_stacked_on_cube_1`
(this is a stacking task, not free-form lifting). The reward `goal_tracking_*`
terms read the implicit stack target `cube_1.pos_w + [0, 0, CUBE_SIZE]`
directly from the scene — no CommandsCfg / no command_manager dependency.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##


@configclass
class StackCubeSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + Franka (filled by subclass) + two cubes + ee_frame + lights."""

    # robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # end-effector sensor (LiftCube convention): filled by subclass via __post_init__.
    ee_frame: FrameTransformerCfg = MISSING
    # two cubes: filled by the per-robot subclass via __post_init__.
    cube_0: RigidObjectCfg = MISSING
    cube_1: RigidObjectCfg = MISSING

    # Contact sensors on the two Franka fingertips, filtered against the two
    # cubes. Per the IsaacLab ContactSensor docs, filter_prim_paths_expr only
    # works reliably when prim_path resolves to a single body per env, so we
    # register one sensor per finger (left/right). Kept from the prior edit so
    # downstream code that reads these sensors continues to work; the new
    # reward set does not use them (LiftCube doesn't read contact forces).
    finger_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
        ],
    )
    finger_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
        ],
    )

    # Table — same SeattleLabTable used by Reach / Lift / Push.
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
class ActionsCfg:
    """Action specs — filled by per-robot subclass.

    * arm_action — `mdp.JointPositionActionCfg` (7-D stock joint-position
      action over the Franka arm joints `panda_joint.*`). Cfg in
      `config/franka/joint_pos_env_cfg.py` uses `scale=0.5`,
      `use_default_offset=True` (target = 0.5 * action + default_joint_pos).
    * gripper_action — `mdp.BinaryJointPositionActionCfg` over the 2 finger
      joints (open=0.04 m / close=0.0 m).

    Total action dim = 7 (arm joints) + 1 (binary gripper) = 8.
    """

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specs — mirrors LiftCube's PolicyCfg, retargeted to cube_0."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Concatenation order is preserved. Nominal dim = 7 + 7 + 3 + 3 + 8 = 28,
        # but `joint_pos_rel` / `joint_vel_rel` default to ALL 9 robot joints
        # (7 arm + 2 fingers; LiftCube canonical behaviour), so the actual obs
        # width is 9 + 9 + 3 + 3 + 8 = 32. smoke_s5 pins the actual layout.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_0_position = ObsTerm(func=mdp.cube_0_position_in_robot_root_frame)
        target_position = ObsTerm(func=mdp.stack_target_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # LiftCube design — corruption ON
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset terms (§3). DR terms are added by `dr-generator` in §7."""

    # Robot joints: pinned to (1.0, 1.0) by smoke checks; dr-generator may widen.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Per-cube position resets — mirror LiftCube's `reset_object_position`
    # ranges (small additive xy perturbation around each cube's `init_state.pos`).
    reset_cube_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )


@configclass
class RewardsCfg:
    """LiftCube reward recipe restored, retargeted to cube_0 (goal = cube_1 top).

    Five named stages + two regularizers + curriculum re-weight (see CurriculumCfg):

        reaching_object       — 1 - tanh(d_ee_cube_0 / 0.1)            w=+1.0
        lifting_object        — 1.0 if cube_0.z > 0.04 else 0.0        w=+15.0
        goal_tracking_coarse  — lifted * (1 - tanh(d / 0.3))           w=+16.0
        goal_tracking_fine    — lifted * (1 - tanh(d / 0.05))          w=+5.0
        success_bonus         — sparse stack indicator                 w=+200.0
        action_rate           — -||Δa||²                               w=-1e-4 (→ -1e-1 @ 10k)
        joint_vel             — -||q̇||²                                w=-1e-4 (→ -1e-1 @ 10k)

    Goal pose is implicit: `cube_1.pos_w + [0, 0, CUBE_SIZE]`. Composer: sum.
    """

    # NOTE: IsaacLab RewardManager's `* dt` multiplier was removed in this
    # branch (reward_manager.py:150). Weights below are the OLD-style values
    # × dt (= 0.02) to preserve previous effective per-step magnitudes.
    # success_bonus is the only NEW explicit value (200 per user direction).

    reaching_object = RewTerm(
        func=mdp.cube_0_ee_distance,
        params={"std": 0.1},
        weight=0.02,
    )

    lifting_object = RewTerm(
        func=mdp.cube_0_is_lifted,
        params={"minimal_height": 0.04},
        weight=0.1,
    )

    goal_tracking_coarse = RewTerm(
        func=mdp.cube_0_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04},
        weight=0.32,
    )

    goal_tracking_fine = RewTerm(
        func=mdp.cube_0_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04},
        weight=0.0,
    )

    success_bonus = RewTerm(
        func=mdp.cube_0_stacked_bonus,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=200.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-6)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Time-out + success (cube_0 stacked on cube_1) + failure (cube_0 dropped)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Failure: cube_0 falls off the table (z below table_height by drop_margin).
    cube_dropped = DoneTerm(
        func=mdp.cube_0_dropping,
        params={"drop_margin": 0.05, "table_height": 0.0},
    )

    # Success: cube_0 is stacked on top of cube_1.
    #   |cube_0.xy - cube_1.xy| < xy_threshold
    #   |cube_0.z  - cube_1.z - CUBE_SIZE| < z_threshold
    success = DoneTerm(
        func=mdp.cube_0_stacked_on_cube_1,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
    )


@configclass
class CurriculumCfg:
    """Empty — LiftCube's 1000× regularizer ramp at step 10k was suppressing
    the cube-release motion needed to stack. Keeping action_rate / joint_vel
    at their initial −1e-4 weights throughout training."""
    pass


##
# Environment configuration
##


@configclass
class StackCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cube-stacking environment."""

    # Scene
    scene: StackCubeSceneCfg = StackCubeSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # No CommandsCfg — the goal is implicit (cube_0 stacked on cube_1).
    commands = None

    def __post_init__(self):
        """LiftCube timing: 100 Hz physics / decimation 2 / 5 s episode = 250 control steps."""
        self.decimation = 2
        self.episode_length_s = 5.0
        # Simulation — LiftCube canonical
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        # Physics knobs — LiftCube canonical
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
