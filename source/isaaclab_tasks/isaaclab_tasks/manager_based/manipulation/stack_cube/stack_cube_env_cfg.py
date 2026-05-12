# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract config for the StackCube task: Franka stacks a 3-cube tower.

Edit_mode_012: extended from 2-cube to 3-tier tower (cube_2 base on the
table, cube_1 stacks on cube_2, cube_0 stacks on cube_1). Success now ANDs
`cube_0_on_cube_1` and `cube_1_on_cube_2`; cube_dropped fires when ANY of
cube_0/cube_1/cube_2 falls below `table_height - drop_margin`. Observation
gains two NEW terms: `cube_1_position` (robot root frame) and
`cube_2_position` (robot root frame). The existing `target_position` term is
renamed `cube_0_target_position` (top of cube_1); a new
`cube_1_target_position` term points at the top of cube_2. §2 (action) and
§6 (reward) and §7 (DR) are untouched — reward-generator owns the 3-cube
reward redesign in the next phase.
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
    """Scene: ground + table + Franka (filled by subclass) + three cubes + ee_frame + lights."""

    # robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # end-effector sensor (LiftCube convention): filled by subclass via __post_init__.
    ee_frame: FrameTransformerCfg = MISSING
    # three cubes: filled by the per-robot subclass via __post_init__.
    # Tower: cube_2 (base on table) ← cube_1 ← cube_0 (top).
    cube_0: RigidObjectCfg = MISSING
    cube_1: RigidObjectCfg = MISSING
    cube_2: RigidObjectCfg = MISSING

    # Contact sensors on the two Franka fingertips, filtered against all three
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
            "{ENV_REGEX_NS}/Cube_2",
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
            "{ENV_REGEX_NS}/Cube_2",
        ],
    )
    # Hand-body contact sensor (iter 8): used to assert cube_0 has NO contact
    # with the EE/wrist body in addition to the fingertips before success fires.
    hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
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
    """Observation specs — 3-tier tower variant of LiftCube's PolicyCfg.

    Edit_mode_012: extended to expose all three cubes plus two per-pair
    targets so the policy can drive cube_0 onto cube_1 (top of tower) AND
    cube_1 onto cube_2 (base on table). Term order:

        joint_pos                  (9,)  — mdp.joint_pos_rel (all 9 robot joints)
        joint_vel                  (9,)  — mdp.joint_vel_rel (all 9 robot joints)
        cube_0_position            (3,)  — mdp.cube_0_position_in_robot_root_frame
        cube_1_position            (3,)  — mdp.cube_1_position_in_robot_root_frame  (NEW)
        cube_2_position            (3,)  — mdp.cube_2_position_in_robot_root_frame  (NEW)
        cube_0_target_position     (3,)  — mdp.stack_target_position_in_robot_root_frame
                                          (rename of `target_position`; still cube_1.pos + [0,0,CUBE_SIZE])
        cube_1_target_position     (3,)  — mdp.cube_1_stack_target_position_in_robot_root_frame
                                          (NEW; cube_2.pos + [0,0,CUBE_SIZE])
        actions                    (8,)  — mdp.last_action
        Total = 9+9+3+3+3+3+3+8 = 41.

    enable_corruption=True (LiftCube design preserved).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Concatenation order is preserved. Nominal arm-only dim sums to
        # 7+7+3+3+3+3+3+8=37 but `joint_pos_rel` / `joint_vel_rel` default to
        # ALL 9 robot joints (7 arm + 2 fingers; LiftCube canonical), so the
        # actual obs width is 9+9+3+3+3+3+3+8 = 41. smoke_s5 pins the actual
        # layout.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_0_position = ObsTerm(func=mdp.cube_0_position_in_robot_root_frame)
        cube_1_position = ObsTerm(func=mdp.cube_1_position_in_robot_root_frame)
        cube_2_position = ObsTerm(func=mdp.cube_2_position_in_robot_root_frame)
        cube_0_target_position = ObsTerm(func=mdp.stack_target_position_in_robot_root_frame)
        cube_1_target_position = ObsTerm(func=mdp.cube_1_stack_target_position_in_robot_root_frame)
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
    reset_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class RewardsCfg:
    """Restored 2-cube LiftCube-style reward (per user direction after the
    3-cube reward-tune loop plateaued at 8.3% stage-1 / 0% full-tower success).

    The §1/§3/§4/§5 environment stays 3-cube. Only §6 reverts: success_bonus
    is the original geometric `cube_0_stacked_bonus` (fires when cube_0 sits
    on cube_1 regardless of where cube_1 is).

    Five named stages + two regularizers:

        reaching_object       — 1 - tanh(d_ee_cube_0 / 0.1)            w=+0.02
        lifting_object        — 1.0 if cube_0.z > 0.04 else 0.0        w=+0.1
        goal_tracking_coarse  — lifted * (1 - tanh(d / 0.3))           w=+0.32
        goal_tracking_fine    — lifted * (1 - tanh(d / 0.05))          w=+0.0 (dormant)
        success_bonus         — sparse cube_0-on-cube_1 indicator      w=+200.0
        action_rate           — -||Δa||²                               w=-2e-6
        joint_vel             — -||q̇||²                                w=-2e-6

    Goal pose: cube_1.pos_w + [0, 0, CUBE_SIZE]. Composer: sum.

    Expected interaction with the 3-cube env: the policy will stack cube_0 on
    cube_1 (wherever cube_1 sits) ≈19% of episodes (iter_023 baseline). The
    `TerminationsCfg.success` (three-tier tower) requires cube_1 on cube_2 too,
    which this reward does not push toward — so success_rate ≈ 0%.

    NOTE: IsaacLab RewardManager's `* dt` multiplier was REMOVED in this branch
    (reward_manager.py:150). Weights are per-step magnitudes.
    """

    # ----- 2-cube LiftCube reward restored (per user direction after iter 33) ----
    # The §1/§3/§4/§5 environment stays 3-cube (three cubes, 3-tier success
    # termination, new obs terms). Only §6 reverts: cube_0_stacked_bonus is
    # the original geometric check (cube_0 on cube_1, regardless of cube_1
    # location). Expected behavior: ~19% cube_0-on-cube_1 fire rate (iter_023
    # baseline), ~0% full-tower success because the policy ignores cube_2.

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

    success_bonus = RewTerm(
        func=mdp.cube_0_stacked_bonus_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=300.0,
    )

    # Stage-2 shaping: reach + lift cube_2. Both gated on cube_0 CURRENTLY
    # stacked on cube_1. reaching weight dropped 10→1.0 (iter 8) to make
    # lifting the dominant stage-2 signal.
    reaching_cube_2 = RewTerm(
        func=mdp.cube_2_ee_distance_gated_on_cube_0_stacked,
        params={"std": 0.1},
        weight=1.0,
    )
    lifting_cube_2 = RewTerm(
        func=mdp.cube_2_is_lifted_gated_on_cube_0_stacked,
        params={"minimal_height": 0.04},
        weight=10.0,
    )

    # One-time -100 penalty: cube_0 was successfully stacked at some point this
    # episode (success_bonus latch fired) AND the stack subsequently broke.
    # Discourages the policy from disturbing the stack while engaging cube_2.
    stack_broke_penalty = RewTerm(
        func=mdp.cube_0_stack_broken_penalty_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=-100.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-6)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Time-out + success (3-tier tower) + failure (cube_0 dropped).

    Tower order (bottom-up): cube_1 (base on table) → cube_0 → cube_2 (top).
    Success requires BOTH cube_0-on-cube_1 AND cube_2-on-cube_0.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Failure: cube_0 falls off the table (z below table_height by drop_margin).
    cube_dropped = DoneTerm(
        func=mdp.cube_0_dropping,
        params={"drop_margin": 0.05, "table_height": 0.0},
    )

    # Success: 3-tier tower. AND of:
    #   |cube_0.xy - cube_1.xy| < xy_threshold AND |Δz_0_1 - CUBE_SIZE| < z_threshold
    #   |cube_2.xy - cube_0.xy| < xy_threshold AND |Δz_2_0 - CUBE_SIZE| < z_threshold
    success = DoneTerm(
        func=mdp.three_tier_tower_stacked,
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
