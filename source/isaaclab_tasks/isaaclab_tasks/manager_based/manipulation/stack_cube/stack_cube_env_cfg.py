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
    """Observation specs — 3-tier tower, grasping-cube state-machine mux.

    Edit_mode_012: extended to expose all three cubes plus two per-pair
    targets so the policy can drive cube_0 onto cube_1 (top of tower) AND
    cube_1 onto cube_2 (base on table). (Term order is preserved in the §5
    re-author note below.)

    Edit_mode_013 (§5 re-author): the policy obs is collapsed to a 4-term
    layout. `joint_vel` is dropped (the `joint_vel_l2` reward still reads
    `robot.data` directly, not the obs, so the regularizer is unaffected).
    All per-cube absolute positions and per-pair targets are replaced by a
    stateless per-step mux on the predicate `cube_0_on_cube_1`:

      not-yet-stacked: grasping_cube      <- cube_0
                       grasping_target    <- cube_1.xyz + [0,0,CUBE_SIZE]
      stacked:         grasping_cube      <- cube_2
                       grasping_target    <- cube_0.xyz + [0,0,CUBE_SIZE]

    New term order:

        joint_pos                  (9,)  — mdp.joint_pos_rel (all 9 robot joints)
        grasping_cube_position     (3,)  — mdp.grasping_cube_position_in_robot_root_frame
        grasping_target_position   (3,)  — mdp.grasping_target_position_in_robot_root_frame
        actions                    (8,)  — mdp.last_action
        Total = 9+3+3+8 = 23.

    enable_corruption=True (LiftCube design preserved).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Concatenation order: joint_pos -> grasping_cube_position ->
        # grasping_target_position -> actions. `joint_pos_rel` defaults to
        # ALL 9 robot joints (7 arm + 2 fingers; LiftCube canonical), so the
        # actual obs width is 9+3+3+8 = 23. smoke_s5 pins the actual layout.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        grasping_cube_position = ObsTerm(func=mdp.grasping_cube_position_in_robot_root_frame)
        grasping_target_position = ObsTerm(func=mdp.grasping_target_position_in_robot_root_frame)
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
            "pose_range": {"x": (0.45, 0.45), "y": (-0.25, -0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.45), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    reset_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.45), "y": (0.15, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class RewardsCfg:
    """Grasping-cube reward (§6 edit_mode_015): unified reach/lift/align on the
    grasping cube (mux mirrors §5 obs) + retained success_bonus and
    stack_broke_penalty + regularizers.

    Stateless per-step grasping-cube mux (same predicate as §5 obs):
        not-yet-stacked (cube_0 NOT on cube_1):
            grasping_cube   <- cube_0
            target          <- cube_1.xyz + [0, 0, CUBE_SIZE]
        stacked (cube_0 ON cube_1):
            grasping_cube   <- cube_2
            target          <- cube_0.xyz + [0, 0, CUBE_SIZE]

    Term order (weights match the pre-edit_mode_014 cube_0-phase scale):
        reach                  — 1 - tanh(||ee - gc|| / 0.1)            w=0.02
        lift                   — gc.z > 0.04                            w=0.1
        align                  — lifted * (1 - tanh(||gc - target||/0.3)) w=0.32
        success_bonus          — cube_0 stacked latch (once-per-episode) w=300
        stack_broke_penalty    — cube_0 stack broken latch              w=-500
        action_rate                                                     w=-2e-6
        joint_vel                                                       w=-2e-6

    Composer: sum. Legacy cube_0_* / cube_2_* / three_tier_tower_* helpers
    in `mdp/rewards.py` are PRESERVED but unreferenced — kept for rollback.
    """

    reach = RewTerm(
        func=mdp.grasping_cube_ee_distance,
        params={"std": 0.1},
        weight=0.02,
    )
    lift = RewTerm(
        func=mdp.grasping_cube_is_lifted,
        params={"minimal_height": 0.04},
        weight=0.1,
    )
    align = RewTerm(
        func=mdp.grasping_cube_goal_distance,
        params={"std": 0.1, "minimal_height": 0.04},
        weight=0.32,
    )

    success_bonus = RewTerm(
        func=mdp.cube_0_stacked_bonus_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=500.0,
    )

    stack_broke_penalty = RewTerm(
        func=mdp.cube_0_stack_broken_penalty_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=-500.0,
    )

    tower_bonus = RewTerm(
        func=mdp.three_tier_tower_bonus_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=1000.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-6)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Time-out (truncation) + success (3-tier tower) + stack-broken failure.

    Tower order (bottom-up): cube_1 (base on table) → cube_0 → cube_2 (top).
    Success requires BOTH cube_0-on-cube_1 AND cube_2-on-cube_0.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success: 3-tier tower. AND of:
    #   |cube_0.xy - cube_1.xy| < xy_threshold AND |Δz_0_1 - CUBE_SIZE| < z_threshold
    #   |cube_2.xy - cube_0.xy| < xy_threshold AND |Δz_2_0 - CUBE_SIZE| < z_threshold
    success = DoneTerm(
        func=mdp.three_tier_tower_stacked,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
    )

    # Failure: cube_0 was stacked at some point this episode AND is now no
    # longer stacked. Mirrors the `stack_broke_penalty` trigger so the −500
    # penalty fires the same step the episode terminates.
    stack_broken = DoneTerm(
        func=mdp.cube_0_stack_broken,
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
        self.decimation = 6
        self.episode_length_s = 10.0
        # Simulation — LiftCube canonical
        self.sim.dt = 1 / 120  # 100 Hz
        self.sim.render_interval = self.decimation
        # Physics knobs — LiftCube canonical
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
