# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UF850 + Allegro right hand cfg for the Dex-Grasp task.

Mirrors `bidex/env/tasks/InsertDrawer/env_cfg.py:InsertDrawerSceneCfg.robot`
byte-for-byte (right robot only): same USD, same init pos `(-0.274, -0.475, 0.01)`,
same joint init pose, same 9-group ImplicitActuatorCfg blocks. The action
manager is the vendored EMA cumulative-relative joint position term defined in
the parent's `ActionsCfg` (configured against `JOINT_LOWER_LIMIT` /
`JOINT_UPPER_LIMIT` from `mdp/actions_cfg.py`).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.dex_grasp.dex_grasp_env_cfg import DexGraspEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Repo-relative robot USD. The file lives at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/config/uf850/
# so `parents[8]` is the repo root, then `nautilus/assets/ufactory850/...`.
_ROBOT_USD_PATH = str(
    Path(__file__).resolve().parents[8]
    / "nautilus" / "assets" / "ufactory850" / "uf850_allegro_right.usd"
)


# Bidex right-robot init joint pose (verbatim from
# `bidex/env/tasks/InsertDrawer/env_cfg.py:InsertDrawerSceneCfg.robot.init_state.joint_pos`).
ROBOT_INIT_JOINT_POS = {
    "joint1": 0.8,
    "joint2": 0.3,
    "joint3": -0.6,
    "joint4": 0.0,
    "joint5": -0.8,
    "joint6": -1.57,
    # hand -- proximal (f1) / mid (f2) / distal (f3) / tip (f4) for index/middle/pinky;
    # thumb has its own group jth1..jth4.
    "jif1": 0.0,
    "jif2": 0.4,
    "jif3": 0.4,
    "jif4": 0.0,
    "jmf1": 0.0,
    "jmf2": 0.4,
    "jmf3": 0.4,
    "jmf4": 0.0,
    "jpf1": 0.0,
    "jpf2": 0.4,
    "jpf3": 0.4,
    "jpf4": 0.0,
    "jth1": 1.3,
    "jth2": 0.0,
    "jth3": 0.2,
    "jth4": 0.0,
}


# UF850 + Allegro right hand articulation cfg -- bidex right-robot verbatim.
UF850_ALLEGRO_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD_PATH,
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
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=ROBOT_INIT_JOINT_POS,
        pos=(-0.274, -0.475, 0.01),
    ),
    actuators={
        "xArm_1-6": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            stiffness=2000.0,
            damping=16.0,
        ),
        "allegro_hand_1": ImplicitActuatorCfg(
            joint_names_expr=["j.*f1"],
            stiffness=325.0,
            damping=20.0,
        ),
        "allegro_hand_2": ImplicitActuatorCfg(
            joint_names_expr=["j.*f2"],
            stiffness=425.0,
            damping=25.0,
        ),
        "allegro_hand_3": ImplicitActuatorCfg(
            joint_names_expr=["j.*f3"],
            stiffness=245.0,
            damping=15.0,
        ),
        "allegro_hand_4": ImplicitActuatorCfg(
            joint_names_expr=["j.*f4"],
            stiffness=1050.0,
            damping=65.0,
        ),
        "allegro_hand_thumb_1": ImplicitActuatorCfg(
            joint_names_expr=["jth1"],
            stiffness=100.0,
            damping=5.0,
        ),
        "allegro_hand_thumb_2": ImplicitActuatorCfg(
            joint_names_expr=["jth2"],
            stiffness=300.0,
            damping=15.0,
        ),
        "allegro_hand_thumb_3": ImplicitActuatorCfg(
            joint_names_expr=["jth3"],
            stiffness=1270.0,
            damping=100.0,
        ),
        "allegro_hand_thumb_4": ImplicitActuatorCfg(
            joint_names_expr=["jth4"],
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)


@configclass
class UF850DexGraspEnvCfg(DexGraspEnvCfg):
    """UF850 + Allegro right hand specialization for Dex-Grasp."""

    def __post_init__(self):
        super().__post_init__()

        # Robot -- UF850 + Allegro right hand at bidex's exact pose.
        self.scene.robot = UF850_ALLEGRO_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Palm EE frame sensor (analog of stack_cube/lift_box ee_frame). Anchored
        # on the robot's root link; tracks `palm_link` with zero offset.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link_base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/palm_link",
                    name="palm",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )


@configclass
class UF850DexGraspEnvCfg_PLAY(UF850DexGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable obs noise for play.
        self.observations.policy.enable_corruption = False
