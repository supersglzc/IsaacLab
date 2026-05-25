"""Action cfg + per-joint limit constants for the dex_grasp task.

`EMACumulativeRelativeJointPositionActionCfg` is vendored verbatim from
`bidex/env/action_managers/actions_cfg.py`. `JOINT_LOWER_LIMIT` and
`JOINT_UPPER_LIMIT` are the right-hand variants from
`bidex/env/tasks/manager_based_env_cfg.py`. The order matches the USD joint
order:
    joint1..joint6 (arm)
    jif1, jmf1, jpf1, jth1   (proximal)
    jif2, jmf2, jpf2, jth2   (mid)
    jif3, jmf3, jpf3, jth3   (distal)
    jif4, jmf4, jpf4, jth4   (tip)
22 entries total.
"""
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeRelativeJointPositionAction


# Bidex right-hand joint limits (vendored verbatim from
# `bidex/env/tasks/manager_based_env_cfg.py:JOINT_LOWER_LIMIT`).
JOINT_LOWER_LIMIT = [
    -6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
    # jif1, jmf1, jpf1, jth1
    -0.05, -0.05, -0.570, 0.364,
    # jif2, jmf2, jpf2, jth2
    -0.296, -0.296, -0.296, -0.205,
    # jif3, jmf3, jpf3, jth3
    -0.274, -0.274, -0.274, -0.290,
    # jif4, jmf4, jpf4, jth4
    -0.327, -0.327, -0.327, -0.262,
]
JOINT_UPPER_LIMIT = [
    6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
    # jif1, jmf1, jpf1, jth1
    0.570, 0.05, 0.05, 1.497,
    # jif2, jmf2, jpf2, jth2
    1.710, 1.710, 1.710, 1.130,
    # jif3, jmf3, jpf3, jth3
    1.809, 1.809, 1.809, 1.633,
    # jif4, jmf4, jpf4, jth4
    1.718, 1.718, 1.718, 1.820,
]


@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the EMA cumulative-relative joint position action term.

    See :class:`EMACumulativeRelativeJointPositionAction` for the per-step rule.
    """

    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving
    average window.
    """

    joint_lower_limit: list[float] | None = None
    joint_upper_limit: list[float] | None = None
    """The lower and upper limits for the joint positions (applied to the
    post-EMA target). Both must be set together; otherwise no clamp is applied.
    """
