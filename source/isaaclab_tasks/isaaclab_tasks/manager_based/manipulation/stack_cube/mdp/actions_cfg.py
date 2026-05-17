"""Cfg for EMACumulativeDeltaPositionAction — fixed-RPY (position-only) variant.

Edit_mode_014 (§2): swap stock joint-position action for this position-only
EMA EE-delta term. The policy outputs a 3-D `(dx, dy, dz)` delta. The EE
orientation is FIXED at the post-reset value (no rotation channels, no
axis-angle compose).

Inherits `DifferentialInverseKinematicsActionCfg`, so `asset_name`,
`joint_names`, `body_name`, `body_offset`, `controller` all behave the same.
The `controller` field MUST have `command_type="pose"` and
`use_relative_mode=False` — the action term forces absolute IK internally
and will raise otherwise.

Extras beyond the parent:
    scale               3-element scale (sx, sy, sz). Overrides the parent's
                        float scale with a 3-tuple of per-axis position scales.
    alpha               EMA weight in [0, 1]. 1.0 = no smoothing.
    pos_lower_limit     Optional 3-element list[float]. Per-axis position
    pos_upper_limit     clamp on the post-EMA pose target. Both must be set
                        together; otherwise no clamp.
"""
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import (
    EMACumulativeDeltaPositionAction,
    EMACumulativeRelativeJointPositionAction,
)


@configclass
class EMACumulativeDeltaPositionActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMACumulativeDeltaPositionAction

    scale: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Per-axis position scale (sx, sy, sz). Default 0.02 m / unit policy output
    (plugin convention — 2 cm/step is a noticeable Cartesian move). No rotation
    channels in this variant — orientation is locked at the post-reset EE quat.
    """

    alpha: float = 0.5
    """EMA weight in [0, 1]. Default 0.5 (plugin convention — half-and-half blend
    against the previously-applied position target). 1.0 disables smoothing."""

    pos_lower_limit: list[float] | None = None
    pos_upper_limit: list[float] | None = None
    """Optional per-axis (xyz) clamp on the post-EMA position target."""


@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for the JOINT-SPACE variant of the EMA cumulative-delta controller.

    Policy outputs raw deltas in [-1, 1] (clipped internally before any scale).
    Per-step processing:
        a_t = clamp(a_t, -1, 1)
        delta_t = delta_{t-1} + scale * a_t + offset
        target_t = α (q_init + delta_t) + (1-α) target_{t-1}
        target_t = clamp(target_t, [joint_lower_limit, joint_upper_limit])

    Inherits `JointPositionActionCfg`, so `asset_name`, `joint_names`,
    `scale`, `offset`, `use_default_offset` work the same way.
    """

    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction

    scale: float = 0.1
    """Per-step joint-delta scale (rad / unit policy output). Plugin default 0.1
    (5.7°/step at full-scale action) overrides JointPositionActionCfg's 1.0."""

    use_default_offset: bool = False
    """OVERRIDE the parent's True default. This action term anchors on the
    POST-RESET joint pose (captured in `init_joint_pos`) and adds it manually
    inside `process_actions`; if `use_default_offset=True`, the parent's
    super().process_actions adds it via `_offset` and we'd add it twice."""

    alpha: float = 0.5
    """EMA weight in [0, 1]. 1.0 disables smoothing."""

    joint_lower_limit: list[float] | None = None
    joint_upper_limit: list[float] | None = None
    """Optional per-joint clamp on the post-EMA target."""
