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
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeDeltaPositionAction


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
