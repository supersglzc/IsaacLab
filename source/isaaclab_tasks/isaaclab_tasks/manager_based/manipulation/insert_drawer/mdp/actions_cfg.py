"""Action configs for the insert_drawer task.

`EMACumulativeDeltaPositionActionCfg` -- 3-D `(dx, dy, dz)` xyz-only EE delta;
EE orientation is FIXED at the post-reset value.
"""
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeDeltaPositionAction


@configclass
class EMACumulativeDeltaPositionActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMACumulativeDeltaPositionAction

    scale: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Per-axis position scale (sx, sy, sz)."""

    alpha: float = 0.5
    """EMA weight in [0, 1]. 1.0 disables smoothing."""

    pos_lower_limit: list[float] | None = None
    pos_upper_limit: list[float] | None = None
    """Optional per-axis (xyz) clamp on the post-EMA position target."""

    forbidden_xy_half: float | None = None
    """Optional 'no-go' square in xy around the robot root in the same frame
    the box clamp uses (robot-root after body_offset). If set, the EE target
    position is pushed OUT of the square `|x| <= h ∧ |y| <= h` to the nearest
    edge after the box clamp. Z is unconstrained."""
