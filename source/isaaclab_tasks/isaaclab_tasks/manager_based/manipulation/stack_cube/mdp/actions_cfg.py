"""Action term cfgs for stack_cube.

Edit_mode_011: revert to LiftCube's stock `mdp.JointPositionActionCfg`. No
custom cfg class is required — `mdp.JointPositionActionCfg` is re-exported
from `isaaclab.envs.mdp` via `mdp/__init__.py`. Kept as a stub so the existing
`from .actions_cfg import *` in `mdp/__init__.py` continues to import cleanly.
"""
