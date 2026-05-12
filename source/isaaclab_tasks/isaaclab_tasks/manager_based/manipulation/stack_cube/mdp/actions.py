"""Action terms for stack_cube.

Edit_mode_011: revert to LiftCube's stock `mdp.JointPositionActionCfg`. No
custom action term is required — `mdp.JointPositionAction[Cfg]` is re-exported
from `isaaclab.envs.mdp` via the `from isaaclab.envs.mdp import *` line at the
top of `mdp/__init__.py`. This file is kept as a stub so the existing
`from .actions_cfg import *` in `mdp/__init__.py` continues to import cleanly
(actions_cfg.py is also a stub and re-exports nothing).
"""
