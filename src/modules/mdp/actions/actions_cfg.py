from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import muscle_actions

@configclass
class MuscleActionCfg(ActionTermCfg):
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = True

    class_type: type[ActionTerm] = muscle_actions.MuscleAction