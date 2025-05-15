from __future__ import annotations
import torch
from typing import TYPE_CHECKING


from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import ActuatorBase
if TYPE_CHECKING:
    from .muscle_actuator_cfg import MuscleActuatorCfg

class MuscleActuator(ActuatorBase):
    cfg: MuscleActuatorCfg

    def __init__(self, cfg: MuscleActuatorCfg, *args, **kwargs):
        MuscleActuator.is_implicit_model = False

        super().__init__(cfg, *args, **kwargs)

    def reset(self, *args, **kwargs):
        pass

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        pass
