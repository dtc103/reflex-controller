from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import ActuatorBase
if TYPE_CHECKING:
    from .reflex_controller_actuator_cfg import ReflexControllerActuatorCfg

from ..muscle_model.muscle_model import MuscleModel

class ReflexControllerActuator(ActuatorBase):
    cfg: ReflexControllerActuatorCfg

    def __init__(self, cfg: ReflexControllerActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        ReflexControllerActuator.is_implicit_model = False

        self.controller_params = cfg.controller_params


    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        muscle_activations_1 = control_action.joint_positions
        muscle_activations_2 = control_action.joint_velocities
        muscle_activations = torch.concatenate([muscle_activations_1, muscle_activations_2], dim=1)

        with torch.no_grad():
            # TODO
            pass

    def reset(self, *args, **kwargs):
        pass