from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import ActuatorBase
if TYPE_CHECKING:
    from .muscle_actuator_cfg import MuscleActuatorCfg

from ..muscle_model.muscle_model import MuscleModel

class MuscleActuator(ActuatorBase):
    cfg: MuscleActuatorCfg

    def __init__(self, cfg: MuscleActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        MuscleActuator.is_implicit_model = False

        self.muscle_params = cfg.muscle_params
        self.muscle_model = torch.jit.script(MuscleModel(
            self._num_envs, 
            self.num_joints, 
            self.muscle_params['lmin'],
            self.muscle_params['lmax'],
            self.muscle_params['fvmax'],
            self.muscle_params['vmax'],
            self.muscle_params['fpmax'],
            self.muscle_params['fmin'],
            self.muscle_params['lce_max'],
            self.muscle_params['peak_force'],
            self.muscle_params['dt'],
            torch.tensor(self.muscle_params['angles'], device=self._device)[self.joint_indices],
            self.muscle_params['device'],
        ))

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        """
        actuator_pos: Current position of actuator
        """
        # since isaac lab does not allow the definition of own input items, we just 
        # act like the inputted control_action.joint_positions contain the muscle activations for one muscle of each joint
        # and control_action.joint_velocities the activations of the other muscles of the joint
        muscle_activations_1 = control_action.joint_positions# flexor
        muscle_activations_2 = control_action.joint_velocities # extensor
        muscle_activations = torch.concatenate([muscle_activations_1, muscle_activations_2], dim=1)

        with torch.no_grad():
            joint_torques = self.muscle_model.compute(muscle_activations, joint_pos, joint_vel)

            control_action.joint_efforts = joint_torques
            control_action.joint_positions = None
            control_action.joint_velocities = None

            self.computed_effort = joint_torques
            self.applied_effort = self._clip_effort(self.computed_effort)

            return control_action
        
    def reset(self, *args, **kwargs):
        self.muscle_model.reset()