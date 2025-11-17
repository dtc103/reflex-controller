from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import ActuatorBase
if TYPE_CHECKING:
    from .reflex_controller_actuator_cfg import ReflexControllerActuatorCfg

from ..reflex_controller.reflex_controller import ReflexController

class ReflexControllerActuator(ActuatorBase):
    cfg: ReflexControllerActuatorCfg

    def __init__(self, cfg: ReflexControllerActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        ReflexControllerActuator.is_implicit_model = False

        if self.joint_indices != slice(None):
            joint_indices = self.joint_indices.detach().clone()
            muscle_indices = torch.cat([joint_indices, joint_indices + 12]) # 12 is the number of joints that are on the articulation in total
        else:
            joint_indices = torch.arange(self.num_joints, device=self._device)
            muscle_indices = torch.cat([joint_indices, joint_indices + 12])

        self.muscle_params = cfg.muscle_params
        self.reflex_params = cfg.reflex_params
        self.reflex_controller = ReflexController(
            self._num_envs,
            self.num_joints,
            torch.tensor(self.reflex_params["connection_matrix"], device=self._device)[muscle_indices][:, muscle_indices],
            self.reflex_params["delay"], #30ms
            self.muscle_params["lmin"],
            self.muscle_params["lmax"],
            self.muscle_params["fvmax"],
            self.muscle_params["vmax"],
            self.muscle_params["fpmax"],
            self.muscle_params["fmin"],
            self.muscle_params["lce_max"],
            self.muscle_params["peak_force"],
            self.muscle_params["dt"],
            torch.tensor(self.muscle_params["angles"], device=self._device)[joint_indices],
            self.muscle_params["device"]
        )

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        with torch.no_grad():
            torques = self.reflex_controller.compute(joint_pos, joint_vel)

            control_action.joint_efforts = torques
            control_action.joint_positions = None
            control_action.joint_velocities = None

            self.computed_effort = torques
            self.applied_effort = self._clip_effort(self.computed_effort)

            return control_action

    def reset(self, *args, **kwargs):
        pass