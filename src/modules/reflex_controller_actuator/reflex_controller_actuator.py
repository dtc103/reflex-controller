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

        self.muscle_params = cfg.muscle_params
        self.reflex_params = cfg.reflex_params
        self.reflex_controller = ReflexController(
            self._num_envs,
            self.num_joints,
            self.reflex_params["connection_matrix"],
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
            self.muscle_params["angles"],
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