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

        self.reflex_params = cfg.reflex_params
        self.reflex_controller = ReflexController(
            self._num_envs,
            self.num_joints,
            None,
            0.03, #30ms
            self.reflex_params["lmin"],
            self.reflex_params["lmax"],
            self.reflex_params["fvmax"],
            self.reflex_params["vmax"],
            self.reflex_params["fpmax"],
            self.reflex_params["fmin"],
            self.reflex_params["lce_max"],
            self.reflex_params["peak_force"],
            self.reflex_params["dt"],
            self.reflex_params["angles"],
            self.reflex_params["device"]
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