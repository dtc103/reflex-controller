from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.actuators import ActuatorBaseCfg
from .reflex_controller_actuator import ReflexControllerActuator

@configclass
class ReflexControllerActuatorCfg(ActuatorBaseCfg):
    class_type: type = ReflexControllerActuator

    reflex_params: dict = MISSING
    muscle_params: dict = MISSING