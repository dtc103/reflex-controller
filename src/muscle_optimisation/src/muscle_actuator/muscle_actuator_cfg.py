from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab.actuators import ActuatorBaseCfg
from .muscle_actuator import MuscleActuator

@configclass
class MuscleActuatorCfg(ActuatorBaseCfg):
    class_type: type = MuscleActuator

