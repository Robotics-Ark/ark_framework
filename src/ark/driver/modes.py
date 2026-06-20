from enum import Enum, auto


class ControllerMode(Enum):
    JOINT_POSITION = auto()
    JOINT_TORQUE = auto()
