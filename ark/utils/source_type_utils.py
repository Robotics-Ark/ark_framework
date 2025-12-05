from enum import Enum


class SourceType(Enum):
    """Supported source types for object creation."""

    URDF = "urdf"
    PRIMITIVE = "primitive"
    SDF = "sdf"
    MJCF = "mjcf"
    USD = "usd"
