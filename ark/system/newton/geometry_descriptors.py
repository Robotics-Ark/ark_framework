"""Solver-agnostic geometry descriptions for scene building.

This module provides unified data structures for describing scene geometry
in a way that can be adapted to any Newton physics solver (XPBD, MuJoCo, etc.).

The key insight is that users should describe geometry semantically (e.g., "ground plane")
without worrying about solver-specific implementation details. Adapters then translate
these descriptions to solver-compatible representations.

Example:
    >>> descriptor = GeometryDescriptor.from_ground_plane_config({
    ...     "friction": 0.8,
    ...     "restitution": 0.0,
    ...     "thickness": 0.02
    ... })
    >>> # Adapter decides whether to use native plane or box geometry
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class GeometryType(Enum):
    """Solver-agnostic geometry types.

    These represent semantic geometry types that may have different
    implementations depending on the physics solver being used.
    """

    INFINITE_PLANE = "infinite_plane"  # Ground plane, collision floor
    BOX = "box"                        # Rectangular prism
    SPHERE = "sphere"                  # Sphere
    CAPSULE = "capsule"                # Capsule (cylinder with hemispherical ends)
    CYLINDER = "cylinder"              # Cylinder
    MESH = "mesh"                      # Triangle mesh
    SDF = "sdf"                        # Signed distance field
    URDF = "urdf"                      # URDF-defined articulation


@dataclass
class GeometryDescriptor:
    """Unified geometry description that can be adapted to any solver.

    This class encapsulates all information needed to create geometry in a
    solver-agnostic way. Adapters translate these descriptions to solver-specific
    API calls.

    Attributes:
        geometry_type: Semantic type of geometry (plane, box, sphere, etc.)
        parameters: Geometric parameters (size, radius, etc.) - type-specific
        physics: Physical properties (friction, restitution, mass, etc.)
        metadata: Optional metadata (name, tags, etc.)

    Example:
        Creating a ground plane descriptor:

        >>> ground = GeometryDescriptor(
        ...     geometry_type=GeometryType.INFINITE_PLANE,
        ...     parameters={"thickness": 0.02},  # For box fallback
        ...     physics={"friction": 0.8, "restitution": 0.0}
        ... )
    """

    geometry_type: GeometryType
    parameters: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ground_plane_config(cls, cfg: Dict[str, Any]) -> "GeometryDescriptor":
        """Create ground plane descriptor from ARK config format.

        This factory method handles the standard ARK ground_plane configuration
        format and converts it to a unified descriptor.

        Args:
            cfg: Ground plane configuration dict, typically from YAML:
                {
                    "enabled": true,
                    "friction": 0.8,
                    "restitution": 0.0,
                    "thickness": 0.02  # For box fallback if solver needs it
                }

        Returns:
            GeometryDescriptor representing an infinite ground plane

        Example:
            >>> cfg = {"friction": 0.8, "restitution": 0.0, "thickness": 0.02}
            >>> descriptor = GeometryDescriptor.from_ground_plane_config(cfg)
            >>> descriptor.geometry_type
            <GeometryType.INFINITE_PLANE: 'infinite_plane'>
        """
        return cls(
            geometry_type=GeometryType.INFINITE_PLANE,
            parameters={
                "size": cfg.get("size", 100.0),  # Half-extent for box fallback
                "thickness": cfg.get("thickness", 0.02),  # Half-height for box fallback
            },
            physics={
                "friction": cfg.get("friction", 0.8),
                "restitution": cfg.get("restitution", 0.0),
                "density": 0.0,  # Static body (infinite mass)
            },
            metadata={
                "name": "ground",
                "semantic_type": "ground_plane",
            }
        )

    @classmethod
    def from_primitive_config(cls, cfg: Dict[str, Any]) -> "GeometryDescriptor":
        """Create primitive shape descriptor from ARK object config.

        This factory method handles primitive object configurations (boxes, spheres, etc.)
        and converts them to unified descriptors.

        Args:
            cfg: Primitive configuration dict:
                {
                    "shape": "box",  # or "sphere", "capsule", etc.
                    "size": [0.1, 0.1, 0.1],  # Full extents for box
                    "radius": 0.05,  # For sphere/capsule
                    "height": 0.2,  # For cylinder/capsule
                    "mass": 0.1,
                    "friction": 0.8,
                    "restitution": 0.0
                }

        Returns:
            GeometryDescriptor representing the primitive shape

        Example:
            >>> cfg = {"shape": "box", "size": [0.1, 0.1, 0.1], "mass": 0.5}
            >>> descriptor = GeometryDescriptor.from_primitive_config(cfg)
            >>> descriptor.geometry_type
            <GeometryType.BOX: 'box'>
        """
        shape = cfg.get("shape", "box").lower()

        # Map shape string to GeometryType
        shape_map = {
            "box": GeometryType.BOX,
            "sphere": GeometryType.SPHERE,
            "capsule": GeometryType.CAPSULE,
            "cylinder": GeometryType.CYLINDER,
        }

        geometry_type = shape_map.get(shape, GeometryType.BOX)

        # Extract shape-specific parameters
        parameters = {}
        if geometry_type == GeometryType.BOX:
            size = cfg.get("size", [1.0, 1.0, 1.0])
            # Convert full extents to half-extents for Newton API
            parameters["half_extents"] = [abs(float(s)) * 0.5 for s in size]
        elif geometry_type == GeometryType.SPHERE:
            parameters["radius"] = cfg.get("radius", 0.5)
        elif geometry_type in (GeometryType.CAPSULE, GeometryType.CYLINDER):
            parameters["radius"] = cfg.get("radius", 0.05)
            parameters["half_height"] = cfg.get("height", 1.0) * 0.5

        return cls(
            geometry_type=geometry_type,
            parameters=parameters,
            physics={
                "mass": cfg.get("mass", 1.0),
                "friction": cfg.get("friction", 0.8),
                "restitution": cfg.get("restitution", 0.0),
            },
            metadata={
                "name": cfg.get("name", "primitive"),
            }
        )
