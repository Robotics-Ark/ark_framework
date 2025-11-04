"""Scene adapters for solver-specific geometry translation.

This package provides adapters that translate solver-agnostic scene descriptions
(GeometryDescriptors) into solver-specific Newton API calls.

Available adapters:
- base_adapter.SolverSceneAdapter: Abstract base class
- xpbd_adapter.XPBDAdapter: XPBD solver adapter (native ground plane)
- mujoco_adapter.MuJoCoAdapter: MuJoCo solver adapter (box-based ground)
- featherstone_adapter.FeatherstoneAdapter: Featherstone solver adapter
- semiimplicit_adapter.SemiImplicitAdapter: SemiImplicit solver adapter

Usage:
    >>> from ark.system.newton.scene_adapters import XPBDAdapter, MuJoCoAdapter
    >>> adapter = XPBDAdapter(builder)
    >>> adapter.adapt_ground_plane(descriptor)
"""

from ark.system.newton.scene_adapters.base_adapter import SolverSceneAdapter
from ark.system.newton.scene_adapters.xpbd_adapter import XPBDAdapter

__all__ = [
    "SolverSceneAdapter",
    "XPBDAdapter",
]
