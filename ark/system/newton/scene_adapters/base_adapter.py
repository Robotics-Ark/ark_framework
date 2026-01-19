"""Base abstract class for solver-specific scene adapters.

This module defines the interface that all solver adapters must implement.
Adapters translate solver-agnostic GeometryDescriptors into solver-specific
geometry creation API calls.

The adapter pattern allows ARK to support multiple Newton physics solvers
(XPBD, MuJoCo, Featherstone, SemiImplicit) with the same user configuration.

Example:
    >>> from ark.system.newton.scene_adapters import XPBDAdapter, MuJoCoAdapter
    >>> # Both adapters implement the same interface
    >>> xpbd_adapter = XPBDAdapter(builder)
    >>> mujoco_adapter = MuJoCoAdapter(builder)
    >>> # But they handle ground planes differently
    >>> xpbd_adapter.adapt_ground_plane(descriptor)  # Uses native plane
    >>> mujoco_adapter.adapt_ground_plane(descriptor)  # Uses box geometry
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import newton

if TYPE_CHECKING:
    from ark.system.newton.newton_builder import NewtonBuilder
    from ark.system.newton.geometry_descriptors import GeometryDescriptor


# Collision flag - centralize version detection here once
try:
    _COLLIDE_FLAG: int = newton.ShapeFlags.COLLIDE
except AttributeError:
    _COLLIDE_FLAG: int = 1  # Fallback for older Newton versions (bit 0)


class SolverSceneAdapter(ABC):
    """Abstract base class for solver-specific scene adaptation.

    Each Newton physics solver (XPBD, MuJoCo, Featherstone, etc.) gets a
    concrete adapter subclass that knows how to:

    1. Translate generic geometry descriptions to solver-compatible forms
    2. Handle solver-specific constraints (e.g., MuJoCo ground plane limitation)
    3. Create and configure the appropriate solver instance
    4. Handle post-step coordinate reconstruction if needed

    The adapter pattern keeps solver-specific logic isolated, making it easy
    to add new solvers without modifying existing code.

    Attributes:
        builder: NewtonBuilder instance that provides access to Newton's
                 ModelBuilder for adding geometry
        collide_flag: Solver-appropriate collision flag value
    """

    # Class-level collision flag (shared by all instances)
    collide_flag: int = _COLLIDE_FLAG

    def __init__(self, builder: "NewtonBuilder"):
        """Initialize adapter with a scene builder.

        Args:
            builder: NewtonBuilder instance for scene construction
        """
        self.builder = builder

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Human-readable name of the solver (for logging).

        Returns:
            Solver name (e.g., "XPBD", "MuJoCo", "Featherstone")
        """
        pass

    @abstractmethod
    def adapt_ground_plane(self, descriptor: "GeometryDescriptor") -> None:
        """Add ground plane using solver-compatible geometry.

        Different solvers may require different implementations:
        - XPBD, Featherstone: Can use builder.add_ground_plane() natively
        - MuJoCo: Requires explicit box geometry as workaround

        This method must implement the appropriate strategy for this solver.

        Args:
            descriptor: Unified ground plane description containing:
                - parameters: size, thickness
                - physics: friction, restitution
                - metadata: name, semantic_type

        Example:
            XPBD implementation:
            >>> def adapt_ground_plane(self, descriptor):
            ...     self.builder.builder.add_ground_plane()
            ...     log.ok("XPBD: Added native ground plane")

            MuJoCo implementation:
            >>> def adapt_ground_plane(self, descriptor):
            ...     thickness = descriptor.parameters["thickness"]
            ...     self.builder.builder.add_shape_box(
            ...         body=-1, hx=100, hy=thickness, hz=100, ...
            ...     )
            ...     log.ok("MuJoCo: Added ground as box geometry")
        """
        pass

    @abstractmethod
    def create_solver(
        self,
        model: "newton.Model",
        sim_cfg: Dict[str, Any]
    ) -> "newton.solvers.SolverBase":
        """Create and configure the solver instance.

        This method constructs the appropriate solver type with solver-specific
        parameters extracted from the simulation configuration.

        Args:
            model: Finalized Newton model
            sim_cfg: Simulation configuration dictionary, typically containing:
                - solver: Solver name (already known by adapter)
                - solver_iterations: Number of solver iterations
                - substeps: Number of substeps per frame
                - newton_physics: Newton-specific parameters

        Returns:
            Configured solver instance (SolverXPBD, SolverMuJoCo, etc.)

        Example:
            XPBD implementation:
            >>> def create_solver(self, model, sim_cfg):
            ...     iterations = int(sim_cfg.get("solver_iterations", 1))
            ...     # XPBD needs at least 8 iterations for position control
            ...     iterations = max(iterations, 8)
            ...     return newton.solvers.SolverXPBD(model, iterations=iterations)

            MuJoCo implementation:
            >>> def create_solver(self, model, sim_cfg):
            ...     # MuJoCo uses default parameters (20 iterations)
            ...     return newton.solvers.SolverMuJoCo(model)
        """
        pass

    def validate_scene(self, global_config: Dict[str, Any]) -> list[str]:
        """Check for solver-specific incompatibilities in scene configuration.

        This method is called early in initialization to detect configuration
        issues before attempting to build the scene. It can return warnings
        or error messages that help users fix their configs.

        Args:
            global_config: Complete ARK global configuration dictionary

        Returns:
            List of warning/error messages. Empty list if scene is valid.
            Messages should be prefixed with "WARNING:" or "ERROR:" for severity.

        Example:
            >>> def validate_scene(self, global_config):
            ...     issues = []
            ...     sim_cfg = global_config.get("simulator", {}).get("config", {})
            ...     iterations = sim_cfg.get("solver_iterations", 1)
            ...     substeps = sim_cfg.get("substeps", 1)
            ...
            ...     if iterations * substeps < 20:
            ...         issues.append(
            ...             "WARNING: XPBD with TARGET_POSITION needs 20+ "
            ...             f"effective iterations. Current: {iterations * substeps}"
            ...         )
            ...     return issues
        """
        # Base implementation - subclasses can override to add solver-specific checks
        return []

    @property
    def needs_coordinate_reconstruction(self) -> bool:
        """Whether solver needs joint coordinates reconstructed from body state.

        Maximal coordinate solvers (XPBD, MuJoCo) operate on body positions/velocities
        and may drift from joint coordinates. These solvers need eval_ik() called
        after each step to reconstruct joint_q/joint_qd from body state.

        Generalized coordinate solvers (Featherstone) integrate directly in joint
        space and don't need reconstruction.

        Returns:
            True for maximal coordinate solvers (XPBD, MuJoCo).
            False for generalized coordinate solvers (Featherstone).
        """
        return False

    def post_step(
        self,
        model: "newton.Model",
        state: "newton.State",
    ) -> None:
        """Perform solver-specific post-step updates.

        This method is called after each physics step to handle solver-specific
        post-processing. The primary use case is coordinate reconstruction for
        maximal coordinate solvers.

        Override in subclasses that need IK reconstruction or other post-step work.
        The base implementation does nothing (suitable for generalized coord solvers).

        Args:
            model: Newton model instance
            state: Current simulation state after stepping
        """
        pass
