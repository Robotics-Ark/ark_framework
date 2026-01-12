"""XPBD solver adapter with native ground plane support.

XPBD (Extended Position-Based Dynamics) is an iterative, position-based solver
that natively supports infinite ground planes. This adapter uses the standard
builder.add_ground_plane() API without workarounds.

Key characteristics of XPBD solver:
- Iterative constraint solver (Gauss-Seidel-like)
- Linear convergence (needs 20+ effective iterations for stiff constraints)
- Excellent for soft bodies, cloth, deformables
- GPU-parallel (fast for batch simulations)
- Ground plane: Native support via builder.add_ground_plane()

For robot position control, XPBD requires careful iteration tuning:
- Minimum 8 iterations enforced by backend
- Research shows 20+ effective iterations needed (e.g., 2 iter × 10 substeps)
- Lower damping (target_kd ~ 1.0) works better than standard PD gains
"""

from typing import TYPE_CHECKING, Any, Dict

from ark.tools.log import log
from ark.system.newton.scene_adapters.base_adapter import SolverSceneAdapter

if TYPE_CHECKING:
    import newton
    from ark.system.newton.geometry_descriptors import GeometryDescriptor


class XPBDAdapter(SolverSceneAdapter):
    """Adapter for Newton's XPBD solver.

    XPBD uses native ground plane implementation, making this adapter
    straightforward. The main complexity is in solver validation - XPBD
    requires many iterations for stiff position control.

    XPBD is a maximal coordinate solver, so it needs coordinate reconstruction
    after each step to keep joint_q/joint_qd synchronized with body state.

    Example:
        >>> adapter = XPBDAdapter(builder)
        >>> adapter.adapt_ground_plane(descriptor)
        # Uses builder.add_ground_plane() natively
        >>> solver = adapter.create_solver(model, sim_cfg)
        # Returns SolverXPBD with appropriate iteration count
    """

    @property
    def needs_coordinate_reconstruction(self) -> bool:
        """XPBD operates in maximal coordinates, needs IK reconstruction."""
        return True

    @property
    def solver_name(self) -> str:
        """Return solver display name."""
        return "XPBD"

    def adapt_ground_plane(self, descriptor: "GeometryDescriptor") -> None:
        """Add ground plane using XPBD's native support.

        XPBD natively supports infinite collision planes, so we can use
        the standard builder.add_ground_plane() API directly. No workarounds
        or geometry substitution needed.

        Args:
            descriptor: Ground plane description (physics properties applied to plane)
        """
        # XPBD supports native ground plane
        # Note: Newton's add_ground_plane() doesn't accept size parameter - it's always infinite
        self.builder.builder.add_ground_plane()

        log.ok("XPBD adapter: Added native ground plane")

    def create_solver(
        self,
        model: "newton.Model",
        sim_cfg: Dict[str, Any]
    ) -> "newton.solvers.SolverBase":
        """Create XPBD solver with appropriate iteration count.

        XPBD is an iterative solver that needs sufficient iterations for
        convergence, especially for stiff constraints like position-controlled
        robot joints.

        Research findings (from investigation):
        - 4 iterations (ARK default): Robot appears "frozen", ~0.3 rad error
        - 20 effective iterations: Converges successfully
        - "Small Steps" paper: Many substeps better than many iterations

        This method enforces a minimum of 8 iterations for TARGET_POSITION mode
        to prevent the "frozen robot" problem.

        Args:
            model: Finalized Newton model
            sim_cfg: Simulation config containing:
                - solver_iterations: Requested iteration count
                - newton_physics.joint_defaults.mode: Control mode

        Returns:
            Configured SolverXPBD instance
        """
        import newton

        iterations = int(sim_cfg.get("solver_iterations", 1))

        # Check if using TARGET_POSITION control mode
        newton_cfg = sim_cfg.get("newton_physics", {})
        joint_mode = newton_cfg.get("joint_defaults", {}).get("mode", "")

        if joint_mode == "TARGET_POSITION":
            # Enforce minimum 8 iterations for position control stability
            # (Full solution needs 20+ effective iterations via substeps)
            if iterations < 8:
                old_iterations = iterations
                iterations = 8
                log.warning(
                    f"XPBD adapter: Increased solver_iterations from {old_iterations} "
                    f"to {iterations} (minimum for TARGET_POSITION mode)"
                )

        log.info(f"XPBD adapter: Creating solver with {iterations} iterations")

        return newton.solvers.SolverXPBD(model, iterations=iterations)

    def validate_scene(self, global_config: Dict[str, Any]) -> list[str]:
        """Validate XPBD-specific configuration requirements.

        XPBD's main gotcha is iteration count for position control. This method
        checks for common misconfigurations and provides helpful warnings.

        Args:
            global_config: Complete ARK configuration

        Returns:
            List of warning messages (empty if config is optimal)
        """
        issues = super().validate_scene(global_config)

        sim_cfg = global_config.get("simulator", {}).get("config", {})
        iterations = sim_cfg.get("solver_iterations", 1)
        substeps = sim_cfg.get("substeps", 1)
        effective_iters = iterations * substeps

        # Check for TARGET_POSITION mode with insufficient iterations
        newton_cfg = sim_cfg.get("newton_physics", {})
        joint_mode = newton_cfg.get("joint_defaults", {}).get("mode", "")

        if joint_mode == "TARGET_POSITION" and effective_iters < 20:
            issues.append(
                f"WARNING: XPBD with TARGET_POSITION mode needs ~20 effective iterations "
                f"for stability (research-validated). Current: {iterations} × {substeps} = {effective_iters}. "
                f"Consider using: solver_iterations=2, substeps=10 (Small Steps approach) "
                f"or solver: 'mujoco' for better robot control convergence."
            )

        # Check for high damping (common mistake when porting from MuJoCo)
        target_kd = newton_cfg.get("joint_defaults", {}).get("target_kd", 0)
        if target_kd > 20.0:
            issues.append(
                f"WARNING: XPBD target_kd={target_kd} is high. XPBD interprets damping "
                f"differently than MuJoCo. Research shows target_kd=1.0-5.0 works better "
                f"for XPBD. High damping can cause 'frozen robot' behavior."
            )

        return issues

    def post_step(
        self,
        model: "newton.Model",
        state: "newton.State",
    ) -> None:
        """Reconstruct joint coordinates from body state after XPBD step.

        XPBD operates in maximal coordinates (body positions/velocities).
        After stepping, joint_q/joint_qd may drift from body state. This
        method calls eval_ik() to reconstruct consistent joint coordinates.
        """
        import newton

        newton.eval_ik(
            model,
            state,
            state.joint_q,
            state.joint_qd,
        )
