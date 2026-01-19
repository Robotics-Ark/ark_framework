"""MuJoCo solver adapter with box-based ground plane workaround.

MuJoCo is a direct, global solver with quadratic convergence, excellent for
articulated robot control. However, it cannot use Newton's native ground_plane
API - it requires explicit box geometry as a workaround.

Key characteristics of MuJoCo solver:
- Direct/Newton solver (quadratic convergence, <10 iterations needed)
- Excellent for rigid articulated systems (robots, manipulation)
- Validated gradient support (production-ready for RL training)
- Ground plane: Requires box geometry workaround (planes must be on body=-1)
- Default 20 iterations is sufficient (no tuning needed)

Why the ground plane limitation exists:
MuJoCo converts Newton geometry to MJCF format. Newton's add_ground_plane()
creates a procedural infinite plane, but MuJoCo's MJCF requires explicit geom
elements. The converter can't translate procedural planes, causing a geometry
count mismatch error. Solution: Use a large, thin box as ground substitute.
"""

from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from ark.tools.log import log
from ark.system.newton.scene_adapters.base_adapter import SolverSceneAdapter

if TYPE_CHECKING:
    import newton
    from ark.system.newton.geometry_descriptors import GeometryDescriptor


class MuJoCoAdapter(SolverSceneAdapter):
    """Adapter for Newton's MuJoCo solver.

    MuJoCo requires a workaround for ground planes - we substitute a large,
    thin box geometry placed at the world origin. This provides equivalent
    collision behavior for most scenarios.

    MuJoCo uses maximal coordinates internally, so it needs coordinate
    reconstruction after each step to keep joint_q/joint_qd synchronized.

    Example:
        >>> adapter = MuJoCoAdapter(builder)
        >>> adapter.adapt_ground_plane(descriptor)
        # Creates 100m × 100m × 0.02m box instead of infinite plane
        >>> solver = adapter.create_solver(model, sim_cfg)
        # Returns SolverMuJoCo with default parameters
    """

    @property
    def needs_coordinate_reconstruction(self) -> bool:
        """MuJoCo may drift from joint coords, needs IK reconstruction."""
        return True

    @property
    def solver_name(self) -> str:
        """Return solver display name."""
        return "MuJoCo"

    def adapt_ground_plane(self, descriptor: "GeometryDescriptor") -> None:
        """Add ground plane using box geometry workaround for MuJoCo.

        MuJoCo solver cannot handle builder.add_ground_plane() due to MJCF
        conversion limitations. Instead, we create a large, thin box attached
        to the world body (body=-1) positioned so its top surface is at z=0.

        This workaround is transparent to users and provides equivalent collision
        behavior for most robotic scenarios (flat ground, no edge effects).

        Args:
            descriptor: Ground plane description containing:
                - parameters.thickness: Half-height of ground box (default 0.02m)
                - parameters.size: Half-extent in x/z (default 100m)
                - physics.friction: Ground friction coefficient
                - physics.restitution: Ground restitution coefficient

        Implementation details:
            Box dimensions: 200m × 0.04m × 200m (default)
            Position: [0, -thickness, 0] (top surface at y=0 for up_axis=Y)
            Body: -1 (world-fixed, infinite mass)
        """
        import newton

        params = descriptor.parameters
        physics = descriptor.physics

        # Extract ground properties from descriptor
        thickness = params.get("thickness", 0.02)  # Half-height (2cm default)
        size = params.get("size", 100.0)  # Half-extent in x/z (100m default)
        friction = physics.get("friction", 0.8)
        restitution = physics.get("restitution", 0.0)

        # Create shape configuration
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.density = 0.0  # Static body (infinite mass)
        ground_cfg.ke = 1.0e5  # Contact stiffness
        ground_cfg.kd = 1.0e3  # Contact damping
        ground_cfg.mu = friction
        ground_cfg.restitution = restitution

        # Add large flat box as ground substitute
        # Position: top surface at origin (assuming up_axis is Y)
        # Box center is at y = -thickness, so top face is at y = 0
        self.builder.builder.add_shape_box(
            body=-1,  # World body (static)
            hx=size,  # Half-extent in x (100m → 200m total)
            hy=thickness,  # Half-height in y (0.02m → 0.04m total)
            hz=size,  # Half-extent in z (100m → 200m total)
            xform=wp.transform(
                wp.vec3(0.0, -thickness, 0.0),  # Position (center below origin)
                wp.quat_identity()  # No rotation
            ),
            cfg=ground_cfg
        )

        log.ok(
            f"MuJoCo adapter: Added ground as box geometry "
            f"({size*2:.0f}m × {thickness*2:.3f}m × {size*2:.0f}m, "
            f"friction={friction:.2f})"
        )

    def create_solver(
        self,
        model: "newton.Model",
        sim_cfg: Dict[str, Any]
    ) -> "newton.solvers.SolverBase":
        """Create MuJoCo solver optimized for grasping.

        MuJoCo solver is configured based on Newton's panda_hydro example
        which demonstrates successful gripper-object interaction. Key settings:
        - use_mujoco_contacts=False: Use Newton's contact system
        - impratio=1000.0: High implicit/explicit friction cone ratio for grasping
        - cone="elliptic": Elliptic friction cone (more accurate than pyramidal)

        Args:
            model: Finalized Newton model
            sim_cfg: Simulation config with optional newton_physics.mujoco overrides

        Returns:
            Configured SolverMuJoCo instance optimized for manipulation
        """
        import newton

        # Extract MuJoCo-specific settings from config if present
        newton_cfg = sim_cfg.get("newton_physics", {})
        mujoco_cfg = newton_cfg.get("mujoco", {})

        # Default parameters based on panda_hydro example (successful grasping)
        use_mujoco_contacts = mujoco_cfg.get("use_mujoco_contacts", False)
        solver_type = mujoco_cfg.get("solver", "newton")
        integrator = mujoco_cfg.get("integrator", "implicitfast")
        cone = mujoco_cfg.get("cone", "elliptic")
        iterations = int(mujoco_cfg.get("iterations", 15))
        ls_iterations = int(mujoco_cfg.get("ls_iterations", 100))
        njmax = int(mujoco_cfg.get("njmax", 500))
        nconmax = int(mujoco_cfg.get("nconmax", 500))
        # impratio: High value creates stiff friction cone for better grasping
        impratio = float(mujoco_cfg.get("impratio", 1000.0))

        log.info(
            f"MuJoCo adapter: Creating solver (impratio={impratio}, "
            f"cone={cone}, iterations={iterations})"
        )

        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=use_mujoco_contacts,
            solver=solver_type,
            integrator=integrator,
            cone=cone,
            njmax=njmax,
            nconmax=nconmax,
            iterations=iterations,
            ls_iterations=ls_iterations,
            impratio=impratio,
        )

    def validate_scene(self, global_config: Dict[str, Any]) -> list[str]:
        """Validate MuJoCo-specific configuration requirements.

        MuJoCo is production-ready for rigid articulated systems but doesn't
        support soft bodies or particles. This method checks for incompatible
        features.

        Args:
            global_config: Complete ARK configuration

        Returns:
            List of warning/error messages (empty if scene is compatible)
        """
        issues = super().validate_scene(global_config)

        # Check if user is trying to use XPBD-specific features
        sim_cfg = global_config.get("simulator", {}).get("config", {})
        newton_cfg = sim_cfg.get("newton_physics", {})

        # Soft bodies warning
        if "soft_bodies" in newton_cfg:
            issues.append(
                "WARNING: MuJoCo solver doesn't support soft bodies "
                "(XPBD feature). Use solver='xpbd' for deformables."
            )

        # Particles warning
        if "particles" in newton_cfg:
            issues.append(
                "WARNING: MuJoCo solver doesn't support particle systems. "
                "Use solver='xpbd' or 'implicitm pm' for particles."
            )

        return issues

    def post_step(
        self,
        model: "newton.Model",
        state: "newton.State",
    ) -> None:
        """Reconstruct joint coordinates from body state after MuJoCo step.

        MuJoCo operates with its own internal state representation. After
        stepping, joint_q/joint_qd may drift from body state. This method
        calls eval_ik() to reconstruct consistent joint coordinates.
        """
        import newton

        newton.eval_ik(
            model,
            state,
            state.joint_q,
            state.joint_qd,
        )
