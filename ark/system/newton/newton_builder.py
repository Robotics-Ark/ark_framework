# newton_builder.py
"""
Newton scene builder following ARK's design patterns.

Provides a fluent API for building Newton physics scenes while tracking metadata
for integration with ARK's messaging layer. Similar in spirit to mjcf_builder.py
but designed around Newton's articulation-based architecture and conventions.

Key features:
- Articulation-centric scene construction
- Method chaining for clean, readable scene descriptions
- Automatic tracking of bodies, joints, and articulations for spawn state generation
- Support for Newton-specific features (particles, USD/URDF/MJCF loading, env replication)
- Follows Newton conventions: Z-up, (x,y,z,w) quaternions, generalized coordinates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import newton
import numpy as np
import warp as wp

from ark.tools.log import log


# ----------------------------- Data Structures -----------------------------


@dataclass
class ArticulationSpec:
    """Metadata for a single articulation (robot/mechanism)."""

    name: str
    key: Optional[str] = None  # Newton articulation key
    root_body_idx: Optional[int] = None
    body_indices: List[int] = field(default_factory=list)
    joint_indices: List[int] = field(default_factory=list)
    joint_q_start: Optional[int] = None
    joint_q_count: int = 0
    joint_qd_start: Optional[int] = None
    joint_qd_count: int = 0


@dataclass
class JointSpec:
    """Metadata for joint tracking (for spawn state generation)."""

    name: str
    joint_type: newton.JointType
    articulation: Optional[str] = None
    parent_body_idx: int = -1
    child_body_idx: int = -1
    q_start: int = 0  # Index into joint_q array
    q_size: int = 0  # Number of position DOFs
    qd_start: int = 0  # Index into joint_qd array
    qd_size: int = 0  # Number of velocity DOFs
    default_q: Optional[List[float]] = None


@dataclass
class BodySpec:
    """Metadata for body tracking."""

    name: str
    body_idx: int
    articulation: Optional[str] = None
    xform: wp.transform = field(default_factory=lambda: wp.transform([0, 0, 0], [0, 0, 0, 1]))
    mass: float = 0.0


# --------------------------- Newton Builder ----------------------------


class NewtonBuilder:
    """
    Fluent API for building Newton physics scenes with ARK integration.

    Wraps Newton's ModelBuilder with:
    - Named entity tracking (bodies, joints, articulations)
    - Spawn state generation (joint_q, body_q initialization)
    - Method chaining for readable scene construction
    - Integration with ARK's YAML config system

    Example:
        builder = (
            NewtonBuilder("robot_scene")
            .set_gravity(-9.81)
            .add_ground_plane()
            .add_articulation("panda")
            .load_urdf("panda", "panda.urdf", xform=wp.transform([0,0,0.5]))
            .add_articulation("cube")
            .add_simple_object("cube", shape="box", size=[0.05,0.05,0.05], xform=...)
        )

        model, metadata = builder.finalize()
        spawn_state = builder.make_spawn_state()
    """

    def __init__(
        self,
        model_name: str = "world",
        up_axis: newton.Axis = newton.Axis.Z,
        gravity: float = -9.81
    ):
        """
        Initialize Newton scene builder.

        Args:
            model_name: Name for the scene/model
            up_axis: Up direction (default: Z-up following Newton conventions)
            gravity: Gravity magnitude along up axis (default: -9.81)
        """
        self.model_name = model_name
        self.up_axis = up_axis
        self.gravity_magnitude = gravity

        # Create underlying Newton builder
        self.builder = newton.ModelBuilder(up_axis=up_axis, gravity=gravity)

        # Articulation tracking
        self._articulations: Dict[str, ArticulationSpec] = {}
        self._current_articulation: Optional[str] = None

        # Entity tracking for spawn state generation
        self._bodies: Dict[str, BodySpec] = {}
        self._joints: List[JointSpec] = []
        self._joint_defaults: Dict[str, List[float]] = {}

        # Counter for auto-naming
        self._unnamed_body_count = 0
        self._unnamed_joint_count = 0

    # ---------- Configuration ----------

    def set_gravity(self, magnitude: float) -> "NewtonBuilder":
        """Set gravity magnitude along up axis."""
        self.gravity_magnitude = magnitude
        # Newton's builder doesn't have a setter, gravity is set at init
        # So we'll track it and use it when creating bodies if needed
        return self

    def set_default_shape_config(self, **kwargs) -> "NewtonBuilder":
        """
        Set default shape configuration.

        Args:
            **kwargs: Shape config parameters (density, friction, restitution, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.builder.default_shape_cfg, key):
                setattr(self.builder.default_shape_cfg, key, value)
        return self

    def set_default_joint_config(self, **kwargs) -> "NewtonBuilder":
        """
        Set default joint configuration.

        Args:
            **kwargs: Joint config parameters (target_ke, target_kd, armature, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.builder.default_joint_cfg, key):
                setattr(self.builder.default_joint_cfg, key, value)
        return self

    # ---------- Articulation Management ----------

    def add_articulation(self, name: str, key: Optional[str] = None) -> "NewtonBuilder":
        """
        Start a new articulation group (robot/mechanism).

        Subsequent add_body and add_joint calls will be associated with this articulation
        until a new articulation is started.

        Args:
            name: Unique name for this articulation (for ARK tracking)
            key: Optional Newton articulation key (auto-generated if None)

        Returns:
            Self for method chaining
        """
        if name in self._articulations:
            log.warning(f"Articulation '{name}' already exists, switching context")
        else:
            # Create Newton articulation
            artic_key = key if key is not None else name
            self.builder.add_articulation(artic_key)

            # Track metadata
            self._articulations[name] = ArticulationSpec(
                name=name,
                key=artic_key,
                joint_q_start=len(self.builder.joint_q),
                joint_qd_start=len(self.builder.joint_qd)
            )

        self._current_articulation = name
        return self

    def set_current_articulation(self, name: str) -> "NewtonBuilder":
        """Switch to an existing articulation context."""
        if name not in self._articulations:
            raise ValueError(f"Articulation '{name}' not found. Call add_articulation('{name}') first.")
        self._current_articulation = name
        return self

    # ---------- Bodies ----------

    def add_body(
        self,
        name: Optional[str] = None,
        xform: Union[wp.transform, Tuple, List, None] = None,
        mass: float = 1.0,
        com: Optional[Vec3] = None,
        inertia: Optional[Mat33] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add a rigid body to the scene.

        Args:
            name: Body name (auto-generated if None)
            xform: Initial transform (position + quaternion xyzw)
            mass: Body mass in kg
            com: Center of mass offset
            inertia: 3x3 inertia tensor
            **kwargs: Additional args passed to ModelBuilder.add_body()

        Returns:
            Self for method chaining
        """
        # Handle transform input
        if xform is None:
            xform = wp.transform([0, 0, 0], [0, 0, 0, 1])
        elif isinstance(xform, (tuple, list)):
            # Convert list/tuple to wp.transform
            if len(xform) == 3:
                xform = wp.transform(xform, [0, 0, 0, 1])
            elif len(xform) == 7:
                xform = wp.transform(xform[:3], xform[3:])

        # Auto-generate name if needed
        if name is None:
            name = f"body_{self._unnamed_body_count}"
            self._unnamed_body_count += 1

        # Add to Newton builder
        body_idx = self.builder.add_body(
            xform=xform,
            mass=mass,
            com=com,
            I_m=inertia,
            **kwargs
        )

        # Track metadata
        self._bodies[name] = BodySpec(
            name=name,
            body_idx=body_idx,
            articulation=self._current_articulation,
            xform=xform,
            mass=mass
        )

        # Associate with current articulation
        if self._current_articulation:
            artic = self._articulations[self._current_articulation]
            artic.body_indices.append(body_idx)
            if artic.root_body_idx is None:
                artic.root_body_idx = body_idx

        return self

    # ---------- Shapes ----------

    def add_shape_plane(
        self,
        body: Union[str, int],
        width: float = 10.0,
        length: float = 10.0,
        **kwargs
    ) -> "NewtonBuilder":
        """Add plane shape to body."""
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_plane(body=body_idx, width=width, length=length, **kwargs)
        return self

    def add_ground_plane(
        self,
        size: float = 1000.0,
        **kwargs
    ) -> "NewtonBuilder":
        """Add infinite ground plane."""
        self.builder.add_ground_plane(size=size, **kwargs)
        return self

    def add_shape_box(
        self,
        body: Union[str, int],
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add box shape to body.

        Args:
            body: Body name or index
            hx, hy, hz: Half-extents along each axis (Newton convention)
            **kwargs: Additional shape parameters
        """
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_box(body=body_idx, hx=hx, hy=hy, hz=hz, **kwargs)
        return self

    def add_shape_sphere(
        self,
        body: Union[str, int],
        radius: float = 0.5,
        **kwargs
    ) -> "NewtonBuilder":
        """Add sphere shape to body."""
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_sphere(body=body_idx, radius=radius, **kwargs)
        return self

    def add_shape_capsule(
        self,
        body: Union[str, int],
        radius: float = 0.5,
        half_height: float = 1.0,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add capsule shape to body.

        Args:
            body: Body name or index
            radius: Capsule radius
            half_height: Half-height excluding hemispherical caps (Newton convention)
            **kwargs: Additional shape parameters
        """
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_capsule(
            body=body_idx,
            radius=radius,
            half_height=half_height,
            **kwargs
        )
        return self

    def add_shape_cylinder(
        self,
        body: Union[str, int],
        radius: float = 0.5,
        half_height: float = 1.0,
        **kwargs
    ) -> "NewtonBuilder":
        """Add cylinder shape to body."""
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_cylinder(
            body=body_idx,
            radius=radius,
            half_height=half_height,
            **kwargs
        )
        return self

    def add_shape_mesh(
        self,
        body: Union[str, int],
        mesh: newton.Mesh,
        **kwargs
    ) -> "NewtonBuilder":
        """Add mesh shape to body."""
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_mesh(body=body_idx, mesh=mesh, **kwargs)
        return self

    def add_shape_sdf(
        self,
        body: Union[str, int],
        sdf: newton.SDF,
        **kwargs
    ) -> "NewtonBuilder":
        """Add SDF (signed distance field) shape to body."""
        body_idx = self._resolve_body_idx(body)
        self.builder.add_shape_sdf(body=body_idx, sdf=sdf, **kwargs)
        return self

    # ---------- Joints ----------

    def add_joint_revolute(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        axis: Union[newton.Axis, wp.vec3] = newton.Axis.Z,
        limit_lower: float = -1e6,
        limit_upper: float = 1e6,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add revolute (hinge) joint.

        Args:
            name: Joint name (auto-generated if None)
            parent: Parent body name or index (-1 for world)
            child: Child body name or index
            axis: Rotation axis
            limit_lower: Lower joint limit (radians)
            limit_upper: Upper joint limit (radians)
            **kwargs: Additional joint parameters (target_ke, target_kd, etc.)
        """
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        # Add to Newton builder
        joint_idx = self.builder.add_joint_revolute(
            parent=parent_idx,
            child=child_idx,
            axis=axis,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            **kwargs
        )

        # Track metadata
        self._track_joint(
            name=name,
            joint_type=newton.JointType.REVOLUTE,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=1,
            qd_dofs=1
        )

        return self

    def add_joint_prismatic(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        axis: Union[newton.Axis, wp.vec3] = newton.Axis.Z,
        limit_lower: float = -1e6,
        limit_upper: float = 1e6,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add prismatic (slider) joint.

        Args:
            name: Joint name
            parent: Parent body
            child: Child body
            axis: Sliding axis
            limit_lower: Lower position limit (meters)
            limit_upper: Upper position limit (meters)
            **kwargs: Additional joint parameters
        """
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        joint_idx = self.builder.add_joint_prismatic(
            parent=parent_idx,
            child=child_idx,
            axis=axis,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            **kwargs
        )

        self._track_joint(
            name=name,
            joint_type=newton.JointType.PRISMATIC,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=1,
            qd_dofs=1
        )

        return self

    def add_joint_ball(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add ball (spherical) joint.

        Ball joints have 3 rotational DOFs represented as quaternions.
        Position DOFs: 4 (quaternion), Velocity DOFs: 3 (angular velocity).
        """
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        joint_idx = self.builder.add_joint_ball(
            parent=parent_idx,
            child=child_idx,
            **kwargs
        )

        self._track_joint(
            name=name,
            joint_type=newton.JointType.BALL,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=4,  # Quaternion
            qd_dofs=3  # Angular velocity
        )

        return self

    def add_joint_free(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add free (6-DOF) joint for floating base robots/objects.

        Free joints have 6 DOFs: 3 translation + 3 rotation.
        Position DOFs: 7 (pos + quat), Velocity DOFs: 6 (linear + angular).
        """
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        joint_idx = self.builder.add_joint_free(
            parent=parent_idx,
            child=child_idx,
            **kwargs
        )

        self._track_joint(
            name=name,
            joint_type=newton.JointType.FREE,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=7,  # Position + quaternion
            qd_dofs=6  # Linear + angular velocity
        )

        return self

    def add_joint_fixed(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        **kwargs
    ) -> "NewtonBuilder":
        """Add fixed joint (rigid connection between bodies)."""
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        joint_idx = self.builder.add_joint_fixed(
            parent=parent_idx,
            child=child_idx,
            **kwargs
        )

        self._track_joint(
            name=name,
            joint_type=newton.JointType.FIXED,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=0,
            qd_dofs=0
        )

        return self

    def add_joint_d6(
        self,
        name: Optional[str] = None,
        parent: Union[str, int] = -1,
        child: Union[str, int] = -1,
        linear_axes: Optional[List] = None,
        angular_axes: Optional[List] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Add D6 (generic 6-DOF configurable) joint.

        Args:
            name: Joint name
            parent: Parent body
            child: Child body
            linear_axes: List of linear axis configs (up to 3)
            angular_axes: List of angular axis configs (up to 3)
            **kwargs: Additional joint parameters
        """
        parent_idx = self._resolve_body_idx(parent) if parent != -1 else -1
        child_idx = self._resolve_body_idx(child)

        if name is None:
            name = f"joint_{self._unnamed_joint_count}"
            self._unnamed_joint_count += 1

        joint_idx = self.builder.add_joint_d6(
            parent=parent_idx,
            child=child_idx,
            linear_axes=linear_axes or [],
            angular_axes=angular_axes or [],
            **kwargs
        )

        # D6 DOF count depends on axes configuration
        num_linear = len(linear_axes) if linear_axes else 0
        num_angular = len(angular_axes) if angular_axes else 0
        total_dofs = num_linear + num_angular

        self._track_joint(
            name=name,
            joint_type=newton.JointType.D6,
            parent_idx=parent_idx,
            child_idx=child_idx,
            q_dofs=total_dofs,
            qd_dofs=total_dofs
        )

        return self

    # ---------- Asset Loading ----------

    def load_urdf(
        self,
        name: str,
        file: Union[str, Path],
        xform: Optional[wp.transform] = None,
        floating: bool = False,
        collapse_fixed_joints: bool = False,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Load a URDF file as an articulation.

        Args:
            name: Articulation name for tracking
            file: Path to URDF file
            xform: Initial transform
            floating: If True, add free joint for floating base
            collapse_fixed_joints: Merge fixed-joint-connected bodies
            **kwargs: Additional args for add_urdf

        Returns:
            Self for method chaining
        """
        # Ensure articulation context
        if name not in self._articulations:
            self.add_articulation(name)
        else:
            self.set_current_articulation(name)

        xform = xform or wp.transform([0, 0, 0], [0, 0, 0, 1])

        # Load URDF
        self.builder.add_urdf(
            source=str(file),
            xform=xform,
            floating=floating,
            collapse_fixed_joints=collapse_fixed_joints,
            **kwargs
        )

        # Update articulation metadata
        artic = self._articulations[name]
        artic.joint_q_count = len(self.builder.joint_q) - artic.joint_q_start
        artic.joint_qd_count = len(self.builder.joint_qd) - artic.joint_qd_start

        log.info(f"Loaded URDF '{file}' as articulation '{name}' "
                 f"({artic.joint_q_count} DOFs)")

        return self

    def load_usd(
        self,
        name: str,
        file: Union[str, Path],
        xform: Optional[wp.transform] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Load a USD (Universal Scene Description) file.

        Args:
            name: Articulation name
            file: Path to USD file
            xform: Initial transform
            **kwargs: Additional args for add_usd
        """
        if name not in self._articulations:
            self.add_articulation(name)
        else:
            self.set_current_articulation(name)

        xform = xform or wp.transform([0, 0, 0], [0, 0, 0, 1])

        self.builder.add_usd(
            source=str(file),
            xform=xform,
            **kwargs
        )

        artic = self._articulations[name]
        artic.joint_q_count = len(self.builder.joint_q) - artic.joint_q_start
        artic.joint_qd_count = len(self.builder.joint_qd) - artic.joint_qd_start

        log.info(f"Loaded USD '{file}' as articulation '{name}'")

        return self

    def load_mjcf(
        self,
        name: str,
        file: Union[str, Path],
        xform: Optional[wp.transform] = None,
        ignore_names: Optional[List[str]] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """
        Load a MuJoCo MJCF XML file.

        Args:
            name: Articulation name
            file: Path to MJCF file
            xform: Initial transform
            ignore_names: List of body names to skip
            **kwargs: Additional args for add_mjcf
        """
        if name not in self._articulations:
            self.add_articulation(name)
        else:
            self.set_current_articulation(name)

        xform = xform or wp.transform([0, 0, 0], [0, 0, 0, 1])

        self.builder.add_mjcf(
            source=str(file),
            xform=xform,
            ignore_names=ignore_names or [],
            **kwargs
        )

        artic = self._articulations[name]
        artic.joint_q_count = len(self.builder.joint_q) - artic.joint_q_start
        artic.joint_qd_count = len(self.builder.joint_qd) - artic.joint_qd_start

        log.info(f"Loaded MJCF '{file}' as articulation '{name}'")

        return self

    # ---------- High-Level Convenience Methods ----------

    def add_simple_object(
        self,
        name: str,
        shape: str = "box",
        size: Union[List[float], float] = 0.1,
        xform: Optional[wp.transform] = None,
        mass: float = 1.0,
        free: bool = True,
        **shape_kwargs
    ) -> "NewtonBuilder":
        """
        Convenience method to add a simple geometric object.

        Args:
            name: Object name
            shape: Shape type ("box", "sphere", "capsule", "cylinder")
            size: Size parameter (half-extents for box, radius for sphere, etc.)
            xform: Initial transform
            mass: Object mass
            free: If True, add free joint for 6-DOF motion
            **shape_kwargs: Additional shape parameters (friction, restitution, etc.)

        Returns:
            Self for method chaining
        """
        # Create articulation for this object
        self.add_articulation(name)

        # Add body
        self.add_body(name=name, xform=xform, mass=mass)

        # Add free joint if requested
        if free:
            self.add_joint_free(name=f"{name}_root", parent=-1, child=name)

        # Add shape
        if shape == "box":
            if isinstance(size, (list, tuple)):
                hx, hy, hz = size[0], size[1], size[2]
            else:
                hx = hy = hz = size
            self.add_shape_box(body=name, hx=hx, hy=hy, hz=hz, **shape_kwargs)
        elif shape == "sphere":
            radius = size if isinstance(size, (int, float)) else size[0]
            self.add_shape_sphere(body=name, radius=radius, **shape_kwargs)
        elif shape == "capsule":
            if isinstance(size, (list, tuple)):
                radius, half_height = size[0], size[1]
            else:
                radius = half_height = size
            self.add_shape_capsule(body=name, radius=radius, half_height=half_height, **shape_kwargs)
        elif shape == "cylinder":
            if isinstance(size, (list, tuple)):
                radius, half_height = size[0], size[1]
            else:
                radius = half_height = size
            self.add_shape_cylinder(body=name, radius=radius, half_height=half_height, **shape_kwargs)
        else:
            raise ValueError(f"Unknown shape type: {shape}")

        return self

    # ---------- Particles ----------

    def add_particle(
        self,
        pos: Union[wp.vec3, Tuple, List],
        vel: Union[wp.vec3, Tuple, List] = (0, 0, 0),
        mass: float = 1.0,
        radius: Optional[float] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """Add a single particle to the scene."""
        radius = radius or self.builder.default_particle_radius
        self.builder.add_particle(pos=pos, vel=vel, mass=mass, radius=radius, **kwargs)
        return self

    def add_particles(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        **kwargs
    ) -> "NewtonBuilder":
        """Add multiple particles at once."""
        self.builder.add_particles(
            positions=positions,
            velocities=velocities,
            masses=masses,
            radii=radii,
            **kwargs
        )
        return self

    def add_particle_grid(
        self,
        lower: Union[wp.vec3, Tuple, List],
        upper: Union[wp.vec3, Tuple, List],
        spacing: float,
        **kwargs
    ) -> "NewtonBuilder":
        """Add a regular grid of particles."""
        self.builder.add_particle_grid(lower=lower, upper=upper, spacing=spacing, **kwargs)
        return self

    # ---------- Environment Replication ----------

    def replicate_articulation(
        self,
        num_envs: int,
        spacing: Union[float, Tuple[float, float, float]] = 2.0
    ) -> "NewtonBuilder":
        """
        Replicate the current scene for parallel environments.

        Newton's replicate() creates multiple copies of the scene for vectorized
        simulation (useful for RL training).

        Args:
            num_envs: Number of environment copies
            spacing: Spacing between environments (scalar or (x,y,z) tuple)

        Returns:
            Self for method chaining
        """
        if isinstance(spacing, (int, float)):
            spacing = (spacing, spacing, 0.0)

        # Note: Newton's replicate works on the whole builder, not per-articulation
        # For now, we'll just call it directly
        # TODO: Implement per-articulation replication if needed
        log.warning("replicate_articulation() replicates the entire scene, not just one articulation")

        # We can't actually replicate here because it would duplicate everything
        # This should be called after all articulations are added
        # Store parameters for later
        self._replicate_params = {
            "num_envs": num_envs,
            "spacing": spacing
        }

        return self

    # ---------- Spawn State Generation ----------

    def set_joint_defaults(
        self,
        joint_defaults: Dict[str, Union[float, List[float]]]
    ) -> "NewtonBuilder":
        """
        Set default joint positions for spawn state generation.

        Args:
            joint_defaults: Dict mapping joint names to default positions
                           (scalar for 1-DOF, list for multi-DOF joints)

        Returns:
            Self for method chaining
        """
        for name, value in joint_defaults.items():
            if isinstance(value, (int, float)):
                self._joint_defaults[name] = [float(value)]
            else:
                self._joint_defaults[name] = [float(v) for v in value]
        return self

    def make_spawn_state(
        self,
        name: str = "spawn",
        joint_positions: Optional[Dict[str, Union[float, List[float]]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate initial spawn state for the scene.

        Similar to mjcf_builder's make_spawn_keyframe(), but returns numpy arrays
        instead of XML. Includes both generalized (joint_q) and maximal (body_q)
        coordinates.

        Args:
            name: State name (for logging/debugging)
            joint_positions: Optional dict of joint positions (overrides defaults)

        Returns:
            Dict with keys:
                - joint_q: Generalized positions (numpy array)
                - joint_qd: Generalized velocities (numpy array, all zeros)
                - body_q: Maximal coordinates (numpy array, computed via FK)
                - body_qd: Body velocities (numpy array, all zeros)
        """
        # Merge defaults with overrides
        merged_defaults = dict(self._joint_defaults)
        if joint_positions:
            for k, v in joint_positions.items():
                if isinstance(v, (int, float)):
                    merged_defaults[k] = [float(v)]
                else:
                    merged_defaults[k] = [float(x) for x in v]

        # Build joint_q array
        joint_q = []
        for joint_spec in self._joints:
            jname = joint_spec.name
            q_size = joint_spec.q_size

            if jname in merged_defaults:
                vals = merged_defaults[jname]
                if len(vals) != q_size:
                    raise ValueError(
                        f"Joint '{jname}' expects {q_size} DOFs, got {len(vals)}"
                    )
                joint_q.extend(vals)
            else:
                # Default values based on joint type
                if joint_spec.joint_type == newton.JointType.FREE:
                    # Free joint: [x, y, z, qx, qy, qz, qw]
                    # Use body's initial transform if available
                    body_spec = self._bodies.get(
                        next((b.name for b in self._bodies.values()
                             if b.body_idx == joint_spec.child_body_idx), None)
                    )
                    if body_spec:
                        xform = body_spec.xform
                        joint_q.extend([
                            xform.p[0], xform.p[1], xform.p[2],  # position
                            xform.q[0], xform.q[1], xform.q[2], xform.q[3]  # quat (xyzw)
                        ])
                    else:
                        joint_q.extend([0, 0, 0, 0, 0, 0, 1])
                elif joint_spec.joint_type == newton.JointType.BALL:
                    # Ball joint: [qx, qy, qz, qw]
                    joint_q.extend([0, 0, 0, 1])
                else:
                    # Revolute/prismatic: single value
                    joint_q.extend([0.0] * q_size)

        joint_q = np.array(joint_q, dtype=np.float32)
        joint_qd = np.zeros_like(joint_q)

        log.ok(f"Generated spawn state '{name}' with {len(joint_q)} generalized DOFs")

        return {
            "name": name,
            "joint_q": joint_q,
            "joint_qd": joint_qd,
            # Note: body_q and body_qd would require calling eval_fk after finalize
            # We'll add that capability after finalize() is implemented
        }

    # ---------- Utilities ----------

    def get_body_idx(self, name: str) -> int:
        """Get body index by name."""
        if name not in self._bodies:
            raise ValueError(f"Body '{name}' not found")
        return self._bodies[name].body_idx

    def get_articulation(self, name: str) -> ArticulationSpec:
        """Get articulation metadata by name."""
        if name not in self._articulations:
            raise ValueError(f"Articulation '{name}' not found")
        return self._articulations[name]

    def joint_order(self) -> List[str]:
        """Return list of joint names in qpos order."""
        return [j.name for j in self._joints]

    def _resolve_body_idx(self, body: Union[str, int]) -> int:
        """Convert body name or index to index."""
        if isinstance(body, str):
            return self.get_body_idx(body)
        return body

    def _track_joint(
        self,
        name: str,
        joint_type: newton.JointType,
        parent_idx: int,
        child_idx: int,
        q_dofs: int,
        qd_dofs: int
    ):
        """Internal helper to track joint metadata."""
        q_start = len(self._joints)  # Simplified - would need actual index tracking

        joint_spec = JointSpec(
            name=name,
            joint_type=joint_type,
            articulation=self._current_articulation,
            parent_body_idx=parent_idx,
            child_body_idx=child_idx,
            q_start=q_start,
            q_size=q_dofs,
            qd_start=q_start,  # Simplified
            qd_size=qd_dofs
        )

        self._joints.append(joint_spec)

        # Update articulation metadata
        if self._current_articulation:
            artic = self._articulations[self._current_articulation]
            artic.joint_indices.append(len(self._joints) - 1)

    # ---------- Finalization ----------

    def finalize(self, device: str = "cuda:0") -> Tuple[newton.Model, Dict]:
        """
        Finalize the builder and return Newton model + metadata.

        Args:
            device: Compute device (e.g., "cuda:0", "cpu")

        Returns:
            Tuple of (Newton Model, metadata dict)
        """
        # Apply environment replication if requested
        if hasattr(self, '_replicate_params'):
            params = self._replicate_params
            log.info(f"Replicating scene {params['num_envs']} times...")
            # Newton's replicate needs to be called before finalize
            # but after all entities are added
            # For now, skip as it requires more complex integration
            log.warning("Environment replication not yet implemented in finalize()")

        # Finalize Newton model
        log.info(f"Finalizing Newton model '{self.model_name}' on {device}...")
        model = self.builder.finalize(device=device)

        # Build metadata dict
        metadata = {
            "model_name": self.model_name,
            "articulations": {
                name: {
                    "key": spec.key,
                    "num_bodies": len(spec.body_indices),
                    "num_joints": len(spec.joint_indices),
                    "joint_q_start": spec.joint_q_start,
                    "joint_q_count": spec.joint_q_count,
                    "joint_qd_start": spec.joint_qd_start,
                    "joint_qd_count": spec.joint_qd_count,
                }
                for name, spec in self._articulations.items()
            },
            "bodies": {
                name: {
                    "idx": spec.body_idx,
                    "articulation": spec.articulation,
                    "mass": spec.mass,
                }
                for name, spec in self._bodies.items()
            },
            "joints": [
                {
                    "name": spec.name,
                    "type": spec.joint_type.name,
                    "articulation": spec.articulation,
                    "q_start": spec.q_start,
                    "q_size": spec.q_size,
                }
                for spec in self._joints
            ],
            "num_bodies": model.body_count,
            "num_joints": model.joint_count,
            "num_dofs": model.joint_dof_count,
        }

        log.ok(f"Finalized model with {model.body_count} bodies, "
               f"{model.joint_count} joints, {model.joint_dof_count} DOFs")

        return model, metadata


# Type aliases for convenience
Vec3 = Union[wp.vec3, Tuple[float, float, float], List[float]]
Mat33 = Union[wp.mat33, np.ndarray]
