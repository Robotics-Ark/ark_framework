"""Newton multi-body component integration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence

import newton
import numpy as np
import warp as wp

from ark.tools.log import log
from ark.system.component.sim_component import SimComponent
from arktypes import flag_t, rigid_body_state_t


def _as_transform(position: Sequence[float], orientation: Sequence[float]) -> wp.transform:
    pos = wp.vec3(*(float(v) for v in (position[:3] if len(position) >= 3 else [0.0, 0.0, 0.0])))
    if len(orientation) == 4:
        quat = wp.quat(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
    elif len(orientation) == 3:
        quat = wp.quat_from_euler(wp.vec3(float(orientation[0]), float(orientation[1]), float(orientation[2])), 0, 1, 2)
    else:
        quat = wp.quat_identity()
    return wp.transform(pos, quat)


class SourceType(Enum):
    URDF = "urdf"
    PRIMITIVE = "primitive"
    GROUND_PLANE = "ground_plane"


class NewtonMultiBody(SimComponent):
    """Simulation object backed by Newton physics."""

    def __init__(
        self,
        name: str,
        builder: newton.ModelBuilder,
        global_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, global_config)

        self.builder = builder
        self._model: newton.Model | None = None
        self._state_accessor = lambda: None
        self._body_names: list[str] = []
        self._body_indices: list[int] = []  # Pre-stored body indices (stable across finalization)

        source_str = self.config.get("source", "primitive").lower()
        source = SourceType(source_str) if source_str in SourceType._value2member_map_ else SourceType.PRIMITIVE

        if source is SourceType.URDF:
            self._load_urdf()
        elif source is SourceType.PRIMITIVE:
            self._load_primitive()
        elif source is SourceType.GROUND_PLANE:
            self._load_ground_plane()
        else:
            log.warning(f"Newton multi-body '{self.name}': unsupported source '{source_str}'.")

        self.publisher_name = f"{self.name}/ground_truth/sim"
        if self.publish_ground_truth:
            self.component_channels_init({self.publisher_name: rigid_body_state_t})

    def _load_urdf(self) -> None:
        urdf_path = Path(self.config.get("urdf_path", ""))
        if not urdf_path.is_absolute():
            base_dir = Path(self.config.get("class_dir", ""))
            if base_dir.is_file():
                base_dir = base_dir.parent
            urdf_path = base_dir / urdf_path

        # Validate URDF file exists
        if not urdf_path.exists():
            log.error(f"Newton multi-body '{self.name}': URDF file not found at '{urdf_path}'")
            log.error(f"  Full path: {urdf_path.resolve()}")
            return

        pre_body_count = self.builder.body_count
        try:
            self.builder.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False)
            log.ok(f"Newton multi-body '{self.name}': Loaded URDF '{urdf_path.name}' successfully")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Newton multi-body '{self.name}': Failed to load URDF '{urdf_path}': {exc}")
            return

        self._body_names = list(self.builder.body_key[pre_body_count : self.builder.body_count])
        log.info(f"Newton multi-body '{self.name}': Loaded {len(self._body_names)} bodies from URDF")

    def _load_primitive(self) -> None:
        pos = self.config.get("base_position", [0.0, 0.0, 0.0])
        orn = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        mass = float(self.config.get("mass", 0.0))
        transform = _as_transform(pos, orn)
        armature = self.config.get("armature", None)

        size = self.config.get("size", [1.0, 1.0, 1.0])

        # Validate size array
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            log.warning(
                f"Newton multi-body '{self.name}': 'size' must be a list of 3 elements, "
                f"got {size}. Using default [1.0, 1.0, 1.0]"
            )
            size = [1.0, 1.0, 1.0]

        hx, hy, hz = [abs(float(component)) * 0.5 for component in size]

        shape_cfg = self.builder.default_shape_cfg.copy()
        if "contact_ke" in self.config:
            shape_cfg.ke = float(self.config["contact_ke"])
        if "contact_kd" in self.config:
            shape_cfg.kd = float(self.config["contact_kd"])
        if "friction" in self.config:
            shape_cfg.mu = float(self.config["friction"])
        if "restitution" in self.config:
            shape_cfg.restitution = float(self.config["restitution"])
        if "rolling_friction" in self.config:
            shape_cfg.rolling_friction = float(self.config["rolling_friction"])
        if "torsional_friction" in self.config:
            shape_cfg.torsional_friction = float(self.config["torsional_friction"])
        if "thickness" in self.config:
            shape_cfg.thickness = float(self.config["thickness"])
        if "contact_margin" in self.config:
            shape_cfg.contact_margin = float(self.config["contact_margin"])

        # Hydroelastic contact parameters (SDF-based volumetric contacts)
        # Required for stable grasping without penetration
        if self.config.get("is_hydroelastic"):
            shape_cfg.is_hydroelastic = True
            if "k_hydro" in self.config:
                shape_cfg.k_hydro = float(self.config["k_hydro"])
            if "sdf_max_resolution" in self.config:
                shape_cfg.sdf_max_resolution = int(self.config["sdf_max_resolution"])
            log.info(
                f"Newton multi-body '{self.name}': Hydroelastic contacts enabled "
                f"(k_hydro={getattr(shape_cfg, 'k_hydro', 'default')}, "
                f"sdf_max_resolution={getattr(shape_cfg, 'sdf_max_resolution', 'default')})"
            )

        # Static bodies (mass=0) are attached to the world body (-1) for proper
        # collision detection with MuJoCo solver. Dynamic bodies get their own body.
        if mass == 0.0:
            # Static body - attach shape directly to world body
            body_idx = -1
            self.builder.add_shape_box(body_idx, hx=hx, hy=hy, hz=hz, xform=transform, cfg=shape_cfg)
            self._body_indices = []  # No dynamic body to track
            self._body_names = []
            log.info(
                f"Newton multi-body '{self.name}': Created static box attached to world "
                f"(size=[{hx*2:.3f}, {hy*2:.3f}, {hz*2:.3f}], pos={pos})"
            )
        else:
            # Dynamic body - create a new body and attach shape
            if armature is not None:
                body_idx = self.builder.add_body(
                    xform=transform, mass=mass, armature=float(armature), key=self.name
                )
            else:
                body_idx = self.builder.add_body(xform=transform, mass=mass, key=self.name)

            self.builder.add_shape_box(body_idx, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)

            # Store the body index directly (stable across finalization)
            self._body_indices = [body_idx]
            self._body_names = [self.name]  # Keep for logging

            log.info(
                f"Newton multi-body '{self.name}': Created primitive box "
                f"(size=[{hx*2:.3f}, {hy*2:.3f}, {hz*2:.3f}], mass={mass:.3f}), "
                f"body_idx={body_idx}"
            )

    def _load_ground_plane(self) -> None:
        """Add a ground plane collision surface to the scene."""
        self.builder.add_ground_plane()
        # Ground plane doesn't create a named body, so no body_names to track
        self._body_names = []
        log.info(f"Newton multi-body '{self.name}': Added ground plane")

    def bind_runtime(self, model: newton.Model, state_accessor) -> None:
        self._model = model
        self._state_accessor = state_accessor

        # If body indices were already stored during loading (primitives), use them directly
        if self._body_indices:
            log.info(
                f"Newton multi-body '{self.name}': "
                f"Using pre-stored body indices: {self._body_indices}"
            )
            return

        # Fallback: Resolve indices from body names (for URDF-loaded bodies)
        if not self._body_names:
            log.debug(f"Newton multi-body '{self.name}': No bodies to bind (e.g., ground plane)")
            return

        key_to_index = {name: idx for idx, name in enumerate(model.body_key)}
        self._body_indices = [key_to_index[name] for name in self._body_names if name in key_to_index]

        # Warn if bodies weren't found in model
        missing_bodies = [name for name in self._body_names if name not in key_to_index]
        if missing_bodies:
            log.warning(
                f"Newton multi-body '{self.name}': {len(missing_bodies)} body/bodies "
                f"not found in finalized model: {missing_bodies}. "
                f"This may happen if collapse_fixed_joints renamed bodies."
            )

        log.info(
            f"Newton multi-body '{self.name}': "
            f"Bound {len(self._body_indices)}/{len(self._body_names)} bodies to model"
        )

    def get_object_data(self) -> Dict[str, Any]:
        state = self._state_accessor()
        if state is None:
            log.warning(f"Newton multi-body '{self.name}': State accessor returned None")
            return {
                "name": self.name,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "lin_velocity": [0.0, 0.0, 0.0],
                "ang_velocity": [0.0, 0.0, 0.0],
            }
        if not self._body_indices:
            # No bodies to query (e.g., ground plane or unbound object)
            return {
                "name": self.name,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "lin_velocity": [0.0, 0.0, 0.0],
                "ang_velocity": [0.0, 0.0, 0.0],
            }

        body_idx = self._body_indices[0]
        pose = state.body_q.numpy()[body_idx]
        vel = state.body_qd.numpy()[body_idx] if state.body_qd is not None else np.zeros(6)
        position = pose[:3].tolist()
        orientation = pose[3:].tolist()
        linear_velocity = vel[:3].tolist()
        angular_velocity = vel[3:].tolist()

        return {
            "name": self.name,
            "position": position,
            "orientation": orientation,
            "lin_velocity": linear_velocity,
            "ang_velocity": angular_velocity,
        }

    def pack_data(self, data_dict: Dict[str, Any]) -> Dict[str, rigid_body_state_t]:
        msg = rigid_body_state_t()
        msg.name = data_dict["name"]
        msg.position = data_dict["position"]
        msg.orientation = data_dict["orientation"]
        msg.lin_velocity = data_dict["lin_velocity"]
        msg.ang_velocity = data_dict["ang_velocity"]
        return {self.publisher_name: msg}

    def reset_component(self, channel, msg) -> flag_t:
        log.warning(f"Reset not implemented for Newton multi-body '{self.name}'.")
        return flag_t()
