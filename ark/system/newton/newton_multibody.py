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
        self._body_indices: list[int] = []

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
        if not urdf_path.exists():
            log.error(f"Newton multi-body '{self.name}': URDF '{urdf_path}' not found.")
            return
        pre_body_count = self.builder.body_count
        try:
            self.builder.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False)
        except Exception as exc:  # noqa: BLE001
            log.error(f"Newton multi-body '{self.name}': failed to load URDF: {exc}")
            return
        self._body_names = list(self.builder.body_key[pre_body_count : self.builder.body_count])

    def _load_primitive(self) -> None:
        pos = self.config.get("base_position", [0.0, 0.0, 0.0])
        orn = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        mass = float(self.config.get("mass", 0.0))
        transform = _as_transform(pos, orn)

        body_idx = self.builder.add_body(xform=transform, mass=mass, key=self.name)
        size = self.config.get("size", [1.0, 1.0, 1.0])
        if len(size) == 3:
            hx, hy, hz = [abs(float(component)) * 0.5 for component in size]
        else:
            hx = hy = hz = 0.5
        self.builder.add_shape_box(body_idx, hx=hx, hy=hy, hz=hz)
        self._body_names = [self.name]

    def _load_ground_plane(self) -> None:
        """Add a ground plane collision surface to the scene."""
        self.builder.add_ground_plane()
        # Ground plane doesn't create a named body, so no body_names to track
        self._body_names = []

    def bind_runtime(self, model: newton.Model, state_accessor) -> None:
        self._model = model
        self._state_accessor = state_accessor
        if not self._body_names:
            return
        key_to_index = {name: idx for idx, name in enumerate(model.body_key)}
        self._body_indices = [key_to_index[name] for name in self._body_names if name in key_to_index]

    def get_object_data(self) -> Dict[str, Any]:
        state = self._state_accessor()
        if state is None or not self._body_indices:
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
