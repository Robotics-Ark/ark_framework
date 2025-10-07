"""Newton camera driver placeholder returning synthetic imagery."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ark.system.driver.sensor_driver import CameraDriver


class NewtonCameraDriver(CameraDriver):
    """Camera driver that exposes Newton scene transforms."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        attached_body_id: Optional[int] = None,
    ) -> None:
        super().__init__(component_name, component_config, True)

        sim_cfg = component_config.get("sim_config", {})
        self.width = int(component_config.get("width", sim_cfg.get("width", 640)))
        self.height = int(component_config.get("height", sim_cfg.get("height", 480)))
        self.fov = float(sim_cfg.get("fov", 60.0))
        self.near = float(sim_cfg.get("near", 0.1))
        self.far = float(sim_cfg.get("far", 100.0))

        attach_cfg = sim_cfg.get("attach", {})
        self._parent_body = attach_cfg.get("parent_name")
        self._parent_link = attach_cfg.get("parent_link")
        self._offset_position = attach_cfg.get("position", [0.0, 0.0, 0.0])
        self._offset_orientation = attach_cfg.get("orientation", [0.0, 0.0, 0.0, 1.0])

        self._model = None
        self._state_accessor: Callable[[], Any] = lambda: None
        self._body_index: Optional[int] = attached_body_id

    def bind_runtime(self, model, state_accessor: Callable[[], Any]) -> None:
        self._model = model
        self._state_accessor = state_accessor
        if self._body_index is not None or self._parent_body is None:
            return
        key_to_index = {name: idx for idx, name in enumerate(model.body_key)}
        self._body_index = key_to_index.get(self._parent_body)

    def get_images(self) -> Dict[str, np.ndarray]:
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        return {"color": color, "depth": depth}

    def shutdown_driver(self) -> None:
        super().shutdown_driver()

