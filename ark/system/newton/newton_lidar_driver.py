"""Newton LiDAR driver using RaycastSensor for range measurements."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ark.tools.log import log
from ark.system.driver.sensor_driver import LiDARDriver

try:
    from newton.sensors import RaycastSensor
    RAYCAST_AVAILABLE = True
except ImportError:
    log.warning("Newton RaycastSensor not available - LiDAR scans will be placeholders")
    RAYCAST_AVAILABLE = False


class LiDARType(Enum):
    """Supported LiDAR mounting types."""

    FIXED = "fixed"
    ATTACHED = "attached"


class NewtonLiDARDriver(LiDARDriver):
    """LiDAR driver using Newton's RaycastSensor for range measurements.

    Simulates a 2D planar LiDAR by using RaycastSensor with height=1.
    The sensor casts rays horizontally in a fan pattern and returns
    range measurements.

    Supports two mounting modes:
    - FIXED: Static position and orientation
    - ATTACHED: LiDAR attached to a robot body with offset transform
    """

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        attached_body_id: Optional[int] = None,
    ) -> None:
        super().__init__(component_name, component_config, True)

        sim_cfg = component_config.get("sim_config", {})

        # LiDAR parameters
        self.num_rays = int(sim_cfg.get("num_rays", 360))
        self.linear_range = float(sim_cfg.get("linear_range", 10.0))
        self.angular_range = float(sim_cfg.get("angular_range", 360.0))  # degrees

        # Validate parameters
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be > 0 for {component_name}")
        if self.linear_range <= 0:
            raise ValueError(f"linear_range must be > 0 for {component_name}")
        if not (0 < self.angular_range <= 360):
            raise ValueError(f"angular_range must be > 0 and <= 360 for {component_name}")

        # Determine LiDAR type
        lidar_type_str = sim_cfg.get("lidar_type", "fixed")
        try:
            self.lidar_type = LiDARType(lidar_type_str.lower())
        except ValueError:
            log.warning(f"Newton LiDAR '{component_name}': Unknown lidar_type '{lidar_type_str}', defaulting to FIXED")
            self.lidar_type = LiDARType.FIXED

        # Pre-compute angles array (in radians, relative to sensor frame)
        # Angles are centered around 0, spanning from -angular_range/2 to +angular_range/2
        angular_range_rad = math.radians(self.angular_range)
        if self.angular_range == 360:
            # Don't repeat the same angle
            self._angles = np.linspace(-angular_range_rad / 2, angular_range_rad / 2, self.num_rays, endpoint=False)
        else:
            self._angles = np.linspace(-angular_range_rad / 2, angular_range_rad / 2, self.num_rays, endpoint=True)

        # Fixed LiDAR parameters
        if self.lidar_type == LiDARType.FIXED:
            fix_cfg = sim_cfg.get("fix", {})
            self.current_position = np.array(fix_cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            yaw_deg = fix_cfg.get("yaw", 0.0)
            self.current_yaw = math.radians(yaw_deg)

        # Attached LiDAR parameters
        else:
            attach_cfg = sim_cfg.get("attach", {})
            self._parent_name = attach_cfg.get("parent_name")
            self._parent_link = attach_cfg.get("parent_link")
            self._offset_position = np.array(attach_cfg.get("offset_translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            self._offset_yaw = math.radians(attach_cfg.get("offset_yaw", 0.0))
            self.current_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.current_yaw = 0.0

        # Runtime bindings
        self._model = None
        self._state_accessor: Callable[[], Any] = lambda: None
        self._raycast_sensor: Optional[Any] = None
        self._body_index: Optional[int] = attached_body_id

        log.info(
            f"Newton LiDAR driver '{component_name}': Initialized "
            f"(rays={self.num_rays}, range={self.linear_range}m, "
            f"angular={self.angular_range}deg, type={self.lidar_type.value})"
        )

    def bind_runtime(
        self,
        model: Any,
        state_accessor: Callable[[], Any],
    ) -> None:
        """Bind to Newton model after finalization.

        Args:
            model: Finalized newton.Model
            state_accessor: Callable returning current newton.State
        """
        self._model = model
        self._state_accessor = state_accessor

        # Resolve parent body index for ATTACHED mode
        if self.lidar_type == LiDARType.ATTACHED and self._body_index is None:
            if self._parent_name and hasattr(model, 'body_key'):
                key_to_index = {name: idx for idx, name in enumerate(model.body_key)}
                search_key = self._parent_link or self._parent_name
                self._body_index = key_to_index.get(search_key)

                if self._body_index is None:
                    log.warning(
                        f"Newton LiDAR driver '{self.component_name}': "
                        f"Parent body '{search_key}' not found in model."
                    )

        # Create RaycastSensor configured for planar scanning
        # Use height=1 for 2D LiDAR, width=num_rays
        if RAYCAST_AVAILABLE and self._model is not None:
            try:
                # Compute initial camera direction based on yaw
                direction = np.array([
                    math.cos(self.current_yaw),
                    math.sin(self.current_yaw),
                    0.0
                ], dtype=np.float32)

                # Up vector is Z (planar LiDAR scans horizontally)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                # FOV in radians for horizontal scan
                fov_rad = math.radians(self.angular_range)

                self._raycast_sensor = RaycastSensor(
                    model=self._model,
                    camera_position=tuple(self.current_position),
                    camera_direction=tuple(direction),
                    camera_up=tuple(up),
                    fov_radians=fov_rad,
                    width=self.num_rays,
                    height=1,  # Single row for 2D LiDAR
                    max_distance=self.linear_range,
                )
                log.ok(f"Newton LiDAR driver '{self.component_name}': RaycastSensor created")
            except Exception as e:
                log.warning(f"Newton LiDAR driver '{self.component_name}': Failed to create RaycastSensor: {e}")
                self._raycast_sensor = None

    def _update_position(self) -> None:
        """Update LiDAR position and orientation for ATTACHED mode."""
        if self.lidar_type != LiDARType.ATTACHED or self._body_index is None:
            return

        state = self._state_accessor()
        if state is None or state.body_q is None:
            return

        # Get body transform
        body_q = state.body_q.numpy()
        if self._body_index >= len(body_q):
            return

        body_transform = body_q[self._body_index]
        body_pos = body_transform[:3]
        body_quat = body_transform[3:7]  # xyzw

        # Get body yaw
        body_rot = R.from_quat(body_quat)
        body_euler = body_rot.as_euler('xyz')
        body_yaw = body_euler[2]

        # Apply offset
        offset_pos_rotated = np.array([
            self._offset_position[0] * math.cos(body_yaw) - self._offset_position[1] * math.sin(body_yaw),
            self._offset_position[0] * math.sin(body_yaw) + self._offset_position[1] * math.cos(body_yaw),
            self._offset_position[2]
        ], dtype=np.float32)

        self.current_position = (body_pos + offset_pos_rotated).astype(np.float32)
        self.current_yaw = body_yaw + self._offset_yaw

        # Update RaycastSensor pose
        if self._raycast_sensor is not None:
            direction = np.array([
                math.cos(self.current_yaw),
                math.sin(self.current_yaw),
                0.0
            ], dtype=np.float32)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            self._raycast_sensor.update_camera_pose(
                position=tuple(self.current_position),
                direction=tuple(direction),
                up=tuple(up),
            )

    def get_scan(self) -> Dict[str, np.ndarray]:
        """Get LiDAR scan data.

        Returns:
            Dictionary with keys:
            - "angles": 1D array of angles in radians (in LiDAR's reference frame)
            - "ranges": 1D array of range values in meters (-1 for no hit)
        """
        # Update position for attached LiDAR
        if self.lidar_type == LiDARType.ATTACHED:
            self._update_position()

        # Get range measurements
        ranges = self._get_ranges()

        return {
            "angles": self._angles.copy(),
            "ranges": ranges,
        }

    def _get_ranges(self) -> np.ndarray:
        """Get range measurements from RaycastSensor."""
        if self._raycast_sensor is None:
            # Return placeholder (all -1 = no hit)
            return np.full(self.num_rays, -1.0, dtype=np.float32)

        state = self._state_accessor()
        if state is None:
            return np.full(self.num_rays, -1.0, dtype=np.float32)

        # Evaluate raycast
        self._raycast_sensor.eval(state)
        depth = self._raycast_sensor.get_depth_image_numpy()

        # depth is shape (1, num_rays), flatten to (num_rays,)
        ranges = depth.flatten().astype(np.float32)

        # RaycastSensor uses -1.0 for no hit, which matches expected LiDAR output
        return ranges

    def shutdown_driver(self) -> None:
        """Clean up resources."""
        self._raycast_sensor = None
        self._model = None
        super().shutdown_driver()
