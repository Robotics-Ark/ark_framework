"""Isaac Sim camera driver for ARK sensors (e.g., IntelRealSense)."""

from __future__ import annotations

from math import cos, sin, radians
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ark.system.driver.sensor_driver import CameraDriver
from ark.tools.log import log


def _yaw_pitch_roll_to_pose(
    target: Tuple[float, float, float],
    distance: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[list[float], list[float]]:
    """Convert target/distance/ypr to position/quaternion in Isaac frame."""
    yaw, pitch, roll = map(radians, (yaw_deg, pitch_deg, roll_deg))
    # Camera looks toward target; position offset in spherical coords
    pos = [
        target[0] + distance * cos(pitch) * cos(yaw),
        target[1] + distance * cos(pitch) * sin(yaw),
        target[2] + distance * sin(pitch),
    ]
    # Isaac uses wxyz; core.utils returns (w, x, y, z)
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    quat = euler_angles_to_quat((roll, pitch, yaw), degrees=False)
    # convert to [x,y,z,w]
    return pos, [quat[1], quat[2], quat[3], quat[0]]


class IsaacCameraDriver(CameraDriver):
    """Camera driver that renders RGB-D frames from Isaac Sim."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        world: Any,
    ) -> None:
        self.world = world
        self._camera = None
        self._resolution = (640, 480)
        super().__init__(
            component_name=component_name, component_config=component_config
        )
        self._create_camera_prim()

    def _create_camera_prim(self) -> None:
        # from omni.isaac.sensor import Camera
        from isaacsim.sensors.camera import Camera

        sim_cfg = self.config.get("sim_config", {})
        self._resolution = (
            int(self.config.get("width", 640)),
            int(self.config.get("height", 480)),
        )
        prim_path = sim_cfg.get("prim_path", f"/World/{self.component_name}")

        camera_type = self.config.get("camera_type", "fixed").lower()
        position = sim_cfg.get("position")
        orientation = sim_cfg.get("orientation")  # xyzw

        if camera_type == "fixed":
            fix_cfg = sim_cfg.get("fix", {})
            target = fix_cfg.get("camera_target_position", [0.0, 0.0, 0.0])
            distance = fix_cfg.get("distance", 1.0)
            yaw = fix_cfg.get("yaw", 0.0)
            pitch = fix_cfg.get("pitch", 0.0)
            roll = fix_cfg.get("roll", 0.0)
            position, orientation = _yaw_pitch_roll_to_pose(
                target, distance, yaw, pitch, roll
            )
        elif camera_type == "attached":
            attach_cfg = sim_cfg.get("attach", {})
            position = attach_cfg.get("offset_translation", [0.0, 0.0, 0.0])
            orientation = attach_cfg.get("offset_rotation", [0.0, 0.0, 0.0, 1.0])
            parent_prim = attach_cfg.get("parent_prim")
            if parent_prim:
                prim_path = f"{parent_prim}/{self.component_name}"
        else:
            log.warn(
                f"Unsupported camera_type '{camera_type}' for Isaac; falling back to fixed."
            )
            position = sim_cfg.get("position", [0.0, 0.0, 1.0])
            orientation = sim_cfg.get("orientation", [0.0, 0.0, 0.0, 1.0])

        self._camera = Camera(
            prim_path=prim_path,
            position=position,
            frequency=self.config.get("frequency", 20),
            resolution=self._resolution,
            orientation=orientation,
        )

        # self.world.reset()
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()

        # TODO add depth image as well

    def get_images(self) -> Dict[str, np.ndarray]:
        """Return RGB and depth arrays."""
        # Trigger a render; Isaac sensor API returns RGBA + depth
        rgb = self._camera.get_rgba()[:, :, :3]  # drop alpha
        print("Image shape:", rgb.shape)
        depth = self._camera.get_depth()
        image_out = dict(color=np.asarray(rgb), depth=np.asarray(depth))
        if rgb is None:
            log.warn(f"Camera {self.component_name} has no rgb frames yet.")
            image_out["color"] = np.zeros((*self._resolution[::-1], 3))
        if depth is None:
            log.warn(f"Camera {self.component_name} has no depth frames yet.")
            image_out["depth"] = np.zeros(self._resolution[::-1])

        return image_out

    def shutdown_driver(self) -> None:
        # Nothing to clean up explicitly; the backend/app handles shutdown.
        pass
