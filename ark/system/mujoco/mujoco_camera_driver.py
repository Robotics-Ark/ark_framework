"""@file mujoco_camera_driver.py
@brief Camera driver for the MuJoCo simulator."""

from enum import Enum
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from ark.system.driver.sensor_driver import CameraDriver


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> List[float]:
    """!Convert a rotation matrix to Euler angles.

    @param rotation_matrix ``3x3`` rotation matrix in row-major order.
    @return Euler angles ``[roll, pitch, yaw]`` in degrees.
    @rtype List[float]
    """
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler("xyz", degrees=True)
    return euler_angles


class CameraType(Enum):
    """Supported camera models."""

    FIXED = "fixed"
    ATTACHED = "attached"


class MujocoCameraDriver(CameraDriver):
    """Camera driver implementation for MuJoCo."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        attached_body_id: Optional[int] = None,
        builder: Any | None = None,
    ) -> None:
        """!Create a new camera driver.

        @param component_name Name of the camera component.
        @param component_config Configuration dictionary for the camera.
        @param attached_body_id ID of the body to attach the camera to.
        @param builder Optional MJCF builder instance.
        @return ``None``
        """
        super().__init__(component_name, component_config, True)

        self.name = component_name
        self.parent = "__WORLD__"  # All cameras are fixed to the world
        self.width = component_config.get("width", 100)
        self.height = component_config.get("height", 100)

        sim_config = component_config.get("sim_config", {})
        self.fov = sim_config.get("fov", 45)  # Default field of view

        if "quaternion" in sim_config:
            quaternion = sim_config.get("quaternion")
            self.quaternion = [
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2],
            ]
        else:
            self.quaternion = [1.0, 0.0, 0.0, 0.0]

        self.position = sim_config.get("position", [0.0, 0.0, 1.5])

        if builder is not None:
            builder.load_camera(
                name=self.name,
                parent=self.parent,
                pos=self.position,
                quat=self.quaternion,
                fov=self.fov,
            )

    def update_ids(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """!Update IDs and create a renderer.

        @param model MuJoCo model instance.
        @param data MuJoCo data instance.
        @return ``None``
        """
        self.model = model
        self.data = data
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.name)
        self.renderer = mujoco.Renderer(self.model, self.width, self.height)

    def get_xml_config(self) -> tuple[str, str, Optional[str]]:
        """!Return the XML configuration snippet for this camera."""
        return self.xml_config

    def get_images(self) -> Dict[str, np.ndarray]:
        """!Capture the current color and depth images.

        @return Dictionary containing ``color`` and ``depth`` arrays.
        @rtype Dict[str, np.ndarray]
        """
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb_image = self.renderer.render()

        # Flip the RGB image (MuJoCo uses bottom-left as the origin)
        rgb_image = np.flipud(rgb_image.copy())
        return {
            "color": rgb_image,
            "depth": np.zeros(rgb_image.shape[:2], dtype=np.float32),
        }

    def shutdown_driver(self) -> None:
        """!Shut down the camera driver."""
        super().shutdown_driver()
