"""@file pybullet_camera_driver.py
@brief Camera driver for the PyBullet simulator.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List

from ark.tools.log import log
from ark.system.driver.sensor_driver import CameraDriver

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import mujoco


def rotation_matrix_to_euler(R_world):
    """!Convert a rotation matrix to Euler angles.

    @param R_world ``3x3`` rotation matrix in row-major order.
    @return Euler angles ``[roll, pitch, yaw]`` in degrees.
    @rtype List[float]
    """
    r = R.from_matrix(R_world)
    euler_angles = r.as_euler("xyz", degrees=True)
    return euler_angles


class CameraType(Enum):
    """Supported camera models."""

    FIXED = "fixed"
    ATTACHED = "attached"


class MujocoCameraDriver(CameraDriver):
    """Camera driver implementation for Mujoco."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        attached_body_id: int = None,
        client: Any = None,
    ) -> None:
        """!Create a new camera driver.

        @param component_name Name of the camera component.
        @param component_config Configuration dictionary for the camera.
        @param attached_body_id ID of the body to attach the camera to.
        @param client Optional PyBullet client.
        @return ``None``
        """
        super().__init__(
            component_name, component_config, True
        )  # simulation is always True
        
        self.name = component_name
        self.parent = "__WORLD__" #HARDCODED FOR NOW: as all cameras are fixed to the world

        sim_config = component_config.get("sim_config", {})
        self.fov = sim_config.get("fov", 45)  # Default field of view

        if "quaternion" in sim_config:
            self.quaternion = sim_config.get("quaternion")
            self.quaternion = [self.quaternion[3], self.quaternion[0], self.quaternion[1], self.quaternion[2]]
        else:
            self.quaternion = [1.0, 0.0, 0.0, 0.0]
            
        self.position = sim_config.get("position", [0.0, 0.0, 1.5])
        
        print(client)
        client.load_camera(name=self.name, parent=self.parent, pos=self.position, quat=self.quaternion, fov=self.fov)

    def update_ids(self, model, data) -> None:
        self.model = model
        self.data = data
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        self.renderer = mujoco.Renderer(self.model, self.width, self.height)

    def get_xml_config(self) -> tuple[str, str, Optional[str]]:
        return self.xml_config

    def get_images(self) -> Dict[str, np.ndarray]:
        print("MUJOCOCameraDriver get_images")
        self.renderer.update_scene(self.data, camera=self.cam)
        rgb = self.renderer.render()
        print("MUJOCOCameraDriver get_images rgb shape", rgb.shape)

        import imageio

        imageio.imwrite("frame.png", rgb)

        # Flip the RGB and depth images (MuJoCo uses bottom-left as the origin)
        rgb = np.flipud(rgb.copy())
        # depth = np.flipud(self.depth.copy())
        return {
            "color": rgb,
            "depth": np.zeros(rgb.shape[:2], dtype=np.float32),  # Placeholder for depth
        }

    def shutdown_driver(self) -> None:
        return super().shutdown_driver()
