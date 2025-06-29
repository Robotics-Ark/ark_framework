"""@file pybullet_lidar_driver.py
@brief LiDAR driver implementation for PyBullet.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List

from ark.tools.log import log
from ark.system.driver.sensor_driver import LiDARDriver

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

"""
Example Config:
class_dir: "examples/sensors/lidar" # Directory where the class is located
type: "LiDAR"                       # Type of sensor
  sim_config:
    lidar_type: "attached" # Fixed or attached to another body
    num_rays: 360          # Number of rays
    linear_range: 10.0     # Maximum range in meters
    angular_range: 360.0   # Field of view in degrees
    fix:
      position: [0.0, 0.0, 1.0]  # Position in meters
      yaw: 0.0                   # Yaw angle (rotation about Z-axis) in degrees
    attach:
      parent_name: ""SimpleTwoWheelCa"      # Name of the parent body to attach to
      parent_link: lidar_link               # Link name of the parent body to attach to. Remove this config param to attach it to the base
      offset_translation: [0.0, 0.0, 0.02]  # Offset translation from the parent body in meters
      offset_yaw: 0.0                       # Offset yaw angle (rotation about Z-axis) in degrees
"""

class LiDARType(Enum):
    """Types of LiDAR supported in the simulation."""
    FIXED = "fixed"
    ATTACHED = "attached"
    
    
class BulletLiDARDriver(LiDARDriver):
    """LiDAR driver for the PyBullet simulator."""

    def __init__(self, 
                 component_name: str,
                 component_config: Dict[str, Any],
                 attached_body_id: int = None,
                 client: Any = None,
                 ) -> None:
        """Initialize the BulletLiDARDriver.

        @param component_name Name of the LiDAR component.
        @param component_config Dictionary containing LiDAR configuration (e.g., number of rays, range).
        @param attached_body_id Optional PyBullet body ID to which the LiDAR is attached.
        @param client PyBullet client ID for multi-client simulations.
        """
        super().__init__(component_name, component_config, True) # sim is always True for pybullet
        self.client = client
        self.attached_body_id = attached_body_id

        sim_config = self.config.get("sim_config", {})
 
        self.num_rays = sim_config.get("num_rays", 360)
        self.linear_range = sim_config.get("linear_range", 10.0)
        self.angular_range = sim_config.get("angular_range", 360.0)
        self.lidar_type = sim_config.get("lidar_type", "fixed")

        # Check config
        assert(self.num_rays > 0), f"num_rays should be greater than 0 for {self.component_name}"
        assert(self.linear_range > 0), f"linear_range should be greater than 0 for {self.component_name}"
        assert(self.angular_range > 0 and self.angular_range <= 360), f"angular_range should >0 and <= 360 for {self.component_name}"
        assert(self.lidar_type in ["fixed", "attached"]), f"lidar_type should be either 'fixed' or 'attached' for {self.component_name}"

        try:
            self.lidar_type = LiDARType(self.lidar_type)
        except ValueError:
            raise ValueError(f"Invalid lidar type for {self.component_name} !")

        if self.lidar_type == LiDARType.FIXED:
            fix_config = sim_config.get("fix", {})
            self.current_position =  fix_config.get("position", [0, 0, 0])

            yaw = fix_config.get("yaw", 0)
            yaw = np.deg2rad(yaw)
            self.current_orientation = self.client.getQuaternionFromEuler([0, 0, yaw])
            
        elif self.lidar_type == LiDARType.ATTACHED:
            # assert attached body exists
            assert self.attached_body_id is not None
            
            attach_config = sim_config.get("attach", {})
            self.parent_name = attach_config.get("parent_name", "SimpleTwoWheelCar")
            self.parent_link = attach_config.get("parent_link", None)
            self.offset_translation = attach_config.get("offset_translation", [0, 0, 0])
            self.offset_yaw = np.deg2rad(attach_config.get("offset_yaw", 0))

            # Get all link names and indices
            num_joints = p.getNumJoints(self.attached_body_id)
            self.link_info = {}
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.attached_body_id, i)
                link_name = joint_info[12].decode('utf-8')  # joint_info[12] is the link name
                self.link_info[link_name] = i

            # Get the parent link ID
            self.parent_link_id = self.link_info.get(self.parent_link, None)

            # extract position and orientation of link
            try:
                if p.getNumJoints(self.attached_body_id) == 0 or self.parent_link_id is None:
                    position, orientation = p.getBasePositionAndOrientation(self.attached_body_id)
                else:
                    link_state = p.getLinkState(bodyUniqueId = self.attached_body_id, 
                                                linkIndex = self.parent_link_id,
                                                computeForwardKinematics=True)
                    position = link_state[0]
                    orientation = link_state[1]
            except:
                log.error("Could not find link to attach " + self.component_name + " to " + self.parent_name + " !")
            
            self.offset_rot = self.client.getQuaternionFromEuler([0, 0, self.offset_yaw]) 
            position, orientation = self.client.multiplyTransforms(position,
                                                                   orientation,
                                                                   self.offset_translation,
                                                                   self.offset_rot)
            # update position and orientation
            self.current_position = position
            self.current_orientation = orientation
    
    def _update_position(self) -> Any:
        """!Update the LiDAR pose when attached to another body.

        This queries the pose of the attachment link and applies the configured
        offset.  ``self.current_position`` and ``self.current_orientation`` are
        updated accordingly.
        """
        if self.lidar_type == LiDARType.ATTACHED:
            if p.getNumJoints(self.attached_body_id) == 0 or self.parent_link_id is None:
                position, orientation = p.getBasePositionAndOrientation(self.attached_body_id)
            else:
                link_state = p.getLinkState(bodyUniqueId = self.attached_body_id, 
                                            linkIndex = self.parent_link_id,
                                            computeForwardKinematics=True)
                                            
                position = link_state[0]
                orientation = link_state[1]
            self.current_position, self.current_orientation = self.client.multiplyTransforms(position,
                                                                                             orientation,
                                                                                             self.offset_translation,
                                                                                             self.offset_rot)

    def get_scan(self) -> Dict[str, np.ndarray]:
        """!Retrieve a simulated LiDAR scan from PyBullet.

        The returned dictionary contains the keys ``angles`` and ``ranges``.  A
        range value of ``-1`` indicates that no hit was recorded for the
        corresponding angle.

        @return Dictionary with keys ``angles`` and ``ranges``.
        @rtype Dict[str, np.ndarray]
        """
        if self.lidar_type == LiDARType.ATTACHED:
            self._update_position()
        
        # Get current yaw
        euler = p.getEulerFromQuaternion(self.current_orientation)
        yaw = euler[2]

        # Set angular range
        angular_range = np.deg2rad(self.angular_range)
        min_angle = yaw - angular_range / 2
        max_angle = yaw + angular_range / 2

        # Don't repeat the same angle if the range is 360 degrees
        if self.angular_range == 360:
            endpoint = False
        else:
            endpoint = True

        # Generate angles
        angles = np.linspace(min_angle, max_angle, self.num_rays, endpoint=endpoint)            

        # Ray directions (2D plane, xy only)
        dx = np.cos(angles)
        dy = np.sin(angles)
        directions = np.stack([dx, dy, np.zeros_like(dx)], axis=1)  # shape (num_rays, 3)                
     
        # Ray start and end positions
        ray_starts = np.array(self.current_position).reshape(1, 3)  # shape (1, 3)
        ray_starts = ray_starts.repeat(self.num_rays, axis=0)  # shape (num_rays, 3)
        ray_ends = ray_starts + directions * self.linear_range

        # Perform ray casting
        results = p.rayTestBatch(ray_starts.tolist(), ray_ends.tolist())

        # Extract distances
        ranges = []
        for i, result in enumerate(results):
            hit = result[0]
            hit_position = result[3]
            if hit != -1:
                dist = np.linalg.norm(np.array(hit_position) - np.array(ray_starts[i]))
            else:
                dist = -1
            ranges.append(dist)
        ranges = np.array(ranges)

        # Convert angles to the LiDAR's reference frame
        angles = angles - yaw
        scan = {"angles": angles, "ranges": ranges}
        return scan
    
    def shutdown_driver(self) -> None:
        """!Shutdown the LiDAR driver.

        Currently no additional resources are allocated, so this is a no-op.
        """
        # nothing to worry about here
        pass
