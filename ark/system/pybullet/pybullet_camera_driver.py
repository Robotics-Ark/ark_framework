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

def rotation_matrix_to_euler(R_world):
    """!Convert a rotation matrix to Euler angles.

    @param R_world ``3x3`` rotation matrix in row-major order.
    @return Euler angles ``[roll, pitch, yaw]`` in degrees.
    @rtype List[float]
    """
    r = R.from_matrix(R_world)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

class CameraType(Enum):
    """Supported camera models."""
    FIXED = "fixed"
    ATTACHED = "attached"
    
    
class BulletCameraDriver(CameraDriver):
    """Camera driver implementation for PyBullet."""

    def __init__(self,
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
        super().__init__(component_name, component_config, True)  # simulation is always True
        self.client = client
        self.attached_body_id = attached_body_id
        
        try:
            self.camera_type = CameraType(self.config["camera_type"])
        except ValueError:
            raise ValueError(f"Invalid camera type for {self.component_name} !")

        self.visual_body_id = None
        self.attached_body_id = attached_body_id
        
        self.visualize = self.config["sim_config"].get("visualize", False)
        self.urdf_path = self.config["sim_config"].get("urdf_path", None)
        self.fov = self.config['sim_config'].get('fov', 60)
        self.near_val = self.config['sim_config'].get('near_val', 0.1)
        self.far_val = self.config['sim_config'].get('far_val', 100.0)
        
        if self.camera_type == CameraType.FIXED:
            self.camera_target_position = self.config['sim_config']["fix"]['camera_target_position']
            self.distance = self.config['sim_config']["fix"]['distance']
            self.yaw = self.config['sim_config']["fix"]['yaw']
            self.pitch = self.config['sim_config']["fix"]['pitch']
            self.roll = self.config['sim_config']["fix"]['roll']
            self.up_axis_index = self.config['sim_config']["fix"]['up_axis_index']
            
            view_matrix = self.client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.camera_target_position,
                                                                        distance=self.distance,
                                                                        yaw=self.yaw,
                                                                        pitch=self.pitch,
                                                                        roll=self.roll,
                                                                        upAxisIndex=self.up_axis_index)
            # for visualization of the camera
            view_matrix_np = np.array(view_matrix).reshape(4, 4).T
            self.current_position = -view_matrix_np[:3, :3].T @ view_matrix_np[:3, 3]
            # Hack 
            # R_test = np.array([-0.4480736, -0.7992300, -0.4005763, 0.0000000, -0.4480736, 0.8939967, -0.8939967, 0.4005763, 0.2007700]).reshape(3, 3)
            # R = np.eye(4)
            # R[:3, :3] = R_test
            # Ry = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            # T_curr = np.eye(4)
            # T_curr[:3, :3] = view_matrix_np[:3, :3].T
            # T_curr[:3, 3] = self.current_position
            # T = T_curr @ R
            self.current_orientation = self.client.getQuaternionFromEuler(rotation_matrix_to_euler(view_matrix_np[:3, :3].T))
            
        
        elif self.camera_type == CameraType.ATTACHED:
            # assert attached body exists
            assert self.attached_body_id is not None
            
            self.parent_name = self.config["sim_config"]['attach']["parent_name"]
            self.parent_link = self.config["sim_config"]['attach'].get("parent_link", None)
            self.offset_translation = self.config["sim_config"]['attach'].get("offset_translation", [0, 0, 0])
            self.offset_rotation = self.config["sim_config"]['attach'].get("offset_rotation", [0, 0, 0])
            self.rel_camera_target = self.config["sim_config"]['attach'].get("rel_camera_target", [1, 0, 0])

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
                if self.parent_link is None or self.parent_link_id is None:
                    position, orientation = p.getBasePositionAndOrientation(self.attached_body_id)
                else:
                    link_state = p.getLinkState(bodyUniqueId = self.attached_body_id, 
                                                linkIndex = self.parent_link_id,
                                                computeForwardKinematics=True)
                    position = link_state[0]
                    orientation = link_state[1]
            except:
                log.error("Could not find link to attach " + self.component_name + " to " + self.parent_name + " !")
            if len(self.offset_rotation) == 3:# euler
                offset_rot = self.client.getQuaternionFromEuler(self.offset_rotation)
            else: # quaternion
                offset_rot = self.offset_rotation
            position, orientation = self.client.multiplyTransforms(position,
                                                                   orientation,
                                                                   self.offset_translation,
                                                                   offset_rot)
            # update position and orientation
            self.current_position = position
            self.current_orientation = orientation
            
        if self.visualize:
            # TODO urdf
            visual_shape = self.client.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.005, 0.02, 0.01], # x,y,z
                rgbaColor=[1, 0, 0, 1],  # Red color
            )
            self.visual_body_id = self.client.createMultiBody(
                baseVisualShapeIndex=visual_shape,
                basePosition=self.current_position,
                baseOrientation=self.current_orientation,
            )
                
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.aspect = self.width / self.height

        # check if color stream is enabled
        if self.config['streams'].get('color'):
            if self.config['streams']['color']['enable']:
                self.color_stream = True
        else:
            self.color_stream = False
        
        # check if depth stream is enabled
        if self.config['streams'].get('depth'):
            if self.config['streams']['depth']['enable']:
                self.depth_stream = True
        else:
            self.depth_stream = False

        # check if infrared stream is enabled
        if self.config['streams'].get('infrared'):
            if self.config['streams']['infrared']['enable']:
                log.warn("Infrared stream is not supported in pybullet !")
        self.infrared_stream = False

        # check if segmentation stream is enabled
        if self.config['streams'].get('segmentation'):
            if self.config['streams']['segmentation']['enable']:
                self.segmentation_stream = True
        else:
            self.segmentation_stream = False

    
    def _update_position(self) -> Any:
        """!Update internal pose information.

        When the camera is attached to a body this queries PyBullet for the
        current link pose and updates ``self.current_position`` and
        ``self.current_orientation``.
        """
        if self.camera_type == CameraType.ATTACHED:
            if self.parent_link is None or self.parent_link_id is None:
                    position, orientation = p.getBasePositionAndOrientation(self.attached_body_id)
            else:
                link_state = p.getLinkState(bodyUniqueId = self.attached_body_id, 
                                            linkIndex = self.parent_link_id,
                                            computeForwardKinematics=True)
                position = link_state[0]
                orientation = link_state[1]
            if len(self.offset_rotation) == 3:# euler
                offset_rot = self.client.getQuaternionFromEuler(self.offset_rotation)
            else: # quaternion
                offset_rot = self.offset_rotation
            self.current_position, self.current_orientation = self.client.multiplyTransforms(position,
                                                                                             orientation,
                                                                                             self.offset_translation,
                                                                                             offset_rot)
            # update visualization
            if self.visualize:
                self.client.resetBasePositionAndOrientation(self.visual_body_id, self.current_position, self.current_orientation)


    def get_images(self):
        """!Capture camera images from the simulator.

        Depending on the enabled streams the returned dictionary can contain the
        keys ``color``, ``depth`` and ``segmentation``.

        @return Dictionary mapping stream names to ``numpy.ndarray`` images.
        @rtype Dict[str, np.ndarray]
        """
        if self.camera_type == CameraType.ATTACHED:
            self._update_position()
            
            cam_target = tuple(a + b for a, b in zip(tuple(self.current_position), 
                                                     p.rotateVector(self.current_orientation, self.rel_camera_target)))
            cam_up_vector = p.rotateVector(self.current_orientation, [0, 0, 1])
            view_matrix = self.client.computeViewMatrix(cameraEyePosition=self.current_position,
                                                        cameraTargetPosition=cam_target,
                                                        cameraUpVector=cam_up_vector,
                                                        )
        elif self.camera_type == CameraType.FIXED:
            view_matrix = self.client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.camera_target_position,
                                                                        distance=self.distance,
                                                                        yaw=self.yaw,
                                                                        pitch=self.pitch,
                                                                        roll=self.roll,
                                                                        upAxisIndex=self.up_axis_index)
                                
        projection_matrix = self.client.computeProjectionMatrixFOV(fov=self.fov,
                                                                   aspect=self.aspect,
                                                                   nearVal=self.near_val,
                                                                   farVal=self.far_val)

        _, _, rgb_img, depth_img, segmentation_img = self.client.getCameraImage(width=self.width, 
                                                                                height=self.height,
                                                                                viewMatrix=view_matrix,
                                                                                projectionMatrix=projection_matrix)
        
        # pack image into dictionary
        images = {}
        if self.color_stream:
            # convert to rgb to bgr
            bgr_image = rgb_img[..., :3][:, :, ::-1]
            images["color"] = bgr_image
        if self.depth_stream:
            # Convert to meters
            depth_img = (self.far_val * self.near_val) / (self.far_val - (self.far_val - self.near_val) * depth_img)
            images["depth"] = depth_img
        if self.segmentation_stream:
            images["segmentation"] = segmentation_img

        return images
    
    def shutdown_driver(self) -> None:
        """!Clean up any resources used by the driver.

        Called when the simulator is shutting down.  The PyBullet camera driver
        currently does not allocate additional resources so the method is empty.
        """
        # nothing to worry about here
        pass
