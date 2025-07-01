"""@file pybullet_multibody.py
@brief Abstractions for multi-body objects in PyBullet.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from enum import Enum
from pathlib import Path
import yaml
import os

from ark.tools.log import log
from ark.system.component.sim_component import SimComponent
from arktypes import flag_t, rigid_body_state_t

class SourceType(Enum):
    """Supported source types for object creation."""
    URDF = "urdf"
    PRIMITIVE = "primitive"
    SDF = "sdf"
    MJCF = "mjcf"

class PyBulletMultiBody(SimComponent):
    """Utility class for creating PyBullet multi-body objects."""
    def __init__(self,
                 name: str,
                 client: Any,
                 global_config: Dict[str, Any] = None,
                 ) -> None:
        """Instantiate a PyBulletMultiBody object.

        @param name Name of the object.
        @param client Bullet client used for creation.
        @param global_config Global configuration dictionary.
        @return ``None``
        """

        super().__init__(name, global_config)
        self.client = client
        source_str = self.config["source"]
        source_type = getattr(SourceType, source_str.upper())
        
        if source_type == SourceType.URDF:
            urdf_path = self.config["urdf_path"]
            if not urdf_path:
                log.error("Selected loading object " + name + " from URDF, but no URDF was provided. Check your config again.")
            
            # If URDF path is provided, load the URDF
            base_position = self.config.get("base_position", [0, 0, 0])  # Default to (0, 0, 0) if not provided
            base_orientation = self.config.get("base_orientation", [0, 0, 0, 1])  # Default to identity quaternion if not provided
            if len(base_orientation) == 3: # Convert euler angles to quaternion if provided
                base_orientation = self.client.getQuaternionFromEuler(base_orientation)
                
            global_scaling = self.config.get("global_scaling", 1.0) # Default is 1.0

            # Load the URDF into the PyBullet simulation
            self.ref_body_id = client.loadURDF(fileName=urdf_path, 
                                               basePosition=base_position, 
                                               baseOrientation=base_orientation, 
                                               globalScaling=global_scaling,
                                               useMaximalCoordinates=1)
            
            # If there is any additional configuration for visual, collision, or dynamics, apply them
            vis = self.config.get("visual")
            if vis:
                vis_shape_type = getattr(client, vis["shape_type"].upper())
                vis_opts = vis["visual_shape"]
                vid = client.createVisualShape(vis_shape_type, **vis_opts)
                client.changeVisualShape(self.ref_body_id, -1, visualShapeIndex=vid)  # Change the visual shape

            col = self.config.get("collision")
            if col:
                col_shape_type = getattr(client, col["shape_type"].upper())
                col_opts = col["collision_shape"]
                cid = client.createCollisionShape(col_shape_type, **col_opts)
                client.changeCollisionShape(self.ref_body_id, -1, collisionShapeIndex=cid)  # Change the collision shape

            dynamics = self.config.get("dynamics")
            if dynamics:
                client.changeDynamics(self.ref_body_id, -1, **dynamics)  # Apply dynamics settings if present
        elif source_type == SourceType.PRIMITIVE:
            # Fall back to the original primitive creation if no URDF path is provided
            vis = self.config.get("visual")
            if vis:
                vis_shape_type = getattr(client, vis["shape_type"].upper())
                vis_opts = vis["visual_shape"]
                vid = client.createVisualShape(vis_shape_type, **vis_opts)
            else:
                vid = -1
            col = self.config.get("collision")
            if col:
                col_shape_type = getattr(client, col["shape_type"].upper())
                col_opts = col["collision_shape"]
                cid = client.createCollisionShape(col_shape_type, **col_opts)
            else:
                cid = -1
            kwargs = dict(
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
            )
            # pybullet format
            multi_body = self.config["multi_body"]
            multi_body["basePosition"] = self.config["base_position"]
            multi_body["baseOrientation"] = self.config["base_orientation"]
            kwargs = {**kwargs, **multi_body}
            self.ref_body_id = client.createMultiBody(**kwargs)

            dynamics = self.config.get("dynamics")
            if dynamics:
                client.changeDynamics(self.ref_body_id, -1, **dynamics)
        elif source_type == SourceType.SDF:
            raise NotImplementedError
        elif source_type == SourceType.MJCF:
            raise NotImplementedError    
        else:
            log.error("Unknown source specification. Check your config file.")

        # setup communication 
        self.publisher_name = self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init({self.publisher_name: rigid_body_state_t})

    def get_object_data(self):
        """!Return the current state of the simulated object.

        @return Dictionary with position, orientation and velocities of the
                object.
        @rtype Dict[str, Any]
        """
        position, orientation = self.client.getBasePositionAndOrientation(self.ref_body_id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.ref_body_id)
        return {
            "name": self.name,
            "position": position,
            "orientation": orientation,
            "lin_velocity": lin_vel,
            "ang_velocity": ang_vel
        }
    
    def pack_data(self, data_dict):
        """!Convert a state dictionary to a ``rigid_body_state_t`` message.

        @param data_dict Dictionary as returned by :func:`get_object_data`.
        @return Mapping suitable for :class:`MultiChannelPublisher`.
        @rtype Dict[str, rigid_body_state_t]
        """
        msg = rigid_body_state_t()
        msg.name = data_dict["name"]
        msg.position = data_dict["position"]
        msg.orientation = data_dict["orientation"]
        msg.lin_velocity = data_dict["lin_velocity"]
        msg.ang_velocity = data_dict["ang_velocity"]
        return {
            self.publisher_name: msg
        }
        

    def reset_component(self, channel, msg) -> None:
        """!Reset the object pose using a message.

        @param channel LCM channel on which the reset request was received.
        @param msg ``rigid_body_state_t`` containing the desired pose.
        @return ``flag_t`` acknowledging the reset.
        """
        new_pos = msg.position
        new_orn = msg.orientation
        log.info(f"Resetting object {self.name} to position: {new_pos}")
        log.info("PyBullet does not support resetting with velocities, Only using positions.")
        self.client.resetBasePositionAndOrientation(self.ref_body_id, new_pos, new_orn)
        log.ok(f"Reset object  {self.name} completed at: {new_pos}")
        
        return flag_t()
