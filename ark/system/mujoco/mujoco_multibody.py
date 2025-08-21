"""@file mujoco_multibody.py
@brief Abstractions for multi-body objects in MuJoCo.
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
import mujoco

import textwrap


class SourceType(Enum):
    """Supported source types for object creation."""

    URDF = "urdf"
    PRIMITIVE = "primitive"
    SDF = "sdf"
    MJCF = "mjcf"


SHAPE_MAP = {
    "GEOM_BOX": "box",
    "GEOM_SPHERE": "sphere",
    "GEOM_CAPSULE": "capsule",
    "GEOM_CYLINDER": "cylinder",
    # add more if you need
}


class MujocoMultiBody(SimComponent):

    def __init__(
        self,
        name: str,
        client: Any,
        global_config: Dict[str, Any] = None,
    ):
        super().__init__(name, global_config)
        
        source_str = self.config["source"]
        source_type = getattr(SourceType, source_str.upper())

        if source_type == SourceType.URDF:
            raise NotImplementedError(
                "Loading from URDF is not implemented for MujocoMultiBody."
            )
        elif source_type == SourceType.PRIMITIVE:
            vis = self.config.get("visual")
            if vis:
                # map the type to mujoco shape
                vis_shape_type = SHAPE_MAP[vis["shape_type"].upper()]

                
                vis_opts = vis["visual_shape"]
                print(f"Visual options: {vis_opts}, shape type: {vis_shape_type}")

                # multiply halfExtents by 2 to get real size
                if vis_shape_type == "box":
                    extents_size = [s * 2 for s in vis_opts["halfExtents"]]
                
                if vis_shape_type == "sphere":
                    extents_size = [vis_opts["radius"]]

                rgba = vis_opts.get("rgbaColor", [1, 1, 1, 1])  # default to white if not provided
            else:
                raise ValueError(
                    "Visual configuration is required for primitive shapes."
                )

            col = self.config.get("collision")
            if col:
                log.warning(
                    "Collision shapes are not supported in MujocoMultiBody yet, it is defaulted to visual sizes"
                )


            multi_body = self.config["multi_body"]
            base_position = self.config["base_position"]
            base_orientation = self.config["base_orientation"]
            base_orientation = [base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]] # swap orientation to be xyzw

            if multi_body["baseMass"] == 0:
                free = False
                mass = 0.001  # default small mass for fixed base
            else:
                free = True
                mass = multi_body["baseMass"]


            client.load_object(
                name=name,
                shape=vis_shape_type,
                size=extents_size,
                pos=base_position,
                quat=base_orientation,
                rgba=rgba,
                free=free,
                mass=mass
            )
        

        # setup communication
        self.publisher_name = self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init(
                {self.publisher_name: rigid_body_state_t}
            )


    def update_ids(self, model, data) -> None:
        self.model = model
        self.data = data
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)

    def get_xml_config(self) -> tuple[str, str, Optional[str]]:
        return self.xml_config

    def pack_data(self, data_dict) -> dict[str, Any]:
        """Pack object data into the message format."""
        msg = rigid_body_state_t()
        msg.name = data_dict["name"]
        msg.position = data_dict["position"]
        msg.orientation = data_dict["orientation"]
        msg.lin_velocity = data_dict["lin_velocity"]
        msg.ang_velocity = data_dict["ang_velocity"]
        return {self.publisher_name: msg}

    def get_object_data(self) -> Any:
        """Retrieve the current state of the simulated object."""
        pos = self.data.xpos[self.id]  # shape (3,) [x, y, z]
        orn = self.data.xquat[self.id]  # shape (4,) [w, x, y, z]
        # convert to xyzw
        orn = [orn[1], orn[2], orn[3], orn[0]]

        vel = self.data.cvel[self.id]  # shape (3,) [vx, vy, vz, wx, wy, wz]
        lin_vel = vel[:3]  # linear velocity
        ang_vel = vel[3:]  # angular velocity
        return {
            "name": self.name,
            "position": pos.tolist(),
            "orientation": orn,
            "lin_velocity": lin_vel.tolist(),
            "ang_velocity": ang_vel.tolist(),
        }

    def reset_component(self, channel, msg) -> None:
        raise NotImplementedError(
            "Resetting components is not implemented for MujocoMultiBody."
        )
