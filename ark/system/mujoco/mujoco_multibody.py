from enum import Enum
from typing import Any, Optional

import mujoco

from ark.tools.log import log
from ark.system.component.sim_component import SimComponent
from arktypes import rigid_body_state_t


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
    """MuJoCo multi-body simulation component."""

    def __init__(
        self,
        name: str,
        builder: Any,
        global_config: dict[str, Any] | None = None,
    ) -> None:
        """!Initialize a multi-body object.

        @param name Name of the component.
        @param builder MJCF builder used to generate the object.
        @param global_config Global configuration dictionary.
        @return ``None``
        """
        super().__init__(name, global_config)

        source_str = self.config["source"]
        source_type = getattr(SourceType, source_str.upper())

        if source_type == SourceType.URDF:
            raise NotImplementedError(
                "Loading from URDF is not implemented for MujocoMultiBody."
            )
        elif source_type == SourceType.PRIMITIVE:
            visual_config = self.config.get("visual")
            if visual_config:
                visual_shape_type = SHAPE_MAP[visual_config["shape_type"].upper()]

                visual_shape = visual_config["visual_shape"]

                if visual_shape_type == "box":
                    extents_size = [s * 1 for s in visual_shape["halfExtents"]]

                if visual_shape_type == "sphere":
                    extents_size = [visual_shape["radius"]]

                rgba = visual_shape.get("rgbaColor", [1, 1, 1, 1])
            else:
                raise ValueError(
                    "Visual configuration is required for primitive shapes."
                )

            collision_config = self.config.get("collision")
            if collision_config:
                log.warning(
                    "Collision shapes are not supported in MujocoMultiBody yet, it is defaulted to visual sizes"
                )

            multibody_config = self.config["multi_body"]
            base_position = self.config["base_position"]
            base_orientation = self.config["base_orientation"]
            base_orientation = [
                base_orientation[1],
                base_orientation[2],
                base_orientation[3],
                base_orientation[0],
            ]

            if multibody_config["baseMass"] == 0:
                free = False
                mass = 0.001
            else:
                free = True
                mass = multibody_config["baseMass"]

            builder.load_object(
                name=name,
                shape=visual_shape_type,
                size=extents_size,
                pos=base_position,
                quat=base_orientation,
                rgba=rgba,
                free=free,
                mass=mass,
            )

        self.publisher_name = self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init(
                {self.publisher_name: rigid_body_state_t}
            )

    def update_ids(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """!Update internal identifiers from MuJoCo.

        @param model MuJoCo model instance.
        @param data MuJoCo data instance.
        @return ``None``
        """
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)

    def get_xml_config(self) -> tuple[str, str, Optional[str]]:
        """!Return the XML configuration snippet for this object."""
        return self.xml_config

    def pack_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """!Pack object data into the message format."""
        msg = rigid_body_state_t()
        msg.name = data["name"]
        msg.position = data["position"]
        msg.orientation = data["orientation"]
        msg.lin_velocity = data["lin_velocity"]
        msg.ang_velocity = data["ang_velocity"]
        return {self.publisher_name: msg}

    def get_object_data(self) -> Any:
        """!Retrieve the current state of the simulated object."""
        position = self.data.xpos[self.body_id]
        orientation = self.data.xquat[self.body_id]
        orientation = [orientation[1], orientation[2], orientation[3], orientation[0]]

        velocity = self.data.cvel[self.body_id]
        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]
        return {
            "name": self.name,
            "position": position.tolist(),
            "orientation": orientation,
            "lin_velocity": linear_velocity.tolist(),
            "ang_velocity": angular_velocity.tolist(),
        }

    def reset_component(self, channel: str, msg: Any) -> None:
        """!Reset the component (not implemented)."""
        raise NotImplementedError(
            "Resetting components is not implemented for MujocoMultiBody."
        )
