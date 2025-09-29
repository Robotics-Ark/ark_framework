"""@file pybullet_multibody.py
@brief Abstractions for multi-body objects in PyBullet.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import genesis as gs

from ark.tools.log import log
from ark.system.component.sim_component import SimComponent
from arktypes import flag_t, rigid_body_state_t


class SourceType(Enum):
    """Supported source types for object creation."""

    URDF = "urdf"
    PRIMITIVE = "primitive"
    SDF = "sdf"
    MJCF = "mjcf"


class GenesisMultiBody(SimComponent):
    """Utility class for creating Genesis multi-body objects."""

    def __init__(
        self,
        name: str,
        client: Any,
        global_config: dict[str, Any] | None = None,
    ) -> None:
        """Instantiate a GenesisMultiBody object.

        @param name Name of the object.
        @param client Genesis client used for creation.
        @param global_config Global configuration dictionary.
        @return ``None``
        """

        super().__init__(name, global_config)
        self.client = client
        self.body: Any | None = None
        source_str = self.config["source"]
        source_type = getattr(SourceType, source_str.upper())
            
        if source_type == SourceType.PRIMITIVE:
            # Fall back to the original primitive creation if no URDF path is provided
            vis = self.config.get("visual", {})
            vis_shape_type = str(vis.get("shape_type", "GEOM_BOX")).upper()
            vis_opts = vis.get("visual_shape", {})

            col = self.config.get("collision", {})

            mass = self.config.get("multi_body", {}).get("baseMass", 1.0)
            color = vis_opts.get("rgbaColor", [1, 0, 0, 1])  # Default to red if not provided

            if vis_shape_type == "GEOM_SPHERE":
                radius = vis_opts.get("radius", 0.5)
                self.body = self.client.add_entity(
                    gs.morphs.Sphere(
                        pos=self.config.get("base_position", [0, 0, 0]),
                        quat=self.config.get("base_orientation", [0, 0, 0, 1]),
                        radius=radius,
                        fixed=True if mass == 0 else False,
                    ),
                )
            elif vis_shape_type == "GEOM_BOX":
                size = vis_opts.get("halfExtents", [0.5, 0.5, 0.5])
                # Convert half extents to full size
                size = [2 * s for s in size]
                self.body = self.client.add_entity(
                    gs.morphs.Box(
                        pos=self.config.get("base_position", [0, 0, 0]),
                        quat=self.config.get("base_orientation", [0, 0, 0, 1]),
                        size=size,
                        fixed=True if mass == 0 else False,
                    ),
                )
            else:
                log.warn(
                    f"Unsupported primitive type '{vis_shape_type}' for Genesis multi-body; no entity created."
                )

            # Set mass for dynamic objects (mass > 0)
            if mass != 0 and self.body is not None:
                self.body.set_mass(mass)
    
        elif source_type == SourceType.SDF:
            raise ValueError("Not Supported for Genesis")
        elif source_type == SourceType.MJCF:
            raise ValueError("Please use Robot for MJCF files in Genesis")
        else:
            log.error("Unknown source specification. Check your config file.")

        # setup communication
        self.publisher_name = self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init(
                {self.publisher_name: rigid_body_state_t}
            )

    def get_object_data(self) -> dict[str, Any]:
        """!Return the current state of the simulated object.

        @return Dictionary with position, orientation and velocities of the
                object.
        @rtype Dict[str, Any]
        """
        if self.body is None:
            raise RuntimeError("Genesis body has not been created yet.")

        position = self.body.get_pos()
        orientation = self.body.get_quat()
        lin_vel = self.body.get_vel()
        ang_vel = self.body.get_ang()
        return {
            "name": self.name,
            "position": position,
            "orientation": orientation,
            "lin_velocity": lin_vel,
            "ang_velocity": ang_vel,
        }

    def pack_data(self, data_dict: dict[str, Any]) -> dict[str, rigid_body_state_t]:
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
        return {self.publisher_name: msg}

    def reset_component(self, channel: str, msg: rigid_body_state_t) -> flag_t:
        """!Reset the object pose using a message.

        @param channel LCM channel on which the reset request was received.
        @param msg ``rigid_body_state_t`` containing the desired pose.
        @return ``flag_t`` acknowledging the reset.
        """
        new_pos = msg.position
        new_orn = msg.orientation
        log.info(f"Resetting object {self.name} to position: {new_pos}")
        if self.body is None:
            raise RuntimeError("Cannot reset object before it has been created.")

        self.body.set_pos(new_pos)
        self.body.set_quat(new_orn)
        log.ok(f"Reset object {self.name} completed at: {new_pos}")

        return flag_t()
