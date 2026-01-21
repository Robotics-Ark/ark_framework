from __future__ import annotations

from typing import Any

import numpy as np
from ark.system.component.sim_component import SimComponent
from ark.tools.log import log
from ark.utils import lazy
from ark.utils.source_type_utils import SourceType
from arktypes import flag_t, rigid_body_state_t


class IsaacSimObject(SimComponent):
    """Generic Isaac Sim object loader and pose publisher.

    This component abstracts loading and managing objects in Isaac Sim using USD, URDF, or simple primitives.
    It automatically loads the asset, creates the corresponding prim hierarchy, attaches a transform handle
    (`XFormPrim`) for pose access, and optionally publishes ground-truth simulation state.

    Supported types are:
      - `USD`: load a USD asset via reference.
      - `URDF`: import a URDF model into the stage.
      - `PRIMITIVE`: create a DynamicCuboid representing a simple shape.

    Attributes:
        world (Any): Reference to the simulation world.
        _prim_path (str): Path to the Isaac Sim prim representing this object.
        _xform (XFormPrim): Transform handle used for reading and setting poses.
        publisher_name (str): LCM topic name for ground-truth publishing.
        state_publisher (dict[str, Publisher]): Optional publisher for state output.
    """

    def __init__(self, name: str, world: Any, global_config: dict[str, Any]) -> None:
        """Initialize and load the object into the Isaac Sim stage.

        Depending on the configured source type, the object is created by:
        - Adding a USD reference.
        - Importing a URDF.
        - Constructing a primitive (DynamicCuboid) and optionally disabling physics if the mass is zero.
            TODO - Other shapes needs to be added based on config

        Args:
            name (str): Unique component name.
            world (Any): Simulation world or scene container.
            global_config (dict[str, Any]): Global configuration dict.

        """
        self.world = world
        self._prim_path = None
        self._xform = None

        super().__init__(name=name, global_config=global_config)

        self._prim_path = self.config.get("prim_path", f"/World/{name}")
        source_str = self.config["source"]
        source_type = getattr(SourceType, source_str.upper())

        if source_type == SourceType.USD:
            usd_path = self.config.get("usd_path")
            if not usd_path:
                raise ValueError(
                    f"USD source selected for '{name}' but no usd_path provided."
                )

            lazy.omni.isaac.core.utils.stage.add_reference_to_stage(
                str(usd_path), self._prim_path
            )

        elif source_type == SourceType.URDF:
            urdf_path = self.config.get("urdf_path")
            if not urdf_path:
                raise ValueError(
                    f"URDF source selected for '{name}' but no urdf_path provided."
                )

            lazy.isaacsim.asset.importer.urdf.import_urdf(
                str(urdf_path), prim_path=self._prim_path
            )

        elif source_type == SourceType.PRIMITIVE:
            size = self.config.get("size", [1.0, 1.0, 1.0])
            visual = self.config.get("visual", {})
            visual_shape = visual.get("visual_shape", {}) if isinstance(visual, dict) else {}
            half_extents = visual_shape.get("halfExtents")
            if half_extents is not None:
                scale = np.array(half_extents)
            else:
                scale = np.array(size)
            color = visual_shape.get("rgbaColor", [0.6, 0.6, 0.6, 1.0])
            object = lazy.isaacsim.core.api.objects.DynamicCuboid(
                name=name,
                position=np.array(self.config["base_position"]),
                prim_path=self._prim_path,
                scale=scale,
                size=1.0,
                color=np.array(color[:3]),
            )
            base_orientation = self.config.get("base_orientation")
            if base_orientation is not None:
                object.set_world_pose(
                    position=np.array(self.config["base_position"]),
                    orientation=np.array(base_orientation),
                )
            mass = self.config.get("mass")
            if mass is None:
                multi_body = self.config.get("multi_body", {})
                if isinstance(multi_body, dict):
                    mass = multi_body.get("baseMass")
            if mass == 0:
                object.disable_rigid_body_physics()

            self.world.scene.add(object)

        else:
            raise RuntimeError(f"Unsupported object source type '{source_type}'.")

        # Wrap prim for pose access
        self._xform = lazy.isaacsim.core.prims.XFormPrim(
            prim_paths_expr=self._prim_path, name=name
        )

        # Setup publisher for ground truth
        self.publisher_name = f"{self.namespace}/" + self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init(
                {self.publisher_name: rigid_body_state_t}
            )

    def get_object_data(self) -> dict[str, Any]:
        """Retrieve the object's current pose in world coordinates.

        Returns:
            dict[str, Any]: Information describing the object's state:
                {
                    "name": str,
                    "position": ndarray,
                    "orientation": ndarray,
                    "lin_velocity": list[float],
                    "ang_velocity": list[float],
                }
        """
        position, orientation = self._xform.get_world_poses()
        return {
            "name": self.name,
            "position": position.flatten(),
            "orientation": orientation.flatten(),
            "lin_velocity": [0.0, 0.0, 0.0],
            "ang_velocity": [0.0, 0.0, 0.0],
        }

    def pack_data(self, data_dict: dict[str, Any]) -> dict[str, rigid_body_state_t]:
        """Convert pose data into a rigid-body LCM message.

        Takes the raw pose data generated by :meth:`get_object_data`,
        constructs a `rigid_body_state_t` message, and returns it mapped
        to the component's publisher name.

        Args:
            data_dict (dict[str, Any]): Pose information from
                :meth:`get_object_data`.

        Returns:
            dict[str, rigid_body_state_t]: Mapping from publisher topic
            to LCM message ready for transmission.
        """
        msg = rigid_body_state_t()
        msg.name = data_dict["name"]
        msg.position = data_dict["position"]
        msg.orientation = data_dict["orientation"]
        msg.lin_velocity = data_dict["lin_velocity"]
        msg.ang_velocity = data_dict["ang_velocity"]
        return {self.publisher_name: msg}

    def reset_component(self, channel, msg) -> flag_t:
        """Reset the object's world pose via an incoming LCM message.

        This method updates the underlying prim's world position and
        orientation using data from the received state message.

        Args:
            channel (str): LCM channel the message arrived at.
            msg (rigid_body_state_t): Message containing new position and orientation values.

        Returns:
            flag_t: Status flag indicating completion.

        """
        if self._xform:
            self._xform.set_world_pose(msg.position, msg.orientation)
            log.ok(f"Reset object {self.name} to position {msg.position}")
        else:
            log.warn(f"No XFormPrim available to reset object {self.name}.")
        return flag_t()
