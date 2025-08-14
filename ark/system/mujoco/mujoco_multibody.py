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
        print(global_config)
        super().__init__(name, global_config)

        self.config = self._load_config_section(
            global_config=global_config, name=name, type="objects"
        )

        # Pose
        pos = self.config.get("base_position", [0, 0, 0])
        quat = self.config.get("base_orientation", [0, 0, 0, 1])  # wxyz in MuJoCo (same order)

        if self.config.get("source") == "primitive":
            # Visual/collision (we’ll just build one geom; you can split into two if needed)
            vis = self.config.get("visual", {})
            col = self.config.get("collision", {})
            stype = vis.get("shape_type") or col.get("shape_type") or "GEOM_BOX"
            geom_type = SHAPE_MAP.get(stype, "box")

            # Size: MuJoCo uses half-sizes directly for boxes and radii/half-lengths otherwise
            vis_shape = vis.get("visual_shape", {})
            col_shape = col.get("collision_shape", {})
            half_extents = vis_shape.get("halfExtents") or col_shape.get("halfExtents")
            # RGBA (optional)
            rgba = vis_shape.get("rgbaColor", [0.6, 0.6, 0.6, 1.0])

            # “Static” if baseMass == 0  → no joint. Otherwise, give it a free joint.
            mb = self.config.get("multi_body", {})
            base_mass = mb.get("baseMass", 0)

            # Build <geom size="..."> attribute depending on type
            if geom_type == "box":
                if not half_extents:
                    raise ValueError("box needs visual/collision.halfExtents")
                size_attr = (
                    f'size="{half_extents[0]} {half_extents[1]} {half_extents[2]}"'
                )
            elif geom_type in ("sphere",):
                r = vis_shape.get("radius") or col_shape.get("radius")
                if r is None:
                    raise ValueError(f"{geom_type} needs radius")
                size_attr = f'size="{r}"'
            elif geom_type in ("capsule", "cylinder"):
                r = vis_shape.get("radius") or col_shape.get("radius")
                hl = vis_shape.get("halfLength") or col_shape.get("halfLength")
                if r is None or hl is None:
                    raise ValueError(f"{geom_type} needs radius and halfLength")
                size_attr = f'size="{r} {hl}"'
            else:
                raise ValueError(f"Unsupported geom type: {geom_type}")

            joint_xml = '<joint type="free"/>' if base_mass and base_mass > 0 else ""
            body_xml = f"""
            <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
            {joint_xml}
            <geom name="{name}_geom" type="{geom_type}" {size_attr} rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            </body>
            """
            xml_config = None, textwrap.dedent(body_xml).strip(), None

        elif self.config.get("source") == "urdf":
            raise NotImplementedError(
                "URDF source is not implemented for Mujoco yet."
            )
        elif self.config.get("source") == "xml":
            raise ValueError(f"Unsupported source type: {self.config.get('source')}")
            # body_xml = f"""<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
            #                 <include file="{self.config.get("body_path", "")}"/>
            #             </body>"""
            # asset_xml = f"""
            #             <include file="{self.config.get("asset_path", "")}"/>
            #              """

            # default_xml = self.config.get("default_path", None)

            # # print(f"Adding XML object {name} with path {body_xml}")
            # xml_config = textwrap.dedent(asset_xml).strip(), textwrap.dedent(body_xml).strip(), default_xml

        else:
            raise ValueError(f"Unsupported source type: {self.config.get('source')}")

        self.xml_config = xml_config

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
        print("======", self.id)

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