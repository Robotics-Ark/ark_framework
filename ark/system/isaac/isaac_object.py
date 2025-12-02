"""Isaac Sim object helper with LCM ground-truth publishing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ark.system.component.sim_component import SimComponent
from ark.tools.log import log
from ark.utils.source_type_utils import SourceType
from arktypes import flag_t, rigid_body_state_t


class IsaacSimObject(SimComponent):
    """Generic Isaac Sim object that can load USD / URDF / primitives and publish pose."""

    def __init__(self, name: str, world: Any, global_config: Dict[str, Any]) -> None:
        self.world = world
        self._prim_path = None
        self._xform = None

        super().__init__(name=name, global_config=global_config)

        sim_cfg = self.config
        prim_path = sim_cfg.get("prim_path", f"/World/{name}")
        source_str = sim_cfg["source"]
        source_type = getattr(SourceType, source_str.upper())

        if source_type == SourceType.USD:
            usd_path = sim_cfg.get("usd_path")
            if not usd_path:
                raise ValueError(
                    f"USD source selected for '{name}' but no usd_path provided."
                )
            from omni.isaac.core.utils.stage import add_reference_to_stage

            add_reference_to_stage(str(usd_path), prim_path)
            self._prim_path = prim_path

        elif source_type == SourceType.URDF:
            urdf_path = sim_cfg.get("urdf_path")
            if not urdf_path:
                raise ValueError(
                    f"URDF source selected for '{name}' but no urdf_path provided."
                )
            from isaacsim.asset.importer.urdf import import_urdf

            import_urdf(str(urdf_path), prim_path=prim_path)
            self._prim_path = prim_path

        elif source_type == SourceType.PRIMITIVE:
            from isaacsim.core.api.materials.omni_pbr import OmniPBR
            from isaacsim.core.api.objects import VisualCuboid

            textured_material = OmniPBR(
                prim_path="/World/visual_cube_material",
                name="omni_pbr",
                color=np.array([1, 0, 0]),
                texture_scale=[1.0, 1.0],
                texture_translate=[0.5, 0],
            )

            prim_path = f"/World/{name}"
            self.world.scene.add(
                VisualCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array(sim_cfg.get("base_position", [0, 0, 0])),
                    size=1.0,
                    color=np.array([255, 255, 255]),
                    visual_material=textured_material,
                )
            )
            self._prim_path = prim_path

        else:
            raise RuntimeError(f"Unsupported object source type '{source_type}'.")

        # Wrap prim for pose access
        from isaacsim.core.prims import XFormPrim

        self._xform = XFormPrim(prim_paths_expr=self._prim_path, name=name)

        # Setup publisher for ground truth
        self.publisher_name = f"{self.namespace}/" + self.name + "/ground_truth/sim"
        if self.publish_ground_truth:
            self.state_publisher = self.component_channels_init(
                {self.publisher_name: rigid_body_state_t}
            )

    def get_object_data(self) -> Dict[str, Any]:
        """Return current pose (velocities zeroed for now)."""
        position, orientation = self._xform.get_world_poses()
        return {
            "name": self.name,
            "position": position.flatten(),
            "orientation": orientation.flatten(),
            "lin_velocity": [0.0, 0.0, 0.0],
            "ang_velocity": [0.0, 0.0, 0.0],
        }

    def pack_data(self, data_dict: Dict[str, Any]) -> dict[str, rigid_body_state_t]:
        msg = rigid_body_state_t()
        msg.name = data_dict["name"]
        msg.position = data_dict["position"]
        msg.orientation = data_dict["orientation"]
        msg.lin_velocity = data_dict["lin_velocity"]
        msg.ang_velocity = data_dict["ang_velocity"]
        return {self.publisher_name: msg}

    def reset_component(self, channel, msg) -> flag_t:
        """Reset object pose via LCM."""
        if self._xform:
            try:
                self._xform.set_world_pose(msg.position, msg.orientation)
                log.ok(f"Reset object {self.name} to position {msg.position}")
            except Exception as exc:
                log.warn(f"Failed to reset object {self.name}: {exc}")
        else:
            log.warn(f"No XFormPrim available to reset object {self.name}.")
        return flag_t()
