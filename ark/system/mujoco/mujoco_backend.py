"""@file mujoco_backend.py
@brief Backend implementation for running simulations in Mujoco.
"""

import importlib.util
import sys, ast, os
import math
import cv2
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer

from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.system.pybullet.pybullet_robot_driver import BulletRobotDriver
from ark.system.pybullet.pybullet_camera_driver import BulletCameraDriver
from ark.system.pybullet.pybullet_multibody import PyBulletMultiBody
from ark.system.driver.sensor_driver import SensorType
from arktypes import *

import textwrap

SHAPE_MAP = {
    "GEOM_BOX": "box",
    "GEOM_SPHERE": "sphere",
    "GEOM_CAPSULE": "capsule",
    "GEOM_CYLINDER": "cylinder",
    # add more if you need
}


def import_class_from_directory(path: Path) -> tuple[type, Optional[type]]:
    """!Load a class from ``path``.

    The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name.  If a ``Drivers`` class is present in the module
    its ``MUJOCO_DRIVER`` attribute is returned alongside the main class.

    @param path Path to the directory containing the module.
    @return Tuple ``(cls, driver_cls)`` where ``driver_cls`` is ``None`` when no
            driver is defined.
    @rtype Tuple[type, Optional[type]]
    """
    # Extract the class name from the last part of the directory path (last directory name)
    class_name = path.name
    file_path = path / f"{class_name}.py"
    # get the full absolute path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    # for imports
    module_dir = os.path.dirname(file_path)
    sys.path.insert(0, module_dir)
    # Extract class names from the AST
    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    # check if Sensor_Drivers is in the class_names
    if "Drivers" in class_names:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(class_names[0], file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_names[0]] = module
        spec.loader.exec_module(module)

        class_ = getattr(module, class_names[0])
        sys.path.pop(0)

        drivers = class_.MUJOCO_DRIVER
        class_names.remove("Drivers")

    # Retrieve the class from the module (has to be list of one)
    class_ = getattr(module, class_names[0])

    if len(class_names) != 1:
        raise ValueError(
            f"Expected exactly two class definition in {file_path}, but found {len(class_names)}."
        )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)

    # Retrieve the class from the module (has to be list of one)
    class_ = getattr(module, class_names[0])
    sys.path.pop(0)

    # Return the class
    return class_, drivers


def convert_urdf_to_mjcf(urdf_path: str) -> str:
    """
    Uses the `urdf2mjcf` CLI if installed: `pip install urdf2mjcf`.
    Returns MJCF xml string.
    """
    outdir = tempfile.mkdtemp(prefix="urdf2mjcf_")
    out_xml = os.path.join(outdir, "converted.mjcf.xml")
    # urdf2mjcf <in> -o <out>
    subprocess.run(["urdf2mjcf", urdf_path, "-o", out_xml], check=True)
    return open(out_xml, "r", encoding="utf-8").read()


class MujocoBackend(SimulatorBackend):

    def initialize(self) -> None:
        self.world_model_dict = {}
        # set gravity if specified in the global config

        gravity = self.global_config["simulator"]["config"].get(
            "gravity", [0, 0, -9.81]
        )
        self.world_model_dict["gravity"] = self.set_gravity(gravity)

        self.world_model_dict["objects"] = []
        if self.global_config.get("objects", None):
            for obj_name, obj_config in self.global_config["objects"].items():
                print(f"Adding object {obj_name}")
                self.world_model_dict["objects"].append(
                    self.add_sim_component(obj_name, obj_config)
                )

        world_xml = self.build_world(self.world_model_dict)
        self.model, self.data = self.compile_model(world_xml)

        if self.global_config.get("connection_mode", "GUI") == "GUI":
            # Launch the viewer in GUI mode
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            # Launch the viewer in passive mode (headless)
            raise NotImplementedError(
                "Headless mode is not implemented for Mujoco yet."
            )

    def compile_model(self, xml: str):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    def build_world(self, world_xml: dict) -> str:
        bodies = "\n".join(world_xml["objects"])
        gravity = world_xml["gravity"]
        return f"""
        <mujoco model="dyn_world">
        <compiler angle="degree"/>
            {gravity}
        <worldbody>
            {bodies}
        </worldbody>
        </mujoco>
        """

    def set_gravity(self, gravity: tuple[float, float, float]) -> str:
        return f"""
        <option gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/>
        """

    def reset_simulator(self) -> None:
        pass

    def add_robot(
        self,
        name: str,
        global_config: dict[str, Any],
    ) -> None:
        pass

    def add_sensor(
        self,
        name: str,
        sensor_type: SensorType,
        global_config: dict[str, Any],
    ) -> None:
        pass

    def add_sim_component(
        self,
        name: str,
        obj_config: dict[str, Any],
    ) -> str:
        """
        Returns an MJCF <body>...</body> snippet for a single object
        based on a PyBullet-like config.
        """
        cfg = obj_config

        # Pose
        pos = cfg.get("base_position", [0, 0, 0])
        quat = cfg.get("base_orientation", [0, 0, 0, 1])  # wxyz in MuJoCo (same order)

        if cfg.get("source") == "primitive":
            # Visual/collision (we’ll just build one geom; you can split into two if needed)
            vis = cfg.get("visual", {})
            col = cfg.get("collision", {})
            stype = vis.get("shape_type") or col.get("shape_type") or "GEOM_BOX"
            geom_type = SHAPE_MAP.get(stype, "box")

            # Size: MuJoCo uses half-sizes directly for boxes and radii/half-lengths otherwise
            vis_shape = vis.get("visual_shape", {})
            col_shape = col.get("collision_shape", {})
            half_extents = vis_shape.get("halfExtents") or col_shape.get("halfExtents")
            # RGBA (optional)
            rgba = vis_shape.get("rgbaColor", [0.6, 0.6, 0.6, 1.0])

            # “Static” if baseMass == 0  → no joint. Otherwise, give it a free joint.
            mb = cfg.get("multi_body", {})
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
            return textwrap.dedent(body_xml).strip()

        elif cfg.get("source") == "urdf":
            # Load the URDF file and convert it to MJCF
            urdf_path = cfg.get("urdf_path")
            if not urdf_path:
                raise ValueError("URDF path must be specified for URDF source.")
            if not Path(urdf_path).exists():
                raise FileNotFoundError(f"URDF file {urdf_path} does not exist.")

            scaling = cfg.get("global_scaling", 1.0)

            mjcf_str = convert_urdf_to_mjcf(urdf_path)

            root = ET.fromstring(mjcf_str)
            worldbody = root.find("worldbody")
            if worldbody is None:
                raise ValueError("Converted MJCF missing worldbody section.")

            if scaling != 1.0:
                for geom in worldbody.iter("geom"):
                    size = geom.get("size")
                    if size:
                        vals = [float(v) * scaling for v in size.split()]
                        geom.set("size", " ".join(map(str, vals)))
                    pos_attr = geom.get("pos")
                    if pos_attr:
                        pos_vals = [float(v) * scaling for v in pos_attr.split()]
                        geom.set("pos", " ".join(map(str, pos_vals)))
                for body in worldbody.iter("body"):
                    pos_attr = body.get("pos")
                    if pos_attr:
                        pos_vals = [float(v) * scaling for v in pos_attr.split()]
                        body.set("pos", " ".join(map(str, pos_vals)))

            child_xml = "".join(
                ET.tostring(child, encoding="unicode") for child in worldbody
            )
            body_xml = f"""
            <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
            {child_xml}
            </body>
            """
            return textwrap.dedent(body_xml).strip()

    def remove(self, name: str) -> None:
        pass

    def step(self) -> None:
        """!Step the simulator forward by one time step."""
        mujoco.mj_step(self.model, self.data)
        # print("Stepping simulation...")
        self.viewer.sync()

    def shutdown_backend(self) -> None:
        self.viewer.close()
