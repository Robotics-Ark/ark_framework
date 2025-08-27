# mjcf_builder.py
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import os
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

# ----------------------------- Utilities -----------------------------


def _attrs(el: ET.Element, **kwargs):
    """Set attributes on an XML element, converting lists/tuples to space-separated strings."""
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        else:
            v = str(v)
        el.set(k, v)
    return el


def _euler_xyz_to_quat(
    ex: float, ey: float, ez: float, *, degrees: bool = True
) -> list[float]:
    """Convert intrinsic XYZ Euler to quaternion [w, x, y, z]."""
    r = R.from_euler('xyz', [ex, ey, ez], degrees=degrees)
    q = r.as_quat()  # returns [x, y, z, w]
    # Convert to [w, x, y, z]
    return [q[3], q[0], q[1], q[2]]


# ------------------------------ Data ---------------------------------


@dataclass
class BodySpec:
    name: str
    pos: Optional[List[float]] = None
    quat: Optional[List[float]] = None
    euler: Optional[List[float]] = None
    child_bodies: List["BodySpec"] = field(default_factory=list)
    geoms: List[Dict] = field(default_factory=list)
    joints: List[Dict] = field(default_factory=list)
    sites: List[Dict] = field(default_factory=list)
    cameras: List[Dict] = field(default_factory=list)


# --------------------------- MJCF Builder ----------------------------


class MJCFBuilder:
    """
    Minimal MJCF builder that supports:
      - assets, bodies, geoms, joints, sites, cameras, actuators, tendons, equalities, contacts
      - wrapping includes in a poseable body (optional free root joint)
      - tracking joint order and body poses
      - creating consolidated keyframes (e.g., 'spawn') with initial positions
      - specifying initial robot base pose and internal joint configuration at spawn
    """

    def __init__(self, model_name: str = "world"):
        self.model_name = model_name

        # Root & top-level sections
        self.root = ET.Element("mujoco")
        self.root.set("model", model_name)

        self.compiler = ET.SubElement(self.root, "compiler")
        _attrs(self.compiler, angle="degree", coordinate="local")

        self.option = ET.SubElement(self.root, "option")
        _attrs(self.option, timestep="0.002")

        self.size = ET.SubElement(self.root, "size")
        self.asset = ET.SubElement(self.root, "asset")
        self.default = ET.SubElement(self.root, "default")
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.sensor = ET.SubElement(self.root, "sensor")
        self.tendon = ET.SubElement(self.root, "tendon")
        self.equality = ET.SubElement(self.root, "equality")
        self.contact = ET.SubElement(self.root, "contact")
        self.keyframe = ET.SubElement(self.root, "keyframe")

        # Asset bookkeeping
        self._materials: Dict[str, Dict] = {}
        self._textures: Dict[str, Dict] = {}
        self._meshes: Dict[str, Dict] = {}
        self._robots: Dict[str, Dict] = {}

        # For quick lookup to attach children under an existing body
        self._bodies: Dict[str, ET.Element] = {"__WORLD__": self.worldbody}

        # Track joint order and body pose for keyframe synthesis
        self._joint_order: List[Dict] = []  # each: {name, body, type, size}
        self._body_pose: Dict[str, Dict] = {}  # body -> {pos:[3], quat:[4]}

        # Global defaults for make_spawn_keyframe (merged with per-call overrides)
        self._joint_defaults: Dict[str, List[float]] = {}

    # ---------- Global configuration ----------
    def set_compiler(self, **kwargs) -> "MJCFBuilder":
        _attrs(self.compiler, **kwargs)
        return self

    def set_option(self, **kwargs) -> "MJCFBuilder":
        _attrs(self.option, **kwargs)
        return self

    # ---------- Assets ----------
    def add_texture(self, name: str, **kwargs) -> "MJCFBuilder":
        if name not in self._textures:
            tex = ET.SubElement(self.asset, "texture")
            _attrs(tex, name=name, **kwargs)
            self._textures[name] = kwargs
        return self

    def add_material(self, name: str, **kwargs) -> "MJCFBuilder":
        if name not in self._materials:
            mat = ET.SubElement(self.asset, "material")
            _attrs(mat, name=name, **kwargs)
            self._materials[name] = kwargs
        return self

    def add_mesh(
        self, name: str, file: Optional[str] = None, **kwargs
    ) -> "MJCFBuilder":
        if name not in self._meshes:
            m = ET.SubElement(self.asset, "mesh")
            _attrs(m, name=name, file=file, **kwargs)
            self._meshes[name] = {"file": file, **kwargs}
        return self

    # ---------- Bodies / Robots / Objects ----------
    @staticmethod
    def _joint_qpos_size(self, jtype: Optional[str]) -> int:
        if jtype == "free":
            return 7
        if jtype == "ball":
            return 4
        return 1  # revolute/slide/hinge/etc.

    def add_body(
        self,
        name: str,
        parent: str = "__WORLD__",
        pos: Optional[List[float]] = None,
        quat: Optional[List[float]] = None,
        euler: Optional[List[float]] = None,
    ) -> "MJCFBuilder":
        parent_el = self._bodies[parent]
        b = ET.SubElement(parent_el, "body")
        _attrs(b, name=name, pos=pos, quat=quat, euler=euler)
        self._bodies[name] = b

        # Record pose for keyframe building (priority: quat > euler > identity)
        if quat is not None:
            q = quat
        elif euler is not None:
            deg = self.compiler.get("angle", "radian") == "degree"
            q = _euler_xyz_to_quat(*euler, degrees=deg)
        else:
            q = [1, 0, 0, 0]
        p = pos if pos is not None else [0, 0, 0]
        self._body_pose[name] = {"pos": p, "quat": q}
        return self

    def add_geom(self, body: str, **kwargs) -> "MJCFBuilder":
        b = self._bodies[body]
        g = ET.SubElement(b, "geom")
        _attrs(g, **kwargs)
        return self

    def add_joint(self, body: str, **kwargs) -> "MJCFBuilder":
        """
        Adds a joint under `body`. If no name is provided, auto-generate one.
        Records joint order and qpos size to help building keyframes later.
        """
        b = self._bodies[body]
        j = ET.SubElement(b, "joint")

        jtype = kwargs.get("type")
        jname = kwargs.get("name")
        if jname is None:
            count = sum(1 for jinfo in self._joint_order if jinfo.get("body") == body)
            jname = f"{body}_joint_{count}"
            kwargs["name"] = jname

        _attrs(j, **kwargs)

        if jtype is not None:
            self._joint_order.append(
                {
                    "name": jname,
                    "body": body,
                    "type": jtype,
                    "size": self._joint_qpos_size(jtype),
                }
            )
        return self

    def add_site(self, body: str, **kwargs) -> "MJCFBuilder":
        b = self._bodies[body]
        s = ET.SubElement(b, "site")
        _attrs(s, **kwargs)
        return self

    # ---------- Cameras ----------
    def add_camera(
        self, parent: str = "__WORLD__", name: Optional[str] = None, **kwargs
    ) -> "MJCFBuilder":
        p = self._bodies[parent]
        c = ET.SubElement(p, "camera")
        _attrs(c, name=name, **kwargs)
        return self

    # ---------- Sensors ----------
    def add_sensor(self, stype: str, **kwargs) -> "MJCFBuilder":
        s = ET.SubElement(self.sensor, stype)
        _attrs(s, **kwargs)
        return self

    # ---------- High-level loaders ----------
    def load_object(
        self,
        name: str,
        shape: str,
        size: Union[List[float], float],
        pos=(0, 0, 0),
        quat: Optional[List[float]] = None,
        density: Optional[float] = None,
        mass: Optional[float] = None,
        rgba: Optional[List[float]] = None,
        free: bool = True,
        **geom_kwargs,
    ) -> "MJCFBuilder":
        """Convenience: create a body + (optional) free joint + geom."""
        self.add_body(name=name, pos=pos, quat=quat)
        if free:
            self.add_joint(name, type="free", name=f"{name}_root")
        self.add_geom(
            name,
            type=shape,
            size=size,
            density=density,
            mass=mass,
            rgba=rgba,
            **geom_kwargs,
        )
        return self

    def load_robot_from_spec(
        self, root: BodySpec, parent: str = "__WORLD__"
    ) -> "MJCFBuilder":
        """Recursively create a robot from a BodySpec tree."""

        def _recurse(spec: BodySpec, parent_body: str):
            self.add_body(
                spec.name,
                parent=parent_body,
                pos=spec.pos,
                quat=spec.quat,
                euler=spec.euler,
            )
            for j in spec.joints:
                self.add_joint(spec.name, **j)
            for g in spec.geoms:
                self.add_geom(spec.name, **g)
            for s in spec.sites:
                self.add_site(spec.name, **s)
            for cam in spec.cameras:
                self.add_camera(parent=spec.name, **cam)
            for child in spec.child_bodies:
                _recurse(child, spec.name)

        _recurse(root, parent)
        return self

    def include(self, file: str, parent: str | None = None) -> "MJCFBuilder":
        """
        Insert <include file="..."/> either at the root (parent=None),
        inside worldbody (parent="__WORLD__"), or inside a specific body.
        """
        if parent is None:
            inc_parent = self.root
        elif parent == "__WORLD__":
            inc_parent = self.worldbody
        else:
            inc_parent = self._bodies[parent]
        inc = ET.SubElement(inc_parent, "include")
        inc.set("file", file)
        return self

    def include_robot(
        self,
        name: str,
        file: str,
        parent: str = "__WORLD__",
        pos: Optional[List[float]] = None,
        quat: Optional[List[float]] = None,
        euler: Optional[List[float]] = None,
        fixed_base: bool = False,
        root_joint_name: Optional[str] = None,
        *,
        qpos: Optional[List[float]] = None,
    ) -> "MJCFBuilder":
        """
        Wrap an <include> in a named body so you can position the whole robot and
        (optionally) give it a free root joint to control base pose via keyframes.

        fixed_base:
            If True, the robot's base is fixed to the world (no root joint).
            If False, a free joint named ``root_joint_name`` (or ``f"{name}_root"``) is added.

        qpos:
            Flat list of the robot's *internal* generalized coordinates (excluding any free base).
            Stored as a single packed block for make_spawn_keyframe().
        """

        # Ensure internal bookkeeping exists
        if not hasattr(self, "_joint_order"):
            self._joint_order = []
        if not hasattr(self, "_joint_defaults"):
            self._joint_defaults = {}

        # 1) Create a wrapper body for the robot so we can position the entire model
        self.add_body(name, parent=parent, pos=pos, quat=quat, euler=euler)

        # 2) Optionally give the wrapper body a free joint to control base pose
        if not fixed_base:
            jname = root_joint_name if root_joint_name is not None else f"{name}_root"

            # Be defensive about builder signatures to avoid 'name' collisions.
            # Preferred: add_joint(body=..., ...attrs)
            try:
                self.add_joint(body=name, type="free", name=jname)
            except TypeError:
                # Fallback: some builders use (parent_or_body_name, **attrs)
                self.add_joint(name, **{"type": "free", "name": jname})

        # 3) Merge the referenced MJCF file into this model
        tree = ET.parse(file)
        root = tree.getroot()

        # Merge compiler attributes (e.g., meshdir) relative to the robot file
        comp = root.find("compiler")
        if comp is not None:
            attrs = dict(comp.attrib)
            if "meshdir" in attrs:
                # Normalize to the directory of the robot file to avoid double prefixes
                robot_dir = os.path.dirname(os.path.abspath(file))
                attrs["meshdir"] = robot_dir
            # Ensure self.compiler exists (your builder likely created it at init)
            _attrs(self.compiler, **attrs)

        # Merge assets, defaults, tendons, equalities, actuators, contacts
        for sec_name, target in [
            ("asset", self.asset),
            ("default", self.default),
            ("tendon", self.tendon),
            ("equality", self.equality),
            ("actuator", self.actuator),
            ("contact", self.contact),
        ]:
            src = root.find(sec_name)
            if src is not None:
                for child in src:
                    target.append(copy.deepcopy(child))

        # Insert robot bodies under the wrapper body
        wb = root.find("worldbody")
        if wb is not None:
            wrapper = self._bodies[name]
            for body in wb:
                wrapper.append(copy.deepcopy(body))

        # 4) Record this robot's internal qpos as a single packed block
        if qpos is not None:
            block_name = f"{name}/*"  # synthetic ID for packed internal qpos
            block_vals = [float(x) for x in qpos]
            self._joint_order.append(
                {
                    "name": block_name,
                    "body": name,
                    "type": "packed",
                    "size": len(block_vals),
                }
            )
            self._joint_defaults[block_name] = block_vals

        return self

    # ---------- Keyframes ----------
    def add_keyframe(
        self,
        name: str,
        qpos: Optional[List[float]] = None,
        qvel: Optional[List[float]] = None,
        ctrl: Optional[List[float]] = None,
        act: Optional[List[float]] = None,
        mocap_pos: Optional[List[float]] = None,
        mocap_quat: Optional[List[float]] = None,
    ) -> "MJCFBuilder":
        """
        Adds a single <key .../> inside <keyframe>.
        Lists are serialized as space-separated strings, e.g. qpos="0 0 0 1 0 0 0 ..."
        """
        k = ET.SubElement(self.keyframe, "key")
        _attrs(
            k,
            name=name,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            act=act,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )
        return self

    def add_keyframes(self, keys: List[Dict]) -> "MJCFBuilder":
        """Bulk add keyframes: each dict can include any fields accepted by add_keyframe()."""
        for k in keys:
            self.add_keyframe(**k)
        return self

    def joint_order(self) -> List[str]:
        """Returns the joint names in qpos order (useful for building qpos vectors)."""
        return [j["name"] for j in self._joint_order]

    def make_spawn_keyframe(
        self,
        name: str = "spawn",
        joint_defaults: Optional[Dict[str, Union[List[float], float]]] = None,
    ) -> "MJCFBuilder":
        """
        Assemble a single keyframe named `name` that sets:
          - free joints to the current body pose (pos + quat)
          - other joints to 0, unless specified in defaults
        'joint_defaults' merges over any defaults previously provided via include_robot(..., joint_qpos=...).

        joint_defaults can map joint_name -> scalar (1-dof) or list (4/7-dof).
        """
        # Merge global defaults with per-call overrides
        merged_defaults: Dict[str, List[float]] = dict(self._joint_defaults)
        if joint_defaults:
            for k, v in joint_defaults.items():
                if isinstance(v, (int, float)):
                    merged_defaults[k] = [float(v)]
                else:
                    merged_defaults[k] = [float(x) for x in v]

        qpos: List[float] = []

        for j in self._joint_order:
            jn, jb, jt, sz = j["name"], j["body"], j["type"], j["size"]

            # If user provided explicit values for this joint
            if jn in merged_defaults:
                val = merged_defaults[jn]
                # allow scalar for 1-dof
                if len(val) == 1 and sz > 1:
                    raise ValueError(
                        f"Default for joint '{jn}' must have length {sz}, got 1"
                    )
                if len(val) != sz:
                    raise ValueError(
                        f"joint_defaults[{jn}] length {len(val)} != expected {sz}"
                    )
                qpos.extend(val)
                continue

            # Otherwise derive a sensible default
            if jt == "free":
                pose = self._body_pose.get(jb, {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]})
                p = pose["pos"]
                q = pose["quat"]  # [w, x, y, z]
                qpos.extend([p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
            elif jt == "ball":
                qpos.extend([1.0, 0.0, 0.0, 0.0])  # identity quaternion
            else:
                qpos.append(0.0)  # revolute/slide default

        self.add_keyframe(name=name, qpos=qpos)
        return self

    # ---------- Serialization ----------
    def to_string(self, pretty: bool = True) -> str:
        """Return the MJCF XML as a string."""

        def indent(elem, level=0):
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                for e in elem:
                    indent(e, level + 1)
                if not e.tail or not e.tail.strip():
                    e.tail = i
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

        if pretty:
            indent(self.root)
        return ET.tostring(self.root, encoding="unicode")

    def update_on_load(self, model):
        for robots in self.robots:
            root_joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "panda_root"
            )
            pass


# --------------------------- Example usage ---------------------------

if __name__ == "__main__":
    # Minimal example demonstrating include + spawn keyframe with base pose + joint config
    builder = (
        MJCFBuilder("demo_world")
        .set_compiler(angle="radian", meshdir="franka_emika_panda")
        .set_option(timestep="0.002")
    )

    # Ground
    builder.add_body("floor", pos=[0, 0, 0]).add_geom(
        "floor", type="plane", size=[10, 10, 0.1], rgba=[0.8, 0.8, 0.8, 1]
    )

    # Include a Franka Panda, wrapped in a poseable body with a free root joint
    # Provide its internal joint list in the correct qpos order and initial joint angles.
    builder.include_robot(
        name="panda",
        file="franka_emika_panda/panda.xml",
        pos=[0.3, -0.2, 0.0],  # initial XYZ
        euler=[0, 0, 0],  # initial orientation (XYZ Euler)
        fixed_base=False,  # add a free joint controlling base pose
        root_joint_name="panda_root",
        internal_joints=[
            # order MUST match the MuJoCo qpos order inside the included file
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        joint_qpos={
            # radians for hinges; these will be baked into the 'spawn' keyframe
            "joint1": 0.0,
            "joint2": -0.6,
            "joint3": 0.0,
            "joint4": -2.2,
            "joint5": 0.0,
            "joint6": 1.6,
            "joint7": 0.8,
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
    )

    # A free object with its own root free joint
    builder.load_object(
        name="cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        pos=[0.6, 0.0, 0.15],
        quat=[1, 0, 0, 0],
        rgba=[0.2, 0.6, 1.0, 1.0],
        free=True,
    )

    # Create a consolidated keyframe capturing initial positions & robot joint angles
    builder.make_spawn_keyframe(name="spawn")

    xml_text = builder.to_string(pretty=True)
    print(xml_text)
