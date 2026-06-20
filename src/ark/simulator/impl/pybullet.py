import numpy as np
from typing import Any, Callable
from functools import cached_property
from gymnasium.spaces import Box
from pybullet_utils.bullet_client import BulletClient

from ark.driver import ControllerMode
from ark.envs.spaces.sensor_space import JointState, Limits
from ark.envs.spaces.image_space import RGBImage, DepthImage, RGBDImage
from ark.envs.spaces.geometry_space import RigidTransform
from ark.simulator.base import Simulator, SimulatedWorld
from ark.simulator.driver import (
    SimulatedJointGroupDriver,
    SimulatedSensorDriver,
    SimulatedRobotDriver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _joint_indices_from_names(
    client: BulletClient, body_id: int, joint_names: list[str]
) -> list[int]:
    name_map: dict[str, int] = {}
    for i in range(client.getNumJoints(body_id)):
        name_map[client.getJointInfo(body_id, i)[1].decode()] = i
    missing = [n for n in joint_names if n not in name_map]
    if missing:
        raise ValueError(f"Joints not found in body {body_id}: {missing}")
    return [name_map[n] for n in joint_names]


def _position_limits(info) -> tuple[float, float]:
    lo, hi = info[8], info[9]
    if lo == 0.0 and hi == 0.0:
        return -np.inf, np.inf
    return lo, hi


def _velocity_limit(info) -> float:
    v = info[11]
    return v if v > 0.0 else 100.0


def _effort_limit(info) -> float:
    e = info[10]
    return e if e > 0.0 else 1000.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PybulletSimulator(Simulator):
    """PyBullet physics backend.

    asset_cfg schema::

        robots:
          - name: franka
            urdf: path/to/franka.urdf
            base_position: [0, 0, 0]        # optional
            base_orientation: [0, 0, 0, 1]  # xyzw, optional
            use_fixed_base: true             # optional, default false
            joint_groups:
              - name: arm
                joint_names: [joint1, joint2, ...]
                control_mode: JOINT_POSITION  # or JOINT_TORQUE
            sensors:                          # optional robot-attached sensors
              - name: wrist_ft
                type: ft
                joint_name: ft_joint
        objects:
          - name: table
            urdf: path/to/table.urdf
            base_position: [1, 0, 0]
            use_fixed_base: true
            track_pose: false               # set true to publish pose
        sensors:                            # standalone (e.g. fixed cameras)
          - name: overhead_cam
            type: rgb                       # rgb | depth | rgbd
            height: 480
            width: 640
            camera_kwargs: {}               # passed to getCameraImage

    dr_cfg schema::

        gravity:
          z_range: [-10.5, -8.5]           # uniform sample each episode
        bodies:
          - name: franka                    # asset name from asset_cfg
            links: [-1]                    # link indices (-1 = base); omit for all
            mass_scale_range: [0.9, 1.1]   # multiplier on original mass
            lateral_friction_range: [0.5, 1.5]
    """

    def _init_simulator(self):
        import pybullet
        mode_name = str(self._sim_cfg.get("connection_mode", "DIRECT")).upper()
        mode = getattr(pybullet, mode_name, None)
        if mode is None:
            raise ValueError(f"Invalid PyBullet connection mode: {mode_name!r}")
        self._client = BulletClient(connection_mode=mode)
        # Populated on first reset_simulator()
        self._body_ids: dict[str, int] = {}
        self._initial_base_states: dict[int, tuple[list, list]] = {}
        self._initial_joint_states: dict[int, list[tuple[float, float]]] = {}
        self._original_dynamics: dict[str, dict[int, dict[str, float]]] = {}
        self._world: SimulatedWorld | None = None

    def reset_simulator(self) -> SimulatedWorld:
        gx = self._gravity.get("x", 0.0)
        gy = self._gravity.get("y", 0.0)
        gz = self._gravity.get("z", -9.81)
        self._client.setGravity(gx, gy, gz)
        self._client.setTimeStep(self._time_step_sec)

        if self._world is None:
            self._load_assets()
            self._world = self._build_world()
        else:
            self._restore_physics_state()

        return self._world

    # ------------------------------------------------------------------
    # Asset loading (first reset only)
    # ------------------------------------------------------------------

    def _load_assets(self):
        for robot_spec in self._asset_cfg.get("robots", []):
            body_id = self._load_urdf(robot_spec)
            name = robot_spec["name"]
            self._body_ids[name] = body_id
            self._store_initial_state(body_id, robot_spec)
            self._store_original_dynamics(name, body_id)

        for obj_spec in self._asset_cfg.get("objects", []):
            body_id = self._load_urdf(obj_spec)
            name = obj_spec["name"]
            self._body_ids[name] = body_id
            self._store_initial_state(body_id, obj_spec)
            self._store_original_dynamics(name, body_id)

    def _load_urdf(self, spec: dict) -> int:
        pos = spec.get("base_position", [0.0, 0.0, 0.0])
        orn = spec.get("base_orientation", [0.0, 0.0, 0.0, 1.0])  # xyzw
        fixed = spec.get("use_fixed_base", False)
        return self._client.loadURDF(spec["urdf"], pos, orn, useFixedBase=fixed)

    def _store_initial_state(self, body_id: int, spec: dict):
        pos = list(spec.get("base_position", [0.0, 0.0, 0.0]))
        orn = list(spec.get("base_orientation", [0.0, 0.0, 0.0, 1.0]))
        self._initial_base_states[body_id] = (pos, orn)
        n = self._client.getNumJoints(body_id)
        self._initial_joint_states[body_id] = [
            (self._client.getJointState(body_id, i)[0],
             self._client.getJointState(body_id, i)[1])
            for i in range(n)
        ]

    def _store_original_dynamics(self, name: str, body_id: int):
        links = [-1] + list(range(self._client.getNumJoints(body_id)))
        dyn: dict[int, dict[str, float]] = {}
        for link_id in links:
            info = self._client.getDynamicsInfo(body_id, link_id)
            dyn[link_id] = {"mass": info[0], "lateral_friction": info[1]}
        self._original_dynamics[name] = dyn

    # ------------------------------------------------------------------
    # Physics state reset (subsequent resets)
    # ------------------------------------------------------------------

    def _restore_physics_state(self):
        for body_id, (pos, orn) in self._initial_base_states.items():
            self._client.resetBasePositionAndOrientation(body_id, pos, orn)
            self._client.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
        for body_id, joint_states in self._initial_joint_states.items():
            for joint_idx, (pos, vel) in enumerate(joint_states):
                self._client.resetJointState(body_id, joint_idx, pos, vel)
        # Restore original dynamics so DR starts from a clean slate each episode
        self._apply_dynamics(self._original_dynamics)

    def _apply_dynamics(self, dynamics: dict[str, dict[int, dict[str, float]]]):
        for name, link_dyn in dynamics.items():
            body_id = self._body_ids.get(name)
            if body_id is None:
                continue
            for link_id, props in link_dyn.items():
                self._client.changeDynamics(
                    body_id, link_id,
                    mass=props["mass"],
                    lateralFriction=props["lateral_friction"],
                )

    # ------------------------------------------------------------------
    # Build SimulatedWorld (first reset only)
    # ------------------------------------------------------------------

    def _build_world(self) -> SimulatedWorld:
        world = SimulatedWorld()

        for robot_spec in self._asset_cfg.get("robots", []):
            robot_name = robot_spec["name"]
            body_id = self._body_ids[robot_name]

            group_drivers: dict[str, SimulatedJointGroupDriver] = {}
            for group_spec in robot_spec.get("joint_groups", []):
                group_name = group_spec["name"]
                joint_names = group_spec["joint_names"]
                mode_str = group_spec.get("control_mode", "JOINT_POSITION")
                control_mode = ControllerMode[mode_str]
                group_drivers[group_name] = PybulletJointGroupDriver(
                    self._client, body_id, joint_names, control_mode
                )

            robot_sensor_drivers: dict[str, SimulatedSensorDriver] = {}
            for sensor_spec in robot_spec.get("sensors", []):
                sensor_name = sensor_spec["name"]
                sensor_type = sensor_spec["type"].lower()
                if sensor_type == "ft":
                    robot_sensor_drivers[sensor_name] = PybulletFTSensorDriver(
                        self._client, body_id, sensor_spec["joint_name"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported robot sensor type: {sensor_type!r}"
                    )

            world.robot_drivers[robot_name] = SimulatedRobotDriver(
                robot_name, group_drivers, robot_sensor_drivers
            )

        pose_space = RigidTransform()
        for obj_spec in self._asset_cfg.get("objects", []):
            if obj_spec.get("track_pose", False):
                obj_name = obj_spec["name"]
                body_id = self._body_ids[obj_name]
                world.object_pose_getters[obj_name] = self._make_pose_getter(body_id)

        for sensor_spec in self._asset_cfg.get("sensors", []):
            sensor_name = sensor_spec["name"]
            sensor_type = sensor_spec["type"].lower()
            h = sensor_spec.get("height", 480)
            w = sensor_spec.get("width", 640)
            cam_kwargs = sensor_spec.get("camera_kwargs", {})
            if sensor_type == "rgb":
                world.sensor_drivers[sensor_name] = PybulletRGBSensorDriver(
                    self._client, h, w, **cam_kwargs
                )
            elif sensor_type == "depth":
                world.sensor_drivers[sensor_name] = PybulletDepthSensorDriver(
                    self._client, h, w, **cam_kwargs
                )
            elif sensor_type == "rgbd":
                world.sensor_drivers[sensor_name] = PybulletRGBDSensorDriver(
                    self._client, h, w, **cam_kwargs
                )
            else:
                raise ValueError(f"Unsupported standalone sensor type: {sensor_type!r}")

        return world

    def _make_pose_getter(self, body_id: int) -> Callable[[], dict[str, np.ndarray]]:
        def _get_pose() -> dict[str, np.ndarray]:
            pos, orn = self._client.getBasePositionAndOrientation(body_id)
            # PyBullet returns xyzw; Rotation space uses scipy xyzw convention
            return {
                "translation": np.array(pos, dtype=np.float32),
                "rotation": np.array(orn, dtype=np.float32),
            }
        return _get_pose

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def domain_randomize(self, rng: np.random.Generator) -> None:
        if not self._dr_cfg:
            return

        if "gravity" in self._dr_cfg:
            grav_cfg = self._dr_cfg["gravity"]
            gx = self._gravity.get("x", 0.0)
            gy = self._gravity.get("y", 0.0)
            gz = self._gravity.get("z", -9.81)
            if "z_range" in grav_cfg:
                gz = float(rng.uniform(*grav_cfg["z_range"]))
            self._client.setGravity(gx, gy, gz)

        for body_cfg in self._dr_cfg.get("bodies", []):
            body_name = body_cfg["name"]
            body_id = self._body_ids.get(body_name)
            if body_id is None:
                continue
            orig = self._original_dynamics.get(body_name, {})
            links: list[int] = body_cfg.get("links", list(orig.keys()))
            for link_id in links:
                link_orig = orig.get(link_id, {})
                kwargs: dict[str, Any] = {}
                if "mass_scale_range" in body_cfg:
                    scale = float(rng.uniform(*body_cfg["mass_scale_range"]))
                    kwargs["mass"] = link_orig.get("mass", 1.0) * scale
                if "lateral_friction_range" in body_cfg:
                    kwargs["lateralFriction"] = float(
                        rng.uniform(*body_cfg["lateral_friction_range"])
                    )
                if kwargs:
                    self._client.changeDynamics(body_id, link_id, **kwargs)

    # ------------------------------------------------------------------
    # Core step / close
    # ------------------------------------------------------------------

    def step_simulator(self):
        self._client.stepSimulation()

    def close(self):
        self._client.disconnect()


# ---------------------------------------------------------------------------
# Joint group driver
# ---------------------------------------------------------------------------

class PybulletJointGroupDriver(SimulatedJointGroupDriver):

    def __init__(
        self,
        client: BulletClient,
        body_id: int,
        joint_names: list[str],
        control_mode: ControllerMode,
    ):
        super().__init__(joint_names, control_mode)
        self._client = client
        self._body_id = body_id
        self._joint_indices = _joint_indices_from_names(client, body_id, joint_names)

        if control_mode == ControllerMode.JOINT_TORQUE:
            # Disable default velocity motors so torque control is authoritative.
            # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12644
            self._client.setJointMotorControlArray(
                self._body_id,
                self._joint_indices,
                self._client.VELOCITY_CONTROL,
                forces=[0] * self.dof,
            )

    @cached_property
    def state_space(self) -> JointState:
        pos_lo, pos_hi, vel_hi, eff_hi = [], [], [], []
        for idx in self._joint_indices:
            info = self._client.getJointInfo(self._body_id, idx)
            lo, hi = _position_limits(info)
            pos_lo.append(lo)
            pos_hi.append(hi)
            vel_hi.append(_velocity_limit(info))
            eff_hi.append(_effort_limit(info))

        return JointState(
            joint_names=self.joint_names,
            position_limits=Limits(
                lower=np.array(pos_lo, dtype=np.float32),
                upper=np.array(pos_hi, dtype=np.float32),
            ),
            velocity_limits=Limits(
                lower=-np.array(vel_hi, dtype=np.float32),
                upper=np.array(vel_hi, dtype=np.float32),
            ),
            effort_limits=Limits(
                lower=-np.array(eff_hi, dtype=np.float32),
                upper=np.array(eff_hi, dtype=np.float32),
            ),
        )

    def is_ready(self) -> bool:
        return True

    def get_state(self) -> dict[str, np.ndarray]:
        position, velocity, effort = [], [], []
        for state in self._client.getJointStates(self._body_id, self._joint_indices):
            position.append(state[0])
            velocity.append(state[1])
            effort.append(state[3])
        return {
            "position": np.array(position, dtype=np.float32),
            "velocity": np.array(velocity, dtype=np.float32),
            "effort": np.array(effort, dtype=np.float32),
        }

    def set_target(self, target: np.ndarray):
        if self.control_mode == ControllerMode.JOINT_POSITION:
            self._client.setJointMotorControlArray(
                bodyUniqueId=self._body_id,
                jointIndices=self._joint_indices,
                controlMode=self._client.POSITION_CONTROL,
                targetPositions=target.tolist(),
            )
        elif self.control_mode == ControllerMode.JOINT_TORQUE:
            self._client.setJointMotorControlArray(
                bodyUniqueId=self._body_id,
                jointIndices=self._joint_indices,
                controlMode=self._client.TORQUE_CONTROL,
                forces=target.tolist(),
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")


# ---------------------------------------------------------------------------
# Sensor drivers
# ---------------------------------------------------------------------------

class PybulletSensorDriver(SimulatedSensorDriver):

    def __init__(self, client: BulletClient):
        self._client = client


class PybulletFTSensorDriver(PybulletSensorDriver):
    """Force-torque sensor reading from a joint's reaction forces."""

    def __init__(self, client: BulletClient, body_id: int, joint_name: str):
        super().__init__(client)
        self._body_id = body_id
        indices = _joint_indices_from_names(client, body_id, [joint_name])
        self._joint_index = indices[0]
        self._client.enableJointForceTorqueSensor(
            self._body_id, self._joint_index, enableSensor=1
        )

    @cached_property
    def state_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        joint_state = self._client.getJointState(self._body_id, self._joint_index)
        return np.array(joint_state[2], dtype=np.float32)  # (Fx, Fy, Fz, Mx, My, Mz)


class _PybulletCameraBase(PybulletSensorDriver):

    def __init__(self, client: BulletClient, height: int, width: int, **kwargs):
        super().__init__(client)
        self._height = height
        self._width = width
        self._kwargs = {"width": width, "height": height, **kwargs}

    def reset_view_matrix(self, view_matrix: list[float]):
        self._kwargs["viewMatrix"] = list(view_matrix)

    def _capture(self):
        w, h, rgb, depth, seg = self._client.getCameraImage(**self._kwargs)
        def _buf(buf, dtype, shape):
            if buf is None:
                return None
            arr = (
                np.frombuffer(buf, dtype=dtype)
                if isinstance(buf, (bytes, bytearray, memoryview))
                else np.asarray(buf, dtype=dtype)
            )
            return arr.reshape(shape)
        rgba = _buf(rgb, np.uint8, (h, w, 4))
        dep  = _buf(depth, np.float32, (h, w))
        return rgba, dep


class PybulletRGBSensorDriver(_PybulletCameraBase):

    @cached_property
    def state_space(self) -> RGBImage:
        return RGBImage(self._height, self._width, dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        rgba, _ = self._capture()
        if rgba is None:
            raise RuntimeError("PyBullet camera returned no RGB pixels.")
        return np.ascontiguousarray(rgba[:, :, :3])


class PybulletDepthSensorDriver(_PybulletCameraBase):

    @cached_property
    def state_space(self) -> DepthImage:
        return DepthImage(self._height, self._width, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        _, dep = self._capture()
        if dep is None:
            raise RuntimeError("PyBullet camera returned no depth pixels.")
        return dep


class PybulletRGBDSensorDriver(_PybulletCameraBase):

    @cached_property
    def state_space(self) -> RGBDImage:
        return RGBDImage(
            self._height, self._width,
            rgb_dtype=np.uint8, depth_dtype=np.float32,
        )

    def get_state(self) -> dict[str, np.ndarray]:
        rgba, dep = self._capture()
        if rgba is None:
            raise RuntimeError("PyBullet camera returned no RGB pixels.")
        if dep is None:
            raise RuntimeError("PyBullet camera returned no depth pixels.")
        return {
            "rgb": np.ascontiguousarray(rgba[:, :, :3]),
            "depth": dep,
        }
