from __future__ import annotations

import numpy as np
from typing import Any, Callable
from functools import cached_property
from gymnasium.spaces import Box

from ark.driver import ControllerMode
from ark.envs.spaces.sensor_space import JointState, Limits
from ark.envs.spaces.image_space import RGBImage, DepthImage, RGBDImage
from ark.simulator.base import Simulator, SimulatedWorld
from ark.simulator.driver import (
    SimulatedJointGroupDriver,
    SimulatedSensorDriver,
    SimulatedRobotDriver,
)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MujocoSimulator(Simulator):

    def _init_simulator(self):
        import mujoco  # lazy import so pybullet users don't need mujoco installed
        self._mujoco = mujoco
        mjcf_path: str = self._asset_cfg["mjcf"]
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._time_step_sec
        gx = self._gravity.get("x", 0.0)
        gy = self._gravity.get("y", 0.0)
        gz = self._gravity.get("z", -9.81)
        self._model.opt.gravity[:] = [gx, gy, gz]

        # Build world once; subsequent resets only reset data
        self._world = self._build_world()

        # Store originals for DR restore
        self._original_body_mass = self._model.body_mass.copy()
        self._original_dof_damping = self._model.dof_damping.copy()
        self._original_geom_friction = self._model.geom_friction.copy()
        self._original_gravity = self._model.opt.gravity.copy()

    def reset_simulator(self) -> SimulatedWorld:
        self._mujoco.mj_resetData(self._model, self._data)
        self._mujoco.mj_forward(self._model, self._data)
        return self._world

    def step_simulator(self):
        self._mujoco.mj_step(self._model, self._data)

    def close(self):
        # MuJoCo model/data are garbage-collected; renderers need explicit cleanup
        for driver in _flatten_sensor_drivers(self._world):
            if isinstance(driver, _MujocoCameraBase):
                driver.close()

    # ------------------------------------------------------------------
    # Build world (once at init)
    # ------------------------------------------------------------------

    def _build_world(self) -> SimulatedWorld:
        world = SimulatedWorld()

        for robot_spec in self._asset_cfg.get("robots", []):
            robot_name = robot_spec["name"]

            group_drivers: dict[str, SimulatedJointGroupDriver] = {}
            for group_spec in robot_spec.get("joint_groups", []):
                group_name = group_spec["name"]
                joint_names = group_spec["joint_names"]
                actuator_names = group_spec.get("actuator_names", [])
                mode_str = group_spec.get("control_mode", "JOINT_POSITION")
                control_mode = ControllerMode[mode_str]
                group_drivers[group_name] = MujocoJointGroupDriver(
                    self._model, self._data,
                    joint_names, actuator_names, control_mode,
                )

            robot_sensor_drivers: dict[str, SimulatedSensorDriver] = {}
            for sensor_spec in robot_spec.get("sensors", []):
                sensor_name = sensor_spec["name"]
                sensor_type = sensor_spec["type"].lower()
                if sensor_type == "ft":
                    robot_sensor_drivers[sensor_name] = MujocoFTSensorDriver(
                        self._model, self._data, sensor_spec["sensor_names"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported robot sensor type: {sensor_type!r}"
                    )

            world.robot_drivers[robot_name] = SimulatedRobotDriver(
                robot_name, group_drivers, robot_sensor_drivers
            )

        for obj_name in self._asset_cfg.get("objects", []):
            world.object_pose_getters[obj_name] = self._make_pose_getter(obj_name)

        for sensor_spec in self._asset_cfg.get("sensors", []):
            sensor_name = sensor_spec["name"]
            sensor_type = sensor_spec["type"].lower()
            h = sensor_spec.get("height", 480)
            w = sensor_spec.get("width", 640)
            cam_name = sensor_spec.get("camera_name", "")
            if sensor_type == "rgb":
                world.sensor_drivers[sensor_name] = MujocoRGBSensorDriver(
                    self._model, self._data, cam_name, h, w
                )
            elif sensor_type == "depth":
                world.sensor_drivers[sensor_name] = MujocoDepthSensorDriver(
                    self._model, self._data, cam_name, h, w
                )
            elif sensor_type == "rgbd":
                world.sensor_drivers[sensor_name] = MujocoRGBDSensorDriver(
                    self._model, self._data, cam_name, h, w
                )
            else:
                raise ValueError(
                    f"Unsupported standalone sensor type: {sensor_type!r}"
                )

        return world

    def _make_pose_getter(
        self, body_name: str
    ) -> Callable[[], dict[str, np.ndarray]]:
        body_id = self._model.body(body_name).id

        def _get_pose() -> dict[str, np.ndarray]:
            pos = self._data.xpos[body_id].astype(np.float32)
            # MuJoCo xquat is (w, x, y, z); scipy/Rotation convention is (x, y, z, w)
            q = self._data.xquat[body_id]
            orn = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
            return {"translation": pos.copy(), "rotation": orn}

        return _get_pose

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def domain_randomize(self, rng: np.random.Generator) -> None:
        if not self._dr_cfg:
            return

        # Restore nominal values first so perturbations don't accumulate
        self._model.body_mass[:] = self._original_body_mass
        self._model.dof_damping[:] = self._original_dof_damping
        self._model.geom_friction[:] = self._original_geom_friction
        self._model.opt.gravity[:] = self._original_gravity

        if "gravity" in self._dr_cfg:
            grav_cfg = self._dr_cfg["gravity"]
            gz = float(self._original_gravity[2])
            if "z_range" in grav_cfg:
                gz = float(rng.uniform(*grav_cfg["z_range"]))
            self._model.opt.gravity[2] = gz

        for body_cfg in self._dr_cfg.get("bodies", []):
            body_id = self._model.body(body_cfg["name"]).id
            if "mass_scale_range" in body_cfg:
                scale = float(rng.uniform(*body_cfg["mass_scale_range"]))
                self._model.body_mass[body_id] *= scale

        for dof_cfg in self._dr_cfg.get("dofs", []):
            jnt_id = self._model.joint(dof_cfg["name"]).id
            dof_adr = self._model.jnt_dofadr[jnt_id]
            if "damping_range" in dof_cfg:
                self._model.dof_damping[dof_adr] = float(
                    rng.uniform(*dof_cfg["damping_range"])
                )

        for geom_cfg in self._dr_cfg.get("geoms", []):
            geom_id = self._model.geom(geom_cfg["name"]).id
            if "friction_scale_range" in geom_cfg:
                scale = float(rng.uniform(*geom_cfg["friction_scale_range"]))
                self._model.geom_friction[geom_id] *= scale

        # Propagate any inertial changes through the kinematic tree
        self._mujoco.mj_setConst(self._model, self._data)


def _flatten_sensor_drivers(world: SimulatedWorld):
    """Yield all sensor drivers from a world for cleanup."""
    yield from world.sensor_drivers.values()
    for robot in world.robot_drivers.values():
        for name in robot.sensor_names:
            yield robot.sensor_driver(name)


# ---------------------------------------------------------------------------
# Joint group driver
# ---------------------------------------------------------------------------

class MujocoJointGroupDriver(SimulatedJointGroupDriver):
    """Controls a named set of 1-DoF joints (hinge or slide) via actuators."""

    def __init__(
        self,
        model,
        data,
        joint_names: list[str],
        actuator_names: list[str],
        control_mode: ControllerMode,
    ):
        super().__init__(joint_names, control_mode)
        self._model = model
        self._data = data

        self._qpos_addrs = [
            int(model.jnt_qposadr[model.joint(n).id]) for n in joint_names
        ]
        self._dof_addrs = [
            int(model.jnt_dofadr[model.joint(n).id]) for n in joint_names
        ]
        self._actuator_ids = [model.actuator(n).id for n in actuator_names]

    @cached_property
    def state_space(self) -> JointState:
        import mujoco as mj
        pos_lo, pos_hi, vel_hi, eff_hi = [], [], [], []
        for name in self.joint_names:
            jnt_id = self._model.joint(name).id
            limited = bool(self._model.jnt_limited[jnt_id])
            if limited:
                lo, hi = self._model.jnt_range[jnt_id]
            else:
                lo, hi = -np.inf, np.inf
            pos_lo.append(float(lo))
            pos_hi.append(float(hi))
            vel_hi.append(100.0)  # MuJoCo doesn't store per-joint vel limit
            eff_hi.append(np.inf)

        # Override effort limits from actuator force ranges where available
        for i, act_id in enumerate(self._actuator_ids):
            if act_id >= 0 and bool(self._model.actuator_forcelimited[act_id]):
                lo, hi = self._model.actuator_forcerange[act_id]
                eff_hi[i] = float(hi)

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
        pos = np.array([self._data.qpos[a] for a in self._qpos_addrs], dtype=np.float32)
        vel = np.array([self._data.qvel[a] for a in self._dof_addrs], dtype=np.float32)
        eff = np.array([self._data.qfrc_actuator[a] for a in self._dof_addrs], dtype=np.float32)
        return {"position": pos, "velocity": vel, "effort": eff}

    def set_target(self, target: np.ndarray):
        if not self._actuator_ids:
            return
        if self.control_mode in (ControllerMode.JOINT_POSITION, ControllerMode.JOINT_TORQUE):
            for act_id, val in zip(self._actuator_ids, target):
                self._data.ctrl[act_id] = float(val)
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")


# ---------------------------------------------------------------------------
# FT sensor driver
# ---------------------------------------------------------------------------

class MujocoFTSensorDriver(SimulatedSensorDriver):
    """Reads one or more named MuJoCo sensors and concatenates their outputs."""

    def __init__(self, model, data, sensor_names: list[str]):
        self._model = model
        self._data = data
        self._sensor_addrs: list[tuple[int, int]] = []  # (adr, dim) per sensor
        total_dim = 0
        for name in sensor_names:
            sid = model.sensor(name).id
            adr = int(model.sensor_adr[sid])
            dim = int(model.sensor_dim[sid])
            self._sensor_addrs.append((adr, dim))
            total_dim += dim
        self._total_dim = total_dim

    @cached_property
    def state_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(self._total_dim,), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        chunks = [
            self._data.sensordata[adr:adr + dim]
            for adr, dim in self._sensor_addrs
        ]
        return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# Camera sensor drivers
# ---------------------------------------------------------------------------

class _MujocoCameraBase(SimulatedSensorDriver):

    def __init__(self, model, data, camera_name: str, height: int, width: int):
        import mujoco
        self._model = model
        self._data = data
        self._camera_name = camera_name
        self._height = height
        self._width = width
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def _render_rgb(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self._camera_name)
        return self._renderer.render().copy()  # (H, W, 3) uint8

    def _render_depth(self) -> np.ndarray:
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._data, camera=self._camera_name)
        depth = self._renderer.render().copy()  # (H, W) float32
        self._renderer.disable_depth_rendering()
        return depth

    def close(self):
        self._renderer.close()


class MujocoRGBSensorDriver(_MujocoCameraBase):

    @cached_property
    def state_space(self) -> RGBImage:
        return RGBImage(self._height, self._width, dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        return self._render_rgb()


class MujocoDepthSensorDriver(_MujocoCameraBase):

    @cached_property
    def state_space(self) -> DepthImage:
        return DepthImage(self._height, self._width, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return self._render_depth()


class MujocoRGBDSensorDriver(_MujocoCameraBase):

    @cached_property
    def state_space(self) -> RGBDImage:
        return RGBDImage(
            self._height, self._width,
            rgb_dtype=np.uint8, depth_dtype=np.float32,
        )

    def get_state(self) -> dict[str, np.ndarray]:
        rgb = self._render_rgb()
        depth = self._render_depth()
        return {"rgb": rgb, "depth": depth}
