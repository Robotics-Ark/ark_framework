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
# Helpers
# ---------------------------------------------------------------------------

def _label_index(labels, name: str) -> int:
    """Find index of name in a Newton label list/array."""
    for i, label in enumerate(labels):
        if str(label) == name:
            return i
    raise ValueError(f"Name {name!r} not found. Available: {[str(l) for l in labels]}")


def _body_q_to_pose(body_q_np: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation_xyzw) from a Newton body_q numpy array.

    warp stores wp.transform as 7 contiguous float32 values per element:
        [px, py, pz, qx, qy, qz, qw]   (position then quaternion, scalar-last)
    """
    flat = body_q_np.view(np.float32).reshape(body_q_np.shape[0], 7)
    pos = flat[idx, :3].copy().astype(np.float32)
    rot = flat[idx, 3:].copy().astype(np.float32)  # xyzw — matches scipy convention
    return pos, rot


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class NewtonSimulator(Simulator):

    def _init_simulator(self):
        import warp as wp
        import newton
        import newton.solvers
        self._wp = wp
        self._newton = newton
        self._newton_solvers = newton.solvers

        device = str(self._sim_cfg.get("device", "cpu"))
        wp.init()

        # Build model from asset config
        builder = newton.ModelBuilder()
        self._populate_builder(builder)
        self._model = builder.finalize(requires_grad=False)

        # Override gravity from our config (builder may set defaults)
        gx = self._gravity.get("x", 0.0)
        gy = self._gravity.get("y", 0.0)
        gz = self._gravity.get("z", -9.81)
        self._model.gravity[0] = wp.vec3(gx, gy, gz)

        # Select solver
        solver_name = str(self._sim_cfg.get("solver", "MuJoCo"))
        solver_cls = getattr(self._newton_solvers, f"Solver{solver_name}", None)
        if solver_cls is None:
            raise ValueError(
                f"Unknown Newton solver {solver_name!r}. "
                f"Options: MuJoCo, XPBD, Featherstone, SemiImplicit"
            )
        self._solver = solver_cls(self._model)

        # Two alternating state buffers (Newton pattern)
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()

        # Compute forward kinematics so initial state has valid body poses
        newton.eval_fk(
            self._model,
            self._state_0.joint_q,
            self._state_0.joint_qd,
            self._state_0,
        )

        # Snapshot initial state for reset
        self._initial_state = self._model.state()
        self._initial_state.assign(self._state_0)

        # Store original physics parameters for DR restore
        self._original_body_mass = self._model.body_mass.numpy().copy()
        self._original_body_inv_mass = self._model.body_inv_mass.numpy().copy()
        self._original_joint_kd = self._model.joint_target_kd.numpy().copy()
        self._original_shape_mu = self._model.shape_mu.numpy().copy()
        self._original_gravity = np.array([gx, gy, gz], dtype=np.float32)

        # Build world (drivers hold callables into self._state_0 / self._control)
        self._world = self._build_world()

    def _populate_builder(self, builder):
        import warp as wp

        gx = self._gravity.get("x", 0.0)
        gy = self._gravity.get("y", 0.0)
        gz = self._gravity.get("z", -9.81)

        if self._asset_cfg.get("ground_plane", True):
            builder.add_ground_plane()

        for robot_spec in self._asset_cfg.get("robots", []):
            pos = robot_spec.get("base_position", [0.0, 0.0, 0.0])
            orn = robot_spec.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
            xform = wp.transform(
                p=wp.vec3(*pos),
                q=wp.quat(*orn),  # xyzw
            )
            floating = bool(robot_spec.get("floating", False))
            collapse = bool(robot_spec.get("collapse_fixed_joints", True))

            if "urdf" in robot_spec:
                builder.add_urdf(
                    source=robot_spec["urdf"],
                    xform=xform,
                    floating=floating,
                    collapse_fixed_joints=collapse,
                    enable_self_collisions=False,
                )
            elif "mjcf" in robot_spec:
                builder.add_mjcf(
                    source=robot_spec["mjcf"],
                    xform=xform,
                    floating=floating,
                    collapse_fixed_joints=collapse,
                )
            else:
                raise ValueError(
                    f"Robot '{robot_spec.get('name')}' must specify 'urdf' or 'mjcf'."
                )

        for obj_spec in self._asset_cfg.get("objects", []):
            pos = obj_spec.get("base_position", [0.0, 0.0, 0.0])
            orn = obj_spec.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
            xform = wp.transform(p=wp.vec3(*pos), q=wp.quat(*orn))
            if "urdf" in obj_spec:
                builder.add_urdf(source=obj_spec["urdf"], xform=xform)
            elif "mjcf" in obj_spec:
                builder.add_mjcf(source=obj_spec["mjcf"], xform=xform)

    # ------------------------------------------------------------------
    # Build world (once at init)
    # ------------------------------------------------------------------

    def _build_world(self) -> SimulatedWorld:
        world = SimulatedWorld()
        joint_labels = list(self._model.joint_label)
        body_labels = list(self._model.body_label)
        shape_labels = list(self._model.shape_label)

        # Current-state accessor: always returns self._state_0 (the active state
        # after each step/swap), so drivers never hold a stale reference.
        get_state = lambda: self._state_0
        get_control = lambda: self._control

        for robot_spec in self._asset_cfg.get("robots", []):
            robot_name = robot_spec["name"]

            group_drivers: dict[str, SimulatedJointGroupDriver] = {}
            for group_spec in robot_spec.get("joint_groups", []):
                group_name = group_spec["name"]
                joint_names = group_spec["joint_names"]
                mode_str = group_spec.get("control_mode", "JOINT_POSITION")
                control_mode = ControllerMode[mode_str]

                q_addrs, qd_addrs = [], []
                for jname in joint_names:
                    jidx = _label_index(joint_labels, jname)
                    q_addrs.append(int(self._model.joint_q_start[jidx]))
                    qd_addrs.append(int(self._model.joint_qd_start[jidx]))

                group_drivers[group_name] = NewtonJointGroupDriver(
                    self._model, get_state, get_control,
                    joint_names, q_addrs, qd_addrs, control_mode,
                )

            world.robot_drivers[robot_name] = SimulatedRobotDriver(
                robot_name, group_drivers, {}
            )

        for obj_spec in self._asset_cfg.get("objects", []):
            if obj_spec.get("track_pose", False):
                obj_name = obj_spec["name"]
                body_idx = _label_index(body_labels, obj_name)
                world.object_pose_getters[obj_name] = self._make_pose_getter(body_idx)

        for sensor_spec in self._asset_cfg.get("sensors", []):
            sensor_name = sensor_spec["name"]
            sensor_type = sensor_spec["type"].lower()
            if sensor_type == "contact":
                pattern = sensor_spec.get("shape_pattern", "*")
                world.sensor_drivers[sensor_name] = NewtonContactSensorDriver(
                    self._model, get_state, lambda: self._contacts, pattern
                )
            elif sensor_type in ("rgb", "depth", "rgbd"):
                cam_body = sensor_spec.get("camera_name", "")
                h = sensor_spec.get("height", 480)
                w = sensor_spec.get("width", 640)
                if sensor_type == "rgb":
                    world.sensor_drivers[sensor_name] = NewtonRGBSensorDriver(
                        self._model, get_state, cam_body, h, w
                    )
                elif sensor_type == "depth":
                    world.sensor_drivers[sensor_name] = NewtonDepthSensorDriver(
                        self._model, get_state, cam_body, h, w
                    )
                else:
                    world.sensor_drivers[sensor_name] = NewtonRGBDSensorDriver(
                        self._model, get_state, cam_body, h, w
                    )
            else:
                raise ValueError(f"Unsupported Newton sensor type: {sensor_type!r}")

        return world

    def _make_pose_getter(
        self, body_idx: int
    ) -> Callable[[], dict[str, np.ndarray]]:
        def _get_pose() -> dict[str, np.ndarray]:
            body_q_np = self._state_0.body_q.numpy()
            pos, rot = _body_q_to_pose(body_q_np, body_idx)
            return {"translation": pos, "rotation": rot}
        return _get_pose

    # ------------------------------------------------------------------
    # Core sim API
    # ------------------------------------------------------------------

    def reset_simulator(self) -> SimulatedWorld:
        self._state_0.assign(self._initial_state)
        self._state_1.assign(self._initial_state)
        self._newton.eval_fk(
            self._model,
            self._state_0.joint_q,
            self._state_0.joint_qd,
            self._state_0,
        )
        return self._world

    def step_simulator(self):
        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(
            self._state_0, self._state_1,
            self._control, self._contacts,
            self._time_step_sec,
        )
        # Swap: state_0 is always the current (post-step) state
        self._state_0, self._state_1 = self._state_1, self._state_0

    def close(self):
        pass  # warp resources are garbage-collected

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def domain_randomize(self, rng: np.random.Generator) -> None:
        if not self._dr_cfg:
            return

        import warp as wp

        # Restore nominal values before sampling new perturbations
        self._model.body_mass.assign(self._original_body_mass)
        self._model.body_inv_mass.assign(self._original_body_inv_mass)
        self._model.joint_target_kd.assign(self._original_joint_kd)
        self._model.shape_mu.assign(self._original_shape_mu)
        self._model.gravity[0] = wp.vec3(*self._original_gravity.tolist())

        body_labels = list(self._model.body_label)
        joint_labels = list(self._model.joint_label)
        shape_labels = list(self._model.shape_label)

        if "gravity" in self._dr_cfg:
            gz = float(self._original_gravity[2])
            if "z_range" in self._dr_cfg["gravity"]:
                gz = float(rng.uniform(*self._dr_cfg["gravity"]["z_range"]))
            gx, gy = float(self._original_gravity[0]), float(self._original_gravity[1])
            self._model.gravity[0] = wp.vec3(gx, gy, gz)

        body_mass_np = self._model.body_mass.numpy().copy()
        body_inv_mass_np = self._model.body_inv_mass.numpy().copy()
        for body_cfg in self._dr_cfg.get("bodies", []):
            bidx = _label_index(body_labels, body_cfg["name"])
            if "mass_scale_range" in body_cfg:
                scale = float(rng.uniform(*body_cfg["mass_scale_range"]))
                orig_mass = float(self._original_body_mass[bidx])
                new_mass = orig_mass * scale
                body_mass_np[bidx] = new_mass
                body_inv_mass_np[bidx] = 1.0 / new_mass if new_mass > 0.0 else 0.0
        self._model.body_mass.assign(body_mass_np)
        self._model.body_inv_mass.assign(body_inv_mass_np)

        kd_np = self._model.joint_target_kd.numpy().copy()
        for joint_cfg in self._dr_cfg.get("joints", []):
            jidx = _label_index(joint_labels, joint_cfg["name"])
            if "damping_range" in joint_cfg:
                kd_np[jidx] = float(rng.uniform(*joint_cfg["damping_range"]))
        self._model.joint_target_kd.assign(kd_np)

        mu_np = self._model.shape_mu.numpy().copy()
        for shape_cfg in self._dr_cfg.get("shapes", []):
            sidx = _label_index(shape_labels, shape_cfg["name"])
            if "friction_range" in shape_cfg:
                mu_np[sidx] = float(rng.uniform(*shape_cfg["friction_range"]))
        self._model.shape_mu.assign(mu_np)


# ---------------------------------------------------------------------------
# Joint group driver
# ---------------------------------------------------------------------------

class NewtonJointGroupDriver(SimulatedJointGroupDriver):
    """Controls 1-DoF joints (revolute or prismatic) in a Newton model.

    State is always read from the *current* state buffer (state_0) via
    get_state(), which the simulator updates after each step/swap.
    """

    def __init__(
        self,
        model,
        get_state: Callable,
        get_control: Callable,
        joint_names: list[str],
        q_addrs: list[int],
        qd_addrs: list[int],
        control_mode: ControllerMode,
    ):
        super().__init__(joint_names, control_mode)
        self._model = model
        self._get_state = get_state
        self._get_control = get_control
        self._q_addrs = q_addrs
        self._qd_addrs = qd_addrs

    @cached_property
    def state_space(self) -> JointState:
        joint_labels = list(self._model.joint_label)
        pos_lo, pos_hi = [], []
        for name in self.joint_names:
            jidx = _label_index(joint_labels, name)
            lo = float(self._model.joint_limit_lower[jidx])
            hi = float(self._model.joint_limit_upper[jidx])
            # Newton uses 0/0 when limits are not set
            if lo == 0.0 and hi == 0.0:
                lo, hi = -np.inf, np.inf
            pos_lo.append(lo)
            pos_hi.append(hi)

        dof = len(self.joint_names)
        return JointState(
            joint_names=self.joint_names,
            position_limits=Limits(
                lower=np.array(pos_lo, dtype=np.float32),
                upper=np.array(pos_hi, dtype=np.float32),
            ),
            velocity_limits=Limits(
                lower=np.full(dof, -100.0, dtype=np.float32),
                upper=np.full(dof,  100.0, dtype=np.float32),
            ),
            effort_limits=Limits(
                lower=np.full(dof, -np.inf, dtype=np.float32),
                upper=np.full(dof,  np.inf, dtype=np.float32),
            ),
        )

    def is_ready(self) -> bool:
        return True

    def get_state(self) -> dict[str, np.ndarray]:
        state = self._get_state()
        q_np = state.joint_q.numpy()
        qd_np = state.joint_qd.numpy()
        pos = np.array([q_np[a]  for a in self._q_addrs],  dtype=np.float32)
        vel = np.array([qd_np[a] for a in self._qd_addrs], dtype=np.float32)
        # Newton does not expose per-joint actuator effort directly;
        # return zeros here. Override if your solver exposes joint forces.
        eff = np.zeros(len(self.joint_names), dtype=np.float32)
        return {"position": pos, "velocity": vel, "effort": eff}

    def set_target(self, target: np.ndarray):
        control = self._get_control()
        if self.control_mode == ControllerMode.JOINT_POSITION:
            ctrl_np = control.joint_target_q.numpy().copy()
            for addr, val in zip(self._q_addrs, target):
                ctrl_np[addr] = float(val)
            control.joint_target_q.assign(ctrl_np)
        elif self.control_mode == ControllerMode.JOINT_TORQUE:
            ctrl_np = control.joint_f.numpy().copy()
            for addr, val in zip(self._qd_addrs, target):
                ctrl_np[addr] = float(val)
            control.joint_f.assign(ctrl_np)
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")


# ---------------------------------------------------------------------------
# Sensor drivers
# ---------------------------------------------------------------------------

class NewtonContactSensorDriver(SimulatedSensorDriver):
    """Reports total contact force (Fx, Fy, Fz) on a set of shapes.

    Uses Newton's SensorContact; ``shape_pattern`` is passed to
    ``sensing_shapes`` (supports glob-style wildcards, e.g. ``"*foot*"``).
    """

    def __init__(self, model, get_state: Callable, get_contacts: Callable, shape_pattern: str):
        from newton.sensors import SensorContact
        self._sensor = SensorContact(
            model, sensing_shapes=shape_pattern, measure_total=True
        )
        self._get_state = get_state
        self._get_contacts = get_contacts

    @cached_property
    def state_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        self._sensor.update(self._get_state(), self._get_contacts())
        force = self._sensor.total_force.numpy()  # (sensing_count, 3)
        return force.sum(axis=0).astype(np.float32)  # sum over all sensing shapes


class _NewtonCameraBase(SimulatedSensorDriver):

    def __init__(self, model, get_state: Callable, camera_name: str, height: int, width: int):
        from newton.sensors import SensorTiledCamera
        self._get_state = get_state
        self._height = height
        self._width = width
        self._camera = SensorTiledCamera(model=model, width=width, height=height)
        self._camera_name = camera_name

    def _render(self) -> np.ndarray:
        self._camera.render(self._get_state())
        return self._camera.image.numpy()  # (H, W, C) or (H, W) depending on mode


class NewtonRGBSensorDriver(_NewtonCameraBase):

    @cached_property
    def state_space(self) -> RGBImage:
        return RGBImage(self._height, self._width, dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        img = self._render()
        return img[:, :, :3].astype(np.uint8)


class NewtonDepthSensorDriver(_NewtonCameraBase):

    @cached_property
    def state_space(self) -> DepthImage:
        return DepthImage(self._height, self._width, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return self._render().astype(np.float32)


class NewtonRGBDSensorDriver(_NewtonCameraBase):

    @cached_property
    def state_space(self) -> RGBDImage:
        return RGBDImage(
            self._height, self._width,
            rgb_dtype=np.uint8, depth_dtype=np.float32,
        )

    def get_state(self) -> dict[str, np.ndarray]:
        # SensorTiledCamera renders once; split channels
        img = self._render()
        return {
            "rgb": img[:, :, :3].astype(np.uint8),
            "depth": img[:, :, 3].astype(np.float32) if img.shape[2] > 3
                     else np.zeros((self._height, self._width), dtype=np.float32),
        }
