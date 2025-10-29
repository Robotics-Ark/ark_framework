"""Newton robot driver bridging ARK control with the Newton simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import newton
import numpy as np
import warp as wp

from ark.tools.log import log
from ark.system.driver.robot_driver import ControlType, SimRobotDriver


def _as_quaternion(values: Sequence[float]) -> wp.quat:
    if len(values) == 4:
        return wp.quat(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    if len(values) == 3:
        return wp.quat_from_euler(wp.vec3(float(values[0]), float(values[1]), float(values[2])), 0, 1, 2)
    return wp.quat_identity()


def _as_transform(position: Sequence[float], orientation: Sequence[float]) -> wp.transform:
    pos = wp.vec3(float(position[0]), float(position[1]), float(position[2])) if len(position) >= 3 else wp.vec3()
    quat = _as_quaternion(orientation)
    return wp.transform(pos, quat)


class NewtonRobotDriver(SimRobotDriver):
    """Driver that exposes Newton articulation controls to ARK."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        builder: newton.ModelBuilder,
    ) -> None:
        super().__init__(component_name, component_config, True)

        self.builder = builder
        self._model: newton.Model | None = None
        self._control: newton.Control | None = None
        self._state_accessor: Callable[[], newton.State] = lambda: None
        self._dt: float = 0.0

        self._joint_names: list[str] = []
        self._joint_index_map: dict[str, int] = {}
        self._body_names: list[str] = []

        self._joint_q_start: np.ndarray | None = None
        self._joint_qd_start: np.ndarray | None = None
        self._joint_dof_dim: np.ndarray | None = None

        self._last_commanded_torque: dict[str, float] = {}
        self._torque_groups = {
            name
            for name, cfg in self.config.get("joint_groups", {}).items()
            if cfg.get("control_mode") == ControlType.TORQUE.value
        }

        self.base_position = self.config.get("base_position", [0.0, 0.0, 0.0])
        self.base_orientation = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        self.initial_configuration = list(self.config.get("initial_configuration", []))

        self._load_into_builder()

    def _resolve_path(self, key: str) -> Path:
        raw = self.config.get(key)
        if raw is None:
            raise ValueError(f"Newton robot driver requires '{key}' in config.")
        path = Path(raw)
        class_dir = Path(self.config.get("class_dir", ""))
        if class_dir.is_file():
            class_dir = class_dir.parent
        if not path.is_absolute():
            path = class_dir / path
        return path

    def _load_into_builder(self) -> None:
        pre_joint_count = self.builder.joint_count
        pre_body_count = self.builder.body_count

        urdf_path = self._resolve_path("urdf_path")

        # Validate URDF file exists
        if not urdf_path.exists():
            log.error(f"Newton robot driver: URDF file not found at '{urdf_path}'")
            log.error(f"  Full path: {urdf_path.resolve()}")
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        xform = _as_transform(self.base_position, self.base_orientation)
        floating = bool(self.config.get("floating", False))
        enable_self_collisions = bool(self.config.get("enable_self_collisions", False))
        collapse_fixed = bool(self.config.get("collapse_fixed_joints", False))

        try:
            self.builder.add_urdf(
                str(urdf_path),
                xform=xform,
                floating=floating,
                enable_self_collisions=enable_self_collisions,
                collapse_fixed_joints=collapse_fixed,
            )
            log.ok(f"Newton robot driver: Loaded URDF '{urdf_path.name}' successfully")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Newton robot driver: Failed to load URDF '{urdf_path}': {exc}")
            raise

        self._joint_names = list(
            self.builder.joint_key[pre_joint_count : self.builder.joint_count]
        )
        self._body_names = list(
            self.builder.body_key[pre_body_count : self.builder.body_count]
        )

    def bind_runtime(
        self,
        model: newton.Model,
        control: newton.Control,
        state_accessor: Callable[[], newton.State],
        dt: float,
    ) -> None:
        self._model = model
        self._control = control
        self._state_accessor = state_accessor
        self._dt = dt

        key_to_index = {name: idx for idx, name in enumerate(model.joint_key)}
        missing_joints = []
        for name in self._joint_names:
            if name in key_to_index:
                self._joint_index_map[name] = key_to_index[name]
            else:
                missing_joints.append(name)

        # Warn about joint mapping issues
        if missing_joints:
            log.warning(
                f"Newton robot driver '{self.component_name}': "
                f"{len(missing_joints)} joint(s) from URDF not found in model: {missing_joints}"
            )

        log.info(
            f"Newton robot driver '{self.component_name}': "
            f"Mapped {len(self._joint_index_map)}/{len(self._joint_names)} joints to model"
        )

        self._joint_q_start = model.joint_q_start.numpy()
        self._joint_qd_start = model.joint_qd_start.numpy()
        self._joint_dof_dim = model.joint_dof_dim.numpy()

        self._apply_initial_configuration()

    def _apply_initial_configuration(self) -> None:
        """Apply initial joint configuration to model and control."""
        if not self.initial_configuration or not self._joint_names:
            log.info(f"Newton robot driver '{self.component_name}': No initial configuration to apply")
            return
        if not isinstance(self.initial_configuration, (list, tuple)):
            log.warning(
                f"Newton robot driver '{self.component_name}': "
                f"initial_configuration is not a list/tuple, skipping"
            )
            return

        # Validate initial configuration length
        if len(self.initial_configuration) != len(self._joint_names):
            log.warning(
                f"Newton robot driver '{self.component_name}': "
                f"initial_configuration length ({len(self.initial_configuration)}) != "
                f"joint count ({len(self._joint_names)}). Using available values."
            )

        # Set both joint positions and targets
        for idx, joint_name in enumerate(self._joint_names):
            if idx >= len(self.initial_configuration):
                break
            target = self.initial_configuration[idx]
            self._write_joint_position(joint_name, target)
            self._write_joint_target(joint_name, target)

        # Synchronize body poses with updated joint positions
        state = self._state_accessor()
        if state is not None and self._model is not None:
            try:
                # Get the current state arrays
                joint_q_np = self._model.joint_q.numpy()
                joint_qd_np = self._model.joint_qd.numpy()
                # Evaluate forward kinematics to update body poses
                newton.eval_fk(
                    self._model,
                    wp.array(joint_q_np, dtype=wp.float32, device=self._model.joint_q.device),
                    wp.array(joint_qd_np, dtype=wp.float32, device=self._model.joint_qd.device),
                    state
                )
                log.ok(f"Newton robot driver '{self.component_name}': Applied initial configuration and updated FK")
            except Exception as exc:  # noqa: BLE001
                log.error(
                    f"Newton robot driver '{self.component_name}': "
                    f"Failed to evaluate FK after initial configuration: {exc}"
                )

    def _write_joint_position(self, joint_name: str, value: float | Sequence[float]) -> None:
        if self._model is None or self._model.joint_q is None:
            log.warning(f"Newton robot driver: Cannot write joint position, model not initialized")
            return
        idx = self._joint_index_map.get(joint_name)
        if idx is None:
            return  # Joint not mapped, already warned in bind_runtime
        if self._joint_q_start is None:
            log.warning(f"Newton robot driver: joint_q_start array not initialized")
            return

        # Validate array bounds
        if idx >= len(self._joint_q_start) - 1:
            log.error(
                f"Newton robot driver: Joint index {idx} out of bounds for joint_q_start "
                f"(size {len(self._joint_q_start)})"
            )
            return

        start = int(self._joint_q_start[idx])
        end = int(self._joint_q_start[idx + 1])
        width = end - start
        if width <= 0:
            return
        values = value if isinstance(value, Sequence) else [value] * width
        # Get numpy copy, modify, and assign back to device
        joint_q_np = self._model.joint_q.numpy().copy()
        for offset in range(width):
            if offset < len(values):
                joint_q_np[start + offset] = float(values[offset])
        self._model.joint_q.assign(joint_q_np)

    def _write_joint_target(self, joint_name: str, value: float | Sequence[float]) -> None:
        if self._control is None or self._control.joint_target is None:
            log.warning(f"Newton robot driver: Cannot write joint target, control not initialized")
            return
        idx = self._joint_index_map.get(joint_name)
        if idx is None:
            return  # Joint not mapped, already warned in bind_runtime
        if self._joint_qd_start is None:
            log.warning(f"Newton robot driver: joint_qd_start array not initialized")
            return

        # Validate array bounds
        if idx >= len(self._joint_qd_start) - 1:
            log.error(
                f"Newton robot driver: Joint index {idx} out of bounds for joint_qd_start "
                f"(size {len(self._joint_qd_start)})"
            )
            return

        start = int(self._joint_qd_start[idx])
        end = int(self._joint_qd_start[idx + 1])
        width = end - start
        if width <= 0:
            return
        values = value if isinstance(value, Sequence) else [value] * width
        # Get numpy copy, modify, and assign back to device
        joint_target_np = self._control.joint_target.numpy().copy()
        for offset in range(width):
            if offset < len(values):
                joint_target_np[start + offset] = float(values[offset])
        self._control.joint_target.assign(joint_target_np)

    def _write_joint_force(self, joint_name: str, value: float | Sequence[float]) -> None:
        if self._control is None or self._control.joint_f is None:
            log.warning(f"Newton robot driver: Cannot write joint force, control not initialized")
            return
        idx = self._joint_index_map.get(joint_name)
        if idx is None:
            return  # Joint not mapped, already warned in bind_runtime
        if self._joint_qd_start is None:
            log.warning(f"Newton robot driver: joint_qd_start array not initialized")
            return

        # Validate array bounds
        if idx >= len(self._joint_qd_start) - 1:
            log.error(
                f"Newton robot driver: Joint index {idx} out of bounds for joint_qd_start "
                f"(size {len(self._joint_qd_start)})"
            )
            return

        start = int(self._joint_qd_start[idx])
        end = int(self._joint_qd_start[idx + 1])
        width = end - start
        if width <= 0:
            return
        values = value if isinstance(value, Sequence) else [value] * width
        # Get numpy copy, modify, and assign back to device
        joint_f_np = self._control.joint_f.numpy().copy()
        for offset in range(width):
            if offset < len(values):
                joint_f_np[start + offset] = float(values[offset])
        self._control.joint_f.assign(joint_f_np)
        self._last_commanded_torque[joint_name] = float(values[0])

    def check_torque_status(self) -> bool:
        return bool(self._torque_groups)

    def _gather_joint_values(
        self,
        joints: Iterable[str],
        array_getter: Callable[[int], float | Sequence[float]],
    ) -> Dict[str, float | Sequence[float]]:
        result: Dict[str, float | Sequence[float]] = {}
        for joint in joints:
            idx = self._joint_index_map.get(joint)
            if idx is None:
                continue
            result[joint] = array_getter(idx)
        return result

    def pass_joint_positions(self, joints: list[str]) -> dict[str, float | Sequence[float]]:
        state = self._state_accessor()
        if state is None or state.joint_q is None or self._joint_q_start is None:
            return {joint: 0.0 for joint in joints}

        def getter(idx: int) -> float | Sequence[float]:
            start = int(self._joint_q_start[idx])
            end = int(self._joint_q_start[idx + 1])
            width = end - start
            if width <= 0:
                return 0.0
            if width == 1:
                return float(state.joint_q[start])
            return [float(state.joint_q[start + k]) for k in range(width)]

        return self._gather_joint_values(joints, getter)

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float | Sequence[float]]:
        state = self._state_accessor()
        if state is None or state.joint_qd is None or self._joint_qd_start is None:
            return {joint: 0.0 for joint in joints}

        def getter(idx: int) -> float | Sequence[float]:
            start = int(self._joint_qd_start[idx])
            end = int(self._joint_qd_start[idx + 1])
            width = end - start
            if width <= 0:
                return 0.0
            if width == 1:
                return float(state.joint_qd[start])
            return [float(state.joint_qd[start + k]) for k in range(width)]

        return self._gather_joint_values(joints, getter)

    def pass_joint_efforts(self, joints: list[str]) -> dict[str, float]:
        return {joint: self._last_commanded_torque.get(joint, 0.0) for joint in joints}

    def pass_joint_group_control_cmd(
        self,
        control_mode: str,
        cmd: dict[str, float | Sequence[float]],
        **_: Any,
    ) -> None:
        mode = ControlType(control_mode.lower())
        if mode in {ControlType.POSITION, ControlType.VELOCITY}:
            for joint, value in cmd.items():
                self._write_joint_target(joint, value)
        elif mode == ControlType.TORQUE:
            for joint, value in cmd.items():
                self._write_joint_force(joint, value)
        else:
            log.warning(f"Newton robot driver: unsupported control mode '{control_mode}'.")

    def sim_reset(
        self,
        base_pos: list[float],
        base_orn: list[float],
        init_pos: list[float],
    ) -> None:
        self.base_position = base_pos
        self.base_orientation = base_orn
        if init_pos:
            self.initial_configuration = list(init_pos)
        self._apply_initial_configuration()
