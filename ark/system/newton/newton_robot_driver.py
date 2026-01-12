"""Newton robot driver bridging ARK control with the Newton simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import newton
import numpy as np
import warp as wp
from newton.selection import ArticulationView

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

        # ArticulationView for proper Newton control API (UR10 pattern)
        self._articulation_view: ArticulationView | None = None
        self._control_handle: wp.array | None = None

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

        # Store pre-load joint/body counts to identify which joints were added
        pre_joint_count = self.builder.joint_count
        pre_joint_dof_count = len(self.builder.joint_target_ke)

        # parse_visuals_as_colliders: Use visual meshes for collision (more reliable per panda_hydro example)
        parse_visuals = bool(self.config.get("parse_visuals_as_colliders", True))

        try:
            self.builder.add_urdf(
                str(urdf_path),
                xform=xform,
                floating=floating,
                enable_self_collisions=enable_self_collisions,
                collapse_fixed_joints=collapse_fixed,
                parse_visuals_as_colliders=parse_visuals,
            )
            log.ok(f"Newton robot driver: Loaded URDF '{urdf_path.name}' (parse_visuals_as_colliders={parse_visuals})")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Newton robot driver: Failed to load URDF '{urdf_path}': {exc}")
            raise

        # CRITICAL FIX: Apply joint defaults to URDF joints via post-processing
        # Newton's add_urdf() ignores default_joint_cfg, so we must manually apply it
        # to all joints that were just loaded (from pre_joint_dof_count to current)
        post_joint_dof_count = len(self.builder.joint_target_ke)
        num_new_dofs = post_joint_dof_count - pre_joint_dof_count

        if num_new_dofs > 0:
            # Get defaults from builder
            default_cfg = self.builder.default_joint_cfg

            # CRITICAL: Apply initial_configuration to builder.joint_q BEFORE finalize
            # This ensures the model starts in the correct configuration, avoiding
            # the "explosion on load" problem where robot tries to jump from zero to config
            new_joint_count = self.builder.joint_count - pre_joint_count
            if self.initial_configuration:
                for j_idx in range(new_joint_count):
                    if j_idx >= len(self.initial_configuration):
                        break
                    # Get DOF range for this joint
                    joint_idx = pre_joint_count + j_idx
                    q_start = int(self.builder.joint_q_start[joint_idx])
                    q_end = int(self.builder.joint_q_start[joint_idx + 1]) if (joint_idx + 1) < len(self.builder.joint_q_start) else len(self.builder.joint_q)
                    width = q_end - q_start
                    if width <= 0:
                        continue
                    target = self.initial_configuration[j_idx]
                    values = target if isinstance(target, Sequence) else [target] * width
                    for offset in range(min(width, len(values))):
                        self.builder.joint_q[q_start + offset] = float(values[offset])
                log.info(
                    f"Newton robot driver: Applied initial_configuration to builder.joint_q "
                    f"({len(self.initial_configuration)} values)"
                )

            # Apply to all newly loaded joint DOFs
            for i in range(pre_joint_dof_count, post_joint_dof_count):
                self.builder.joint_target_ke[i] = default_cfg.target_ke
                self.builder.joint_target_kd[i] = default_cfg.target_kd
                self.builder.joint_limit_ke[i] = default_cfg.limit_ke
                self.builder.joint_limit_kd[i] = default_cfg.limit_kd
                self.builder.joint_armature[i] = default_cfg.armature
                # Override URDF limits if default_joint_cfg specifies them.
                self.builder.joint_effort_limit[i] = default_cfg.effort_limit
                self.builder.joint_velocity_limit[i] = default_cfg.velocity_limit
                self.builder.joint_friction[i] = default_cfg.friction

                # CRITICAL: Initialize joint_target_pos to match joint_q (which now has initial config)
                # Without this, PD controller has no target to track!
                # This follows Newton's own examples (see example_basic_urdf.py:72)
                self.builder.joint_target_pos[i] = self.builder.joint_q[i]

            log.ok(
                f"Newton robot driver: Applied joint defaults to {num_new_dofs} DOFs "
                f"(ke={default_cfg.target_ke}, kd={default_cfg.target_kd})"
            )

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

        # Create ArticulationView for proper Newton control API (UR10 example pattern)
        # This is CRITICAL for joint target position control to work correctly
        try:
            # Use component name as pattern (e.g., "panda" matches bodies with "panda" in name)
            pattern = f"*{self.component_name}*"
            self._articulation_view = ArticulationView(
                model,
                pattern,
                exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE]
            )

            # Get control handle for joint target positions
            self._control_handle = self._articulation_view.get_attribute("joint_target_pos", control)

            log.ok(
                f"Newton robot driver '{self.component_name}': "
                f"Created ArticulationView (pattern='{pattern}', count={self._articulation_view.count}, "
                f"dofs={self._control_handle.shape if self._control_handle is not None else 'N/A'})"
            )
            # Log body names for debugging EE index configuration
            if hasattr(self._articulation_view, "body_names"):
                log.info(
                    f"Newton robot driver '{self.component_name}': "
                    f"ArticulationView body_names (link indices): {self._articulation_view.body_names}"
                )
        except Exception as exc:
            log.error(
                f"Newton robot driver '{self.component_name}': "
                f"Failed to create ArticulationView: {exc}. Falling back to direct control."
            )
            self._articulation_view = None
            self._control_handle = None

        # NOTE: _apply_initial_configuration() is NOT called here because initial config
        # is already applied to builder.joint_q and builder.joint_target_pos in _load_into_builder()
        # BEFORE finalize(). After model.state(), state.joint_q inherits these values, and
        # the backend's eval_fk with model.joint_q correctly computes body transforms.
        # Calling _apply_initial_configuration() here would be redundant and uses state.joint_q
        # for eval_fk instead of model.joint_q, causing inconsistency.
        # The method is kept for sim_reset() which needs to re-apply config at runtime.

    def _apply_initial_configuration(self) -> None:
        """Apply initial joint configuration to runtime state and control.

        CRITICAL: This writes to state.joint_q (runtime simulation state),
        NOT model.joint_q (static template). The physics solver reads/writes
        state.joint_q, so that's where initial config must be applied.
        """
        if not self.initial_configuration or not self._joint_names:
            log.info(f"Newton robot driver '{self.component_name}': No initial configuration to apply")
            return

        state = self._state_accessor()
        if state is None or state.joint_q is None:
            log.warning(
                f"Newton robot driver '{self.component_name}': "
                f"Cannot apply initial config - state not available yet"
            )
            return

        # Validate configuration length
        if len(self.initial_configuration) != len(self._joint_names):
            log.warning(
                f"Newton robot driver '{self.component_name}': "
                f"initial_configuration length ({len(self.initial_configuration)}) != "
                f"joint count ({len(self._joint_names)}). Using available values."
            )

        # Get state arrays as numpy (mutable copies)
        joint_q_np = state.joint_q.numpy().copy()
        joint_qd_np = state.joint_qd.numpy().copy()

        # Apply initial positions to STATE (not model!)
        for idx, joint_name in enumerate(self._joint_names):
            if idx >= len(self.initial_configuration):
                break

            model_idx = self._joint_index_map.get(joint_name)
            if model_idx is None:
                continue

            if model_idx >= len(self._joint_q_start) - 1:
                continue

            start = int(self._joint_q_start[model_idx])
            end = int(self._joint_q_start[model_idx + 1])
            width = end - start
            if width <= 0:
                continue

            target = self.initial_configuration[idx]
            values = target if isinstance(target, Sequence) else [target] * width

            # Write to state.joint_q
            for offset in range(width):
                if offset < len(values):
                    joint_q_np[start + offset] = float(values[offset])

            # Zero out velocities in state.joint_qd
            vel_start = int(self._joint_qd_start[model_idx])
            vel_end = int(self._joint_qd_start[model_idx + 1])
            for offset in range(vel_end - vel_start):
                joint_qd_np[vel_start + offset] = 0.0

            # Set control target
            self._write_joint_target(joint_name, target)

        # Write modified arrays back to state
        state.joint_q.assign(joint_q_np)
        state.joint_qd.assign(joint_qd_np)

        # Update forward kinematics with STATE arrays
        try:
            newton.eval_fk(
                self._model,
                state.joint_q,  # KEY FIX: Use state.joint_q, not model.joint_q
                state.joint_qd,
                state
            )
            log.ok(
                f"Newton robot driver '{self.component_name}': "
                f"Applied initial configuration to runtime state and updated FK"
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                f"Newton robot driver '{self.component_name}': "
                f"Failed to evaluate FK after initial configuration: {exc}"
            )

    def _write_joint_target(self, joint_name: str, value: float | Sequence[float]) -> None:
        """Write joint target position to Newton control.

        Uses direct assignment to control.joint_target_pos following Newton's
        working examples (e.g., example_ik_cube_stacking.py).
        """
        if self._control is None or self._control.joint_target_pos is None:
            log.warning(f"Newton robot driver: Cannot write joint target, control not initialized")
            return
        idx = self._joint_index_map.get(joint_name)
        if idx is None:
            return  # Joint not mapped, already warned in bind_runtime
        # CRITICAL: Use joint_q_start (position indices) not joint_qd_start (velocity indices)
        # Joint targets are POSITION targets, so they must use position DOF indices
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

        # Direct assignment to control.joint_target_pos
        joint_target_np = self._control.joint_target_pos.numpy().copy()
        for offset in range(width):
            if offset < len(values):
                joint_target_np[start + offset] = float(values[offset])
        self._control.joint_target_pos.assign(joint_target_np)

    def _write_joint_target_velocity(self, joint_name: str, value: float | Sequence[float]) -> None:
        """Write joint target velocity using ArticulationView API."""
        if self._control is None or self._control.joint_target_vel is None:
            log.warning("Newton robot driver: Cannot write joint target velocity, control not initialized")
            return
        idx = self._joint_index_map.get(joint_name)
        if idx is None:
            return
        if self._joint_qd_start is None:
            log.warning("Newton robot driver: joint_qd_start array not initialized")
            return

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

        if self._articulation_view is not None:
            handle_np = self._articulation_view.get_attribute("joint_target_vel", self._control).numpy().copy()
            env_idx = 0
            for offset in range(width):
                if offset < len(values):
                    handle_np[env_idx, start + offset] = float(values[offset])
            handle = wp.array(handle_np, dtype=self._control.joint_target_vel.dtype, device=self._control.joint_target_vel.device)
            self._articulation_view.set_attribute("joint_target_vel", self._control, handle)
        else:
            joint_target_np = self._control.joint_target_vel.numpy().copy()
            for offset in range(width):
                if offset < len(values):
                    joint_target_np[start + offset] = float(values[offset])
            self._control.joint_target_vel.assign(joint_target_np)

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

        # Convert Warp array to numpy once for efficient indexing
        joint_q_np = state.joint_q.numpy()

        def getter(idx: int) -> float | Sequence[float]:
            start = int(self._joint_q_start[idx])
            end = int(self._joint_q_start[idx + 1])
            width = end - start
            if width <= 0:
                return 0.0
            if width == 1:
                return float(joint_q_np[start])
            return [float(joint_q_np[start + k]) for k in range(width)]

        return self._gather_joint_values(joints, getter)

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float | Sequence[float]]:
        state = self._state_accessor()
        if state is None or state.joint_qd is None or self._joint_qd_start is None:
            return {joint: 0.0 for joint in joints}

        # Convert Warp array to numpy once for efficient indexing
        joint_qd_np = state.joint_qd.numpy()

        def getter(idx: int) -> float | Sequence[float]:
            start = int(self._joint_qd_start[idx])
            end = int(self._joint_qd_start[idx + 1])
            width = end - start
            if width <= 0:
                return 0.0
            if width == 1:
                return float(joint_qd_np[start])
            return [float(joint_qd_np[start + k]) for k in range(width)]

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
                if mode == ControlType.VELOCITY:
                    self._write_joint_target_velocity(joint, value)
                else:
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
