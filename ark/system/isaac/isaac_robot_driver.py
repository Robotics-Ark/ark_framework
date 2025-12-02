"""Isaac Sim robot driver for ARK with on-the-fly URDF to USD conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ark.system.driver.robot_driver import SimRobotDriver
from ark.tools.log import log


class IsaacSimRobotDriver(SimRobotDriver):
    """Minimal Isaac Sim driver bridging ARK robot commands to an articulation."""

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        world: Any,
    ) -> None:
        self.world = world
        self._articulation = None
        self._joint_names: List[str] = []
        self._joint_name_to_index: Dict[str, int] = {}

        super().__init__(
            component_name=component_name, component_config=component_config
        )
        self._load_robot()

        self.sim_reset()

    def _load_robot(self) -> None:
        """Import the robot asset, converting URDF to USD in-place when needed."""
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import Articulation

        sim_cfg = self.config.get("config", {})
        prim_path = sim_cfg.get("prim_path", f"/World/{self.component_name}")

        usd_path_cfg = sim_cfg.get("usd_path") or self.config.get("usd_path")
        urdf_path_cfg = sim_cfg.get("urdf_path") or self.config.get("urdf_path")  # TODO

        base_dir = Path(self.config.get("class_dir", ".")).resolve()

        def _resolve(path: str | None) -> Path | None:
            if not path:
                return None
            candidate = Path(path)
            return (
                candidate
                if candidate.is_absolute()
                else (base_dir / candidate).resolve()
            )

        usd_path = _resolve(usd_path_cfg)
        urdf_path = _resolve(urdf_path_cfg)

        if usd_path is None and urdf_path is None:
            raise ValueError(f"Robot '{self.component_name}' needs a USD  or URDF.")

        if usd_path and urdf_path:
            raise ValueError(
                f"Robot '{self.component_name}' provide either a USD  or URDF, but not both."
            )

        if usd_path:
            add_reference_to_stage(str(usd_path), prim_path)
            self._articulation = Articulation(
                prim_paths_expr=prim_path, name=self.component_name
            )

        elif urdf_path:
            import omni.kit.commands

            # Setting up import configuration:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.fix_base = True
            import_config.distance_scale = 1.0

            # Import URDF, prim_path contains the path to the usd prim in the stage.
            status, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                get_articulation_root=True,
            )

            self._articulation = Articulation(prim_path)

        self.world.reset()

        self._joint_names = self._articulation.joint_names[
            :9
        ]  # TODO handle 10 joints and 9 positions
        self._joint_name_to_index = {
            name: self._articulation.get_joint_index(name)
            for idx, name in enumerate(self._joint_names)
        }

        self.base_position = self.config.get("base_position", [0.0, 0.0, 0.0])
        self.base_orientation = self.config.get(
            "base_orientation", [0.0, 0.0, 0.0, 1.0]
        )
        self.initial_configuration = self.config.get(
            "initial_configuration", [0.0] * len(self._articulation.joint_names)
        )

    def check_torque_status(self) -> bool:
        return True

    def pass_joint_positions(self, joints: List[str]) -> Dict[str, float]:
        positions = self._articulation.get_joint_positions().flatten()
        return {
            name: float(positions[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    def pass_joint_velocities(self, joints: List[str]) -> Dict[str, float]:
        velocities = self._articulation.get_joint_velocities().flatten()
        return {
            name: float(velocities[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    def pass_joint_efforts(self, joints: List[str]) -> Dict[str, float]:
        return {name: 0.0 for name in joints}

    def pass_joint_group_control_cmd(
        self, control_mode: str, cmd: Dict[str, float], **kwargs
    ) -> None:
        from omni.isaac.core.utils.types import ArticulationAction

        positions = self._articulation.get_joint_positions()
        velocities = self._articulation.get_joint_velocities()
        efforts = np.zeros_like(positions)

        for joint_name, target in cmd.items():
            if joint_name not in self._joint_name_to_index:
                continue
            idx = self._joint_name_to_index[joint_name]
            if control_mode == "position":
                positions[idx] = target
            elif control_mode == "velocity":
                velocities[idx] = target
            elif control_mode == "torque":
                efforts[idx] = target
            else:
                log.warn(f"Unsupported control_mode '{control_mode}' for Isaac Sim.")

        action = ArticulationAction(
            joint_positions=positions,
            joint_velocities=velocities if control_mode == "velocity" else None,
            joint_efforts=efforts if control_mode == "torque" else None,
        )
        self._articulation.apply_action(action)

    def pass_cartesian_control_cmd(
        self, control_mode: str, position, quaternion, **kwargs
    ) -> None:
        """Compute IK for the end-effector and apply joint targets."""
        from omni.isaac.core.utils.types import ArticulationAction

        print("Z" * 100)
        if control_mode != "position":
            log.warn(f"Cartesian control_mode '{control_mode}' not supported in Isaac.")
            return

        # Choose EE index: use provided idx, otherwise default to last joint
        ee_idx = kwargs.get("end_effector_idx")
        if ee_idx is None:
            ee_idx = len(self._joint_names) - 1

        try:
            ik_positions = self._articulation.compute_inverse_kinematics(
                target_position=position,
                target_orientation=quaternion,
                end_effector_index=ee_idx,
            )
        except Exception as exc:
            log.error(f"Isaac IK failed: {exc}")
            return

        ik_positions = np.array(ik_positions, dtype=float).flatten()

        # Optional gripper control: overwrite last two joints if provided
        gripper_target = kwargs.get("gripper", None)
        if gripper_target is not None and ik_positions.size >= 2:
            ik_positions[-2:] = float(gripper_target)

        action = ArticulationAction(joint_positions=ik_positions)
        self._articulation.apply_action(action)

    def sim_reset(self, *kargs, **kwargs) -> None:
        self._articulation.set_world_poses(
            positions=self.base_position, orientations=self.base_orientation
        )
        if len(self.initial_configuration) > 9:
            q_init = self.initial_configuration[:9]
        else:
            q_init = self.initial_configuration

        self._articulation.set_joint_positions([q_init])
        self._articulation.set_joint_velocities(
            np.zeros_like(self._articulation.get_joint_positions())
        )

    # TODO check eef
    def get_ee_pose(self) -> dict[str, float]:
        """!Get the end-effector pose in the world frame."""
        position, orientation = self._articulation.get_world_poses()
        return {"position": position[0], "orientation": orientation[0]}

    def shutdown_driver(self) -> None:
        pass
