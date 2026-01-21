from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from ark.system.driver.robot_driver import SimRobotDriver
from ark.tools.log import log
from ark.utils import lazy
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics


class IsaacSimRobotDriver(SimRobotDriver):
    """Isaac Sim robot driver connecting ARK robot commands to an articulation.

    The driver acts as the low-level bridge between ARK control messages and
    Isaac Sim physics, ensuring that all robot joints map correctly to
    articulation DOFs.

    """

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any],
        sim_app: Any,
        world: Any,
    ) -> None:
        """Initialize the Isaac Sim robot driver.

        Loads and imports the robot asset, creates the articulation, and
        performs an initial reset.

        Args:
            component_name (str): Robot name in ARK.
            component_config (dict[str, Any]): Configuration containing:.
            sim_app (SimulationApp): Isaac Sim application instance.
            world (World): World instance to which the articulation is added.
        """

        self.sim_app = sim_app
        self.world = world
        self._articulation = None
        self._joint_name_to_index: dict[str, int] = {}
        self.component_name = component_name

        super().__init__(
            component_name=component_name, component_config=component_config
        )
        self._load_robot()
        self.sim_reset()

    def _load_robot(self) -> None:
        """Import the robot asset and construct the robot articulation."""

        self.prim_path = self.config.get("prim_path", f"/World/{self.component_name}")
        urdf_path_cfg = self.config.get("urdf_path")

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

        urdf_path = _resolve(urdf_path_cfg)
        usd_path = None  # For future extensions, to load usd file
        self.urdf_path = urdf_path

        if urdf_path is None:
            raise ValueError(f"Robot '{self.component_name}' needs a URDF file.")

        if usd_path:
            # Load robot from USD file
            lazy.isaacsim.core.utils.stage.add_reference_to_stage(
                str(usd_path), self.prim_path
            )
            self._articulation = lazy.isaacsim.core.prims.Articulation(
                prim_paths_expr=self.prim_path, name=self.component_name
            )
        elif urdf_path:
            # Load robot from URDF file

            # Setting up import configuration:
            status, import_config = lazy.omni.kit.commands.execute(
                "URDFCreateImportConfig"
            )
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            fix_base = self.config.get("use_fixed_base")
            if fix_base is None:
                floating = self.config.get("floating")
                if floating is not None:
                    fix_base = not bool(floating)
            if fix_base is None:
                fix_base = True
            import_config.fix_base = bool(fix_base)

            status, self.prim_path = lazy.omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                get_articulation_root=True,
            )

            # Get stage handle
            stage = lazy.omni.usd.get_context().get_stage()

            # Enable physics
            scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
            # Set gravity
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            # Set solver settings
            PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
            physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
            physxSceneAPI.CreateEnableCCDAttr(True)
            physxSceneAPI.CreateEnableStabilizationAttr(True)
            physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
            physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
            physxSceneAPI.CreateSolverTypeAttr("TGS")

            lazy.omni.timeline.get_timeline_interface().play()
            self.sim_app.update()
            self._articulation = lazy.isaacsim.core.prims.Articulation(self.prim_path)
            self.world.scene.add(self._articulation)
            self._articulation.initialize()

        # Set initial Position and Orientation
        self.base_position = self.config.get("base_position", [0.0, 0.0, 0.0])
        self.base_orientation = self.config.get(
            "base_orientation", [0.0, 0.0, 0.0, 1.0]
        )
        self.initial_configuration = self.config.get(
            "initial_configuration", [0.0] * len(self._articulation.joint_names)
        )

        self._joint_name_to_index = {
            name: self._articulation.get_dof_index(name)
            for name in self._articulation.dof_names
        }

    def check_torque_status(self) -> bool:
        """Check whether torque control is enabled.

        Returns:
            bool: Always True for this minimal driver implementation.
        """
        return True

    def pass_joint_positions(self, joints: list[str]) -> dict[str, float]:
        """Retrieve joint positions for the requested joints.

        Args:
            joints (list[str]): Joint names to query.

        Returns:
            dict[str, float]: Mapping joint_name → position_value.
        """
        positions = self._articulation.get_joint_positions().flatten()
        return {
            name: float(positions[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float]:
        """Retrieve joint velocities for the requested joints.

        Args:
            joints (list[str]): Joint names to query.

        Returns:
            dict[str, float]: Mapping joint_name → velocity_value.
        """
        velocities = self._articulation.get_joint_velocities().flatten()
        return {
            name: float(velocities[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    @staticmethod
    def pass_joint_efforts(joints: list[str]) -> dict[str, float]:
        """Retrieve joint efforts.

        Notes:
            Efforts are currently not simulated and always return 0.0.

        Args:
            joints (list[str]): Joint names to query.

        Returns:
            dict[str, float]: Mapping joint_name → effort_value (0.0).
        """
        # TODO check for the implementation
        return {name: 0.0 for name in joints}

    def pass_joint_group_control_cmd(
        self, control_mode: str, cmd: dict[str, float], **kwargs
    ) -> None:
        """Send a group joint control command to the robot.

        Supported control modes:
            - "position": Sets joint target positions.
            - "velocity": Sets joint target velocities.
            - "torque": Applies joint efforts.

        Args:
            control_mode (str): Control mode type.
            cmd (dict[str, float]): Mapping joint_name → target_value.
            **kwargs: Additional unused parameters for compatibility.
        """

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

        action = lazy.omni.isaac.core.utils.types.ArticulationAction(
            joint_positions=positions,
            joint_velocities=velocities if control_mode == "velocity" else None,
            joint_efforts=efforts if control_mode == "torque" else None,
        )
        self._articulation.apply_action(action)

    def sim_reset(self, *kargs, **kwargs) -> None:
        """Reset the robot articulation.

        Args:
            *kargs: Ignored.
            **kwargs: Ignored.
        """
        self._articulation.set_world_poses(
            positions=np.array([self.base_position]),
            orientations=np.array([self.base_orientation]),
        )
        dof_count = len(self._articulation.joint_names)
        if len(self.initial_configuration) >= dof_count:
            q_init = self.initial_configuration[:dof_count]
        else:
            q_init = list(self.initial_configuration) + [0.0] * (
                dof_count - len(self.initial_configuration)
            )

        self._articulation.set_joint_positions([q_init])
        self._articulation.set_joint_velocities(
            np.zeros_like(self._articulation.get_joint_positions())
        )

    @abstractmethod
    def pass_cartesian_control_cmd(self, *kargs, **kwargs) -> None:
        """Send a Cartesian-space control command.

        Abstract method  must be implemented by subclasses.

        Args:
            *kargs: Implementation-specific arguments.
            **kwargs: Implementation-specific arguments.
        """
        ...
