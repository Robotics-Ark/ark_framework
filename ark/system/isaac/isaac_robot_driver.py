from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from ark.system.driver.robot_driver import SimRobotDriver
from ark.tools.log import log


class IsaacSimRobotDriver(SimRobotDriver):
    """Minimal Isaac Sim driver bridging ARK robot commands to an articulation."""

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any],
        sim_app: Any,
        world: Any,
    ) -> None:

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
        """Import the robot asset, converting URDF to USD in-place when needed."""
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import Articulation
        import omni.kit.commands
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics

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
            add_reference_to_stage(str(usd_path), self.prim_path)
            self._articulation = Articulation(
                prim_paths_expr=self.prim_path, name=self.component_name
            )
        elif urdf_path:
            # Load robot from URDF file
            # Setting up import configuration:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.fix_base = True
            # import_config.distance_scale = 1.0

            status, self.prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                get_articulation_root=True,
            )

            # Get stage handle
            stage = omni.usd.get_context().get_stage()

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

            omni.timeline.get_timeline_interface().play()
            self.sim_app.update()
            self._articulation = Articulation(self.prim_path)
            self.world.scene.add(self._articulation)
            self._articulation.initialize()

        # self.world.step(render=True)

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
        return True

    def pass_joint_positions(self, joints: list[str]) -> dict[str, float]:
        positions = self._articulation.get_joint_positions().flatten()
        return {
            name: float(positions[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float]:
        velocities = self._articulation.get_joint_velocities().flatten()
        return {
            name: float(velocities[self._joint_name_to_index[name]])
            for name in joints
            if name in self._joint_name_to_index
        }

    def pass_joint_efforts(self, joints: list[str]) -> dict[str, float]:
        return {name: 0.0 for name in joints}

    def pass_joint_group_control_cmd(
        self, control_mode: str, cmd: dict[str, float], **kwargs
    ) -> None:
        # TODO check this
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

    def sim_reset(self, *kargs, **kwargs) -> None:
        self._articulation.set_world_poses(
            positions=np.array([self.base_position]),
            orientations=np.array([self.base_orientation]),
        )
        if len(self.initial_configuration) > 9:
            q_init = self.initial_configuration[:9]
        else:
            q_init = self.initial_configuration

        self._articulation.set_joint_positions([q_init])
        self._articulation.set_joint_velocities(
            np.zeros_like(self._articulation.get_joint_positions())
        )

    @abstractmethod
    def pass_cartesian_control_cmd(self, *kargs, **kwargs) -> None: ...

    # TODO check eef
    def get_ee_pose(self) -> dict[str, float]:
        """!Get the end-effector pose in the world frame."""
        position, orientation = self._articulation.get_world_poses()
        return {"position": position[0], "orientation": orientation[0]}

    def shutdown_driver(self) -> None:
        pass
