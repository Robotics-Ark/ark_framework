"""Genesis robot driver implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np

from ark.tools.log import log
from ark.system.driver.robot_driver import ControlType, SimRobotDriver
 

class GenesisRobotDriver(SimRobotDriver):
    """Robot driver that interfaces with the Genesis simulation."""

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any] | None = None,
        client: gs.Scene | None = None,
    ) -> None:
        """Create a Genesis robot driver."""

        if component_config is None:
            raise ValueError("GenesisRobotDriver requires a component configuration.")
        if client is None:
            raise ValueError("GenesisRobotDriver requires an initialized Genesis scene.")

        super().__init__(component_name, component_config, True)

        self.client: gs.Scene = client
        self.ref_body_id: Any | None = None

        base_position_cfg = self.config.get("base_position", [0.0, 0.0, 0.0])
        self.base_position: list[float] = list(base_position_cfg)

        base_orientation_cfg = self.config.get(
            "base_orientation", [0.0, 0.0, 0.0, 1.0]
        )
        self.base_orientation = list(base_orientation_cfg)

        self.joint_names: list[str] = list(self.config.get("joint_names", []))
        if not self.joint_names:
            raise ValueError(
                f"Robot '{component_name}' configuration must define 'joint_names'."
            )

        self.load_robot(self.base_position, self.base_orientation, None)

        if self.ref_body_id is None:
            raise RuntimeError("Robot entity has not been created in Genesis.")

        self.dofs_idx = [
            self.ref_body_id.get_joint(name).dof_idx_local for name in self.joint_names
        ]

    def load_robot(
        self,
        base_position: Sequence[float] | None = None,
        base_orientation: Sequence[float] | None = None,
        q_init: Sequence[float] | None = None,
    ) -> None:
        """Load the robot model into the Genesis simulator."""

        if base_position is None:
            base_position = self.base_position
        if base_orientation is None:
            base_orientation = self.base_orientation

        mjcf_path_cfg = self.config.get("mjcf_path")
        if mjcf_path_cfg is None:
            raise ValueError(
                f"Robot '{self.component_name}' configuration requires 'mjcf_path'."
            )

        mjcf_path = Path(mjcf_path_cfg)
        self.ref_body_id = self.client.add_entity(
            gs.morphs.MJCF(file=str(mjcf_path))
        )

        log.ok(
            f"Initialized robot specified by MJCF '{mjcf_path}' in Genesis simulator."
        )

    #####################
    ##    get infos    ##
    #####################

    def check_torque_status(self) -> bool:
        """!Return ``True`` as simulated robots are always torqued.

        @return Always ``True`` in simulation.
        @rtype bool
        """
        return True  # simulated robot is always torqued in genesis

    def pass_joint_positions(self, joints: list[str]) -> dict[str, float]:
        """Return the current joint positions."""

        pos = {}
        joint_pos = self.ref_body_id.get_dofs_position()
        for idx, name in enumerate(self.joint_names):
            pos[name] = joint_pos[idx]
        return pos

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float]:
        """Return the current joint velocities."""
        vel = {}
        joint_vel = self.ref_body_id.get_dofs_velocity()
        for idx, name in enumerate(self.joint_names):
            vel[name] = joint_vel[idx]
        return vel

    def pass_joint_efforts(self, joints: list[str]) -> dict[str, float]:
        """Return the current joint efforts."""
        eff = {}
        joint_vel = self.ref_body_id.get_dofs_force()
        for idx, name in enumerate(self.joint_names):
            eff[name] = joint_vel[idx]
        return eff

    #####################
    ##     control     ##
    #####################

    def pass_joint_group_control_cmd(
        self,
        control_mode: ControlType | str,
        cmd: Mapping[str, float],
        **kwargs: Any,
    ) -> None:
        """Send a control command to a group of joints."""

    
        self.ref_body_id.control_dofs_position(
            np.array(list(cmd.values())),
            self.dofs_idx
        )


    #####################
    ##      misc.      ##
    #####################

    def sim_reset(
        self,
        base_pos: Sequence[float],
        base_orn: Sequence[float],
        q_init: Sequence[float] | None,
    ) -> None:
        """Reset the robot in the simulator."""

        if self.ref_body_id is None:
            raise RuntimeError("Robot entity has not been initialized in Genesis.")

        self.ref_body_id.set_pos(list(base_pos))
        self.ref_body_id.set_quat(list(base_orn))

        if q_init is not None:
            self.ref_body_id.control_dofs_position(np.array(q_init, dtype=float), self.dofs_idx)

        log.ok(f"Reset robot {self.component_name} completed.")
