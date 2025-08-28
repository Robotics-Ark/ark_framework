from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ark.tools.log import log
from ark.system.driver.robot_driver import SimRobotDriver


class MujocoRobotDriver(SimRobotDriver):
    """Robot driver for MuJoCo simulations."""

    @staticmethod
    def body_subtree(model: mujoco.MjModel, root_body_id: int) -> list[int]:
        """Return IDs of bodies in the subtree rooted at ``root_body_id``."""
        descendants: list[int] = []
        stack = [root_body_id]
        visited = {root_body_id}
        while stack:
            current_id = stack.pop()
            descendants.append(current_id)
            for child in range(model.nbody):
                if model.body_parentid[child] == current_id and child not in visited:
                    visited.add(child)
                    stack.append(child)
        return descendants

    @classmethod
    def joints_for_body(
        cls, model: mujoco.MjModel, body_id: int
    ) -> list[tuple[int, str]]:
        """Return ``(joint_id, joint_name)`` for ``body_id`` and its descendants."""
        joint_ids: list[int] = []
        for current_id in cls.body_subtree(model, body_id):
            start = model.body_jntadr[current_id]
            count = model.body_jntnum[current_id]
            for offset in range(count):
                joint_ids.append(start + offset)
        return [
            (jid, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
            for jid in joint_ids
        ]

    @staticmethod
    def joint_qpos_slice(model: mujoco.MjModel, joint_id: int) -> slice:
        """Return slice in ``data.qpos`` corresponding to ``joint_id``."""
        address = model.jnt_qposadr[joint_id]
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            width = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            width = 4
        else:
            width = 1
        return slice(address, address + width)

    @staticmethod
    def actuators_for_joint(model: mujoco.MjModel, joint_id: int) -> list[int]:
        """Return actuator indices driving any degree of freedom of ``joint_id``."""
        actuator_indices: list[int] = []
        for actuator_index in range(model.nu):
            dof = model.actuator_trnid[actuator_index][0]
            if dof >= 0 and model.dof_jntid[dof] == joint_id:
                actuator_indices.append(actuator_index)
        return actuator_indices

    @staticmethod
    def _joint_widths(model: mujoco.MjModel, joint_index: int) -> tuple[int, int]:
        """Return ``(<qpos_width, qvel_width)`` for ``joint_index``."""
        joint_type = model.jnt_type[joint_index]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1

    @classmethod
    def _joints_in_subtree(cls, model: mujoco.MjModel, root_body_id: int) -> list[int]:
        """List joint IDs under the body subtree in ``qpos`` order."""
        joint_ids: list[int] = []
        for body_id in cls.body_subtree(model, root_body_id):
            start = model.body_jntadr[body_id]
            num = model.body_jntnum[body_id]
            for offset in range(num):
                joint_ids.append(start + offset)
        return sorted(joint_ids, key=lambda jid: model.jnt_qposadr[jid])

    @classmethod
    def get_robot_state(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        root_body_id: int,
    ):
        """Return joint positions, velocities and accelerations.

        Parameters
        ----------
        model
            MuJoCo model instance.
        data
            MuJoCo data instance.
        root_body_id
            Root body identifier for the robot.
        as_dict
            If ``True``, return a dictionary representation.

        Returns
        -------
        dict | tuple
            Either a dictionary with per-joint information or a tuple of
            concatenated ``(qpos, qvel, qacc)`` arrays.
        """
        per_joint: list[dict[str, Any]] = []
        qpos_chunks: list[np.ndarray] = []
        qvel_chunks: list[np.ndarray] = []
        qacc_chunks: list[np.ndarray] = []

        for joint_index in cls._joints_in_subtree(model, root_body_id):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
                or f"joint_{joint_index}"
            )
            qpos_w, qvel_w = cls._joint_widths(model, joint_index)

            qpos_addr = model.jnt_qposadr[joint_index]
            qpos = (
                data.qpos[qpos_addr : qpos_addr + qpos_w]
                if qpos_w > 1
                else np.array([data.qpos[qpos_addr]])
            )

            dof_addr = model.jnt_dofadr[joint_index]
            qvel = (
                data.qvel[dof_addr : dof_addr + qvel_w]
                if qvel_w > 1
                else np.array([data.qvel[dof_addr]])
            )
            qacc = (
                data.qacc[dof_addr : dof_addr + qvel_w]
                if qvel_w > 1
                else np.array([data.qacc[dof_addr]])
            )

            per_joint.append(
                {
                    "id": joint_index,
                    "name": name,
                    "qpos": qpos.copy(),
                    "qvel": qvel.copy(),
                    "qacc": qacc.copy(),
                }
            )
            qpos_chunks.append(qpos)
            qvel_chunks.append(qvel)
            qacc_chunks.append(qacc)

        qpos_cat = np.concatenate(qpos_chunks) if qpos_chunks else np.array([])
        qvel_cat = np.concatenate(qvel_chunks) if qvel_chunks else np.array([])
        qacc_cat = np.concatenate(qacc_chunks) if qacc_chunks else np.array([])

        return {
            "per_joint": per_joint,
            "qpos": qpos_cat,
            "qvel": qvel_cat,
            "qacc": qacc_cat,
        }

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, Any] | None = None,
        builder: Any | None = None,
    ) -> None:
        """Create a robot driver for MuJoCo."""
        super().__init__(component_name, component_config, True)

        self.name = component_name

        class_path = self.config.get("class_path")
        mjcf_path = self.config.get("mjcf_path")
        if mjcf_path:
            mjcf_path = (
                Path(class_path) / mjcf_path
                if class_path is not None
                else Path(mjcf_path)
            )

            if not mjcf_path.is_absolute():
                mjcf_path = Path(self.config["class_dir"]) / mjcf_path

            if not mjcf_path.exists():
                log.error(f"The URDF path '{mjcf_path}' does not exist.")
                log.error(f"Full path: {mjcf_path.resolve()}")
                return

        position = self.config.get("base_position", [0.0, 0.0, 0.0])
        orientation = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        orientation = [orientation[3], orientation[0], orientation[1], orientation[2]]

        fixed_base = self.config.get("use_fixed_base", False)
        root_joint_name = f"{self.name}_root"
        self.initial_positions = self.config.get("initial_configuration", None)

        if builder is not None:
            builder.include_robot(
                name=self.name,
                file=mjcf_path,
                pos=position,
                quat=orientation,
                fixed_base=fixed_base,
                root_joint_name=root_joint_name,
                qpos=self.initial_positions,
            )

    def update_ids(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Update internal IDs from the MuJoCo model."""
        self.model = model
        self.data = data

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        if self.body_id < 0:
            raise ValueError(f"Body '{self.name}' not found in model.")

        self.actuated_joints = {
            f"joint{i}": mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"
            )
            for i in range(1, 8)
        }
        self.gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )

        self.actuated_joints = {**self.actuated_joints, "gripper": self.gripper_id}
        for i, (_, actuator_id) in enumerate(self.actuated_joints.items()):
            data.ctrl[actuator_id] = self.initial_positions[i]

        data.ctrl[self.gripper_id] = 180

    def check_torque_status(self) -> bool:
        """Check the torque status of the robot."""
        raise NotImplementedError(
            "MujocoRobotDriver.check_torque_status is not implemented yet."
        )

    def pass_joint_efforts(self, joints: list[str]) -> dict[str, float]:
        """Retrieve joint efforts (not implemented)."""
        raise NotImplementedError(
            "MujocoRobotDriver.pass_joint_efforts is not implemented yet."
        )

    def pass_joint_group_control_cmd(
        self, control_mode: str, cmd: dict[str, float], **kwargs
    ) -> None:
        """Send a group control command to the robot."""
        for value, actuator_id in zip(cmd.values(), self.actuated_joints.values()):
            self.data.ctrl[actuator_id] = value

    def pass_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        """Return current joint positions for all actuated joints."""
        state = self.get_robot_state(self.model, self.data, self.body_id)
        positions_dict: dict[str, float] = {}
        for i, joint in enumerate(self.actuated_joints):
            positions_dict[joint] = state["qpos"][i]
        return positions_dict

    def pass_joint_velocities(self, joints: list[str]) -> dict[str, float]:
        """Return joint velocities (not implemented)."""
        raise NotImplementedError(
            "MujocoRobotDriver.pass_joint_velocities is not implemented yet."
        )

    def sim_reset(
        self, base_pos: list[float], base_orn: list[float], init_pos: list[float]
    ) -> None:
        """Reset the robot simulation (not implemented)."""
        raise NotImplementedError("MujocoRobotDriver.sim_reset is not implemented yet.")
