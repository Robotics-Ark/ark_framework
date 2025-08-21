"""@file pybullet_robot_driver.py
@brief Robot driver handling PyBullet specific commands.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List
import os
import pybullet as p
from pathlib import Path
import mujoco
import numpy as np
from ark.tools.log import log
from ark.system.driver.robot_driver import SimRobotDriver, ControlType

from ark.system.mujoco.mjcf_builder import MJCFBuilder, BodySpec
from functools import cache

def body_subtree(model, root_body_id):
    """Return a list of body IDs in the subtree rooted at root_body_id (incl. root)."""
    descendants = []
    stack = [root_body_id]
    visited = set(stack)
    while stack:
        b = stack.pop()
        descendants.append(b)
        # children = bodies whose parent is b
        for child in range(model.nbody):
            if model.body_parentid[child] == b and child not in visited:
                visited.add(child)
                stack.append(child)
    return descendants

def joints_for_body(model, body_id):
    """Return all joint IDs (and names) attached to body_id and its descendants."""
    joint_ids = []
    for b in body_subtree(model, body_id):
        start = model.body_jntadr[b]
        count = model.body_jntnum[b]
        for k in range(count):
            jid = start + k
            joint_ids.append(jid)
    return [(jid, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "") 
            for jid in joint_ids]


def joint_qpos_slice(model, joint_id):
    """Return the slice in data.qpos that corresponds to this joint."""
    adr = model.jnt_qposadr[joint_id]
    jtype = model.jnt_type[joint_id]
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        width = 7
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        width = 4
    else:  # hinge or slide
        width = 1
    return slice(adr, adr + width)


def actuators_for_joint(model, joint_id):
    """
    Return actuator indices that drive any DOF of this joint.
    Uses actuator_trnid -> dof_jntid mapping.
    """
    acts = []
    for a in range(model.nu):
        dof = model.actuator_trnid[a][0]  # target dof id
        if dof >= 0 and model.dof_jntid[dof] == joint_id:
            acts.append(a)
    return acts


def _body_subtree(model, root_body_id):
    """All body ids in the subtree rooted at root_body_id (incl. root)."""
    sub = []
    stack = [root_body_id]
    seen = set(stack)
    while stack:
        b = stack.pop()
        sub.append(b)
        for child in range(model.nbody):
            if model.body_parentid[child] == b and child not in seen:
                seen.add(child)
                stack.append(child)
    return sub


def _joint_widths(model, j):
    """(qpos_width, qvel_width) for joint j."""
    t = model.jnt_type[j]
    if t == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if t == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    # hinge or slide
    return 1, 1


def _joints_in_subtree(model, root_body_id):
    """List joint IDs under the body subtree, ordered by qpos address."""
    jids = []
    for b in _body_subtree(model, root_body_id):
        start = model.body_jntadr[b]
        num   = model.body_jntnum[b]
        for k in range(num):
            jids.append(start + k)
    # stable order in generalized coordinates
    return sorted(jids, key=lambda j: model.jnt_qposadr[j])

# --- main ---------------------------------------------------------------
def get_robot_state(model, data, root_body_id, as_dict=True):
    """
    Return joint positions, velocities, accelerations for all joints in the
    robot subtree identified by root_body_id.

    Returns:
      if as_dict:
        {
          'per_joint': [
              {'id': j, 'name': ..., 'qpos': np.array|float, 'qvel': np.array|float, 'qacc': np.array|float}
              ...
          ],
          'qpos': np.ndarray,  # concatenated in generalized order
          'qvel': np.ndarray,
          'qacc': np.ndarray,
        }
      else:
        (qpos_concat, qvel_concat, qacc_concat)
    """
    per_joint = []
    qpos_chunks, qvel_chunks, qacc_chunks = [], [], []

    for j in _joints_in_subtree(model, root_body_id):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        qpos_w, qvel_w = _joint_widths(model, j)

        # qpos slice
        qa = model.jnt_qposadr[j]
        qpos = data.qpos[qa:qa+qpos_w] if qpos_w > 1 else np.array([data.qpos[qa]])

        # qvel/qacc slice (indexed by dof address)
        da = model.jnt_dofadr[j]
        qvel = data.qvel[da:da+qvel_w] if qvel_w > 1 else np.array([data.qvel[da]])
        qacc = data.qacc[da:da+qvel_w] if qvel_w > 1 else np.array([data.qacc[da]])

        per_joint.append({
            "id": j, "name": name,
            "qpos": qpos.copy(),
            "qvel": qvel.copy(),
            "qacc": qacc.copy(),
        })
        qpos_chunks.append(qpos)
        qvel_chunks.append(qvel)
        qacc_chunks.append(qacc)

    qpos_cat = np.concatenate(qpos_chunks) if qpos_chunks else np.array([])
    qvel_cat = np.concatenate(qvel_chunks) if qvel_chunks else np.array([])
    qacc_cat = np.concatenate(qacc_chunks) if qacc_chunks else np.array([])

    if as_dict:
        return {
            "per_joint": per_joint,
            "qpos": qpos_cat,
            "qvel": qvel_cat,
            "qacc": qacc_cat,
        }
    else:
        return qpos_cat, qvel_cat, qacc_cat


class MujocoRobotDriver(SimRobotDriver):
    """
    MujocoRobotDriver is a robot driver for Mujoco simulations.
    """

    def __init__(
        self,
        component_name=str,
        component_config: Dict[str, Any] = None,
        client: Any = None,
    ) -> None:
        """!Create a robot driver for Mujoco.

        @param component_name Name of the robot component.
        @param component_config Configuration dictionary for the robot.
        @param client Mujoco client instance.
        @return ``None``
        """
        super().__init__(component_name, component_config, True)

        print("Initializing MujocoRobotDriver with config:", component_config)

        self.name = component_name

        class_path = self.config.get("class_path", None)
        mjcf_path = self.config.get("mjcf_path", None)
        if mjcf_path:
            # Append the MJCF path to the class path if provided
            if class_path is not None:
                mjcf_path = Path(class_path) / mjcf_path
            else:
                mjcf_path = Path(mjcf_path)


            # Make the MJCF path absolute if it is not already
            if not mjcf_path.is_absolute():
                mjcf_path = Path(self.config["class_dir"]) / mjcf_path

            # Check if the URDF path exists
            if not mjcf_path.exists():
                log.error(f"The URDF path '{mjcf_path}' does not exist.")
                log.error(f"Full path: {mjcf_path.resolve()}")
                return
            
        position = self.config.get("base_position", [0.0, 0.0, 0.0])
        orientation = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        # convert xyzw to wxyz
        orientation = [orientation[3], orientation[0], orientation[1], orientation[2]]

        use_fixed_base = self.config.get("use_fixed_base", False)
        root_joint_name = f"{self.name}_root"
        qpos = self.config.get("initial_configuration", None)


        print(f"Loading Mujoco robot '{self.name}' with MJCF path: {mjcf_path}")

        client.include_robot(
            name=self.name,
            file=mjcf_path,
            pos=position,  # base position (x, y, z)
            quat=orientation,  # base orientation in WXYZ quaternion
            fixed_base=use_fixed_base,
            root_joint_name=root_joint_name,
            qpos=qpos,
        )

        # # Include the Franka Panda robot with an initial base pose and joint configuration
        # builder.include_robot(
        #     name="panda",
        #     file="franka_emika_panda/panda.xml",
        #     pos=[0.0, 0.0, 0.0],  # base position (x, y, z)
        #     quat=[1, 0, 0, 0],     # base orientation in WXYZ Euler angles
        #     fixed_base=True,
        #     root_joint_name="panda_root",
        #     qpos=[
        #         0.0,
        #         -0.6,
        #         0.0,
        #         -2.2,
        #         0.0,
        #         1.6,
        #         0.8,
        #         0.04,
        #         0.04,
        #     ],
        # )
    
    def update_ids(self, model, data) -> None:
        self.model = model
        self.data = data

        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            addr = model.jnt_qposadr[i]
            print(f"Joint {i}: {name}, qpos index {addr}")

        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        print(f"Robot '{self.name}' ID: {self.id}")

        # get all the joint names:
        self.actuated_joints = {}
        state = get_robot_state(self.model, self.data, self.id, as_dict=True)
        for joint in state["per_joint"]:
            joint_name = joint["name"]
            self.actuated_joints[joint_name] = joint["id"]
            

        print(f"Actuated joints for robot '{self.name}': {self.actuated_joints}")

        joints = joints_for_body(model, self.id)
        print("All joints under robot:")
        for jid, name in joints:
            s = joint_qpos_slice(model, jid)
            acts = actuators_for_joint(model, jid)
            print(f"  {jid:>3}  {name:<24} qpos[{s.start}:{s.stop}]  actuators={acts}")

    def check_torque_status(self) -> bool:
        raise NotImplementedError("MujocoRobotDriver.check_torque_status is not implemented yet.")

    def pass_joint_efforts(self, joints: List[str]) -> Dict[str, float]:
        raise NotImplementedError("MujocoRobotDriver.pass_joint_efforts is not implemented yet.")

    def pass_joint_group_control_cmd(self, control_mode: str, cmd: Dict[str, float], **kwargs) -> None:
        raise NotImplementedError("MujocoRobotDriver.pass_joint_group_control_cmd is not implemented yet.")

    def pass_joint_positions(self, positions: Dict[str, float]) -> Dict[str, float]:
        state = get_robot_state(self.model, self.data, self.id, as_dict=True)
        pos = {}
        for i,joint in enumerate(self.actuated_joints):
            pos[joint] = state["qpos"][i]
        return pos

    def pass_joint_velocities(self, joints: List[str]) -> Dict[str, float]:
        raise NotImplementedError("MujocoRobotDriver.pass_joint_velocities is not implemented yet.")
    
    def sim_reset(self, base_pos: List[float], base_orn: List[float], init_pos: List[float]) -> None:
        raise NotImplementedError("MujocoRobotDriver.sim_reset is not implemented yet.")