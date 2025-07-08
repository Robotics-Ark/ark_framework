"""@file pybullet_robot_driver.py
@brief Robot driver handling PyBullet specific commands.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List
import os
import pybullet as p
from pathlib import Path

from ark.tools.log import log
from ark.system.driver.robot_driver import SimRobotDriver, ControlType

# for pybullet setJointMotorControlArray optional arguments
motor_control_kwarg = {
    "position" : "targetPositions",
    "velocity" : "targetVelocities",
    "torque"   : "forces", 
}

class BulletRobotDriver(SimRobotDriver):
    """Robot driver that interfaces with the PyBullet simulation."""

    def __init__(self,
                 component_name = str,
                 component_config: Dict[str, Any] = None,
                 client: Any = None,
                 ) -> None:
        """!Create a robot driver for PyBullet.

        @param component_name Name of the robot component.
        @param component_config Configuration dictionary for the robot.
        @param client Bullet client instance.
        @return ``None``
        """
        super().__init__(component_name, component_config, True)

        self.client = client
        
        self.base_position = self.config.get("base_position", [0.0, 0.0, 0.0])
        self.base_orientation = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])
        if len(self.base_orientation) == 3:
            self.base_orientation = p.getQuaternionFromEuler(self.base_orientation)
        
        self.load_robot(self.base_position, self.base_orientation, None)
        self.initial_configuration = self.config.get("initial_configuration", [0.0] * self.client.getNumJoints(self.ref_body_id))

        self.num_joints = self.client.getNumJoints(self.ref_body_id)
        self.bullet_joint_infos = {}
        # {"name" : {"index"             : ... ,
        #            "type"              : ... ,
        #            "actuated"          : ... ,
        #            "parent_link"       : ... ,
        #            "child_link"        : ... ,
        #            "lower_limit"       : ... ,
        #            "upper_limit"       : ... ,
        #            "effort_limit"      : ... ,
        #            "velocity_limit"    : ... ,
        # 
        #            "joint_axis"        : ... ,  # PyBullet specific
        #            "joint_parent_index": ... ,  # PyBullet specific
        #            "joint_child_index" : ... ,  # PyBullet specific 
        #            }
        # }
        
        self.actuated_joints = {}
        self.joints = {}
        # {"name" : index}

        for joint_index in range(self.num_joints):
            # extract joint information
            joint_info = self.client.getJointInfo(self.ref_body_id, joint_index)
            # (jointIndex, jointName, jointType, jointAxis, jointLowerLimit, jointUpperLimit,
            # jointMaxForce, jointMaxVelocity, linkName, jointType, jointParentindex, jointChildIndex)
            joint_name = joint_info[1].decode("utf-8")
            self.joints[joint_name] = joint_index
            self.bullet_joint_infos[joint_name] = {}
            self.bullet_joint_infos[joint_name]["index"] = joint_index
            self.bullet_joint_infos[joint_name]["type"] = joint_info[2]
            if self.bullet_joint_infos[joint_name]["type"] == 4:
                self.bullet_joint_infos[joint_name]["actuated"] = False
            else:
                self.bullet_joint_infos[joint_name]["actuated"] = True
                self.actuated_joints[joint_name] = joint_index
            self.bullet_joint_infos[joint_name]["parent_link"] = None
            self.bullet_joint_infos[joint_name]["child_link"] = None
            self.bullet_joint_infos[joint_name]["lower_limit"] = joint_info[4]
            self.bullet_joint_infos[joint_name]["upper_limit"] = joint_info[5]
            self.bullet_joint_infos[joint_name]["effort_limit"] = joint_info[6]
            self.bullet_joint_infos[joint_name]["velocity_limit"] = joint_info[7]

            self.bullet_joint_infos[joint_name]["joint_axis"] = joint_info[3]
            self.bullet_joint_infos[joint_name]["joint_parent_index"] = joint_info[10]
            self.bullet_joint_infos[joint_name]["joint_child_index"] = joint_info[11]
    
        self.sim_reset(base_pos=self.base_position, 
                       base_orn=self.base_orientation, 
                       q_init=self.initial_configuration)
        
        # PyBullet specific : extract and save joint group information to handle torque control
        torque_control_groups = {}
        
        for group_name, group_config in self.config.get("joint_groups", {}).items():
            # add control type from enum to internal config dict

            if group_config["control_mode"] == self.client.TORQUE_CONTROL:
                force_limit = group_config.get("force_limit", 0.0)
                torque_control_groups[group_name] = {}
                torque_control_groups[group_name]["force_limit"] = force_limit
                torque_control_groups[group_name]["indices"] = []
                for joint in group_config["joints"]:
                    joint_idx = self.bullet_joint_infos[joint]["index"]
                    torque_control_groups[group_name]["indices"].append(joint_idx)
        
        # Setup torque control
        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12644
        for group_name, group_data in torque_control_groups.items():
            joint_indices = group_data["indices"]
            force_limit = group_data["force_limit"]
            self.client.setJointMotorControlArray(
                self.ref_body_id,
                joint_indices,
                self.client.VELOCITY_CONTROL,
                forces=[force_limit] * len(joint_indices),
            )
       

    def load_robot(self, base_position = None, base_orientation = None, q_init=None) -> None:
        """!Load the robot model into the simulator.

        @param base_position Optional base position ``[x, y, z]``.
        @param base_orientation Optional base orientation as quaternion.
        @param q_init Optional list of initial joint positions.
        """
        kwargs = {}

        kwargs["useFixedBase"] = self.config.get("use_fixed_base", 1)

        if self.config.get("merge_fixed_links", True):
            kwargs["flags"] = p.URDF_MERGE_FIXED_LINKS

        if base_position is not None:
            kwargs["basePosition"] = base_position
        else:
            kwargs["basePosition"] = self.config.get("base_position", [0.0, 0.0, 0.0])

        if base_orientation is not None:
            kwargs["baseOrientation"] = base_orientation
        else:
            kwargs["baseOrientation"] = self.config.get("base_orientation", [0.0, 0.0, 0.0, 1.0])

            

        urdf_path = self.config.get("urdf_path", None)
        mjcf_path = self.config.get("mjcf_path", None)
        class_path = self.config.get("class_path", None)
        if mjcf_path and urdf_path:
            log.error("Both urdf and mjcf paths are provided. Please provide only one.")
            return
        elif mjcf_path:
            self.ref_body_id = self.client.loadMJCF(mjcf_path)[0]
            log.ok("Initialized robot specified by mjcf " + mjcf_path + " in PyBullet simulator.")
        elif urdf_path:
        # Append the URDF path to the class path if provided
            if class_path is not None:
                urdf_path = Path(class_path) / urdf_path
            else:
                urdf_path = Path(urdf_path)

            # Make the URDF path absolute if it is not already
            if not urdf_path.is_absolute():
                urdf_path = Path(self.config["class_dir"]) / urdf_path

            # Check if the URDF path exists
            if not urdf_path.exists():
                log.error(f"The URDF path '{urdf_path}' does not exist.")
                log.error(f"Full path: {urdf_path.resolve()}")
                # print the full path for debugging
                
                return

            # Load the URDF into the simulator
            self.ref_body_id = self.client.loadURDF(str(urdf_path), **kwargs)
            log.ok(f"Initialized robot specified by URDF '{urdf_path}' in PyBullet simulator.")
        

        if q_init is not None:
            for joint in range(self.client.getNumJoints(self.ref_body_id)):
                self.client.resetJointState(self.ref_body_id, joint, q_init[joint], 0.0)
        
        
    #####################
    ##    get infos    ##
    #####################

    def check_torque_status(self) -> bool:
        """!Return ``True`` as simulated robots are always torqued.

        @return Always ``True`` in simulation.
        @rtype bool
        """
        return True  # simulated robot is always torqued in bullet

    def pass_joint_positions(self, joints: List[str]) -> Dict[str, float]:
        """!Return the current joint positions.

        @param joints List of joint names.
        @return Dictionary from joint name to position.
        @rtype Dict[str, float]
        """
        pos = {}
        idx = [self.actuated_joints[joint] for joint in joints]
        # Iterate over each joint index and corresponding joint state to fill dictionaries
        for name, idx in zip(joints, idx):
            state = self.client.getJointState(self.ref_body_id, idx)
            pos[name] = state[0]  # Joint position
        return pos

    def pass_joint_velocities(self, joints: List[str]) -> Dict[str, float]:
        """!Return the current joint velocities.

        @param joints List of joint names.
        @return Dictionary from joint name to velocity.
        @rtype Dict[str, float]
        """
        vel = {}
        idx = [self.actuated_joints[joint] for joint in joints]
        # Iterate over each joint index and corresponding joint state to fill dictionaries
        for name, idx in zip(joints, idx):
            state = self.client.getJointState(self.ref_body_id, idx)
            vel[name] = state[1]  # Joint velocity
        return vel

    def pass_joint_efforts(self, joints: List[str]) -> Dict[str, float]:
        """!Return the current joint efforts.

        @param joints List of joint names.
        @return Dictionary from joint name to effort.
        @rtype Dict[str, float]
        """
        eff = {}
        idx = [self.actuated_joints[joint] for joint in joints]
        # Iterate over each joint index and corresponding joint state to fill dictionaries
        for name, idx in zip(joints, idx):
            state = self.client.getJointState(self.ref_body_id, idx)
            eff[name] = state[3]  # Joint applied force (effort)
        return eff

    #####################
    ##     control     ##
    #####################

    def pass_joint_group_control_cmd(self, control_mode: str, cmd: Dict[str, float], **kwargs) -> None:
        """!Send a control command to a group of joints.

        @param control_mode One of ``position``, ``velocity`` or ``torque``.
        @param cmd Mapping from joint names to command values.
        @param kwargs Additional keyword arguments forwarded to PyBullet.
        @return ``None``
        """
        idx = [self.actuated_joints[joint] for joint in cmd.keys()]
        
        kwargs = {motor_control_kwarg[control_mode]: list(cmd.values())}
        if control_mode == ControlType.POSITION.value:
            control_mode = p.POSITION_CONTROL
        elif control_mode == ControlType.VELOCITY.value:
            control_mode = p.VELOCITY_CONTROL
        elif control_mode == ControlType.TORQUE.value:
            control_mode = p.TORQUE_CONTROL
        else:
            log.error("Invalid control mode. Please use 'position', 'velocity', or 'torque', but received: " + control_mode)

        self.client.setJointMotorControlArray(
            bodyUniqueId=self.ref_body_id,
            jointIndices=idx,
            controlMode=control_mode,
            **kwargs,
        )
    
    #####################
    ##      misc.      ##
    #####################

    def sim_reset(self,
                  base_pos : List[float],
                  base_orn : List[float],
                  q_init : List[float]) -> None:
        """!Reset the robot in the simulator.

        @param base_pos New base position.
        @param base_orn New base orientation quaternion.
        @param q_init List of joint positions after the reset.
        """
        # delete the robot
        self.client.removeBody(self.ref_body_id)
        self.load_robot(base_position=base_pos, base_orientation=base_orn, q_init=q_init)

        log.ok("Reset robot " + self.component_name + " completed.")

        # print the joint positons after reset
        joint_positions = self.pass_joint_positions(list(self.actuated_joints.keys()))
        log.info("Joint positions after reset: " + str(joint_positions))
        return 

