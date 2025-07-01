
"""Robot component abstractions used by the ARK framework."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, Tuple, List, Union
from pathlib import Path
import xml.etree.ElementTree as ET
import time

import numpy as np

from ark.tools.log import log
from ark.system.component.base_component import SimToRealComponent
from ark.system.driver.robot_driver import RobotDriver
from arktypes import flag_t, robot_init_t


def robot_joint_control(func):
    """Decorator applying joint group lookup before calling ``func``."""
    def wrapper(self, group_name: str, target: Dict[str, float], **kwargs):
        # Ensure the class instance has 'joint_groups' and the necessary structure
        if not hasattr(self, 'joint_groups'):
            raise AttributeError("The class must have 'joint_groups' attribute.")
        
        if group_name not in self.joint_groups:
            raise ValueError(f"Group name '{group_name}' not found in joint_groups.")
        
        control_mode = self.joint_groups[group_name]["control_mode"]
        actuated_joints = self.joint_groups[group_name]["actuated_joints"]

        if len(target) != len(actuated_joints):
            log.warning(f"Number of targets ({len(target)}) does not equal number of actuated joints ({len(actuated_joints)}) in group '{group_name}'!")
        kwargs['group_name'] = group_name
        # Call the original function with modified arguments
        return func(self, control_mode, list(actuated_joints), target, **kwargs)
    
    return wrapper

def robot_control(func):
    """Decorator forwarding commands for a joint group to ``func``."""
    def wrapper(self, group_name: str, target: Dict[str, float], **kwargs):
        # Ensure the class instance has 'joint_groups' and the necessary structure
        if not hasattr(self, 'joint_groups'):
            raise AttributeError("The class must have 'joint_groups' attribute.")
        
        if group_name not in self.joint_groups:
            raise ValueError(f"Group name '{group_name}' not found in joint_groups.")
        
        control_mode = self.joint_groups[group_name]["control_mode"]
        actuated_joints = self.joint_groups[group_name]["actuated_joints"]
        kwargs['group_name'] = group_name

        # Call the original function with modified arguments
        return func(self, control_mode, list(actuated_joints), target, **kwargs)
    
    return wrapper

class ControlType(Enum):
    POSITION = "position" 
    VELOCITY = "velocity"
    TORQUE = "torque"
    FIXED = "fixed"



class Robot(SimToRealComponent):
    """High level representation of a robot in ARK."""


    def __init__(self, name: str,
                       global_config: Union[str, Dict[str, Any], Path],
                       driver: RobotDriver,
                       ) -> None:
        """Create a robot component.

        @param name  Unique name of the robot.
        @param global_config  Configuration dictionary or file path.
        @param driver  Concrete :class:`RobotDriver` controlling the hardware or simulator.
        """
        super().__init__(name=name, 
                         global_config=global_config,
                         driver=driver)
        self.robot_config = self._load_config_section(global_config=global_config, name=name, type="robots")
        

        self.joint_infos: Dict[str, Any] = {} # from urdf
        # {"name" : {"index"          : ... ,
        #            "type"           : ... ,
        #            "actuated"       : ... ,
        #            "parent_link"    : ... ,
        #            "child_link"     : ... ,
        #            "lower_limit"    : ... ,
        #            "upper_limit"    : ... ,
        #            "effort_limit"   : ... ,
        #            "velocity_limit" : ... ,
        #            }
        # }

        self.joint_groups: Dict[str, Any] = {} # from urdf
        # { "name" : { "control_mode"    : ... ,
        #              "joints"          : { "name"  : idx,
        #                                    "name"  : idx,
        #                                     ...  
        #                                  },
        #              "actuated_joints" : { "name" : idx
        #                                  },
        #              "end_effector"    : { "ee_name" : idx_ee},
        #                                  }
        #              }

        self._all_actuated_joints = []
        # { "name" : idx,
        #   ...
        # }


        self.initial_configuration = {} # from self.robot_config["initial_configuration"]
        # { "name" : float,
        #   ...
        # }   
        if self.robot_config.get("urdf_path", None) and self.robot_config.get("mjcf_path", None):
            log.error(f"Both 'urdf_path' and 'mjcf_path' are provided for robot '{self.name}'. Please provide only one of them.")
        elif self.robot_config.get("urdf_path", None):
            class_path = self.robot_config.get("class_dir", None)
            urdf_path = self.robot_config["urdf_path"]
            if class_path is None:
                urdf_path = Path(class_path) / urdf_path
            else:
                urdf_path = Path(urdf_path)

            # Make the URDF path absolute if it is not already
            if not urdf_path.is_absolute():
                urdf_path = Path(class_path) / urdf_path

            # Check if the URDF path exists
            if not urdf_path.exists():
                log.error(f"The URDF path '{urdf_path}' does not exist.")
                return

            tree = ET.parse(urdf_path)
            root = tree.getroot()
            elements = root.findall('joint')
        elif self.robot_config.get("mjcf_path", None):
            tree = ET.parse(self.robot_config["mjcf_path"])
            root = tree.getroot()
            elements = root.findall('.//joint')
        
        print(f"Robot '{self.name}' has the following elements (Total: {len(elements)}):")
        print(f"{'Index':<8} {'Joint Name':<20} {'Type':<10}")
        print("=" * 40)  # Separator
        # Iterate over all joints and collect relevant info
        for i, joint in enumerate(elements):
            name = joint.get("name")
            joint_info = {}
            joint_info['index'] = i
            joint_info['type'] = joint.get('type')
            if not joint_info['type'] == "fixed":
                self._all_actuated_joints.append(name)
                joint_info["actuated"] = True
            else:
                joint_info["actuated"] = False

            # Get the parent and child link names (URDF and MJCF have different structures here)
            if self.robot_config.get("urdf_path", None):  # URDF case
                joint_info['parent Link'] = joint.find('parent').get('link')
                joint_info['child Link'] = joint.find('child').get('link')
            elif self.robot_config.get("mjcf_path", None):  # MJCF case
                # In MJCF, 'parent' and 'child' are typically within bodies, we use a different approach
                joint_info['parent Link'] = joint.attrib.get('parent', 'N/A')
                joint_info['child Link'] = joint.attrib.get('child', 'N/A')

            # If joint has limits (revolute or prismatic joints), get the limits
            if joint_info['type'] in ['revolute', 'prismatic']:
                limit = joint.find('limit')
                if limit is not None:
                    joint_info['lower_limit'] = limit.get('lower', None)
                    joint_info['upper_limit'] = limit.get('upper', None)
                    joint_info['effort_limit'] = limit.get('effort', None)
                    joint_info['velocity_limit'] = limit.get('velocity', None)
                else:
                    joint_info['limits'] = 'No limits defined'
            self.joint_infos[joint.get('name')] = joint_info
            # save dict of iniital cofngiruation of joints
            self.initial_configuration[joint.get('name')] = self.robot_config["initial_configuration"][i] 
            #print(f"{i:<8} {name:<20} {joint_info['type']:<10}")
            # Print the joint summary
            print(f"{i:<8} {name:<20} {joint_info['type']:<10}")
            print(f"   Parent Link: {joint_info['parent Link']}")
            print(f"   Child Link: {joint_info['child Link']}")
            if 'lower_limit' in joint_info:
                print(f"   Limits: {joint_info['lower_limit']} to {joint_info['upper_limit']}, "
                    f"Effort: {joint_info['effort_limit']}, Velocity: {joint_info['velocity_limit']}")
            else:
                print(f"   Limits: {joint_info.get('limits', 'None')}")
            print("-" * 40)  # Divider for each joint summary
  
        
        # check if joint group is defined:
        if "joint_groups" in self.robot_config:
            for group_name, group_config in self.robot_config.get("joint_groups", {}).items():
                # add control type from enum to internal config dict
                control_mode = group_config.get("control_mode", {})
                if control_mode == "position":
                    group_config["control_type"] = ControlType.POSITION
                elif control_mode == "velocity":
                    group_config["control_type"] = ControlType.VELOCITY
                elif control_mode == "torque":
                    group_config["control_type"] = ControlType.TORQUE
                elif control_mode == "fixed":
                    group_config["control_type"] = ControlType.FIXED
                    # TODO
                    raise NotImplementedError("TODO - how to manually fix a joint")
                else:
                    raise ValueError(
                        f"control mode '{control_mode}' is not supported"
                    )
                joints = {}
                actuated_joints = {}
                for joint in group_config["joints"]:
                    joint_idx = self.joint_infos[joint]["index"]
                    joints[joint] = joint_idx # {"joint name": joint index}
                    if self.joint_infos[joint]["actuated"]:
                        actuated_joints[joint] = joint_idx
                group_config["joints"] = joints
                group_config["actuated_joints"] = actuated_joints

                # # same for end effector
                # ee = group_config.get("end_effector", "None")# if not provided
                # assert isinstance(ee, str), "end_effector must be either None or a single value joint name."
                # if ee == "None": # if explicitly set to None
                #     ee = None
                
                # group_config["end_effector"] = None
                # if ee is not None:
                #     ee_idx = self.joint_infos[ee]["index"] # idx
                #     group_config["end_effector"] = {ee: ee_idx}
                self.joint_groups[group_name] = group_config  
        else:
            log.warning(f"Using Default Joint Group all in Position Control '{self.name}' !")
            group_config = {}
            group_config["control_mode"] = "position"
            group_config["control_type"] = ControlType.POSITION
            joints = {}
            actuated_joints = {}
            for joint in self.joint_infos.keys():
                joint_idx = self.joint_infos[joint]["index"]
                joints[joint] = joint_idx
                if self.joint_infos[joint]["actuated"]:
                    actuated_joints[joint] = joint_idx
            group_config["joints"] = joints
            group_config["actuated_joints"] = actuated_joints
            self.joint_groups["all"] = group_config

        self.create_service(self.reset_service_name, robot_init_t, flag_t, self.reset_component)
        
        if not self.sim:
            # runs if the robot is real
            try:
                self.freq = self.robot_config["frequency"]
            except:
                log.warning(f"No frequency provided for robot '{self.name}', using default 240Hz !")
                self.freq = 240
            self.create_stepper(self.freq, self.step_component)
            self.create_stepper(self.freq, self.control_robot)
            
        print(self._all_actuated_joints)
    
    @abstractmethod
    def control_robot(self) -> None:
        """Send the currently stored command to the robot driver."""
        print("No call")

    @abstractmethod
    def pack_data(self) -> None:
        """Pack state information for publishing to the client."""

    @abstractmethod
    def get_state(self) -> Any:
        """Retrieve the current robot state from the driver."""

    


    #####################
    ##    get infos    ##
    #####################
    
    def get_joint_limits(self) -> Dict[str, Dict[str, float]]:
        """Return joint limits for all actuated joints."""
        actuated_info = {joint: info for joint, info in self.joint_infos.items() if info["actuated"]}
        return {joint: {"lower_limit": float(info.get("lower_limit", "inf")),
                        "upper_limit": float(info.get("upper_limit", "inf")),
                        "effort_limit": float(info.get("effort_limit", "inf")),
                        "velocity_limit": float(info.get("velocity_limit", "inf"))}
                for joint, info in actuated_info.items()}

    def _get_joint_group_indices(self, joint_group: str) -> Tuple[List[float], List[float]]:
        """Return joint and actuated joint indices for ``joint_group``."""
        return list(self.joint_groups[joint_group]["joints"].values()), list(self.joint_groups[joint_group]["actuated_joints"].values())

    def is_torqued(self) -> bool:
        """Check if the driver currently outputs torque."""
        return self._driver.check_torque_status()

    def get_joint_positions(self) -> Dict[str, float]:
        """Get positions of all actuated joints."""
        return self._driver.pass_joint_positions(self._all_actuated_joints)
        # return self._driver.pass_joint_positions(self._all_actuated_joints)

    def get_joint_velocities(self) -> Dict[str, float]:
        """Get velocities of all actuated joints."""
        return self._driver.pass_joint_velocities(self._all_actuated_joints)

    def get_joint_efforts(self) -> Dict[str, float]:
        """Get efforts of all actuated joints."""
        return self._driver.pass_joint_efforts(self._all_actuated_joints)

    def get_joint_group_positions(self, joint_group: str) -> Dict[str, float]:
        """Get joint positions for a specific group."""
        actuated_joints = self.joint_groups[joint_group]["actuated_joints"]
        return self._driver.pass_joint_positions(actuated_joints)

    def get_joint_group_velocities(self, joint_group: str) -> Dict[str, float]:
        """Get joint velocities for a specific group."""
        actuated_joints = self.joint_groups[joint_group]["actuated_joints"]
        return self._driver.pass_joint_velocities(actuated_joints)

    def get_joint_group_efforts(self, joint_group: str) -> Dict[str, float]:
        """Get joint efforts for a specific group."""
        actuated_joints = self.joint_groups[joint_group]["actuated_joints"]
        return self._driver.pass_joint_efforts(actuated_joints)


    #####################
    ##     control     ##
    #####################
    def control_joint_group(self, control_mode: str, cmd: Dict[str, float], **kwargs) -> None:
        """Forward a joint group command to the driver."""
        self._driver.pass_joint_group_control_cmd(control_mode, cmd, **kwargs)

    #####################
    ##      misc.      ##
    #####################
    
    def reset_component(self, channel=None, msg=None) -> flag_t:
        """Reset the robot to its initial configuration."""
        print("RESET HAS BEEN CALLED")
        self.suspend_communications(services=False)  # Suspend communications to avoid conflicts during reset
        self._is_suspended = True

        
        # # TODO
        # IDEA seperate reset iinto sim and real reset, make user implement real_reset for each robot ?
        if not msg:
            new_pos = self.robot_config["base_position"]
            new_orn = self.robot_config["base_orientation"]
            q_init = self.robot_config["initial_configuration"]
        else:
            new_pos = np.array(msg.position)
            new_orn = np.array(msg.orientation)
            q_init = np.array(msg.q_init)

            nbr_actuated = len(self._all_actuated_joints)

            nbr_init_pos = len(self.robot_config["initial_configuration"])

            if q_init.size == len(self._all_actuated_joints):
                idx = np.linspace(0, q_init.size-1, q_init.size, dtype=np.uint8)
                temp = np.zeros(nbr_init_pos)
                temp[idx] = q_init
                q_init = temp.copy()

            if q_init.size != nbr_init_pos:
                log.error(f"Number of initial positions ({q_init.size}) does not match number of joints ({nbr_init_pos}) or actuated joints ({nbr_actuated}) for robot {self.name}!")
        
        if not self.sim:
            usr_input = input("Start moving robot back to initial position [yes/no] ?")
            if usr_input == "yes":
                log.panda("if this appears big issue")
                raise NotImplementedError("TODO - how to reset a real robot ?")    
        elif self.sim:
            self._driver.sim_reset(base_pos=new_pos, 
                                   base_orn=new_orn, 
                                   q_init=list(q_init))
        self.resume_communications(services=False)
        self._is_suspended = False
        return flag_t()
    
    
    
    def step_component(self):
        """Query state and publish it to the configured channels."""
        data = self.get_state()
        packed = self.pack_data(data)
        self.component_multi_publisher.publish(packed)