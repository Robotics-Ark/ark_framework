from typing import Optional, Any
from control_msgs.msg import JointTrajectoryControllerState
from arktypes import joint_group_command_t
from ark.tools.ros_bridge.ark_ros2_bridge import ArkRos2Bridge
import os
import yaml


class MoveIt2Bridge(ArkRos2Bridge):
    """Bridge ROS2 JointTrajectoryControllerState -> Ark joint_group_command_t."""

    def __init__(
        self,
        ros_controller: str,
        ark_robot_name: str,
        mapping_table: Optional[dict[str, Any]] = None,
        global_config: Optional[dict[str, Any]] = None,
    ):
        sim = self.is_sim_enabled(global_config=global_config)

        # Build topic/channel names
        if sim:
            ros_topic = f"/{ros_controller}_controller/state"
            ark_channel = f"{ark_robot_name}/joint_group_command/sim"
        else:
            ros_topic = f"/{ros_controller}_controller/state"
            ark_channel = f"{ark_robot_name}/joint_group_command"

        # Base mapping (MoveIt2 state -> Ark command)
        moveit2_mapping_table = {
            "ros2_to_ark": [
                {
                    "ros2_channel": ros_topic,
                    "ros2_type": JointTrajectoryControllerState,
                    "ark_channel": ark_channel,
                    "ark_type": joint_group_command_t,
                    "translator_callback": self.moveit2_translator,
                }
            ],
            "ark_to_ros": [],
        }

        # Merge in extra mappings if provided
        if mapping_table:
            moveit2_mapping_table["ros2_to_ark"].extend(mapping_table.get("ros2_to_ark", []))
            moveit2_mapping_table["ark_to_ros"].extend(mapping_table.get("ark_to_ros", []))
            
        # Init parent with the final mapping
        super().__init__(mapping_table=moveit2_mapping_table, global_config=global_config)

    def moveit2_translator(self, ros_msg: JointTrajectoryControllerState, ros_channel: str, ros_type: type[JointTrajectoryControllerState], ark_channel: str, ark_type: type[joint_group_command_t]):
        """Convert joint state positions into Ark command."""
        msg = joint_group_command_t()
        msg.name = "arm"
        msg.n = len(ros_msg.actual.positions)
        msg.cmd = list(ros_msg.actual.positions)
        return msg
    
    def is_sim_enabled(self, global_config: Any) -> Optional[bool]:
        """
        Check if the key 'sim' is True or False in a dict or YAML file.

        Args:
            global_config (Any): Global configuration dictionary or YAML file path.

        Returns:
            bool | None: True/False if 'sim' key exists, None if missing.
        """
        data = None

        if isinstance(global_config, dict):
            data = global_config
        elif isinstance(global_config, str) and os.path.isfile(global_config):
            with open(global_config, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Source must be a dict or a valid YAML file path.")

        if not isinstance(data, dict):
            raise ValueError("YAML/Dict must represent a dictionary at top-level.")

        return data.get("sim", None)
