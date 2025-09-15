from typing import Optional, Dict, Any
from control_msgs.msg import JointTrajectoryControllerState
from arktypes import joint_group_command_t
from ark.tools.ros_bridge.ark_ros2_bridge import ArkRos2Bridge


class MoveIt2Bridge(ArkRos2Bridge):
    """Bridge ROS2 JointTrajectoryControllerState -> Ark joint_group_command_t."""

    def __init__(
        self,
        ros_controller: str,
        ark_robot: str,
        mapping_table: Optional[Dict[str, Any]] = None,
        sim: bool = False,
    ):
        # Build topic/channel names
        if sim:
            ros_topic = f"/{ros_controller}_controller/state"
            ark_channel = f"{ark_robot}/joint_group_command/sim"
        else:
            ros_topic = f"/{ros_controller}_controller/state"
            ark_channel = f"{ark_robot}/joint_group_command"

        # Base mapping (MoveIt state -> Ark command)
        moveit_mapping_table = {
            "ros2_to_ark": [
                {
                    "ros2_channel": ros_topic,
                    "ros2_type": JointTrajectoryControllerState,
                    "ark_channel": ark_channel,
                    "ark_type": joint_group_command_t,
                    "translator_callback": self.moveit_translator,
                }
            ],
            "ark_to_ros": [],
        }

        # Merge in extra mappings if provided
        if mapping_table:
            moveit_mapping_table["ros2_to_ark"].extend(mapping_table.get("ros2_to_ark", []))
            moveit_mapping_table["ark_to_ros"].extend(mapping_table.get("ark_to_ros", []))

        # Init parent with the final mapping
        super().__init__(moveit_mapping_table)

    def moveit_translator(self, ros_msg, ros_channel, ros_type, ark_channel, ark_type):
        """Convert joint state positions into Ark command."""
        msg = ark_type()
        msg.name = "arm"
        msg.n = len(ros_msg.actual.positions)
        msg.cmd = list(ros_msg.actual.positions)
        return msg
