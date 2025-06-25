
import yaml
from ark.tools.log import log
from functools import partial

from typing import Dict, Any, Optional
from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import string_t

import rospy

__doc__ = (
    """ARK to ROS translator"""
)

class ArkRosBridge(BaseNode):
    def __init__(self, mapping_table, node_name="ARK_ROS_Bridge", global_config=None):
        super().__init__(node_name, global_config=global_config)
        self.ros_to_ark_mapping = []

        ros_to_ark_table = mapping_table["ros_to_ark"]
        ark_to_ros_table = mapping_table["ark_to_ros"]

        for mapping in ros_to_ark_table: 
            ros_channel = mapping["ros_channel"]
            ros_type = mapping["ros_type"]
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            translator_callback = mapping["translator_callback"]

            publisher = self.create_publisher(ark_channel, ark_type)
            
            modified_callback = partial(self._generic_ros_to_ark_translator_callback, 
                                        translator_callback=translator_callback, 
                                        ros_channel=ros_channel,
                                        ros_type=ros_type,
                                        ark_channel=ark_channel,
                                        ark_type=ark_type,
                                        publisher=publisher)
            
            rospy.Subscriber(ros_channel, ros_type, modified_callback)

            ros_to_ark_map = {
                "ros_channel": ros_channel,
                "ros_type": ros_type,
                "ark_channel": ark_channel,
                "ark_type": ark_type,
                "translator_callback": translator_callback,
                "publisher": publisher
            }
            self.ros_to_ark_mapping.append(ros_to_ark_map)

        
        self.ark_to_ros_mapping = []

        for mapping in ark_to_ros_table: 
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            ros_channel = mapping["ros_channel"]
            ros_type = mapping["ros_type"]
            translator_callback = mapping["translator_callback"]

            # Create a listener
            publisher = rospy.Publisher(ros_channel, ros_type, queue_size=10)

            modified_callback = partial(self._generic_ark_to_ros_translator_callback,
                            translator_callback=translator_callback, 
                            ark_channel=ark_channel,
                            ark_type=ark_type,
                            ros_channel=ros_channel,
                            ros_type=ros_type,
                            publisher=publisher
                            )

            # Create a ROS publisher
            self.create_subscriber(ark_channel, ark_type, modified_callback)

            ros_to_ark_map = {
                "ros_channel": ros_channel,
                "ros_type": ros_type,
                "ark_channel": ark_channel,
                "ark_type": ark_type,
                "translator_callback": translator_callback,
                "publisher": publisher
            }
            self.ark_to_ros_mapping.append(ros_to_ark_map)


        # Create a minimal ROS node
        rospy.init_node(node_name, anonymous=True)

    def _generic_ros_to_ark_translator_callback(self, ros_msg, translator_callback, ros_channel, ros_type, ark_channel, ark_type, publisher):
        """
        This is the modified callback that includes the ROS channel and ark publisher.
        """
        ark_msg = translator_callback(ros_msg, ros_channel, ros_type, ark_channel, ark_type)
        publisher.publish(ark_msg)
            
    
    def _generic_ark_to_ros_translator_callback(self, t, _, ark_msg,  translator_callback, ark_channel, ark_type, ros_channel, ros_type, publisher):
        """
        This is the modified callback that includes the ark channel and ROS publisher.
        """
        ros_msg = translator_callback(t, ark_channel, ark_msg)
        publisher.publish(ros_msg)
    

    def spin(self) -> None:
        """!
        Runs the node’s main loop, handling ark messages continuously until the node is finished.

        The loop calls `self._ark.handle()` to process incoming messages. If an OSError is encountered,
        the loop will stop and the node will shut down.
        """
        while not self._done and not rospy.is_shutdown():
            try:
                self._lcm.handle_timeout(0)
                # rospy.spin()
            except OSError as e:
                log.warning(f"Ark or ROS threw OSError {e}")
                self._done = True
    
    
    @staticmethod
    def get_cli_doc():
        return __doc__
    
    def shutdown(self) -> None:
        """!
        Shuts down the node by stopping all communication handlers and steppers.

        Iterates through all registered communication handlers and steppers, shutting them down.
        """
        for ch in self._comm_handlers:
            ch.shutdown()
        for s in self._steppers:
            s.shutdown()            
        