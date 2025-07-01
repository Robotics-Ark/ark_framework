
from lcm import LCM
import time
import threading
from ark.client.frequencies.stepper import Stepper
from ark.client.comm_handler.publisher import Publisher
from ark.client.comm_handler.multi_comm_handler import MultiCommHandler
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from ark.tools.log import log

class MultiChannelPublisher(MultiCommHandler):
    """!
    Publisher that manages multiple communication channels.

    @note Internally creates one :class:`Publisher` per channel.
    """

    def __init__(self, channels: Dict[str,type], lcm_instance: LCM) -> None:
        """!
        Initialize the publisher with a list of channels.

        @param channels: Dictionary mapping channel names to their types.
        @type channels: Dict[str, type]
        @param lcm_instance: LCM instance used for publishing.
        """
        
        super().__init__()
        
        
        self.comm_type = "Multi Channel Publisher"
        # iterate through the channels dictionary
        for channel_name, channel_type in channels.items():
            publisher = Publisher(lcm_instance, channel_name, channel_type)
            self._comm_handlers.append(publisher)

    def publish(self, messages_to_publish: Dict[str, Any]) -> None:
        """!
        Publish messages to their respective channels.

        @param messages_to_publish: Mapping of channel names to messages.
        """
        for publisher in self._comm_handlers:
            channel_name = publisher.channel_name
            channel_type = publisher.channel_type
            try: 
                if channel_name not in messages_to_publish:
                    # log.warning(f"Channel '{channel_name}' not found in messages to publish.")
                    continue
                message = messages_to_publish[channel_name]

                if not isinstance(message, channel_type):
                    raise TypeError(
                        f"Incorrect message type for channel '{channel_name}'. "
                        f"Expected {channel_type}, got {type(message)}."
                    )

                publisher.publish(message)
                # log.info(f"Message Published for channel '{channel_name}'.")
            except: 
                log.warning(f"Error Occured when publishing on channel '{channel_name}'.")
                pass