
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
    """
    A publisher that manages multiple communication channels using LCM (Lightweight Communications and Marshalling).

    This class initializes multiple publishers for different channels and handles the publishing of messages
    to these channels. It supports dynamic channel management by accepting a list of channel names and their
    corresponding message types during initialization.

    Attributes:
        _lcm (lcm.LCM): The LCM instance used for publishing messages.
        data (dict): A dictionary to store data for each channel.
        _comm_handlers (list of CommHandler): A list of communication handlers for each channel.
    """

    def __init__(self, channels: List, lcm_instance: LCM) -> None:
        """
        Initializes the MultiChannelPublisher with specified channels.
        """
        
        super().__init__()
        
        
        self.comm_type = "Multi Channel Publisher"
        for channel_name, channel_type in channels:
            publisher = Publisher(lcm_instance, channel_name, channel_type)
            self._comm_handlers.append(publisher)

    def publish(self, messages_to_publish: Dict[str, Any]) -> None:
        """
        Publishes messages to their respective channels.

        This method iterates over all registered communication handlers and publishes the corresponding
        message to each channel. It expects a dictionary where keys are channel names and values are the
        messages to be published.

        """
        for publisher in self._comm_handlers:
            channel_name = publisher.channel_name
            channel_type = publisher.channel_type
            try: 
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