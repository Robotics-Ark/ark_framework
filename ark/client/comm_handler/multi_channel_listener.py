
import lcm 
from lcm import LCM
import time
import threading
from ark.client.frequencies.stepper import Stepper
from ark.client.comm_handler.listener import Listener
from ark.client.comm_handler.subscriber import Subscriber
from ark.client.comm_handler.multi_comm_handler import MultiCommHandler
from typing import List
import copy

class MultiChannelListener(MultiCommHandler):
    def __init__(
        self, channels: dict[str,type], lcm_instance: LCM) -> None:
        """!
        Initialize listeners for multiple channels.

        @param channels: List of ``(channel_name, channel_type)`` tuples.
        @param lcm_instance: LCM instance used for communication.
        """

        super().__init__()

        self.data = {}
        self.blank_data = {}
        self.comm_type = "Multi Channel Listener"

        for channel_name, channel_type in channels.items():
            listener = Listener(lcm_instance, channel_name, channel_type)
            self.channel_data[channel_name] = None
            self.blank_data[channel_name] = None
            self._comm_handlers.append(listener)

    def get(self):
        """!
        Retrieves the current observation from the space.

        @return: The current observation.
        @rtype: Any
        """
        
        # get all the data
        for listener in self._comm_handlers:
            listener_message = listener.get()
            self.data[listener.channel_name] = listener_message
            
        # return it
        return self.data
    
    def empty_data(self):
        """!
        Empties the data dictionary.
        """
        self.data = copy.deepcopy(self.blank_data)
        for listener in self._comm_handlers:
            listener.empty_data()
