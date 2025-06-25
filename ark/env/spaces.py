
from lcm import LCM
from abc import ABC, abstractmethod


from typing import Any, Tuple, List, Dict, Callable
import numpy as np
from numpy.typing import NDArray
import time

from ark.client.comm_handler.multi_channel_listener import MultiChannelListener
from ark.client.comm_handler.multi_channel_publisher import MultiChannelPublisher

from ark.tools.log import log


class Space(ABC):
    """!
    An abstract base class for different types of spaces. This is used as a generic
    interface for both action and observation spaces in the system.

    The `Space` class provides the general structure for subclasses to define space-specific
    shutdown behavior.
    """

    # @abstractmethod
    def shutdown(self) -> None:
        """!
        Abstract method to shut down the space, ensuring any resources are released.

        Subclasses must implement this method to cleanly shut down their specific space.
        """

class ActionSpace(Space):
    """!
    A class representing a space where actions are taken. This space handles publishing
    of actions to a given LCM channel.

    @param action channels: Channel names where actions will be published.
    @type action_channels: List
    @param action_packing: A function that converts any types of action into dictionary
    @type action_packing: Callable
    @param lcm_instance: Communication variable 
    @type lcm_instance: LCM
    """

    def __init__(self, action_channels: List, action_packing: Callable, lcm_instance: LCM):
        self.action_space_publisher = MultiChannelPublisher(action_channels, lcm_instance)
        self.action_packing = action_packing
        self.messages_to_publish = None


    def pack_and_publish(self, action: Any):
        messages_to_publish = self.pack_message(action)
        self.action_space_publisher.publish(messages_to_publish)
    

    def pack_message(self, action: Any) -> Dict[str, Any]:
        """!
        Abstract method to pack the action into a message format suitable for LCM.

        @param action: The action to be packed into a message.
        @type action: Any
        @return: The packed LCM message.
        @type: Dict
        """

        return self.action_packing(action)


class ObservationSpace(Space):
    """!
    A class representing an observation space that listens for observations over LCM
    and processes them.

    @param observation channels: Channel names where observations will be listened
    @type observation_channels: List
    @param observation_unpacking: A function that converts observation dictionary into any types
    @type observation_unpacking: Callable
    @param lcm_instance: Communication variable 
    @type lcm_instance: LCM
    """

    def __init__(
        self, observation_channels: List, observation_unpacking: Callable, lcm_instance: LCM):
        self.observation_space_listener = MultiChannelListener(observation_channels, lcm_instance)
        self.observation_unpacking = observation_unpacking
        self.is_ready = False

    def unpack_message(self, observation_dict: Dict) -> Any:
        '''
        input: a dictionary containg the channel and then message recived on that channel
        returns: Any
        '''
        obs = self.observation_unpacking(observation_dict)
        return obs

    def check_readiness(self):
        lcm_dictionary = self.observation_space_listener.get()
        self.is_ready = not any(value is None for value in lcm_dictionary.values())
        
    def wait_until_observation_space_is_ready(self):
        # Continuously check if the observation space is ready
        while not self.is_ready:
            log.warning('Observation space is getting checked')
            # Perform any necessary steps to check readiness
            self.check_readiness()

            # Pause briefly before checking again
            time.sleep(0.05)

            if not self.is_ready:
                log.warning('Observation space is still not ready. Retrying...')

    def empty_data(self):
        """!
        Empties the data dictionary.
        """
        self.observation_space_listener.empty_data()

    def get_observation(self) -> Any:
        """!
        Retrieves the current observation from the space.

        @return: The current observation.
        @rtype: Any
        """
        assert self.is_ready, 'Observation space is not ready. Call wait_until_observation_space_is_ready() first.'
        
        self.data = self.observation_space_listener.get()

        processed_observation = self.unpack_message(self.data)

        return processed_observation