
from abc import ABC, abstractmethod
from lcm import LCM
from typing import Optional


class CommHandler(ABC):
    """!
    Base class for communication handlers, used for managing communication
    between different nodes.

    This class holds common attributes like the LCM instance, channel name,
    and channel type for communication handlers, and provides an interface
    for shutting down the communication.
    """

    def __init__(self, lcm: LCM, channel_name: str, channel_type: type):
        """!
        Initializes the communication handler with an LCM instance, a channel name,
        and a channel type.

        @param lcm: The LCM instance used for communication.
        @param channel_name: The name of the communication channel.
        @param channel_type: The type of the message expected for this communication channel.
        """
        self._lcm: LCM = lcm
        self.channel_name: str = channel_name
        self.channel_type: type = channel_type
        self.active = True

    def __repr__(self) -> str:
        """!
        Returns a string representation of the communication handler, showing the
        channel name and the type of message it handles.

        @return: A string representation of the handler in the format
                 "channel_name[channel_type]".
        """
        return f"{self.channel_name}[{self.channel_type.__name__}]"

    @abstractmethod
    def get_info(self) -> dict: 
        """!
        Should return a dictionary containing all information about the comms

        This method is abstract and should be implemented in subclasses.
        """

    @abstractmethod
    def suspend(self) -> None:
        """!
        Suspends the comms handler

        TODO
        """
    
    @abstractmethod
    def restart(self) -> None:
        """!
        Reactivates the comms handler

        TODO
        """