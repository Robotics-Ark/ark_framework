
import threading
from copy import deepcopy
from ark.client.comm_handler.subscriber import Subscriber
from typing import Any
from lcm import LCM


class Listener(Subscriber):
    """!
    A class for receiving and processing messages from a specific channel using an LCM subscriber.

    This class listens for messages on a specified channel, and saves the latest message
    and provides methods to retrieve it in a thread-safe manner. It inherits from the `Subscriber` class,
    which handles subscribing to the LCM channel and receiving messages. The `Listener` class ensures
    thread-safety using a `Lock` to protect access to the message.

    @param lcm: The LCM instance used for communication.
    @param channel_name: The name of the channel to subscribe to.
    @param channel_type: The type of the message expected from the channel.
    """

    def __init__(self, lcm: LCM, channel_name: str, channel_type: type) -> None:
        """!
        Initializes the Listener instance, subscribing to the specified channel and setting up a mutex
        to protect access to the received message.

        @param lcm: The LCM instance used for communication.
        @param channel_name: The name of the channel to subscribe to.
        @param channel_type: The type of the message expected from the channel.
        """
        self.mutex: threading.Lock = threading.Lock()
        self._msg: Any = None
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.comm_type = "Listener"
        super().__init__(lcm, channel_name, channel_type, self.callback)

    def received(self) -> bool:
        """!
        Checks whether a message has been received.

        @return: True if a message has been received, False otherwise.
        """
        return self._msg is not None

    def callback(self, t: int, channel_name: str, msg: Any) -> None:
        """!
        Callback function that is called when a new message is received on the subscribed channel.

        This method is invoked by the parent `Subscriber` class when a new message is received.
        It locks the mutex to safely store the received message in the instance.

        @param t: The time stamp when the message was received in nanoseconds.
        @param channel_name: The name of the channel to subscribe to.
        @param msg: The received message.
        """
        with self.mutex:
            self._msg = msg

    def get(self) -> Any:
        """!
        Retrieves the latest received message in a thread-safe manner.

        The method locks the mutex to ensure thread-safe access to the message. It creates and returns
        a deep copy of the message to avoid any unintended modifications to the internal state.

        @return: A deep copy of the latest received message.
        """
        with self.mutex:
            msg = deepcopy(self._msg)
        return msg
    
    def suspend(self):
        """!
        Suspend the listener and clear any cached message.

        @return: ``None``
        """
        self.empty_data()
        return super().suspend()
    
    def empty_data(self):
        """!
        Clear the stored message.
        """
        self._msg = None

    def get_info(self):
        """!
        Return a dictionary describing this listener.
        """
        info  = {
            "comms_type": "Listener",
            "channel_name": self.channel_name,
            "channel_type": self.channel_type.__name__,
            "channel_status": self.active
        }
        return info