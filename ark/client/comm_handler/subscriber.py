
import time
from lcm import LCM
from ark.tools.log import log
from ark.client.comm_handler.comm_handler import CommHandler
from typing import Callable


class Subscriber(CommHandler):
    """!
    A subscriber for listening to messages on a communication channel.

    This class subscribes to a specified communication channel and calls a user-defined
    callback function whenever a new message is received. The message data is passed to the
    callback along with the timestamp and channel name.

    @note: This class is a subclass of `CommHandler` and requires an LCM instance,
           a channel name, and a channel type to function correctly.
    """

    def __init__(
        self,
        lcm: LCM,
        channel_name: str,
        channel_type: type,
        callback: Callable[[int, str, object], None],
        callback_args: list[object] = [],
    ) -> None:
        """!
        Initializes the subscriber with the necessary parameters for subscribing
        to a communication channel and setting up the callback function.

        @param lcm: The LCM instance used for communication.
        @param channel_name: The name of the communication channel.
        @param channel_type: The type of the message expected for this communication channel.
        @param callback: The user-defined callback function to be called with the message data.
        @param callback_args: Additional arguments to be passed to the callback function.
        """
        super().__init__(lcm, channel_name, channel_type)
        self._user_callback: Callable[[int, str, object], None] = callback
        self._callback_args: list[object] = callback_args
        self.comm_type = "Subscriber"
        self.subscribe()
        

    def subscriber_callback(self, channel_name: str, data: bytes) -> None:
        """!
        Callback function to handle incoming messages on the subscribed channel.

        This method decodes the message data, records the timestamp, and calls
        the user-defined callback function with the timestamp, channel name, and message.

        @param channel_name: The name of the communication channel the message was received from.
        @param data: The raw byte data of the message.
        """
        t: int = time.time_ns()
        try:
            msg: object = self.channel_type.decode(data)
            self._user_callback(t, channel_name, msg, *self._callback_args)
        except ValueError as e:
            log.warning(f"failed to decode message on channel '{channel_name}': {e}")

    def subscribe(self):
        """!
        Subscribe to the configured channel.

        @return: ``None``
        """
        self._sub = self._lcm.subscribe(self.channel_name, self.subscriber_callback)
        self._sub.set_queue_capacity(1)  # TODO: configurable
        log.ok(f"subscribed to {self}")
        self.active = True

    def restart(self):
        """!
        Reconnect the subscriber to its channel.
        """
        self.subscribe()
        self.active = True
        

    def suspend(self) -> None:
        """!
        Suspends the subscriber by unsubscribing from the communication channel.

        This method releases the subscription and logs that the subscriber has been unsubscribed.
        """
        if self.active == True:
            self._lcm.unsubscribe(self._sub)
            log.ok(f"unsubscribed from {self}")
            self.active = False

    def get_info(self):
        """!
        Return a dictionary describing this subscriber.
        """
        info = {
            "comms_type": "Subscriber",
            "channel_name": self.channel_name,
            "channel_type": self.channel_type.__name__,
            "channel_status": self.active
        }
        return info