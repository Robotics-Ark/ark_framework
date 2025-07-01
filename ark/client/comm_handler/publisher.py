
from ark.tools.log import log
from ark.client.comm_handler.comm_handler import CommHandler
from lcm import LCM  


class Publisher(CommHandler):
    """!
    A Publisher class that extends the CommHandler base class. This class handles
    the publishing of messages to a specified communication channel using LCM.

    Attributes:
        _enabled (bool): A flag indicating whether the publisher is enabled.
    """

    def __init__(self, lcm: LCM, channel_name: str, channel_type: type) -> None:
        """!
        Initializes the Publisher instance with the LCM instance, channel name,
        and message type. Also sets the publisher as enabled and logs the setup.

        @param lcm: The LCM instance used for communication.
        @param channel_name: The name of the channel for publishing messages.
        @param channel_type: The type of message expected for the channel.
        """
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.comm_type = "Publisher"
        super().__init__(lcm, channel_name, channel_type)
        log.ok(f"setup publisher {self}")

    def publish(self, msg: object) -> None:
        """!
        Publishes a message to the specified channel if the publisher is enabled.

        @param msg: The message object to publish. This should match the expected
                    type for the channel.
        """
        assert type(msg) == self.channel_type, f"Wrong Message Type send to Channel {self.channel_name}"
        
        if self.active:
            self._lcm.publish(self.channel_name, self.channel_type.encode(msg))
        else:
            log.warning(f"publisher {self} is not enabled, cannot publish messages")


    def restart(self) -> None:
        """!
        Restarts the publisher by enabling it again and logging the action.
        """
        self.active = True
        log.ok(f"enabled {self}")

    def suspend(self) -> None:
        """!
        Shuts down the publisher by disabling it and logging the shutdown action.
        """
        self.active = False
        log.ok(f"suspended publisher {self}")

    def get_info(self):
        """!
        Return a dictionary describing this publisher.
        """
        info = {
            "comms_type": "Publisher",
            "channel_name": self.channel_name,
            "channel_type": self.channel_type.__name__,
            "channel_status": self.active
        }
        return info