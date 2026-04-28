from abc import abstractmethod
from ark.reset import ResetableObject
from google.protobuf.message import Message


class ChannelNoise(ResetableObject):

    @abstractmethod
    def apply(self, msg: Message) -> Message:
        """Apply noise to the message."""


class NoNoise:

    def apply(self, msg: Message) -> Message:
        """Return the message unchanged."""
        return msg
