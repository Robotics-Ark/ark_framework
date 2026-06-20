import zenoh
from .channel import Channel
from abc import ABC, abstractmethod


class EndPoint(ABC):

    def __init__(self, channel: Channel, session: zenoh.Session) -> None:
        self._channel = channel
        self._session = session

    @abstractmethod
    def close(self) -> None:
        """Close the endpoint and release any associated resources."""
