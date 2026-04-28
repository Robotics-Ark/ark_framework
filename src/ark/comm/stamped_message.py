from ark.time import Time
from dataclasses import dataclass
from google.protobuf.message import Message


@dataclass(frozen=True, slots=True)
class StampedMessage:
    """A decoded message with the Ark time at which it was received."""

    time: Time
    message: Message | None
