import zenoh
from ark_msgs import Envelope
from google.protobuf.message import Message


def message_from_sample(sample: zenoh.Sample) -> Message:
    """Extract the message from the given sample."""
    env = Envelope()
    env.ParseFromString(bytes(sample.payload))
    return env.extract_message()
