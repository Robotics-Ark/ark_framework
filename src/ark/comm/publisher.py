import zenoh
from ark.time.clock import Clock
from .end_point import EndPoint
from ark_msgs import Envelope
from google.protobuf.message import Message
from ark.data.data_collector import DataCollector


class Publisher(EndPoint):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock: Clock,
        channel: str,
        data_collector: DataCollector | None,
    ):
        super().__init__(node_name, session, clock, channel, data_collector)
        self._pub = self._session.declare_publisher(self._channel)

    def core_registration(self):
        print("..todo: register with ark core..")

    def publish(self, msg: Message | bytes):
        if self._active:

            # Create Envelope
            env = Envelope(
                endpoint_type=Envelope.EndpointType.PUBLISH,
                channel=self._channel,
                src_node_name=self._node_name,
                sent_seq_index=self._seq_index,
                sent_timestamp=self._clock.now(),
            )
            if isinstance(msg, Message):
                env.msg_type = msg.DESCRIPTOR.full_name
                env.payload = msg.SerializeToString()
            elif isinstance(msg, bytes):
                env.msg_type = "__bytes__"
                env.payload = bytes(msg)
            else:
                raise TypeError("msg must be a protobuf Message or bytes")
            env_bytes = env.SerializeToString()

            # Publish envelope
            self._pub.put(env_bytes)

            # Collect data if enabled
            if self._data_collector is not None:
                self._data_collector.append(env_bytes)

            # Increment sequence index
            self._seq_index += 1

    def close(self):
        super().close()
        self._pub.undeclare()
