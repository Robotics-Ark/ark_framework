import zenoh
from ark.time.clock import Clock
from ark_msgs import Envelope
from collections.abc import Callable
from ark.comm.end_point import EndPoint
from google.protobuf.message import Message
from ark.data.data_collector import DataCollector


class Subscriber(EndPoint):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock: Clock,
        channel: str,
        data_collector: DataCollector | None,
        callback: Callable[[Message | bytes], None],
    ):
        super().__init__(node_name, session, clock, channel, data_collector)
        self._callback = callback
        self._sub = self._session.declare_subscriber(self._channel, self._on_sample)

    def core_registration(self):
        print("..todo: register with ark core..")

    def _on_sample(self, sample: zenoh.Sample):
        if self._active:

            # Retreive Envelope from sample and mark as RECEIVE
            env = Envelope()
            env.ParseFromString(bytes(sample.payload))
            env.endpoint_type = Envelope.EndpointType.RECEIVE
            env.dst_node_name = self._node_name
            env.recv_timestamp = self._clock.now()
            env.recv_seq_index = self._seq_index

            # Collect data if enabled
            if self._data_collector:
                self._data_collector.append(env.SerializeToString())

            # Invoke user callback
            self._callback(env.extract_message())

            # Increment sequence index
            self._seq_index += 1

    def close(self):
        super().close()
        self._sub.undeclare()
