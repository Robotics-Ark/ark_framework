import zenoh
from ark_msgs import Envelope
from google.protobuf.message import Message
from ark.data.data_collector import DataCollector
from ark.comm.end_point import EndPoint


class Querier(EndPoint):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock,
        channel: str,
        data_collector: DataCollector | None,
    ):
        super().__init__(node_name, session, clock, channel, data_collector)
        self._querier = self._session.declare_querier(self._channel)

    def core_registration(self):
        print("..todo: register with ark core..")

    def query(
        self,
        req: Message | bytes,
        timeout: float = 10.0,
    ) -> Message:
        """Send a query message and wait for the first OK response."""
        if not self._active:
            raise RuntimeError("Querier is not active")

        # Create Envelope for the request
        req_env = Envelope(
            endpoint_type=Envelope.EndpointType.REQUEST,
            channel=self._channel,
            src_node_name=self._node_name,
            sent_seq_index=self._seq_index,
            sent_timestamp=self._clock.now(),
        )

        if isinstance(req, Message):
            req_env.msg_type = req.DESCRIPTOR.full_name
            req_env.payload = req.SerializeToString()
        elif isinstance(req, bytes):
            req_env.msg_type = "__bytes__"
            req_env.payload = bytes(req)
        else:
            raise TypeError("req must be a protobuf Message or bytes")

        replies = self._querier.get(value=req_env.SerializeToString(), timeout=timeout)

        for reply in replies:
            if reply.ok is None:
                continue

            resp_env = Envelope()
            resp_env.ParseFromString(bytes(reply.ok))
            resp_env.dst_node_name = self._node_name
            resp_env.recv_timestamp = self._clock.now()

            resp = resp_env.extract_message()

            self._seq_index += 1

            if self._data_collector:
                self._data_collector.append(req_env.SerializeToString())
                self._data_collector.append(resp_env.SerializeToString())

            return resp

        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {timeout}s"
            )

    def close(self):
        super().close()
        self._querier.undeclare()
