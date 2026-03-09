import zenoh
from google.protobuf.message import Message
from ark_msgs import Envelope
from ark.time.clock import Clock
from ark.comm.end_point import EndPoint
from ark.data.data_collector import DataCollector
from ark_msgs.registry import msgs
from typing import Callable


class Queryable(EndPoint):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock: Clock,
        channel: str,
        handler: Callable[[Message], Message],
        data_collector: DataCollector | None = None,
    ):
        super().__init__(node_name, session, clock, channel, data_collector)
        self._handler = handler
        self._queryable = self._session.declare_queryable(self._channel, self._on_query)

    def core_registration(self):
        print("..todo: register with ark core..")

    def _on_query(self, query: zenoh.Query) -> None:
        # If we were closed, ignore queries
        if not self._active:
            return

        try:
            # Zenoh query may or may not include a payload.
            # For your use-case, the request is always in query.value (bytes)
            raw = bytes(query.value) if query.value is not None else b""
            if not raw:
                return  # nothing to do

            req_env = Envelope()
            req_env.ParseFromString(raw)

            # Decode request protobuf
            req_type = msgs.get(req_env.payload_msg_type)
            if req_type is None:
                # Unknown message type: ignore (or reply error later)
                return

            req_msg = req_type()
            req_msg.ParseFromString(req_env.payload)

            # Call user handler
            resp_msg: Message = self._handler(req_msg)

            resp_env = Envelope.pack(self._node_name, self._clock, resp_msg)
            query.reply(resp_env.SerializeToString())
            self._seq_index += 1

        except Exception:
            # Keep it minimal: don't kill the zenoh callback thread
            # You can add logging here if desired
            return
