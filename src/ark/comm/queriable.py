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
        self._queryable = self._session.declare_queryable(self._channel,
                                                          self._on_query,
                                                          complete=False)
        print(f"Declared queryable on channel: {self._channel}")

    def core_registration(self):
        print("..todo: register with ark core..")

    def _on_query(self, query: zenoh.Query) -> None:
        # If we were closed, ignore queries
        if not self._active:
            print("Received query on closed Queryable, ignoring")
            return

        try:
            # Zenoh query may or may not include a payload.
            # For your use-case, the request is always in query.value (bytes)
            raw = bytes(query.payload) if query.payload is not None else b""
            if not raw:
                print("Received query with no payload, ignoring")
                return  # nothing to do

            req_env = Envelope()
            req_env.ParseFromString(raw)

            # Decode request protobuf
            # req_type = msgs.get(req_env.payload_msg_type)
            req_type = msgs.get(req_env.msg_type)
            if req_type is None:
                # Unknown message type: ignore (or reply error later)
                print(f"Unknown message type '{req_env.msg_type}' in query, ignoring")
                return

            req_msg = req_type()
            req_msg.ParseFromString(req_env.payload)

            # Call user handler
            resp_msg: Message = self._handler(req_msg)

            # Pack envelope for response
            resp_env = Envelope()
            resp_env.endpoint_type = Envelope.EndpointType.RESPONSE
            resp_env.sent_timestamp = self._clock.now()
            resp_env.sent_seq_index = self._seq_index
            resp_env.src_node_name = self._node_name
            resp_env.channel = self._channel
            resp_env.msg_type = resp_msg.DESCRIPTOR.full_name
            resp_env.payload = resp_msg.SerializeToString()

            self._seq_index += 1

            with query: 
                query.reply(query.key_expr, resp_env.SerializeToString())

            if self._data_collector:
                self._data_collector.append(req_env.SerializeToString())
                self._data_collector.append(resp_env.SerializeToString())

        except Exception:
            # Keep it minimal: don't kill the zenoh callback thread
            # You can add logging here if desired
            print("Error processing query:")
            # write the traceback to stdout for debugging
            import traceback
            traceback.print_exc()

            return
