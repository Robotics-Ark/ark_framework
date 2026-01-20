import zenoh
from ark.time.clock import Clock
from ark.core.registerable import Registerable
from ark.data.data_collector import DataCollector


class EndPoint(Registerable):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock: Clock,
        channel: str,
        data_collector: DataCollector | None,
    ):
        self._node_name = node_name
        self._session = session
        self._clock = clock
        self._channel = channel
        self._data_collector = data_collector
        self._active = True
        self._seq_index = 0

    def is_active(self) -> bool:
        return self._active

    def reset(self):
        self._active = True

    def close(self):
        self._active = False
