import zenoh
import struct


class SimTime:
    def __init__(
        self, session: zenoh.Session, clock_channel_name: str, time_step_ns: int
    ):
        self._pub = session.declare_publisher(clock_channel_name)
        self._sim_time_ns = None
        self._time_step_ns = int(time_step_ns)

    def reset(self):
        self._sim_time_ns = 0
        self._pub.put(struct.pack("<q", self._sim_time_ns))

    def tick(self, n: int = 1):
        if self._sim_time_ns is None:
            raise RuntimeError("SimTime not initialized. Call reset() first.")
        self._sim_time_ns += self._time_step_ns * int(n)
        self._pub.put(struct.pack("<q", self._sim_time_ns))
