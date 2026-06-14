import struct
import time
import zenoh
import threading
from typing import Callable
from dataclasses import dataclass
from ark.parameters import get_sim

_TIME_STRUCT = struct.Struct("<q")


@dataclass(frozen=True, slots=True)
class Time:
    nanosec: int

    @property
    def sec(self) -> float:
        return self.nanosec / 1e9

    def as_bytes(self) -> bytes:
        return _TIME_STRUCT.pack(self.nanosec)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Time":
        (nanosec,) = _TIME_STRUCT.unpack(data)
        return cls(nanosec=nanosec)

    @classmethod
    def from_sec(cls, sec: float) -> "Time":
        return cls(nanosec=round(sec * 1e9))

    def __add__(self, other: "Time") -> "Time":
        return Time(nanosec=self.nanosec + other.nanosec)

    def __sub__(self, other: "Time") -> "Time":
        return Time(nanosec=self.nanosec - other.nanosec)

    def __lt__(self, other: "Time") -> bool:
        return self.nanosec < other.nanosec

    def __le__(self, other: "Time") -> bool:
        return self.nanosec <= other.nanosec

    def __gt__(self, other: "Time") -> bool:
        return self.nanosec > other.nanosec

    def __ge__(self, other: "Time") -> bool:
        return self.nanosec >= other.nanosec

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Time) and self.nanosec == other.nanosec


class SimulatedTime:
    """Publishes simulated time ticks on the Zenoh network.

    Owned by SimulatorNode, which calls reset() before the first step and
    tick() after each physics step. Not a ResetObject — the SimulatorNode
    handles its lifecycle through the reset protocol.
    """

    def __init__(self, env_name: str, time_step_sec: float, session: zenoh.Session):
        self._time_step = Time.from_sec(time_step_sec)
        self._time_pub = session.declare_publisher(f"_ark/{env_name}/time")
        self._current: Time | None = None

    @property
    def time_step(self) -> Time:
        return self._time_step

    def reset(self):
        self._current = Time(nanosec=0)
        self._time_pub.put(self._current.as_bytes())

    def tick(self):
        if self._current is None:
            raise RuntimeError(
                "SimulatedTime.reset() must be called before tick()."
            )
        self._current += self._time_step
        self._time_pub.put(self._current.as_bytes())

    def close(self):
        self._time_pub.undeclare()


_SIM_CHANNEL = "_ark/{env_name}/sim"


class Clock:

    def __init__(self, env_name: str, session: zenoh.Session):
        # Always initialise the condition variable so _sim_now and
        # wait_until are safe to call regardless of the current mode.
        self._sim_time: Time | None = None
        self._sim_time_cv = threading.Condition()

        # Sim-time subscription is always active; it becomes meaningful
        # as soon as a SimulatorNode starts publishing.
        self._time_sub = session.declare_subscriber(
            f"_ark/{env_name}/time", self._on_time_sample
        )

        # Set initial mode from the env parameter server, then subscribe
        # so we react automatically if the executor hot-swaps sim ↔ real.
        self.sim = get_sim(env_name, session)
        self._update_time_source(self.sim)
        self._sim_sub = session.declare_subscriber(
            _SIM_CHANNEL.format(env_name=env_name), self._on_sim_change
        )

    def _update_time_source(self, sim: bool) -> None:
        self.sim = sim
        self.now = self._sim_now if sim else self.wall_now

    def _on_sim_change(self, sample: zenoh.Sample) -> None:
        payload = bytes(sample.payload)
        new_sim = bool(payload[0]) if payload else self.sim
        if new_sim == self.sim:
            return
        self._update_time_source(new_sim)
        if not new_sim:
            # Clear stale sim timestamp so _sim_now doesn't serve old data
            # if sim is re-enabled later.
            with self._sim_time_cv:
                self._sim_time = None

    def _on_time_sample(self, sample: zenoh.Sample) -> None:
        t = Time.from_bytes(bytes(sample.payload))
        with self._sim_time_cv:
            self._sim_time = t
            self._sim_time_cv.notify_all()

    def _sim_now(self) -> Time:
        with self._sim_time_cv:
            while self._sim_time is None:
                self._sim_time_cv.wait()
            return Time(nanosec=self._sim_time.nanosec)

    def wall_now(self) -> Time:
        return Time(nanosec=time.time_ns())

    def wait_until(self, target: Time) -> None:
        if not self.sim:
            raise RuntimeError("Simulated time is not enabled.")
        with self._sim_time_cv:
            while self._sim_time is None or self._sim_time < target:
                self._sim_time_cv.wait()


class Sleep:

    def __init__(self, clock: Clock):
        self._clock = clock

    def _wall_sleep(self, dur: Time) -> None:
        time.sleep(dur.sec)

    def _sim_sleep(self, dur: Time) -> None:
        now = self._clock.now()
        self._clock.wait_until(now + dur)

    def __call__(self, dur: Time) -> None:
        if dur.nanosec <= 0:
            return
        if self._clock.sim:
            self._sim_sleep(dur)
        else:
            self._wall_sleep(dur)


class Rate:
    def __init__(self, clock: Clock, hz: float):
        self._clock = clock
        self._sleep = Sleep(clock)
        self._time_step = Time(nanosec=round(1e9 / hz))
        self._next: Time | None = None

    def reset(self) -> None:
        self._next = None

    def sleep(self) -> None:
        now = self._clock.now()
        if self._next is None:
            self._next = now + self._time_step

        remaining = self._next - now
        if remaining.nanosec > 0:
            self._sleep(remaining)
            self._next += self._time_step
        else:
            self._next = now + self._time_step


class Stepper(threading.Thread):

    def __init__(
        self, clock: Clock, hz: float, callback: Callable[[Time], None]
    ) -> None:
        super().__init__(daemon=True)
        self._clock = clock
        self._rate = Rate(clock, hz)
        self.reset = self._rate.reset
        self._callback = callback
        self._closed = False
        self.start()

    def run(self):
        while not self._closed:
            self._rate.sleep()
            if self._closed:
                break
            self._callback(self._clock.now())

    def close(self):
        self._closed = True
