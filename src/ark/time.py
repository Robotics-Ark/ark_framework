import time
import zenoh
import struct
import threading
from typing import Callable
from dataclasses import dataclass

TIME_CHANNEL = "time"  # Channel name for publishing simulated time updates
TIME_FMT = "<q"  # little-endian 64-bit signed integer (nanoseconds)
TIME_BYTES_SIZE = 8  # Size of the time data in bytes


@dataclass
class Time:
    nanosec: int

    @property
    def sec(self) -> float:
        return self.nanosec / 1e9

    def as_bytes(self) -> bytes:
        return struct.pack(TIME_FMT, self.nanosec)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Time":
        if len(data) != TIME_BYTES_SIZE:
            raise ValueError(
                f"Invalid data length for Time. Expected {TIME_BYTES_SIZE} bytes."
            )
        nanosec = struct.unpack(TIME_FMT, data)[0]
        return cls(nanosec=nanosec)

    def __add__(self, other: object) -> "Time":
        if not isinstance(other, Time):
            return NotImplemented
        return Time(nanosec=self.nanosec + other.nanosec)

    def __sub__(self, other: object) -> "Time":
        if not isinstance(other, Time):
            return NotImplemented
        return Time(nanosec=self.nanosec - other.nanosec)

    def __iadd__(self, other: object) -> "Time":
        if not isinstance(other, Time):
            return NotImplemented
        self.nanosec += other.nanosec
        return self

    def __isub__(self, other: object) -> "Time":
        if not isinstance(other, Time):
            return NotImplemented
        self.nanosec -= other.nanosec
        return self

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self.nanosec < other.nanosec

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self.nanosec <= other.nanosec

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self.nanosec > other.nanosec

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self.nanosec >= other.nanosec

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self.nanosec == other.nanosec


class SimulatedTime:

    def __init__(self, session: zenoh.Session, time_step: float):
        """Initialize the SimulatedTime.

        Parameters:
        -----------
        session: zenoh.Session
            The zenoh session to use for publishing time updates.
        time_step: float
            The time step in seconds for each tick of the simulated time.
        """
        self._time_pub = session.declare_publisher(TIME_CHANNEL)
        self._sim_time: Time | None = None
        self._time_step = Time(nanosec=round(time_step * 1e9))

    @property
    def time_step(self) -> Time:
        """Get the time step as a Time object."""
        return self._time_step

    def _publish_time(self):
        """Publish the current simulated time in nanoseconds."""
        if self._sim_time is None:
            raise RuntimeError("Simulated time not initialized.")
        self._time_pub.put(self._sim_time.as_bytes())

    def reset(self):
        """Reset the simulated time to zero and publish the update."""
        self._sim_time = Time(nanosec=0)
        self._publish_time()

    def tick(self):
        """Advance the simulated time by one time step and publish the update."""
        if self._sim_time is None:
            raise RuntimeError("SimulatedTime not initialized. Call reset() first.")
        self._sim_time += self._time_step
        self._publish_time()


class Clock:

    def __init__(self, sim: bool, session: zenoh.Session):
        """Initialize the Clock.

        Parameters:
        -----------
        sim: bool
            Whether to use simulated time.
        session: zenoh.Session
            The zenoh session to use for subscribing to time updates.
        """
        self._sim = sim
        self._sim_time: Time | None = None

        if self._sim:
            self._sim_time_cv = threading.Condition()
            self.now = self._sim_now
            self._time_sub = session.declare_subscriber(
                TIME_CHANNEL, self._on_time_sample
            )
        else:
            self._time_sub = None
            self._sim_time_cv = None
            self.now = self.wall_now

    @property
    def sim(self) -> bool:
        """Check if the clock is using simulated time."""
        return self._sim

    def _on_time_sample(self, sample: zenoh.Sample):
        """Callback for handling incoming time samples from the TIME_CHANNEL."""
        payload = bytes(sample.payload)
        t = Time.from_bytes(payload)
        with self._sim_time_cv:
            self._sim_time = t
            self._sim_time_cv.notify_all()

    def _sim_now(self) -> Time:
        """Get the current simulated time as a Time object."""
        if not self._sim:
            raise RuntimeError("Simulated time is not enabled.")
        with self._sim_time_cv:
            while self._sim_time is None:
                self._sim_time_cv.wait()
            return Time(nanosec=self._sim_time.nanosec)

    def wall_now(self) -> Time:
        """Get the current wall-clock time as a Time object."""
        return Time(nanosec=time.time_ns())

    def wait_until(self, target: Time) -> None:
        """Block until the simulated time reaches the target time."""
        if not self._sim:
            raise RuntimeError("Simulated time is not enabled.")
        with self._sim_time_cv:
            while self._sim_time is None or self._sim_time < target:
                self._sim_time_cv.wait()


class Sleep:

    def __init__(self, clock: Clock):
        self._clock = clock
        self._sleep = self._sim_sleep if clock.sim else self._wall_sleep

    def _wall_sleep(self, dur: Time) -> None:
        if dur.nanosec <= 0:
            return
        time.sleep(dur.sec)

    def _sim_sleep(self, dur: Time) -> None:
        if dur.nanosec <= 0:
            return
        now = self._clock.now()
        self._clock.wait_until(now + dur)

    def __call__(self, dur: Time) -> None:
        self._sleep(dur)


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
