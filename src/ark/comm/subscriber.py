import queue
import threading
import zenoh
from gymnasium import Space
from typing import Callable
from ark.time import Clock
from .end_point import EndPoint
from .codec.registry import sample_codec
from .queryable_space import QueryableSpace
from .stamped_sample import StampedSample
from .channel import Channel
from ark.noise import NOISE_TYPE, normalise_noise

_SENTINEL = object()


class Subscriber(EndPoint):

    def __init__(
        self,
        channel: Channel,
        space: Space,
        callback: Callable[[StampedSample], None],
        session: zenoh.Session,
        check: bool,
        noise: NOISE_TYPE = None,
    ):
        super().__init__(channel, session)
        self._space = space
        self._callback = callback
        self._check = check
        self._noises = normalise_noise(noise)
        self._clock = Clock(channel.env_name, session)
        self._codec = sample_codec.get(self._space)
        self._delivery_queue: queue.Queue = queue.Queue()
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop, daemon=True
        )
        self._delivery_thread.start()
        self._z_sub = self._session.declare_subscriber(
            self._channel.name, self._on_callback
        )
        self._z_sub_qr = QueryableSpace(self._channel, "subscriber", self._space, self._session)

    def _on_callback(self, sample: zenoh.Sample):
        t = self._clock.now()
        raw = self._codec.decode(sample.payload)
        self._delivery_queue.put((t, raw))

    def _delivery_loop(self):
        while True:
            item = self._delivery_queue.get()
            if item is _SENTINEL:
                break
            t, sample = item
            for noise in self._noises:
                sample = noise.apply(sample)
            if self._check and not self._space.contains(sample):
                raise ValueError(
                    f"Sample {sample} does not conform to the space {self._space}"
                )
            self._callback(StampedSample(t, sample))

    def close(self):
        self._z_sub.undeclare()
        self._delivery_queue.put(_SENTINEL)
        self._delivery_thread.join()
        self._z_sub_qr.undeclare()
