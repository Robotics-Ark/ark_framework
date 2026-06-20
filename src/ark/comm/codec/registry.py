from __future__ import annotations

from typing import Any

from gymnasium import Space

from .sample import SampleDecoder, SampleEncoder
from .space import SpaceDecoder, SpaceEncoder
from .payload import payload_bytes


class _SampleCodec:
    def __init__(self, space: Space):
        self._encoder = SampleEncoder(space)
        self._decoder = SampleDecoder(space)

    def encode(self, sample: Any) -> bytes:
        return self._encoder.encode(sample)

    def decode(self, payload: Any) -> Any:
        return self._decoder.decode(payload_bytes(payload))


class _SpaceCodec:
    def __init__(self):
        self._encoder = SpaceEncoder()
        self._decoder = SpaceDecoder()

    def encode(self, space: Space) -> bytes:
        return self._encoder.encode(space)

    def decode(self, payload: Any) -> Space:
        return self._decoder.decode(payload_bytes(payload))


class _SampleCodecRegistry:
    def get(self, space: Space) -> _SampleCodec:
        return _SampleCodec(space)


class _SpaceCodecRegistry:
    def get(self, _: Any = None) -> _SpaceCodec:
        return _SpaceCodec()


sample_codec = _SampleCodecRegistry()
space_codec = _SpaceCodecRegistry()
