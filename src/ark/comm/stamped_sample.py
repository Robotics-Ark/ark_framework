from typing import Any
from ark.time import Time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StampedSample:
    """A decoded sample with the Ark time at which it was received."""

    time: Time
    sample: Any
