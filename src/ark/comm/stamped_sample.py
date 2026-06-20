from typing import Any
from ark.time import Time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StampedSample:
    time: Time
    sample: Any
