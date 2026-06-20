from __future__ import annotations

from typing import Any


def payload_bytes(payload: Any) -> bytes:
    if payload is None:
        raise ValueError("Expected a Zenoh payload, got None.")
    if hasattr(payload, "payload"):
        payload = payload.payload
    return bytes(payload)
