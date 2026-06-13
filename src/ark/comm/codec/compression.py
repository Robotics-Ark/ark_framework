from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image as PILImage


class NoCompression:
    tag: int = 0

    def compress(self, array: np.ndarray) -> bytes:
        arr = array if array.flags.c_contiguous else np.ascontiguousarray(array)
        return arr.tobytes()

    def decompress(self, data: bytes, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


class JPEGCompression:
    tag: int = 1

    def __init__(self, quality: int = 85) -> None:
        if not 1 <= quality <= 95:
            raise ValueError(f"JPEG quality must be between 1 and 95, got {quality}.")
        self.quality = quality

    def compress(self, array: np.ndarray) -> bytes:
        pil = PILImage.fromarray(array)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.quality)
        return buf.getvalue()

    def decompress(self, data: bytes, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        pil = PILImage.open(io.BytesIO(data))
        return np.array(pil, dtype=np.dtype(dtype)).reshape(shape)


_REGISTRY: dict[int, NoCompression | JPEGCompression] = {
    NoCompression.tag: NoCompression(),
    JPEGCompression.tag: JPEGCompression(),
}


def get_decompressor(tag: int) -> NoCompression | JPEGCompression:
    try:
        return _REGISTRY[tag]
    except KeyError:
        raise ValueError(f"Unknown compression tag: {tag}.")
