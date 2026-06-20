from __future__ import annotations

import math
from typing import Any

import numpy as np

from .binary import Reader, Writer


def shape_size(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def write_dtype(writer: Writer, dtype: Any) -> None:
    writer.string(np.dtype(dtype).str)


def read_dtype(reader: Reader) -> np.dtype:
    return np.dtype(reader.string())


def write_shape(writer: Writer, shape: tuple[int, ...]) -> None:
    writer.uint(len(shape))
    for dim in shape:
        writer.uint(int(dim))


def read_shape(reader: Reader) -> tuple[int, ...]:
    return tuple(reader.uint() for _ in range(reader.uint()))


def write_array(writer: Writer, value: Any, dtype: Any, shape: tuple[int, ...]) -> None:
    dtype = np.dtype(dtype)
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"Array shape {array.shape} does not match {shape}.")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    writer.bytes(array.tobytes(order="C"))


def read_array(
    reader: Reader, dtype: Any, shape: tuple[int, ...], *, copy: bool = False
) -> np.ndarray:
    dtype = np.dtype(dtype)
    data = reader.bytes(shape_size(shape) * dtype.itemsize)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def write_bool_array(writer: Writer, value: Any, shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != shape:
        raise ValueError(f"Boolean array shape {array.shape} does not match {shape}.")
    packed = np.packbits(array.reshape(-1), bitorder="little")
    writer.bytes(packed.tobytes(order="C"))


def read_bool_array(reader: Reader, shape: tuple[int, ...]) -> np.ndarray:
    n = shape_size(shape)
    packed = reader.bytes((n + 7) // 8)
    return (
        np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=n, bitorder="little")
        .view(np.bool_)
        .reshape(shape)
    )


def write_bound(writer: Writer, value: Any, dtype: Any, shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"Bound shape {array.shape} does not match {shape}.")
    is_scalar = bool(array.size and np.all(array == array.reshape(-1)[0]))
    writer.bool(is_scalar)
    if is_scalar:
        write_array(writer, array.reshape(-1)[0], dtype, ())
    else:
        write_array(writer, array, dtype, shape)


def read_bound(reader: Reader, dtype: Any, shape: tuple[int, ...]) -> Any:
    if reader.bool():
        return read_array(reader, dtype, (), copy=True).item()
    return read_array(reader, dtype, shape, copy=True)
