from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Text, Tuple
from gymnasium.vector import utils as vector_utils

try:
    from gymnasium.spaces import Graph, GraphInstance, OneOf, Sequence
except ImportError:
    Graph = GraphInstance = OneOf = Sequence = None

from .arrays import (
    read_array,
    read_bool_array,
    write_array,
    write_bool_array,
)
from .binary import BytesLike, Reader, Writer


def encode_sample(space: Space, sample: Any) -> bytes:
    writer = Writer()
    _write_sample(writer, space, sample)
    return writer.finish()


def decode_sample(space: Space, payload: BytesLike, *, copy: bool = False) -> Any:
    reader = Reader(payload)
    sample = _read_sample(reader, space, copy=copy)
    reader.finish()
    return sample


def _write_sample(writer: Writer, space: Space, sample: Any) -> None:
    if isinstance(space, Box):
        if np.dtype(space.dtype) == np.dtype(np.bool_):
            write_bool_array(writer, sample, space.shape)
        else:
            write_array(writer, sample, space.dtype, space.shape)
    elif isinstance(space, Discrete):
        value = int(sample)
        offset = value - int(space.start)
        if offset < 0 or offset >= int(space.n):
            raise ValueError(f"Discrete sample {sample!r} is outside {space}.")
        writer.uint(offset)
    elif isinstance(space, MultiBinary):
        write_bool_array(writer, sample, space.shape)
    elif isinstance(space, MultiDiscrete):
        array = np.asarray(sample)
        if array.shape != space.shape:
            raise ValueError(
                f"MultiDiscrete sample shape {array.shape} does not match {space.shape}."
            )
        write_array(writer, array, space.dtype, space.shape)
    elif isinstance(space, Text):
        if not space.contains(sample):
            raise ValueError(f"Text sample {sample!r} is outside {space}.")
        writer.string(sample)
    elif isinstance(space, Dict):
        if not isinstance(sample, Mapping):
            raise TypeError("Dict samples must be mappings.")
        for key, child in space.spaces.items():
            _write_sample(writer, child, sample[key])
    elif isinstance(space, Tuple):
        if len(sample) != len(space.spaces):
            raise ValueError(f"Tuple sample must have {len(space.spaces)} values.")
        for child, value in zip(space.spaces, sample):
            _write_sample(writer, child, value)
    elif Sequence is not None and isinstance(space, Sequence):
        items = (
            tuple(vector_utils.iterate(space.stacked_feature_space, sample))
            if space.stack
            else tuple(sample)
        )
        writer.uint(len(items))
        for item in items:
            _write_sample(writer, space.feature_space, item)
    elif Graph is not None and isinstance(space, Graph):
        _write_graph(writer, space, sample)
    elif OneOf is not None and isinstance(space, OneOf):
        index, value = sample
        index = int(index)
        if index < 0 or index >= len(space.spaces):
            raise ValueError(f"OneOf index {index} is outside {len(space.spaces)}.")
        writer.uint(index)
        _write_sample(writer, space.spaces[index], value)
    else:
        raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}.")


def _read_sample(reader: Reader, space: Space, *, copy: bool) -> Any:
    if isinstance(space, Box):
        if np.dtype(space.dtype) == np.dtype(np.bool_):
            return read_bool_array(reader, space.shape)
        return read_array(reader, space.dtype, space.shape, copy=copy)
    if isinstance(space, Discrete):
        offset = reader.uint()
        if offset >= int(space.n):
            raise ValueError(f"Decoded Discrete offset {offset} is outside {space}.")
        return np.asarray(int(space.start) + offset, dtype=space.dtype)[()]
    if isinstance(space, MultiBinary):
        return read_bool_array(reader, space.shape).astype(space.dtype, copy=False)
    if isinstance(space, MultiDiscrete):
        return read_array(reader, space.dtype, space.shape, copy=copy)
    if isinstance(space, Text):
        value = reader.string()
        if not space.contains(value):
            raise ValueError(f"Decoded Text sample {value!r} is outside {space}.")
        return value
    if isinstance(space, Dict):
        return {
            key: _read_sample(reader, child, copy=copy)
            for key, child in space.spaces.items()
        }
    if isinstance(space, Tuple):
        return tuple(_read_sample(reader, child, copy=copy) for child in space.spaces)
    if Sequence is not None and isinstance(space, Sequence):
        items = tuple(
            _read_sample(reader, space.feature_space, copy=copy)
            for _ in range(reader.uint())
        )
        if not space.stack:
            return items
        out = vector_utils.create_empty_array(space.feature_space, n=len(items))
        if not items:
            return out
        return vector_utils.concatenate(space.feature_space, items, out)
    if Graph is not None and isinstance(space, Graph):
        return _read_graph(reader, space, copy=copy)
    if OneOf is not None and isinstance(space, OneOf):
        index = reader.uint()
        if index >= len(space.spaces):
            raise ValueError(f"Decoded OneOf index {index} is out of range.")
        return np.int64(index), _read_sample(reader, space.spaces[index], copy=copy)
    raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}.")


def _write_graph(writer: Writer, space: Graph, sample: GraphInstance) -> None:
    if not isinstance(sample, GraphInstance):
        raise TypeError("Graph samples must be GraphInstance values.")
    nodes = np.asarray(sample.nodes)
    writer.uint(len(nodes))
    _write_feature_array(writer, space.node_space, nodes, len(nodes))

    if space.edge_space is None:
        if sample.edges is not None or sample.edge_links is not None:
            raise ValueError("Graph sample has edges but the space has no edge_space.")
        return

    has_edges = sample.edges is not None or sample.edge_links is not None
    writer.bool(has_edges)
    if not has_edges:
        return
    if sample.edges is None or sample.edge_links is None:
        raise ValueError("Graph edges and edge_links must both be present.")
    edges = np.asarray(sample.edges)
    links = np.asarray(sample.edge_links, dtype=np.int64)
    if links.shape != (len(edges), 2):
        raise ValueError("Graph edge_links must have shape (num_edges, 2).")
    writer.uint(len(edges))
    _write_feature_array(writer, space.edge_space, edges, len(edges))
    write_array(writer, links, np.int64, (len(edges), 2))


def _read_graph(reader: Reader, space: Graph, *, copy: bool) -> GraphInstance:
    n_nodes = reader.uint()
    nodes = _read_feature_array(reader, space.node_space, n_nodes, copy=copy)
    if space.edge_space is None or not reader.bool():
        return GraphInstance(nodes=nodes, edges=None, edge_links=None)
    n_edges = reader.uint()
    edges = _read_feature_array(reader, space.edge_space, n_edges, copy=copy)
    links = read_array(reader, np.int64, (n_edges, 2), copy=True)
    return GraphInstance(nodes=nodes, edges=edges, edge_links=links)


def _write_feature_array(writer: Writer, space: Space, value: Any, count: int) -> None:
    if isinstance(space, Box):
        write_array(writer, value, space.dtype, (count,) + space.shape)
    elif isinstance(space, Discrete):
        write_array(writer, value, space.dtype, (count,))
    else:
        raise TypeError("Graph feature spaces must be Box or Discrete.")


def _read_feature_array(
    reader: Reader, space: Space, count: int, *, copy: bool
) -> np.ndarray:
    if isinstance(space, Box):
        return read_array(reader, space.dtype, (count,) + space.shape, copy=copy)
    if isinstance(space, Discrete):
        return read_array(reader, space.dtype, (count,), copy=copy)
    raise TypeError("Graph feature spaces must be Box or Discrete.")


class SampleEncoder:
    def __init__(self, space: Space):
        self.space = space

    def encode(self, sample: Any) -> bytes:
        return encode_sample(self.space, sample)

    def __call__(self, sample: Any) -> bytes:
        return self.encode(sample)


class SampleDecoder:
    def __init__(self, space: Space, *, copy: bool = False):
        self.space = space
        self.copy = copy

    def decode(self, payload: BytesLike) -> Any:
        return decode_sample(self.space, payload, copy=self.copy)

    def __call__(self, payload: BytesLike) -> Any:
        return self.decode(payload)
