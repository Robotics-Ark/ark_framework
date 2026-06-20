from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Text, Tuple
from gymnasium.vector import utils as vector_utils

try:
    from gymnasium.spaces import Graph, GraphInstance, OneOf, Sequence
except ImportError:
    Graph = GraphInstance = OneOf = Sequence = None

from ark.envs.spaces.image_space import DepthImage, GrayscaleImage, Image, RGBImage

from .arrays import read_array, read_bool_array, write_array, write_bool_array
from .binary import BytesLike, Reader, Writer
from .compression import JPEGCompression, get_decompressor

_JPEG = JPEGCompression(quality=85)


# ---------------------------------------------------------------------------
# Public one-shot helpers (not for hot paths)
# ---------------------------------------------------------------------------

def encode_sample(space: Space, sample: Any) -> bytes:
    return SampleEncoder(space).encode(sample)


def decode_sample(space: Space, payload: BytesLike, *, copy: bool = False) -> Any:
    return SampleDecoder(space, copy=copy).decode(payload)


# ---------------------------------------------------------------------------
# Encoder / decoder compilation
# ---------------------------------------------------------------------------

def _compile_encoder(space: Space) -> Callable[[Writer, Any], None]:
    """Return a closure that writes a sample for *space* into a Writer.

    All isinstance checks and dtype comparisons happen here at construction
    time, so the returned callable is branch-free for leaf spaces.
    """
    # Image subclasses must be checked before Box (they inherit from it).
    if isinstance(space, RGBImage):
        dtype, shape = space.dtype, space.shape
        def _enc_rgb(writer: Writer, sample: Any) -> None:
            arr = np.asarray(sample, dtype=dtype)
            compressed = _JPEG.compress(arr)
            writer.uint(_JPEG.tag)
            writer.sized_bytes(compressed)
        return _enc_rgb

    if isinstance(space, GrayscaleImage):
        dtype, shape = space.dtype, space.shape
        def _enc_gray(writer: Writer, sample: Any) -> None:
            arr = np.asarray(sample, dtype=dtype)
            compressed = _JPEG.compress(arr)
            writer.uint(_JPEG.tag)
            writer.sized_bytes(compressed)
        return _enc_gray

    if isinstance(space, (DepthImage, Image)):
        # Raw bytes — no compression for depth or unknown Image subclasses.
        dtype, shape = space.dtype, space.shape
        return lambda writer, sample: write_array(writer, sample, dtype, shape)

    if isinstance(space, Box):
        if np.dtype(space.dtype) == np.dtype(np.bool_):
            shape = space.shape
            return lambda writer, sample: write_bool_array(writer, sample, shape)
        dtype, shape = space.dtype, space.shape
        return lambda writer, sample: write_array(writer, sample, dtype, shape)

    if isinstance(space, Discrete):
        start = int(space.start)
        return lambda writer, sample: writer.uint(int(sample) - start)

    if isinstance(space, MultiBinary):
        shape = space.shape
        return lambda writer, sample: write_bool_array(writer, sample, shape)

    if isinstance(space, MultiDiscrete):
        dtype, shape = space.dtype, space.shape
        return lambda writer, sample: write_array(writer, np.asarray(sample), dtype, shape)

    if isinstance(space, Text):
        return lambda writer, sample: writer.string(sample)

    if isinstance(space, Dict):
        child_fns = [(key, _compile_encoder(child)) for key, child in space.spaces.items()]
        def _enc_dict(writer: Writer, sample: Any) -> None:
            for key, fn in child_fns:
                fn(writer, sample[key])
        return _enc_dict

    if isinstance(space, Tuple):
        child_fns = [_compile_encoder(child) for child in space.spaces]
        def _enc_tuple(writer: Writer, sample: Any) -> None:
            for fn, val in zip(child_fns, sample):
                fn(writer, val)
        return _enc_tuple

    if Sequence is not None and isinstance(space, Sequence):
        feature_fn = _compile_encoder(space.feature_space)
        is_stack = space.stack
        stacked_fs = space.stacked_feature_space if is_stack else None
        def _enc_seq(writer: Writer, sample: Any) -> None:
            items = (
                tuple(vector_utils.iterate(stacked_fs, sample))
                if is_stack
                else tuple(sample)
            )
            writer.uint(len(items))
            for item in items:
                feature_fn(writer, item)
        return _enc_seq

    if Graph is not None and isinstance(space, Graph):
        _space = space
        return lambda writer, sample: _write_graph(writer, _space, sample)

    if OneOf is not None and isinstance(space, OneOf):
        child_fns = [_compile_encoder(child) for child in space.spaces]
        def _enc_one_of(writer: Writer, sample: Any) -> None:
            index = int(sample[0])
            writer.uint(index)
            child_fns[index](writer, sample[1])
        return _enc_one_of

    raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}.")


def _compile_decoder(space: Space, *, copy: bool) -> Callable[[Reader], Any]:
    """Return a closure that reads a sample for *space* from a Reader."""
    # Image subclasses must be checked before Box (they inherit from it).
    if isinstance(space, (RGBImage, GrayscaleImage)):
        dtype, shape = space.dtype, space.shape
        def _dec_image(reader: Reader) -> np.ndarray:
            tag = reader.uint()
            data = bytes(reader.sized_bytes())
            return get_decompressor(tag).decompress(data, dtype, shape)
        return _dec_image

    if isinstance(space, (DepthImage, Image)):
        dtype, shape = space.dtype, space.shape
        return lambda reader: read_array(reader, dtype, shape, copy=copy)

    if isinstance(space, Box):
        if np.dtype(space.dtype) == np.dtype(np.bool_):
            shape = space.shape
            return lambda reader: read_bool_array(reader, shape)
        dtype, shape = space.dtype, space.shape
        return lambda reader: read_array(reader, dtype, shape, copy=copy)

    if isinstance(space, Discrete):
        start = int(space.start)
        scalar_type = space.dtype.type
        return lambda reader: scalar_type(start + reader.uint())

    if isinstance(space, MultiBinary):
        dtype = space.dtype
        shape = space.shape
        return lambda reader: read_bool_array(reader, shape).astype(dtype, copy=False)

    if isinstance(space, MultiDiscrete):
        dtype, shape = space.dtype, space.shape
        return lambda reader: read_array(reader, dtype, shape, copy=copy)

    if isinstance(space, Text):
        return lambda reader: reader.string()

    if isinstance(space, Dict):
        child_fns = [(key, _compile_decoder(child, copy=copy)) for key, child in space.spaces.items()]
        def _dec_dict(reader: Reader) -> dict:
            return {key: fn(reader) for key, fn in child_fns}
        return _dec_dict

    if isinstance(space, Tuple):
        child_fns = [_compile_decoder(child, copy=copy) for child in space.spaces]
        return lambda reader: tuple(fn(reader) for fn in child_fns)

    if Sequence is not None and isinstance(space, Sequence):
        feature_fn = _compile_decoder(space.feature_space, copy=copy)
        is_stack = space.stack
        fs = space.feature_space
        def _dec_seq(reader: Reader) -> Any:
            n = reader.uint()
            items = tuple(feature_fn(reader) for _ in range(n))
            if not is_stack:
                return items
            out = vector_utils.create_empty_array(fs, n=n)
            if not items:
                return out
            return vector_utils.concatenate(fs, items, out)
        return _dec_seq

    if Graph is not None and isinstance(space, Graph):
        _space = space
        return lambda reader: _read_graph(reader, _space, copy=copy)

    if OneOf is not None and isinstance(space, OneOf):
        child_fns = [_compile_decoder(child, copy=copy) for child in space.spaces]
        def _dec_one_of(reader: Reader) -> tuple:
            index = reader.uint()
            return np.int64(index), child_fns[index](reader)
        return _dec_one_of

    raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}.")


# ---------------------------------------------------------------------------
# Graph helpers (kept separate — graph spaces are not hot-path)
# ---------------------------------------------------------------------------

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


def _read_feature_array(reader: Reader, space: Space, count: int, *, copy: bool) -> np.ndarray:
    if isinstance(space, Box):
        return read_array(reader, space.dtype, (count,) + space.shape, copy=copy)
    if isinstance(space, Discrete):
        return read_array(reader, space.dtype, (count,), copy=copy)
    raise TypeError("Graph feature spaces must be Box or Discrete.")


# ---------------------------------------------------------------------------
# Public encoder / decoder classes
# ---------------------------------------------------------------------------

class SampleEncoder:
    def __init__(self, space: Space) -> None:
        self._writer = Writer()
        self._fn = _compile_encoder(space)

    def encode(self, sample: Any) -> bytes:
        self._writer.clear()
        self._fn(self._writer, sample)
        return self._writer.finish()

    def __call__(self, sample: Any) -> bytes:
        return self.encode(sample)


class SampleDecoder:
    def __init__(self, space: Space, *, copy: bool = False) -> None:
        self._fn = _compile_decoder(space, copy=copy)

    def decode(self, payload: BytesLike) -> Any:
        reader = Reader(payload)
        result = self._fn(reader)
        reader.finish()
        return result

    def __call__(self, payload: BytesLike) -> Any:
        return self.decode(payload)
