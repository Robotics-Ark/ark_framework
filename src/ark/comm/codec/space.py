from __future__ import annotations

from collections import OrderedDict

from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Text, Tuple

try:
    from gymnasium.spaces import Graph, OneOf, Sequence
except ImportError:
    Graph = OneOf = Sequence = None

from ark.envs.spaces.image_space import DepthImage, GrayscaleImage, Image, RGBImage

from .arrays import (
    read_array,
    read_bound,
    read_dtype,
    read_shape,
    write_array,
    write_bound,
    write_dtype,
    write_shape,
)
from .binary import BytesLike, Reader, Writer
from .tags import VERSION, SpaceTag


def encode_space(space: Space) -> bytes:
    writer = Writer()
    writer.uint(VERSION)
    _write_space(writer, space)
    return writer.finish()


def decode_space(payload: BytesLike) -> Space:
    reader = Reader(payload)
    version = reader.uint()
    if version != VERSION:
        raise ValueError(f"Unsupported space codec version {version}.")
    space = _read_space(reader)
    reader.finish()
    return space


def _write_space(writer: Writer, space: Space) -> None:
    # Image subclasses must be checked before Box (they inherit from it).
    # Image subclasses must be checked before Box (they inherit from it).
    if isinstance(space, RGBImage):
        writer.uint(SpaceTag.RGB_IMAGE)
        writer.uint(space.height)
        writer.uint(space.width)
        write_dtype(writer, space.dtype)
        write_bound(writer, space.high, space.dtype, space.shape)
        return
    if isinstance(space, GrayscaleImage):
        writer.uint(SpaceTag.GRAYSCALE_IMAGE)
        writer.uint(space.height)
        writer.uint(space.width)
        write_dtype(writer, space.dtype)
        write_bound(writer, space.high, space.dtype, space.shape)
        return
    if isinstance(space, DepthImage):
        writer.uint(SpaceTag.DEPTH_IMAGE)
        writer.uint(space.height)
        writer.uint(space.width)
        write_dtype(writer, space.dtype)
        write_bound(writer, space.high, space.dtype, space.shape)
        return
    if isinstance(space, Image):
        writer.uint(SpaceTag.IMAGE)
        writer.uint(space.height)
        writer.uint(space.width)
        writer.uint(space.color_channels)
        write_dtype(writer, space.dtype)
        write_bound(writer, space.high, space.dtype, space.shape)
        return
    if isinstance(space, Box):
        writer.uint(SpaceTag.BOX)
        write_dtype(writer, space.dtype)
        write_shape(writer, space.shape)
        write_bound(writer, space.low, space.dtype, space.shape)
        write_bound(writer, space.high, space.dtype, space.shape)
    elif isinstance(space, Discrete):
        writer.uint(SpaceTag.DISCRETE)
        write_dtype(writer, space.dtype)
        writer.uint(int(space.n))
        writer.int(int(space.start))
    elif isinstance(space, MultiBinary):
        writer.uint(SpaceTag.MULTI_BINARY)
        write_shape(writer, space.shape)
    elif isinstance(space, MultiDiscrete):
        writer.uint(SpaceTag.MULTI_DISCRETE)
        write_dtype(writer, space.dtype)
        write_shape(writer, space.shape)
        write_array(writer, space.nvec, space.dtype, space.shape)
        write_array(writer, space.start, space.dtype, space.shape)
    elif isinstance(space, Text):
        writer.uint(SpaceTag.TEXT)
        writer.uint(space.min_length)
        writer.uint(space.max_length)
        characters = tuple(sorted(space.character_set))
        writer.uint(len(characters))
        for character in characters:
            writer.string(character)
    elif isinstance(space, Dict):
        writer.uint(SpaceTag.DICT)
        writer.uint(len(space.spaces))
        for key, child in space.spaces.items():
            if not isinstance(key, str):
                raise TypeError("Dict space keys must be strings.")
            writer.string(key)
            _write_space(writer, child)
    elif isinstance(space, Tuple):
        writer.uint(SpaceTag.TUPLE)
        writer.uint(len(space.spaces))
        for child in space.spaces:
            _write_space(writer, child)
    elif Sequence is not None and isinstance(space, Sequence):
        writer.uint(SpaceTag.SEQUENCE)
        writer.bool(bool(space.stack))
        _write_space(writer, space.feature_space)
    elif Graph is not None and isinstance(space, Graph):
        writer.uint(SpaceTag.GRAPH)
        _write_space(writer, space.node_space)
        writer.bool(space.edge_space is not None)
        if space.edge_space is not None:
            _write_space(writer, space.edge_space)
    elif OneOf is not None and isinstance(space, OneOf):
        writer.uint(SpaceTag.ONE_OF)
        writer.uint(len(space.spaces))
        for child in space.spaces:
            _write_space(writer, child)
    else:
        raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}.")


def _read_space(reader: Reader) -> Space:
    tag = SpaceTag(reader.uint())
    if tag == SpaceTag.RGB_IMAGE and RGBImage is not None:
        height, width = reader.uint(), reader.uint()
        dtype = read_dtype(reader)
        high = read_bound(reader, dtype, (height, width, 3))
        return RGBImage(height=height, width=width, dtype=dtype, high=high)
    if tag == SpaceTag.GRAYSCALE_IMAGE and GrayscaleImage is not None:
        height, width = reader.uint(), reader.uint()
        dtype = read_dtype(reader)
        high = read_bound(reader, dtype, (height, width))
        return GrayscaleImage(height=height, width=width, dtype=dtype, high=high)
    if tag == SpaceTag.DEPTH_IMAGE and DepthImage is not None:
        height, width = reader.uint(), reader.uint()
        dtype = read_dtype(reader)
        high = read_bound(reader, dtype, (height, width))
        return DepthImage(height=height, width=width, dtype=dtype, high=high)
    if tag == SpaceTag.IMAGE and Image is not None:
        height, width, color_channels = reader.uint(), reader.uint(), reader.uint()
        dtype = read_dtype(reader)
        shape = (height, width) if color_channels == 1 else (height, width, color_channels)
        high = read_bound(reader, dtype, shape)
        return Image(height=height, width=width, color_channels=color_channels, dtype=dtype, high=high)
    if tag == SpaceTag.BOX:
        dtype = read_dtype(reader)
        shape = read_shape(reader)
        low = read_bound(reader, dtype, shape)
        high = read_bound(reader, dtype, shape)
        return Box(low=low, high=high, shape=shape, dtype=dtype)
    if tag == SpaceTag.DISCRETE:
        dtype = read_dtype(reader)
        return Discrete(n=reader.uint(), start=reader.int(), dtype=dtype)
    if tag == SpaceTag.MULTI_BINARY:
        return MultiBinary(read_shape(reader))
    if tag == SpaceTag.MULTI_DISCRETE:
        dtype = read_dtype(reader)
        shape = read_shape(reader)
        nvec = read_array(reader, dtype, shape, copy=True)
        start = read_array(reader, dtype, shape, copy=True)
        return MultiDiscrete(nvec=nvec, start=start, dtype=dtype)
    if tag == SpaceTag.TEXT:
        min_length = reader.uint()
        max_length = reader.uint()
        charset = [reader.string() for _ in range(reader.uint())]
        return Text(max_length=max_length, min_length=min_length, charset=charset)
    if tag == SpaceTag.DICT:
        spaces = OrderedDict()
        for _ in range(reader.uint()):
            spaces[reader.string()] = _read_space(reader)
        return Dict(spaces)
    if tag == SpaceTag.TUPLE:
        return Tuple(tuple(_read_space(reader) for _ in range(reader.uint())))
    if tag == SpaceTag.SEQUENCE and Sequence is not None:
        stack = reader.bool()
        return Sequence(_read_space(reader), stack=stack)
    if tag == SpaceTag.GRAPH and Graph is not None:
        node_space = _read_space(reader)
        edge_space = _read_space(reader) if reader.bool() else None
        return Graph(node_space=node_space, edge_space=edge_space)
    if tag == SpaceTag.ONE_OF and OneOf is not None:
        return OneOf(tuple(_read_space(reader) for _ in range(reader.uint())))
    raise ValueError(f"Unsupported space tag: {int(tag)}.")


class SpaceEncoder:
    def encode(self, space: Space) -> bytes:
        return encode_space(space)

    def __call__(self, space: Space) -> bytes:
        return self.encode(space)


class SpaceDecoder:
    def decode(self, payload: BytesLike) -> Space:
        return decode_space(payload)

    def __call__(self, payload: BytesLike) -> Space:
        return self.decode(payload)
