from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
from gymnasium import Space
from gymnasium.spaces import (
    Box,
    Dict as DictSpace,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Sequence as SequenceSpace,
    Text,
    Tuple as TupleSpace,
)
from gymnasium.vector import utils as vector_utils

BytesLike = bytes | bytearray | memoryview
_SPACE_CODEC_VERSION = 1

# Gymnasium space schema tags.
_SPACE_BOX = 1
_SPACE_DISCRETE = 2
_SPACE_MULTI_BINARY = 3
_SPACE_MULTI_DISCRETE = 4
_SPACE_TEXT = 5
_SPACE_DICT = 6
_SPACE_TUPLE = 7
_SPACE_SEQUENCE = 8
_SPACE_GRAPH = 9
_SPACE_ONE_OF = 10

# ARK space schema tags.
_SPACE_ARK_IMAGE = 101
_SPACE_ARK_GRAYSCALE_IMAGE = 102
_SPACE_ARK_RGB_IMAGE = 103
_SPACE_ARK_DEPTH_IMAGE = 104
_SPACE_ARK_RGBD_IMAGE = 105
_SPACE_ARK_TRANSLATION = 106
_SPACE_ARK_ROTATION = 107
_SPACE_ARK_RIGID_TRANSFORM = 108
_SPACE_ARK_JOINT_STATE = 109
_SPACE_ARK_CONTROLLER = 110

_BOUND_SCALAR = 0
_BOUND_ARRAY = 1

_ARK_SPACES_PACKAGE = "ark.envs.spaces"
_JOINT_STATE_OPTIONAL_FIELDS = ("velocity", "effort", "ext_torque")


class _BinaryWriter:
    """Byte buffer with little-endian bit packing for small bounded integers."""

    __slots__ = ("_data", "_bit_buffer", "_bit_count")

    def __init__(self) -> None:
        self._data = bytearray()
        self._bit_buffer = 0
        self._bit_count = 0

    def write_bits(self, value: int, n_bits: int) -> None:
        if n_bits < 0:
            raise ValueError(f"Cannot write a negative number of bits: {n_bits}.")
        if n_bits == 0:
            # Spaces with exactly one possible value need no payload bits.
            if value != 0:
                raise ValueError(f"Cannot encode non-zero value {value} in 0 bits.")
            return
        if value < 0 or value >= (1 << n_bits):
            raise ValueError(f"Value {value} does not fit in {n_bits} bits.")

        self._bit_buffer |= value << self._bit_count
        self._bit_count += n_bits

        while self._bit_count >= 8:
            self._data.append(self._bit_buffer & 0xFF)
            self._bit_buffer >>= 8
            self._bit_count -= 8

    def write_bytes(self, data: BytesLike) -> None:
        # Raw NumPy buffers are byte-addressed, so they must start after any
        # pending packed-bit byte has been flushed.
        self.align_byte()
        self._data.extend(data)

    def write_varuint(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"Cannot encode negative integer {value}.")

        # Variable-length integers are used only for dynamic counts. Aligning
        # keeps their continuation-bit format independent of packed samples.
        self.align_byte()
        while value >= 0x80:
            self._data.append((value & 0x7F) | 0x80)
            value >>= 7
        self._data.append(value)

    def write_varint(self, value: int) -> None:
        encoded = value * 2 if value >= 0 else (-value * 2) - 1
        self.write_varuint(encoded)

    def align_byte(self) -> None:
        """Flush pending write bits and pad the current byte with zeroes."""
        if self._bit_count:
            self._data.append(self._bit_buffer & 0xFF)
            self._bit_buffer = 0
            self._bit_count = 0

    def finish(self) -> bytes:
        self.align_byte()
        return bytes(self._data)


class _BinaryReader:
    """Reader matching _BinaryWriter's byte and little-endian bit format."""

    __slots__ = ("_buffer", "_pos", "_bit_buffer", "_bit_count")

    def __init__(self, payload: BytesLike) -> None:
        self._buffer = memoryview(payload)
        self._pos = 0
        self._bit_buffer = 0
        self._bit_count = 0

    def read_bits(self, n_bits: int) -> int:
        if n_bits < 0:
            raise ValueError(f"Cannot read a negative number of bits: {n_bits}.")
        if n_bits == 0:
            # Matches write_bits(..., 0): a fixed single-value space uses no bytes.
            return 0

        while self._bit_count < n_bits:
            if self._pos >= len(self._buffer):
                raise ValueError("Payload ended while reading packed bits.")
            self._bit_buffer |= int(self._buffer[self._pos]) << self._bit_count
            self._pos += 1
            self._bit_count += 8

        mask = (1 << n_bits) - 1
        value = self._bit_buffer & mask
        self._bit_buffer >>= n_bits
        self._bit_count -= n_bits
        return value

    def read_bytes(self, n_bytes: int) -> memoryview:
        if n_bytes < 0:
            raise ValueError(f"Cannot read a negative number of bytes: {n_bytes}.")
        # Discard any padding bits from the previous packed-bit field before
        # reading byte-aligned data.
        self.align_byte()
        end = self._pos + n_bytes
        if end > len(self._buffer):
            raise ValueError(
                f"Payload ended while reading {n_bytes} bytes at offset {self._pos}."
            )
        data = self._buffer[self._pos : end]
        self._pos = end
        return data

    def read_varuint(self) -> int:
        # Varints are byte-oriented; packed-bit reads may have consumed only
        # part of the current byte, so start at the next payload byte.
        self.align_byte()
        value = 0
        shift = 0

        while True:
            if self._pos >= len(self._buffer):
                raise ValueError("Payload ended while reading variable-length integer.")
            byte = int(self._buffer[self._pos])
            self._pos += 1

            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    def read_varint(self) -> int:
        encoded = self.read_varuint()
        return encoded // 2 if encoded % 2 == 0 else -((encoded // 2) + 1)

    def align_byte(self) -> None:
        """Discard unread padding bits and resume on the next byte boundary."""
        self._bit_buffer = 0
        self._bit_count = 0

    def finish(self) -> None:
        self.align_byte()
        if self._pos != len(self._buffer):
            raise ValueError(
                f"Payload has {len(self._buffer) - self._pos} unread trailing bytes."
            )


def encode_sample(space: Space, sample: Any) -> bytes:
    """Encode a Gymnasium or ARK space sample into compact binary bytes.

    The receiver must decode with the same space schema. Fixed metadata such as
    dtype, shape, dictionary keys, and tuple layout is intentionally omitted from
    the payload.
    """
    writer = _BinaryWriter()
    _encode(space, sample, writer)
    return writer.finish()


def decode_sample(space: Space, payload: BytesLike, *, copy: bool = False) -> Any:
    """Decode bytes produced by encode_sample for a Gymnasium or ARK space.

    Numeric Box arrays are zero-copy views by default when possible. Pass
    copy=True if decoded arrays must be writable and independent of the payload.
    """
    reader = _BinaryReader(payload)
    sample = _decode(space, reader, copy=copy)
    reader.finish()
    return sample


def encode_space(space: Space) -> bytes:
    """Encode a Gymnasium or ARK space schema into compact binary bytes."""
    writer = _BinaryWriter()
    writer.write_varuint(_SPACE_CODEC_VERSION)
    _encode_space(space, writer)
    return writer.finish()


def decode_space(payload: BytesLike) -> Space:
    """Decode bytes produced by encode_space into a Gymnasium or ARK space."""
    reader = _BinaryReader(payload)
    version = reader.read_varuint()
    if version != _SPACE_CODEC_VERSION:
        raise ValueError(
            f"Unsupported space codec version {version}; "
            f"expected {_SPACE_CODEC_VERSION}."
        )

    space = _decode_space(reader)
    reader.finish()
    return space


def _encode(space: Space, sample: Any, writer: _BinaryWriter) -> None:
    # ARK spaces.
    if _is_ark_space(space, "channel_space", "ChannelSpace"):
        raise TypeError(
            "ARK ChannelSpace and subclasses are internal communication helpers "
            "and are not supported by sample serialization."
        )
    elif _is_ark_space(space, "geometry_space", "Rotation"):
        _encode_rotation(space, sample, writer)
    elif _is_ark_space(space, "geometry_space", "RigidTransform"):
        _encode_rigid_transform(space, sample, writer)
    elif _is_ark_space(space, "image_space", "RGBDImage"):
        _encode_rgbd_image(space, sample, writer)
    # Gymnasium spaces. ARK Box/Dict subclasses use these payload encoders once
    # their concrete schema has been preserved by encode_space/decode_space.
    elif isinstance(space, Box):
        _encode_box_array(space, sample, space.shape, writer)
    elif isinstance(space, Discrete):
        _encode_discrete(space, sample, writer)
    elif isinstance(space, MultiBinary):
        _encode_multi_binary(space, sample, writer)
    elif isinstance(space, MultiDiscrete):
        _encode_multi_discrete(space, sample, writer)
    elif isinstance(space, Text):
        _encode_text(space, sample, writer)
    elif isinstance(space, DictSpace):
        _encode_dict(space, sample, writer)
    elif isinstance(space, TupleSpace):
        _encode_tuple(space, sample, writer)
    elif isinstance(space, SequenceSpace):
        _encode_sequence(space, sample, writer)
    elif isinstance(space, Graph):
        _encode_graph(space, sample, writer)
    elif isinstance(space, OneOf):
        _encode_one_of(space, sample, writer)
    else:
        raise TypeError(f"Unsupported space type: {type(space).__name__}.")


def _decode(space: Space, reader: _BinaryReader, *, copy: bool) -> Any:
    # ARK spaces.
    if _is_ark_space(space, "geometry_space", "Rotation"):
        return _decode_rotation(space, reader, copy=copy)
    # Gymnasium spaces. ARK Box/Dict subclasses use these payload decoders once
    # their concrete schema has been preserved by encode_space/decode_space.
    elif isinstance(space, Box):
        return _decode_box_array(space, space.shape, reader, copy=copy)
    elif isinstance(space, Discrete):
        return _decode_discrete(space, reader)
    elif isinstance(space, MultiBinary):
        return _decode_multi_binary(space, reader)
    elif isinstance(space, MultiDiscrete):
        return _decode_multi_discrete(space, reader)
    elif isinstance(space, Text):
        return _decode_text(space, reader)
    elif isinstance(space, DictSpace):
        return _decode_dict(space, reader, copy=copy)
    elif isinstance(space, TupleSpace):
        return _decode_tuple(space, reader, copy=copy)
    elif isinstance(space, SequenceSpace):
        return _decode_sequence(space, reader, copy=copy)
    elif isinstance(space, Graph):
        return _decode_graph(space, reader, copy=copy)
    elif isinstance(space, OneOf):
        return _decode_one_of(space, reader, copy=copy)
    else:
        raise TypeError(f"Unsupported space type: {type(space).__name__}.")


def _encode_space(space: Space, writer: _BinaryWriter) -> None:
    # ARK spaces. Keep these checks before the Gymnasium base classes: most ARK
    # spaces subclass Box or Dict, so order is what preserves the concrete type.
    if _is_ark_space(space, "image_space", "GrayscaleImage"):
        writer.write_varuint(_SPACE_ARK_GRAYSCALE_IMAGE)
        _encode_ark_image_space(space, writer)
    elif _is_ark_space(space, "image_space", "RGBImage"):
        writer.write_varuint(_SPACE_ARK_RGB_IMAGE)
        _encode_ark_image_space(space, writer)
    elif _is_ark_space(space, "image_space", "DepthImage"):
        writer.write_varuint(_SPACE_ARK_DEPTH_IMAGE)
        _encode_ark_image_space(space, writer)
    elif _is_ark_space(space, "image_space", "Image"):
        writer.write_varuint(_SPACE_ARK_IMAGE)
        _encode_ark_image_space(space, writer, include_color_channels=True)
    elif _is_ark_space(space, "image_space", "RGBDImage"):
        writer.write_varuint(_SPACE_ARK_RGBD_IMAGE)
        _encode_ark_rgbd_image_space(space, writer)
    elif _is_ark_space(space, "geometry_space", "Translation"):
        writer.write_varuint(_SPACE_ARK_TRANSLATION)
        _encode_ark_translation_space(space, writer)
    elif _is_ark_space(space, "geometry_space", "Rotation"):
        writer.write_varuint(_SPACE_ARK_ROTATION)
        _encode_ark_rotation_space(space, writer)
    elif _is_ark_space(space, "geometry_space", "RigidTransform"):
        writer.write_varuint(_SPACE_ARK_RIGID_TRANSFORM)
        _encode_ark_rigid_transform_space(space, writer)
    elif _is_ark_space(space, "sensor_space", "JointState"):
        writer.write_varuint(_SPACE_ARK_JOINT_STATE)
        _encode_ark_joint_state_space(space, writer)
    elif _is_ark_space(space, "sensor_space", "Controller"):
        writer.write_varuint(_SPACE_ARK_CONTROLLER)
        _encode_ark_controller_space(space, writer)
    elif _is_ark_space(space, "channel_space", "ChannelSpace"):
        raise TypeError(
            "ARK ChannelSpace and subclasses are internal communication helpers "
            "and are not supported by space serialization."
        )
    # Gymnasium spaces.
    elif isinstance(space, Box):
        writer.write_varuint(_SPACE_BOX)
        _encode_box_space(space, writer)
    elif isinstance(space, Discrete):
        writer.write_varuint(_SPACE_DISCRETE)
        _encode_discrete_space(space, writer)
    elif isinstance(space, MultiBinary):
        writer.write_varuint(_SPACE_MULTI_BINARY)
        _encode_multi_binary_space(space, writer)
    elif isinstance(space, MultiDiscrete):
        writer.write_varuint(_SPACE_MULTI_DISCRETE)
        _encode_multi_discrete_space(space, writer)
    elif isinstance(space, Text):
        writer.write_varuint(_SPACE_TEXT)
        _encode_text_space(space, writer)
    elif isinstance(space, DictSpace):
        writer.write_varuint(_SPACE_DICT)
        _encode_dict_space(space, writer)
    elif isinstance(space, TupleSpace):
        writer.write_varuint(_SPACE_TUPLE)
        _encode_tuple_space(space, writer)
    elif isinstance(space, SequenceSpace):
        writer.write_varuint(_SPACE_SEQUENCE)
        _encode_sequence_space(space, writer)
    elif isinstance(space, Graph):
        writer.write_varuint(_SPACE_GRAPH)
        _encode_graph_space(space, writer)
    elif isinstance(space, OneOf):
        writer.write_varuint(_SPACE_ONE_OF)
        _encode_one_of_space(space, writer)
    else:
        raise TypeError(f"Unsupported space type: {type(space).__name__}.")


def _decode_space(reader: _BinaryReader) -> Space:
    tag = reader.read_varuint()
    # ARK spaces.
    if tag == _SPACE_ARK_GRAYSCALE_IMAGE:
        return _decode_ark_image_space(reader, "GrayscaleImage", color_channels=1)
    elif tag == _SPACE_ARK_RGB_IMAGE:
        return _decode_ark_image_space(reader, "RGBImage", color_channels=3)
    elif tag == _SPACE_ARK_DEPTH_IMAGE:
        return _decode_ark_image_space(reader, "DepthImage", color_channels=1)
    elif tag == _SPACE_ARK_IMAGE:
        return _decode_ark_base_image_space(reader)
    elif tag == _SPACE_ARK_RGBD_IMAGE:
        return _decode_ark_rgbd_image_space(reader)
    elif tag == _SPACE_ARK_TRANSLATION:
        return _decode_ark_translation_space(reader)
    elif tag == _SPACE_ARK_ROTATION:
        return _decode_ark_rotation_space(reader)
    elif tag == _SPACE_ARK_RIGID_TRANSFORM:
        return _decode_ark_rigid_transform_space(reader)
    elif tag == _SPACE_ARK_JOINT_STATE:
        return _decode_ark_joint_state_space(reader)
    elif tag == _SPACE_ARK_CONTROLLER:
        return _decode_ark_controller_space(reader)
    # Gymnasium spaces.
    elif tag == _SPACE_BOX:
        return _decode_box_space(reader)
    elif tag == _SPACE_DISCRETE:
        return _decode_discrete_space(reader)
    elif tag == _SPACE_MULTI_BINARY:
        return _decode_multi_binary_space(reader)
    elif tag == _SPACE_MULTI_DISCRETE:
        return _decode_multi_discrete_space(reader)
    elif tag == _SPACE_TEXT:
        return _decode_text_space(reader)
    elif tag == _SPACE_DICT:
        return _decode_dict_space(reader)
    elif tag == _SPACE_TUPLE:
        return _decode_tuple_space(reader)
    elif tag == _SPACE_SEQUENCE:
        return _decode_sequence_space(reader)
    elif tag == _SPACE_GRAPH:
        return _decode_graph_space(reader)
    elif tag == _SPACE_ONE_OF:
        return _decode_one_of_space(reader)
    else:
        raise ValueError(f"Unsupported space tag: {tag}.")


def _encode_box_space(space: Box, writer: _BinaryWriter) -> None:
    _write_dtype(writer, space.dtype)
    _write_shape(writer, space.shape)
    _write_box_bound(writer, space.low, space.dtype)
    _write_box_bound(writer, space.high, space.dtype)


def _decode_box_space(reader: _BinaryReader) -> Box:
    dtype = _read_dtype(reader)
    shape = _read_shape(reader)
    low = _read_box_bound(reader, dtype, shape)
    high = _read_box_bound(reader, dtype, shape)
    return Box(low=low, high=high, shape=shape, dtype=dtype)


def _encode_discrete_space(space: Discrete, writer: _BinaryWriter) -> None:
    _write_dtype(writer, space.dtype)
    writer.write_varuint(int(space.n))
    writer.write_varint(int(space.start))


def _decode_discrete_space(reader: _BinaryReader) -> Discrete:
    dtype = _read_dtype(reader)
    n = reader.read_varuint()
    start = reader.read_varint()
    return Discrete(n=n, start=start, dtype=dtype)


def _encode_multi_binary_space(space: MultiBinary, writer: _BinaryWriter) -> None:
    if isinstance(space.n, int):
        writer.write_bits(0, 1)
        writer.write_varuint(space.n)
    else:
        writer.write_bits(1, 1)
        _write_shape(writer, space.shape)


def _decode_multi_binary_space(reader: _BinaryReader) -> MultiBinary:
    is_shaped = bool(reader.read_bits(1))
    n = _read_shape(reader) if is_shaped else reader.read_varuint()
    return MultiBinary(n)


def _encode_multi_discrete_space(space: MultiDiscrete, writer: _BinaryWriter) -> None:
    _write_dtype(writer, space.dtype)
    _write_shape(writer, space.shape)
    _write_ndarray(writer, space.nvec, space.dtype, space.shape)
    _write_ndarray(writer, space.start, space.dtype, space.shape)


def _decode_multi_discrete_space(reader: _BinaryReader) -> MultiDiscrete:
    dtype = _read_dtype(reader)
    shape = _read_shape(reader)
    nvec = _read_ndarray(reader, dtype, shape, copy=True)
    start = _read_ndarray(reader, dtype, shape, copy=True)
    return MultiDiscrete(nvec=nvec, start=start, dtype=dtype)


def _encode_text_space(space: Text, writer: _BinaryWriter) -> None:
    writer.write_varuint(space.min_length)
    writer.write_varuint(space.max_length)

    characters = _text_characters(space)
    writer.write_varuint(len(characters))
    for char in characters:
        _write_string(writer, char)


def _decode_text_space(reader: _BinaryReader) -> Text:
    min_length = reader.read_varuint()
    max_length = reader.read_varuint()
    characters = [_read_string(reader) for _ in range(reader.read_varuint())]
    return Text(max_length=max_length, min_length=min_length, charset=characters)


def _encode_dict_space(space: DictSpace, writer: _BinaryWriter) -> None:
    writer.write_varuint(len(space.spaces))
    for key, subspace in space.spaces.items():
        if not isinstance(key, str):
            raise TypeError(f"Dict space keys must be str, got {type(key).__name__}.")
        _write_string(writer, key)
        _encode_space(subspace, writer)


def _decode_dict_space(reader: _BinaryReader) -> DictSpace:
    spaces = []
    for _ in range(reader.read_varuint()):
        spaces.append((_read_string(reader), _decode_space(reader)))
    return DictSpace(spaces)


def _encode_tuple_space(space: TupleSpace, writer: _BinaryWriter) -> None:
    writer.write_varuint(len(space.spaces))
    for subspace in space.spaces:
        _encode_space(subspace, writer)


def _decode_tuple_space(reader: _BinaryReader) -> TupleSpace:
    return TupleSpace(
        tuple(_decode_space(reader) for _ in range(reader.read_varuint()))
    )


def _encode_sequence_space(space: SequenceSpace, writer: _BinaryWriter) -> None:
    writer.write_bits(int(space.stack), 1)
    _encode_space(space.feature_space, writer)


def _decode_sequence_space(reader: _BinaryReader) -> SequenceSpace:
    stack = bool(reader.read_bits(1))
    return SequenceSpace(_decode_space(reader), stack=stack)


def _encode_graph_space(space: Graph, writer: _BinaryWriter) -> None:
    _encode_space(space.node_space, writer)
    writer.write_bits(int(space.edge_space is not None), 1)
    if space.edge_space is not None:
        _encode_space(space.edge_space, writer)


def _decode_graph_space(reader: _BinaryReader) -> Graph:
    node_space = _decode_space(reader)
    if not isinstance(node_space, (Box, Discrete)):
        raise ValueError("Decoded Graph node_space must be Box or Discrete.")

    edge_space = None
    if reader.read_bits(1):
        edge_space = _decode_space(reader)
        if not isinstance(edge_space, (Box, Discrete)):
            raise ValueError("Decoded Graph edge_space must be Box or Discrete.")

    return Graph(node_space=node_space, edge_space=edge_space)


def _encode_one_of_space(space: OneOf, writer: _BinaryWriter) -> None:
    writer.write_varuint(len(space.spaces))
    for subspace in space.spaces:
        _encode_space(subspace, writer)


def _decode_one_of_space(reader: _BinaryReader) -> OneOf:
    return OneOf(tuple(_decode_space(reader) for _ in range(reader.read_varuint())))


def _encode_ark_image_space(
    space: Space, writer: _BinaryWriter, *, include_color_channels: bool = False
) -> None:
    if not isinstance(space, Box):
        raise TypeError(f"ARK image space must be Box-backed, got {type(space).__name__}.")

    height = _positive_int_attr(space, "height")
    width = _positive_int_attr(space, "width")
    color_channels = _positive_int_attr(space, "color_channels")
    expected_shape = _image_shape(height, width, color_channels)
    if space.shape != expected_shape:
        raise ValueError(
            f"{type(space).__name__} shape {space.shape} does not match "
            f"{expected_shape}."
        )
    _require_zero_low(space, type(space).__name__)

    _write_dtype(writer, space.dtype)
    writer.write_varuint(height)
    writer.write_varuint(width)
    if include_color_channels:
        writer.write_varuint(color_channels)
    _write_box_bound(writer, space.high, space.dtype)


def _decode_ark_base_image_space(reader: _BinaryReader) -> Space:
    Image = _load_ark_attr("image_space", "Image")
    dtype = _read_dtype(reader)
    height = reader.read_varuint()
    width = reader.read_varuint()
    color_channels = reader.read_varuint()
    high = _read_box_bound(reader, dtype, _image_shape(height, width, color_channels))
    return Image(
        height=height,
        width=width,
        color_channels=color_channels,
        dtype=dtype,
        high=high,
    )


def _decode_ark_image_space(
    reader: _BinaryReader, class_name: str, *, color_channels: int
) -> Space:
    image_cls = _load_ark_attr("image_space", class_name)
    dtype = _read_dtype(reader)
    height = reader.read_varuint()
    width = reader.read_varuint()
    high = _read_box_bound(reader, dtype, _image_shape(height, width, color_channels))
    return image_cls(height=height, width=width, dtype=dtype, high=high)


def _encode_ark_rgbd_image_space(space: Space, writer: _BinaryWriter) -> None:
    if not isinstance(space, DictSpace):
        raise TypeError(f"RGBDImage must be Dict-backed, got {type(space).__name__}.")

    rgb_space = space.spaces.get("rgb")
    depth_space = space.spaces.get("depth")
    if not _is_ark_space(rgb_space, "image_space", "RGBImage"):
        raise ValueError("RGBDImage must contain an ARK RGBImage at key 'rgb'.")
    if not _is_ark_space(depth_space, "image_space", "DepthImage"):
        raise ValueError("RGBDImage must contain an ARK DepthImage at key 'depth'.")

    height = _positive_int_attr(rgb_space, "height")
    width = _positive_int_attr(rgb_space, "width")
    if (_positive_int_attr(depth_space, "height"), _positive_int_attr(depth_space, "width")) != (
        height,
        width,
    ):
        raise ValueError("RGBDImage rgb and depth spaces must have matching dimensions.")

    _require_zero_low(rgb_space, "RGBDImage rgb")
    _require_zero_low(depth_space, "RGBDImage depth")

    writer.write_varuint(height)
    writer.write_varuint(width)
    _write_dtype(writer, rgb_space.dtype)
    _write_box_bound(writer, rgb_space.high, rgb_space.dtype)
    _write_dtype(writer, depth_space.dtype)
    _write_box_bound(writer, depth_space.high, depth_space.dtype)


def _decode_ark_rgbd_image_space(reader: _BinaryReader) -> Space:
    RGBDImage = _load_ark_attr("image_space", "RGBDImage")
    height = reader.read_varuint()
    width = reader.read_varuint()

    rgb_dtype = _read_dtype(reader)
    rgb_high = _read_box_bound(reader, rgb_dtype, _image_shape(height, width, 3))
    depth_dtype = _read_dtype(reader)
    depth_high = _read_box_bound(reader, depth_dtype, _image_shape(height, width, 1))

    return RGBDImage(
        height=height,
        width=width,
        rgb_dtype=rgb_dtype,
        depth_dtype=depth_dtype,
        rgb_high=rgb_high,
        depth_high=depth_high,
    )


def _encode_ark_translation_space(space: Space, writer: _BinaryWriter) -> None:
    if not isinstance(space, Box):
        raise TypeError(f"Translation must be Box-backed, got {type(space).__name__}.")
    if space.shape != (3,):
        raise ValueError(f"Translation shape {space.shape} does not match (3,).")

    _write_dtype(writer, space.dtype)
    _write_box_bound(writer, space.low, space.dtype)
    _write_box_bound(writer, space.high, space.dtype)


def _decode_ark_translation_space(reader: _BinaryReader) -> Space:
    Translation = _load_ark_attr("geometry_space", "Translation")
    dtype = _read_dtype(reader)
    low = _read_box_bound(reader, dtype, (3,))
    high = _read_box_bound(reader, dtype, (3,))
    return Translation(low=low, high=high, dtype=dtype)


def _encode_ark_rotation_space(space: Space, writer: _BinaryWriter) -> None:
    _write_dtype(writer, space.dtype)


def _decode_ark_rotation_space(reader: _BinaryReader) -> Space:
    Rotation = _load_ark_attr("geometry_space", "Rotation")
    return Rotation(dtype=_read_dtype(reader))


def _encode_ark_rigid_transform_space(space: Space, writer: _BinaryWriter) -> None:
    if not isinstance(space, DictSpace):
        raise TypeError(f"RigidTransform must be Dict-backed, got {type(space).__name__}.")

    translation = space.spaces.get("translation")
    rotation = space.spaces.get("rotation")
    if not _is_ark_space(translation, "geometry_space", "Translation"):
        raise ValueError(
            "RigidTransform must contain an ARK Translation at key 'translation'."
        )
    if not _is_ark_space(rotation, "geometry_space", "Rotation"):
        raise ValueError("RigidTransform must contain an ARK Rotation at key 'rotation'.")
    if np.dtype(translation.dtype) != np.dtype(rotation.dtype):
        raise ValueError("RigidTransform translation and rotation dtypes must match.")

    _write_dtype(writer, translation.dtype)
    _write_box_bound(writer, translation.low, translation.dtype)
    _write_box_bound(writer, translation.high, translation.dtype)


def _decode_ark_rigid_transform_space(reader: _BinaryReader) -> Space:
    RigidTransform = _load_ark_attr("geometry_space", "RigidTransform")
    dtype = _read_dtype(reader)
    translation_low = _read_box_bound(reader, dtype, (3,))
    translation_high = _read_box_bound(reader, dtype, (3,))
    return RigidTransform(
        translation_low=translation_low,
        translation_high=translation_high,
        dtype=dtype,
    )


def _encode_ark_joint_state_space(space: Space, writer: _BinaryWriter) -> None:
    if not isinstance(space, DictSpace):
        raise TypeError(f"JointState must be Dict-backed, got {type(space).__name__}.")

    joint_names = list(getattr(space, "joint_names"))
    dof = int(getattr(space, "dof"))
    if len(joint_names) != dof:
        raise ValueError("JointState dof does not match the number of joint_names.")

    writer.write_varuint(len(joint_names))
    for name in joint_names:
        if not isinstance(name, str):
            raise TypeError(
                f"JointState joint names must be str, got {type(name).__name__}."
            )
        _write_string(writer, name)

    shape = (dof,)
    position = _require_box_subspace(space, "position", shape)
    dtype = np.dtype(position.dtype)
    _write_dtype(writer, dtype)
    _write_box_bound(writer, position.low, dtype)
    _write_box_bound(writer, position.high, dtype)

    for key in _JOINT_STATE_OPTIONAL_FIELDS:
        subspace = space.spaces.get(key)
        writer.write_bits(int(subspace is not None), 1)
        if subspace is None:
            continue
        subspace = _require_box_subspace(space, key, shape, dtype=dtype)
        _write_box_bound(writer, subspace.low, dtype)
        _write_box_bound(writer, subspace.high, dtype)


def _decode_ark_joint_state_space(reader: _BinaryReader) -> Space:
    JointState = _load_ark_attr("sensor_space", "JointState")
    Limits = _load_ark_attr("sensor_space", "Limits")

    joint_names = [_read_string(reader) for _ in range(reader.read_varuint())]
    dtype = _read_dtype(reader)
    shape = (len(joint_names),)

    position_limits = Limits(
        lower=_read_box_bound(reader, dtype, shape),
        upper=_read_box_bound(reader, dtype, shape),
    )

    optional_limits: dict[str, Any] = {}
    for key in _JOINT_STATE_OPTIONAL_FIELDS:
        if reader.read_bits(1):
            optional_limits[f"{key}_limits"] = Limits(
                lower=_read_box_bound(reader, dtype, shape),
                upper=_read_box_bound(reader, dtype, shape),
            )

    return JointState(
        joint_names=joint_names,
        position_limits=position_limits,
        dtype=dtype,
        **optional_limits,
    )


def _encode_ark_controller_space(space: Space, writer: _BinaryWriter) -> None:
    if not isinstance(space, DictSpace):
        raise TypeError(f"Controller must be Dict-backed, got {type(space).__name__}.")

    n_axes = int(getattr(space, "n_axes"))
    n_buttons = int(getattr(space, "n_buttons"))
    if n_axes < 0 or n_buttons < 0:
        raise ValueError("Controller axis and button counts must be non-negative.")

    axis_dtype = np.dtype(np.float32)
    if n_axes:
        axes = _require_box_subspace(space, "axes", (n_axes,))
        axis_dtype = np.dtype(axes.dtype)
    if n_buttons:
        buttons = space.spaces.get("buttons")
        if not isinstance(buttons, MultiBinary) or buttons.shape != (n_buttons,):
            raise ValueError(
                f"Controller buttons space must be MultiBinary({n_buttons})."
            )

    writer.write_varuint(n_axes)
    writer.write_varuint(n_buttons)
    _write_dtype(writer, axis_dtype)


def _decode_ark_controller_space(reader: _BinaryReader) -> Space:
    Controller = _load_ark_attr("sensor_space", "Controller")
    n_axes = reader.read_varuint()
    n_buttons = reader.read_varuint()
    axis_dtype = _read_dtype(reader)
    return Controller(n_axes=n_axes, n_buttons=n_buttons, axis_dtype=axis_dtype)


def _encode_box_array(
    space: Box, sample: Any, expected_shape: tuple[int, ...], writer: _BinaryWriter
) -> None:
    _write_ndarray(writer, sample, space.dtype, expected_shape, "Box sample")


def _decode_box_array(
    space: Box,
    expected_shape: tuple[int, ...],
    reader: _BinaryReader,
    *,
    copy: bool,
) -> np.ndarray:
    return _read_ndarray(reader, space.dtype, expected_shape, copy=copy)


def _encode_rotation(space: Space, sample: Any, writer: _BinaryWriter) -> None:
    if hasattr(sample, "as_quat"):
        sample = sample.as_quat()
    _write_ndarray(writer, sample, space.dtype, (4,), "Rotation sample")


def _decode_rotation(
    space: Space, reader: _BinaryReader, *, copy: bool
) -> np.ndarray:
    return _read_ndarray(reader, space.dtype, (4,), copy=copy)


def _encode_rigid_transform(
    space: DictSpace, sample: Any, writer: _BinaryWriter
) -> None:
    if not isinstance(sample, Mapping):
        array = np.asarray(sample)
        if array.shape == (7,):
            sample = {"translation": array[:3], "rotation": array[3:]}
        elif array.shape == (4, 4):
            ScipyRotation = _load_ark_attr("geometry_space", "ScipyRotation")
            sample = {
                "translation": array[:3, 3],
                "rotation": ScipyRotation.from_matrix(array[:3, :3]).as_quat(),
            }
        elif hasattr(sample, "translation") and hasattr(sample, "rotation"):
            sample = {
                "translation": sample.translation,
                "rotation": sample.rotation.as_quat(),
            }
        else:
            raise ValueError(
                "RigidTransform sample must be a mapping, 7D vector, 4x4 matrix, "
                "or scipy RigidTransform."
            )

    _encode_dict(space, sample, writer)


def _encode_rgbd_image(space: DictSpace, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, Mapping):
        array = np.asarray(sample)
        rgb_space = space.spaces["rgb"]
        depth_space = space.spaces["depth"]
        expected_shape = rgb_space.shape[:2] + (4,)
        if array.shape != expected_shape:
            raise ValueError(
                f"RGBDImage array sample shape {array.shape} does not match "
                f"{expected_shape}."
            )
        sample = {
            "rgb": array[:, :, :3],
            "depth": array[:, :, 3].reshape(depth_space.shape),
        }

    _encode_dict(space, sample, writer)


def _encode_discrete(space: Discrete, sample: Any, writer: _BinaryWriter) -> None:
    if not space.contains(sample):
        raise ValueError(f"Discrete sample {sample!r} is not contained in {space}.")

    offset = int(sample) - int(space.start)
    n = int(space.n)
    writer.write_bits(offset, _bits_required(n))


def _decode_discrete(space: Discrete, reader: _BinaryReader) -> np.integer:
    n = int(space.n)
    offset = reader.read_bits(_bits_required(n))
    if offset >= n:
        raise ValueError(f"Decoded Discrete offset {offset} is outside range {n}.")
    return np.dtype(space.dtype).type(int(space.start) + offset)


def _encode_multi_binary(
    space: MultiBinary, sample: Any, writer: _BinaryWriter
) -> None:
    array = np.asarray(sample)
    if array.shape != space.shape:
        raise ValueError(
            f"MultiBinary sample shape {array.shape} does not match {space.shape}."
        )
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("MultiBinary samples can only contain 0 or 1 values.")

    _write_uint_array(writer, array.reshape(-1), [1] * array.size)


def _decode_multi_binary(space: MultiBinary, reader: _BinaryReader) -> np.ndarray:
    n_items = _shape_size(space.shape)
    values = _read_uint_array(reader, [1] * n_items)
    return np.asarray(values, dtype=space.dtype).reshape(space.shape)


def _encode_multi_discrete(
    space: MultiDiscrete, sample: Any, writer: _BinaryWriter
) -> None:
    array = np.asarray(sample)
    if array.shape != space.shape:
        raise ValueError(
            f"MultiDiscrete sample shape {array.shape} does not match {space.shape}."
        )

    values: list[int] = []
    bit_widths: list[int] = []
    for value, start, n in zip(
        array.reshape(-1), space.start.reshape(-1), space.nvec.reshape(-1)
    ):
        int_value = _as_integer(value, "MultiDiscrete value")
        offset = int_value - int(start)
        count = int(n)
        if offset < 0 or offset >= count:
            raise ValueError(
                f"MultiDiscrete value {value!r} is not contained in {space}."
            )
        values.append(offset)
        bit_widths.append(_bits_required(count))

    _write_uint_array(writer, values, bit_widths)


def _decode_multi_discrete(space: MultiDiscrete, reader: _BinaryReader) -> np.ndarray:
    bit_widths = [_bits_required(int(n)) for n in space.nvec.reshape(-1)]
    offsets = _read_uint_array(reader, bit_widths)

    values: list[int] = []
    for offset, start, n in zip(
        offsets, space.start.reshape(-1), space.nvec.reshape(-1)
    ):
        count = int(n)
        if offset >= count:
            raise ValueError(
                f"Decoded MultiDiscrete offset {offset} is outside range {count}."
            )
        values.append(int(start) + offset)

    return np.asarray(values, dtype=space.dtype).reshape(space.shape)


def _encode_text(space: Text, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, str) or not space.contains(sample):
        raise ValueError(f"Text sample {sample!r} is not contained in {space}.")

    length = len(sample)
    length_range = space.max_length - space.min_length + 1
    writer.write_bits(length - space.min_length, _bits_required(length_range))

    characters = _text_characters(space)
    if not characters:
        # Text(max_length=0, charset="") is legal. It can only encode "" and
        # therefore needs no character payload.
        if length:
            raise ValueError("Cannot encode non-empty text with an empty charset.")
        return

    char_to_index = {char: i for i, char in enumerate(characters)}
    char_bits = _bits_required(len(characters))
    for char in sample:
        writer.write_bits(char_to_index[char], char_bits)


def _decode_text(space: Text, reader: _BinaryReader) -> str:
    length_range = space.max_length - space.min_length + 1
    length = space.min_length + reader.read_bits(_bits_required(length_range))
    if length > space.max_length:
        raise ValueError(f"Decoded Text length {length} exceeds {space.max_length}.")

    characters = _text_characters(space)
    if not characters:
        # See the matching encode branch: empty charset means only "" is valid.
        if length:
            raise ValueError(
                "Decoded non-empty text for a space with an empty charset."
            )
        return ""

    char_bits = _bits_required(len(characters))
    decoded_chars: list[str] = []
    for _ in range(length):
        index = reader.read_bits(char_bits)
        if index >= len(characters):
            raise ValueError(f"Decoded Text character index {index} is out of range.")
        decoded_chars.append(characters[index])

    return "".join(decoded_chars)


def _encode_dict(space: DictSpace, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, Mapping):
        raise ValueError(f"Dict sample must be a mapping, got {type(sample).__name__}.")
    if sample.keys() != space.spaces.keys():
        raise ValueError(
            f"Dict sample keys {sample.keys()} do not match space keys "
            f"{space.spaces.keys()}."
        )

    # Keys are part of the shared schema, so only values are serialized.
    for key, subspace in space.spaces.items():
        _encode(subspace, sample[key], writer)


def _decode_dict(
    space: DictSpace, reader: _BinaryReader, *, copy: bool
) -> dict[str, Any]:
    return {
        key: _decode(subspace, reader, copy=copy)
        for key, subspace in space.spaces.items()
    }


def _encode_tuple(space: TupleSpace, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, tuple) or len(sample) != len(space.spaces):
        raise ValueError(f"Tuple sample must be a tuple of length {len(space.spaces)}.")

    for subspace, item in zip(space.spaces, sample):
        _encode(subspace, item, writer)


def _decode_tuple(
    space: TupleSpace, reader: _BinaryReader, *, copy: bool
) -> tuple[Any, ...]:
    return tuple(_decode(subspace, reader, copy=copy) for subspace in space.spaces)


def _encode_sequence(space: SequenceSpace, sample: Any, writer: _BinaryWriter) -> None:
    if space.stack:
        # Gymnasium represents stacked sequences as batched arrays/structures;
        # iterate converts them back to per-element samples for recursive coding.
        items = tuple(vector_utils.iterate(space.stacked_feature_space, sample))
    else:
        if not isinstance(sample, tuple):
            raise ValueError("Unstacked Sequence samples must be tuples.")
        items = sample

    writer.write_varuint(len(items))
    for item in items:
        _encode(space.feature_space, item, writer)


def _decode_sequence(space: SequenceSpace, reader: _BinaryReader, *, copy: bool) -> Any:
    length = reader.read_varuint()
    items = tuple(
        _decode(space.feature_space, reader, copy=copy) for _ in range(length)
    )

    if not space.stack:
        return items
    if length == 0:
        # concatenate() needs at least one item, so Gymnasium's helper builds the
        # correct empty stacked structure directly.
        return vector_utils.create_empty_array(space.feature_space, n=0)

    out = vector_utils.create_empty_array(space.feature_space, n=length)
    return vector_utils.concatenate(space.feature_space, items, out)


def _encode_graph(space: Graph, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, GraphInstance):
        raise ValueError(
            f"Graph sample must be a GraphInstance, got {type(sample).__name__}."
        )

    nodes = np.asarray(sample.nodes)
    num_nodes = len(nodes)
    writer.write_varuint(num_nodes)
    _encode_graph_feature_array(space.node_space, nodes, num_nodes, writer)

    if space.edge_space is None:
        # If the schema says there are no edge features, there is no edge section
        # to read or write.
        if sample.edges is not None or sample.edge_links is not None:
            raise ValueError("Graph sample has edges but the Graph edge_space is None.")
        return

    if sample.edges is None or sample.edge_links is None:
        if sample.edges is not None or sample.edge_links is not None:
            raise ValueError("Graph edges and edge_links must both be present or None.")
        # Marker 0 means "edges are absent". Present-but-empty edges use marker 1,
        # which preserves the distinction Gymnasium's GraphInstance can express.
        writer.write_varuint(0)
        return

    edges = np.asarray(sample.edges)
    edge_links = np.asarray(sample.edge_links)
    num_edges = len(edges)

    if edge_links.shape != (num_edges, 2):
        raise ValueError(
            f"Graph edge_links shape {edge_links.shape} does not match "
            f"({num_edges}, 2)."
        )

    writer.write_varuint(num_edges + 1)
    _encode_graph_feature_array(space.edge_space, edges, num_edges, writer)
    _encode_bounded_int_array(edge_links.reshape(-1), num_nodes, writer)


def _decode_graph(space: Graph, reader: _BinaryReader, *, copy: bool) -> GraphInstance:
    num_nodes = reader.read_varuint()
    nodes = _decode_graph_feature_array(space.node_space, num_nodes, reader, copy=copy)

    if space.edge_space is None:
        return GraphInstance(nodes=nodes, edges=None, edge_links=None)

    edge_marker = reader.read_varuint()
    if edge_marker == 0:
        return GraphInstance(nodes=nodes, edges=None, edge_links=None)

    # Non-zero markers are offset by one so zero can mean "edges absent".
    num_edges = edge_marker - 1
    edges = _decode_graph_feature_array(space.edge_space, num_edges, reader, copy=copy)
    edge_link_values = _decode_bounded_int_array(num_edges * 2, num_nodes, reader)
    edge_links = np.asarray(edge_link_values, dtype=np.int32).reshape(num_edges, 2)
    return GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)


def _encode_one_of(space: OneOf, sample: Any, writer: _BinaryWriter) -> None:
    if not isinstance(sample, tuple) or len(sample) != 2:
        raise ValueError("OneOf samples must be (space_index, subspace_sample) tuples.")

    index = int(sample[0])
    if index < 0 or index >= len(space.spaces):
        raise ValueError(f"OneOf index {index} is outside range {len(space.spaces)}.")

    writer.write_bits(index, _bits_required(len(space.spaces)))
    _encode(space.spaces[index], sample[1], writer)


def _decode_one_of(
    space: OneOf, reader: _BinaryReader, *, copy: bool
) -> tuple[np.integer, Any]:
    index = reader.read_bits(_bits_required(len(space.spaces)))
    if index >= len(space.spaces):
        raise ValueError(f"Decoded OneOf index {index} is out of range.")
    return np.int64(index), _decode(space.spaces[index], reader, copy=copy)


def _encode_graph_feature_array(
    feature_space: Box | Discrete,
    values: np.ndarray,
    count: int,
    writer: _BinaryWriter,
) -> None:
    if isinstance(feature_space, Box):
        _encode_box_array(feature_space, values, (count,) + feature_space.shape, writer)
    elif isinstance(feature_space, Discrete):
        expected_shape = (count,)
        array = np.asarray(values)
        if array.shape != expected_shape:
            raise ValueError(
                f"Graph Discrete feature shape {array.shape} does not match "
                f"{expected_shape}."
            )
        _encode_discrete_array(feature_space, array, writer)
    else:
        raise TypeError(
            f"Graph features must be Box or Discrete, got {type(feature_space).__name__}."
        )


def _decode_graph_feature_array(
    feature_space: Box | Discrete,
    count: int,
    reader: _BinaryReader,
    *,
    copy: bool,
) -> np.ndarray:
    if isinstance(feature_space, Box):
        return _decode_box_array(
            feature_space, (count,) + feature_space.shape, reader, copy=copy
        )
    elif isinstance(feature_space, Discrete):
        return _decode_discrete_array(feature_space, count, reader)
    else:
        raise TypeError(
            f"Graph features must be Box or Discrete, got {type(feature_space).__name__}."
        )


def _encode_discrete_array(
    space: Discrete, values: np.ndarray, writer: _BinaryWriter
) -> None:
    n = int(space.n)
    bit_width = _bits_required(n)
    offsets: list[int] = []

    for value in values.reshape(-1):
        int_value = _as_integer(value, "Discrete value")
        offset = int_value - int(space.start)
        if offset < 0 or offset >= n:
            raise ValueError(f"Discrete value {value!r} is not contained in {space}.")
        offsets.append(offset)

    _write_uint_array(writer, offsets, [bit_width] * len(offsets))


def _decode_discrete_array(
    space: Discrete, count: int, reader: _BinaryReader
) -> np.ndarray:
    n = int(space.n)
    offsets = _read_uint_array(reader, [_bits_required(n)] * count)

    values: list[int] = []
    for offset in offsets:
        if offset >= n:
            raise ValueError(f"Decoded Discrete offset {offset} is outside range {n}.")
        values.append(int(space.start) + offset)

    return np.asarray(values, dtype=space.dtype)


def _encode_bounded_int_array(
    values: np.ndarray, n_values: int, writer: _BinaryWriter
) -> None:
    if n_values < 0:
        raise ValueError(f"Cannot encode bounded integers with {n_values} values.")
    if n_values == 0 and values.size:
        raise ValueError("Cannot encode edge links when the graph has no nodes.")

    bit_width = _bits_required(n_values)
    offsets: list[int] = []
    for value in values.reshape(-1):
        offset = int(value)
        if offset < 0 or offset >= n_values:
            raise ValueError(f"Bounded integer value {offset} is outside {n_values}.")
        offsets.append(offset)

    _write_uint_array(writer, offsets, [bit_width] * len(offsets))


def _decode_bounded_int_array(
    count: int, n_values: int, reader: _BinaryReader
) -> list[int]:
    if n_values < 0:
        raise ValueError(f"Cannot decode bounded integers with {n_values} values.")
    if n_values == 0 and count:
        raise ValueError("Cannot decode edge links when the graph has no nodes.")

    bit_width = _bits_required(n_values)
    values = _read_uint_array(reader, [bit_width] * count)
    for value in values:
        if value >= n_values:
            raise ValueError(f"Decoded bounded integer {value} is outside {n_values}.")
    return values


def _write_uint_array(
    writer: _BinaryWriter, values: Any, bit_widths: list[int]
) -> None:
    for value, bit_width in zip(values, bit_widths):
        writer.write_bits(int(value), bit_width)


def _read_uint_array(reader: _BinaryReader, bit_widths: list[int]) -> list[int]:
    return [reader.read_bits(bit_width) for bit_width in bit_widths]


def _write_ndarray(
    writer: _BinaryWriter,
    value: Any,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    name: str = "array",
) -> None:
    dtype = np.dtype(dtype)
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}.")

    if dtype == np.bool_:
        # Bool arrays are bit-packed; raw NumPy would spend a byte per value.
        _write_uint_array(writer, array.reshape(-1), [1] * array.size)
        return

    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    writer.write_bytes(array.tobytes(order="C"))


def _read_ndarray(
    reader: _BinaryReader,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    *,
    copy: bool,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    n_items = _shape_size(shape)

    if dtype == np.bool_:
        # Bit-packed bool arrays are reconstructed as ordinary writable arrays.
        values = _read_uint_array(reader, [1] * n_items)
        return np.asarray(values, dtype=dtype).reshape(shape)

    data = reader.read_bytes(n_items * dtype.itemsize)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def _write_box_bound(writer: _BinaryWriter, value: np.ndarray, dtype: np.dtype) -> None:
    array = np.asarray(value, dtype=dtype)
    if _is_uniform_array(array):
        # Common image/vector spaces use scalar bounds. Preserve that compactly
        # instead of sending full low/high arrays.
        writer.write_varuint(_BOUND_SCALAR)
        _write_scalar(writer, dtype, array.reshape(-1)[0])
    else:
        writer.write_varuint(_BOUND_ARRAY)
        _write_ndarray(writer, array, dtype, array.shape, "Box bound")


def _read_box_bound(
    reader: _BinaryReader, dtype: np.dtype, shape: tuple[int, ...]
) -> Any:
    marker = reader.read_varuint()
    if marker == _BOUND_SCALAR:
        value = _read_scalar(reader, dtype)
        return int(value) if np.dtype(dtype) == np.bool_ else value
    elif marker == _BOUND_ARRAY:
        return _read_ndarray(reader, dtype, shape, copy=True)
    else:
        raise ValueError(f"Unsupported Box bound marker: {marker}.")


def _write_scalar(writer: _BinaryWriter, dtype: np.dtype | type, value: Any) -> None:
    dtype = np.dtype(dtype)
    scalar = np.asarray(value, dtype=dtype).reshape(())
    if dtype == np.bool_:
        writer.write_bits(int(scalar.item()), 1)
    else:
        writer.write_bytes(scalar.tobytes())


def _read_scalar(reader: _BinaryReader, dtype: np.dtype | type) -> Any:
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return np.bool_(reader.read_bits(1))

    data = reader.read_bytes(dtype.itemsize)
    return np.frombuffer(data, dtype=dtype, count=1)[0]


def _write_dtype(writer: _BinaryWriter, dtype: np.dtype | type) -> None:
    _write_string(writer, np.dtype(dtype).str)


def _read_dtype(reader: _BinaryReader) -> np.dtype:
    return np.dtype(_read_string(reader))


def _write_shape(writer: _BinaryWriter, shape: tuple[int, ...]) -> None:
    writer.write_varuint(len(shape))
    for dim in shape:
        writer.write_varuint(int(dim))


def _read_shape(reader: _BinaryReader) -> tuple[int, ...]:
    return tuple(reader.read_varuint() for _ in range(reader.read_varuint()))


def _write_string(writer: _BinaryWriter, value: str) -> None:
    data = value.encode("utf-8")
    writer.write_varuint(len(data))
    writer.write_bytes(data)


def _read_string(reader: _BinaryReader) -> str:
    return bytes(reader.read_bytes(reader.read_varuint())).decode("utf-8")


def _bits_required(n_values: int) -> int:
    if n_values < 0:
        raise ValueError(f"Cannot encode a negative number of values: {n_values}.")
    # Non-power-of-two counts leave unused bit patterns; decoders validate and
    # reject those patterns after reading.
    return max(0, int(n_values) - 1).bit_length()


def _shape_size(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64))


def _is_uniform_array(array: np.ndarray) -> bool:
    return bool(array.size and np.all(array == array.reshape(-1)[0]))


def _as_integer(value: Any, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)

    array = np.asarray(value)
    if array.shape == () and (
        array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer)
    ):
        return int(value)

    raise ValueError(f"{name} {value!r} must be an integer.")


def _text_characters(space: Text) -> tuple[str, ...]:
    return tuple(sorted(space.character_set))


def _is_ark_space(space: Any, module_name: str, class_name: str) -> bool:
    expected_module = f"{_ARK_SPACES_PACKAGE}.{module_name}"
    return any(
        cls.__module__ == expected_module and cls.__name__ == class_name
        for cls in type(space).__mro__
    )


def _load_ark_attr(module_name: str, attr_name: str) -> Any:
    module_path = f"{_ARK_SPACES_PACKAGE}.{module_name}"
    try:
        return getattr(import_module(module_path), attr_name)
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            f"Cannot decode ARK space because {module_path}.{attr_name} "
            "could not be imported."
        ) from exc


def _positive_int_attr(space: Any, attr_name: str) -> int:
    value = int(getattr(space, attr_name))
    if value < 1:
        raise ValueError(
            f"{type(space).__name__}.{attr_name} must be positive, got {value}."
        )
    return value


def _image_shape(
    height: int, width: int, color_channels: int
) -> tuple[int, ...]:
    if color_channels < 1:
        raise ValueError(f"Image color_channels must be positive, got {color_channels}.")
    return (height, width) if color_channels == 1 else (height, width, color_channels)


def _require_zero_low(space: Box, name: str) -> None:
    if not np.all(np.asarray(space.low) == 0):
        raise ValueError(f"{name} low bounds must be zero to encode as an ARK image.")


def _require_box_subspace(
    space: DictSpace,
    key: str,
    shape: tuple[int, ...],
    *,
    dtype: np.dtype | None = None,
) -> Box:
    subspace = space.spaces.get(key)
    if not isinstance(subspace, Box):
        raise ValueError(
            f"{type(space).__name__} must contain a Gymnasium Box at key {key!r}."
        )
    if subspace.shape != shape:
        raise ValueError(
            f"{type(space).__name__}.{key} shape {subspace.shape} "
            f"does not match {shape}."
        )
    if dtype is not None and np.dtype(subspace.dtype) != np.dtype(dtype):
        raise ValueError(
            f"{type(space).__name__}.{key} dtype {subspace.dtype} "
            f"does not match {dtype}."
        )
    return subspace
