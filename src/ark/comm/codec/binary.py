BytesLike = bytes | bytearray | memoryview


class Writer:
    def __init__(self):
        self._data = bytearray()

    def uint(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"Cannot encode negative integer {value}.")
        while value >= 0x80:
            self._data.append((value & 0x7F) | 0x80)
            value >>= 7
        self._data.append(value)

    def int(self, value: int) -> None:
        self.uint(value * 2 if value >= 0 else (-value * 2) - 1)

    def bool(self, value: bool) -> None:
        self._data.append(1 if value else 0)

    def bytes(self, data: BytesLike) -> None:
        self._data.extend(data)

    def sized_bytes(self, data: BytesLike) -> None:
        self.uint(len(data))
        self.bytes(data)

    def string(self, value: str) -> None:
        self.sized_bytes(value.encode("utf-8"))

    def finish(self) -> bytes:
        return bytes(self._data)


class Reader:
    def __init__(self, payload: BytesLike):
        self._buffer = memoryview(payload)
        self._pos = 0

    def uint(self) -> int:
        value = 0
        shift = 0
        while True:
            if self._pos >= len(self._buffer):
                raise ValueError("Payload ended while reading an integer.")
            byte = int(self._buffer[self._pos])
            self._pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    def int(self) -> int:
        value = self.uint()
        return value // 2 if value % 2 == 0 else -((value // 2) + 1)

    def bool(self) -> bool:
        value = self.bytes(1)[0]
        if value not in (0, 1):
            raise ValueError(f"Invalid boolean byte {value}.")
        return bool(value)

    def bytes(self, n_bytes: int) -> memoryview:
        if n_bytes < 0:
            raise ValueError(f"Cannot read a negative byte count: {n_bytes}.")
        end = self._pos + n_bytes
        if end > len(self._buffer):
            raise ValueError(
                f"Payload ended while reading {n_bytes} bytes at offset {self._pos}."
            )
        data = self._buffer[self._pos : end]
        self._pos = end
        return data

    def sized_bytes(self) -> memoryview:
        return self.bytes(self.uint())

    def string(self) -> str:
        return bytes(self.sized_bytes()).decode("utf-8")

    def finish(self) -> None:
        if self._pos != len(self._buffer):
            raise ValueError(
                f"Payload has {len(self._buffer) - self._pos} unread trailing bytes."
            )


BinaryWriter = Writer
BinaryReader = Reader
