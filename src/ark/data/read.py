import struct
from pathlib import Path
from ark_msgs.envelope import Envelope


def read_data(path: str | Path) -> list[Envelope]:
    records: list[Envelope] = []
    path = Path(path)

    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break  # EOF
            if len(len_bytes) != 4:
                raise IOError("Corrupted data file (incomplete length prefix)")

            (msg_len,) = struct.unpack("<I", len_bytes)

            msg_bytes = f.read(msg_len)
            if len(msg_bytes) != msg_len:
                raise IOError("Corrupted data file (incomplete message)")

            env = Envelope()
            env.ParseFromString(msg_bytes)
            records.append(env)

    return records
