import zenoh
import warnings
import threading
from abc import ABC, abstractmethod
from google.protobuf.struct_pb2 import Value


def encode_seed(seed: int | None) -> bytes:
    """Encode an optional seed into bytes."""
    value = Value()
    if isinstance(seed, int):
        value.number_value = seed
    return value.SerializeToString()


def decode_seed(data: bytes) -> int | None:
    """Decode bytes into an optional seed."""
    value = Value()
    value.ParseFromString(data)
    return int(value.number_value) if value.HasField("number_value") else None


class ResetBase(ABC):

    def __init__(self, env_name: str, session: zenoh.Session):
        self._env_name = env_name
        self._session = session
        self._reset_channel = f"_ark/{env_name}/reset"
        self._register_channel = f"{self._reset_channel}/register"
        self._deregister_channel = f"{self._reset_channel}/deregister"
        self._initiate_channel = f"{self._reset_channel}/initiate"
        self._completed_channel = f"{self._reset_channel}/completed"

    @abstractmethod
    def reset(self, seed: int | None = None):
        """Reset the internal state of the object.

        Parameters:
        -----------
        seed: int | None
            An optional seed for random number generation.
        """

    @abstractmethod
    def close(self):
        """Clean up resources and close any open connections or subscriptions."""


class ResetContainer(ResetBase):

    def __init__(self, env_name: str, session: zenoh.Session):
        super().__init__(env_name, session)
        self._members: set[str] = set()
        self._completed: set[str] = set()
        self._mutex = threading.Lock()
        self._reg_qr = self._session.declare_queryable(
            self._register_channel, self._on_register
        )
        self._dereg_sub = self._session.declare_subscriber(
            self._deregister_channel, self._on_deregister
        )
        self._init_pub = self._session.declare_publisher(self._initiate_channel)
        self._completed_sub = self._session.declare_subscriber(
            self._completed_channel, self._on_completed
        )

    def _on_register(self, query: zenoh.Query) -> None:
        with self._mutex:
            new_member_name = f"reset{len(self._members)}"
            self._members.add(new_member_name)
        with query:
            query.reply(query.key_expr, new_member_name.encode("utf-8"))

    def _on_deregister(self, sample: zenoh.Sample) -> None:
        member_name = sample.payload.to_string()
        with self._mutex:
            if member_name not in self._members:
                warnings.warn(
                    f"Deregistration request from unknown member: {member_name!r}"
                )
                return
            self._members.remove(member_name)
            self._completed.discard(member_name)

    def _initiate_reset(self, seed: int | None = None) -> None:
        with self._mutex:
            self._completed.clear()
            self._init_pub.put(encode_seed(seed))

    def _on_completed(self, sample: zenoh.Sample) -> None:
        member_name = sample.payload.to_string()
        if member_name not in self._members:
            warnings.warn(
                f"Reset completion acknowledgement from unknown member: {member_name!r}"
            )
            return

        with self._mutex:
            if member_name in self._completed:
                raise ValueError(
                    f"Duplicate reset completion acknowledgement from member: {member_name!r}"
                )
            self._completed.add(member_name)

    def _block_until_complete(self) -> None:
        completed = False
        while not completed:
            with self._mutex:
                completed = self._completed == self._members

    def reset(self, seed: int | None = None):
        self._initiate_reset(seed)
        self._block_until_complete()

    def close(self):
        self._reg_qr.undeclare()
        self._init_pub.undeclare()
        self._completed_sub.undeclare()


class ResetObject(ResetBase):

    def __init__(self, env_name: str, session: zenoh.Session):
        super().__init__(env_name, session)
        self._reset_object_name: bytes = self._register_reset_object()
        self._init_sub = self._session.declare_subscriber(
            self._initiate_channel, self._on_initiate_reset
        )
        self._completed_pub = self._session.declare_publisher(self._completed_channel)
        self._dereg_pub = self._session.declare_publisher(self._deregister_channel)

    def _register_reset_object(self) -> bytes:
        if hasattr(self, "_reset_object_name"):
            raise RuntimeError("Reset object is already registered.")
        try:
            qr = self._session.declare_querier(self._register_channel)
            for reply in qr.get():
                if reply.ok is None:
                    continue
                return bytes(reply.ok.payload)
            else:
                raise RuntimeError(
                    "No reply received from reset registration queryable."
                )
        finally:
            qr.undeclare()

    def _on_initiate_reset(self, sample: zenoh.Sample) -> None:
        self.reset(decode_seed(bytes(sample.payload)))
        self._acknowledge_reset_completion()

    def _acknowledge_reset_completion(self) -> None:
        self._completed_pub.put(self._reset_object_name)

    def _deregister(self) -> None:
        self._dereg_pub.put(self._reset_object_name)

    def close(self):
        self._deregister()
        self._init_sub.undeclare()
        self._completed_pub.undeclare()
        self._dereg_pub.undeclare()
