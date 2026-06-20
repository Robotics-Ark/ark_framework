import zenoh
import threading
from google.protobuf.struct_pb2 import Value, Struct


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


def encode_initiate_reset(env_name: str, seed: int | None) -> bytes:
    """Encode an initiate reset request into bytes."""
    s = Struct()
    s.fields["env_name"].string_value = env_name
    if seed is not None:
        s.fields["seed"].number_value = seed
    return s.SerializeToString()


def decode_initiate_reset(data: bytes) -> tuple[str, int | None]:
    """Decode bytes into an initiate reset request."""
    s = Struct()
    s.ParseFromString(data)
    env_name = s.fields["env_name"].string_value
    seed = (
        int(s.fields["seed"].number_value)
        if s.fields["seed"].HasField("number_value")
        else None
    )
    return env_name, seed


class ResetObject:

    def __init__(self, env_name: str, session: zenoh.Session):
        self._env_name = env_name
        self._session = session
        self._object_id = self._register_reset_object()
        self._reset_initiate_sub = self._session.declare_subscriber(
            f"_ark/reset/{self._env_name}/initiate_reset", self._on_initiate_reset
        )
        self._acknowledge_reset_completion_pub = self._session.declare_publisher(
            f"_ark/reset/{self._env_name}/reset_completed"
        )

    def _register_reset_object(self) -> str:
        qr = self._session.declare_querier("_ark/reset/register_reset_object")
        try:
            for reply in qr.get(payload=self._env_name.encode("utf-8")):
                if reply.ok is None:
                    continue
                return reply.ok.payload.to_string()
            else:
                raise RuntimeError(
                    "No reply received from reset registration queryable."
                )
        finally:
            qr.undeclare()

    def reset(self, _seed: int | None = None):
        """Reset the object state. Override this method in subclasses."""
        pass

    def _on_initiate_reset(self, sample: zenoh.Sample):
        seed = decode_seed(sample.payload.to_bytes())
        self.reset(seed)
        self._acknowledge_reset_completion()

    def _acknowledge_reset_completion(self):
        self._acknowledge_reset_completion_pub.put(self._object_id.encode("utf-8"))


class ResetGroup:

    def __init__(self, env_name: str, session: zenoh.Session):
        self._env_name = env_name
        self._session = session
        self._reset_objects = set()
        self._lock = threading.Lock()
        self._completed_reset_objects = set()
        self._initiate_reset_pub = self._session.declare_publisher(
            f"_ark/reset/{self._env_name}/initiate_reset"
        )
        self._completed_reset_sub = self._session.declare_subscriber(
            f"_ark/reset/{self._env_name}/reset_completed", self._on_reset_completed
        )

    def add_reset_object(self) -> str:
        new_object_id = f"reset_object_{len(self._reset_objects)}"
        self._reset_objects.add(new_object_id)
        return new_object_id

    def reset(self, seed: int | None = None):
        """Initiate a reset for all registered reset objects."""
        with self._lock:
            self._completed_reset_objects.clear()
        self._initiate_reset_pub.put(encode_seed(seed))
        self._block_until_complete()

    def _on_reset_completed(self, sample: zenoh.Sample):
        object_id = sample.payload.to_string()
        with self._lock:
            self._completed_reset_objects.add(object_id)

    def _block_until_complete(self):
        while True:
            with self._lock:
                if self._completed_reset_objects == self._reset_objects:
                    break


class ResetCoordinator:

    def __init__(self, session: zenoh.Session):
        self._session = session
        self._lock = threading.Lock()
        self._reset_groups: dict[str, ResetGroup] = {}  # key: env_name
        self._register_reset_object_qr = self._session.declare_queryable(
            "_ark/reset/register_reset_object", self._on_register_reset_object
        )
        self._initiate_reset_qr = self._session.declare_queryable(
            "_ark/reset/initiate_reset", self._on_initiate_reset
        )

    def _register_new_object(self, env_name: str) -> str:
        with self._lock:
            if env_name not in self._reset_groups:
                self._reset_groups[env_name] = ResetGroup(env_name, self._session)
            object_id = self._reset_groups[env_name].add_reset_object()
        return object_id

    def _on_register_reset_object(self, query: zenoh.Query) -> None:
        with query:
            env_name = query.payload.to_string()
            object_id = self._register_new_object(env_name)
            query.reply(query.key_expr, object_id.encode("utf-8"))

    def _on_initiate_reset(self, query: zenoh.Query) -> None:
        with query:
            env_name, seed = decode_initiate_reset(query.payload.to_bytes())
            with self._lock:
                if env_name not in self._reset_groups:
                    query.reply_err(
                        f"No reset group found for environment '{env_name}'.".encode(
                            "utf-8"
                        )
                    )
                    return
                reset_group = self._reset_groups[env_name]
            reset_group.reset(seed)
            query.reply(query.key_expr, b"Reset completed.")

    def close(self):
        self._register_reset_object_qr.undeclare()
        self._initiate_reset_qr.undeclare()


class EnvReset:

    def __init__(self, env_name: str, session: zenoh.Session):
        self._env_name = env_name
        self._session = session
        self._reset_initiate_qr = self._session.declare_querier(
            "_ark/reset/initiate_reset"
        )

    def reset(self, seed: int | None = None):
        """Initiate a reset for the environment."""
        payload = encode_initiate_reset(self._env_name, seed)
        for reply in self._reset_initiate_qr.get(payload=payload):
            if reply.ok is not None:
                if reply.ok.payload.to_string() == "success":
                    return
            elif reply.err is not None:
                raise RuntimeError(
                    f"Error initiating reset for environment '{self._env_name}': {reply.err.to_string()}"
                )
        else:
            raise RuntimeError(
                f"No reply received from reset initiation queryable for environment '{self._env_name}'."
            )

    def close(self):
        self._reset_initiate_qr.undeclare()
