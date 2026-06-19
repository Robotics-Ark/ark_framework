import time
import zenoh
from dataclasses import dataclass
from google.protobuf.struct_pb2 import Struct, Value, ListValue

PARAM_TYPE = (
    bool | int | float | str | tuple[bool] | tuple[int] | tuple[float] | tuple[str]
)

SCALAR_ENCODERS = {
    bool: lambda b: Value(bool_value=b),
    int: lambda i: Value(number_value=i),
    float: lambda f: Value(number_value=f),
    str: lambda s: Value(string_value=s),
}

SCALAR_DECODERS = {
    "bool": lambda v: v.bool_value,
    "int": lambda v: int(v.number_value),
    "float": lambda v: v.number_value,
    "str": lambda v: v.string_value,
}


def _param_to_serialized_struct(param: PARAM_TYPE) -> bytes:
    param_type = type(param)

    if param_type in SCALAR_ENCODERS:
        param_value = SCALAR_ENCODERS[param_type](param)
        param_type_name = param_type.__name__
    elif isinstance(param, tuple):
        try:
            elem_type = type(param[0])
        except IndexError:
            raise ValueError("Tuple parameter cannot be empty.")

        if not all(isinstance(el, elem_type) for el in param):
            raise ValueError("All elements in the tuple must be of the same type.")

        if elem_type not in SCALAR_ENCODERS:
            raise ValueError(f"Unsupported element type in tuple: {elem_type.__name__}")

        encode = SCALAR_ENCODERS[elem_type]
        param_list_value = ListValue(values=[encode(el) for el in param])
        param_value = Value(list_value=param_list_value)
        param_type_name = f"tuple[{elem_type.__name__}]"
    else:
        raise ValueError(f"Unsupported parameter type: {param_type.__name__}")

    s = Struct()
    s.fields["type"].string_value = param_type_name
    s.fields["value"].CopyFrom(param_value)
    return s.SerializeToString()


def _decode_param_struct(struct: Struct) -> PARAM_TYPE:
    param_type = struct.fields["type"].string_value
    value = struct.fields["value"]

    if param_type in SCALAR_DECODERS:
        return SCALAR_DECODERS[param_type](value)
    elif param_type.startswith("tuple[") and param_type.endswith("]"):
        elem_type_name = param_type[6:-1]
        decode = SCALAR_DECODERS[elem_type_name]
        return tuple(decode(el) for el in value.list_value.values)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_parameter(
    server_name: str,
    param_name: str,
    session: zenoh.Session,
    timeout: float = 30.0,
    retry_interval: float = 0.05,
) -> PARAM_TYPE:
    deadline = time.monotonic() + timeout
    while True:
        qr = session.declare_querier(f"{server_name}/get_parameter")
        try:
            for z_reply in qr.get(payload=param_name.encode("utf-8")):
                if z_reply.err is not None:
                    err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Parameter query failed: {err}")
                if z_reply.ok is None:
                    continue
                reply = Struct()
                reply.ParseFromString(bytes(z_reply.ok.payload))
                return _decode_param_struct(reply)
        finally:
            qr.undeclare()

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Parameter '{param_name}' not available on server '{server_name}' "
                f"after {timeout}s. Is the environment running?"
            )
        time.sleep(retry_interval)


# Helpers for specific parameters that are commonly used in the framework.
def get_sim(env_name: str, session: zenoh.Session) -> bool:
    return get_parameter(f"{env_name}/env/parameters", "sim", session)


def get_keys(server_name: str, session: zenoh.Session) -> list[str]:
    qr = session.declare_querier(f"{server_name}/get_keys")
    try:
        for z_reply in qr.get():
            if z_reply.err is not None:
                err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                raise RuntimeError(f"Get keys query failed: {err}")
            if z_reply.ok is None:
                continue
            list_value = ListValue()
            list_value.ParseFromString(bytes(z_reply.ok.payload))
            return [v.string_value for v in list_value.values]
    finally:
        qr.undeclare()


def set_parameter(
    server_name: str, param_name: str, param_value: PARAM_TYPE, session: zenoh.Session
) -> None:
    request = Struct()
    request.ParseFromString(_param_to_serialized_struct(param_value))
    request.fields["name"].string_value = param_name

    qr = session.declare_querier(f"{server_name}/set_parameter")
    try:
        for z_reply in qr.get(payload=request.SerializeToString()):
            if z_reply.err is not None:
                err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                raise RuntimeError(f"Parameter set failed: {err}")

            if z_reply.ok is not None:
                return

        raise RuntimeError("Parameter set failed: No reply received.")
    finally:
        qr.undeclare()


@dataclass(frozen=True, slots=True)
class Parameter:
    value: PARAM_TYPE
    env_value: bytes


class ParameterServer:

    def __init__(
        self,
        server_name: str,
        parameters: dict[str, PARAM_TYPE],
        session: zenoh.Session,
        read_only: bool = False,
    ):
        self._server_name = server_name
        self._read_only = read_only
        self._parameters: dict[str, Parameter] = {
            name: Parameter(value=param, env_value=_param_to_serialized_struct(param))
            for name, param in parameters.items()
        }
        self._get_keys_qr = session.declare_queryable(
            f"{self._server_name}/get_keys", self._on_get_keys_query
        )
        self._get_qr = session.declare_queryable(
            f"{self._server_name}/get_parameter", self._on_get_query
        )
        if not self._read_only:
            self._set_qr = session.declare_queryable(
                f"{self._server_name}/set_parameter", self._on_set_query
            )
        else:
            self._set_qr = None

    def set(self, param_name: str, param_value: PARAM_TYPE) -> None:
        self._parameters[param_name] = Parameter(
            value=param_value, env_value=_param_to_serialized_struct(param_value)
        )

    def get(self, param_name: str) -> PARAM_TYPE:
        if param_name not in self._parameters:
            raise KeyError(f"Parameter '{param_name}' not found.")
        return self._parameters[param_name].value

    def keys(self):
        return self._parameters.keys()

    def _on_get_keys_query(self, query: zenoh.Query) -> None:
        with query:
            keys = ListValue(values=[Value(string_value=k) for k in self.keys()])
            query.reply(query.key_expr, keys.SerializeToString())

    def _on_get_query(self, query: zenoh.Query) -> None:
        with query:

            if not query.payload:
                query.reply_err(b"Query payload is required.")
                return

            param_name = query.payload.to_string()
            param = self._parameters.get(param_name)

            if param is None:
                query.reply_err(f"Parameter '{param_name}' not found.".encode("utf-8"))
                return

            query.reply(query.key_expr, param.env_value)

    def _on_set_query(self, query: zenoh.Query) -> None:
        with query:

            if not query.payload:
                query.reply_err(b"Query payload is required.")
                return

            request = Struct()
            request.ParseFromString(bytes(query.payload))

            name = request.fields["name"].string_value
            if not name:
                query.reply_err(b"Parameter name is required.")
                return

            decoded_value = _decode_param_struct(request)
            self._parameters[name] = Parameter(
                value=decoded_value,
                env_value=_param_to_serialized_struct(decoded_value),
            )
            query.reply(query.key_expr, b"")

    def close(self):
        self._get_qr.undeclare()
        self._set_qr.undeclare()


class ParameterClient:

    def __init__(self, server_name: str, session: zenoh.Session):
        self._server_name = server_name
        self._session = session

    def get(self, param_name: str) -> PARAM_TYPE:
        return get_parameter(self._server_name, param_name, self._session)

    def set(self, param_name: str, param_value: PARAM_TYPE) -> None:
        set_parameter(self._server_name, param_name, param_value, self._session)
