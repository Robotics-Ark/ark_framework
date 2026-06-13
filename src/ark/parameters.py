import zenoh
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

    reply = Struct()
    reply.fields["type"].string_value = param_type_name
    reply.fields["value"].CopyFrom(param_value)
    return reply.SerializeToString()


def get_parameter(
    server_name: str, param_name: str, session: zenoh.Session
) -> PARAM_TYPE:
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
            param_type = reply.fields["type"].string_value
            value = reply.fields["value"]

            if param_type in SCALAR_DECODERS:
                decode = SCALAR_DECODERS[param_type]
                return decode(value)

            elif param_type.startswith("tuple[") and param_type.endswith("]"):
                elem_type_name = param_type[6:-1]
                decode = SCALAR_DECODERS[elem_type_name]
                return tuple(decode(el) for el in value.list_value.values)

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


class ParameterServer:

    def __init__(
        self,
        server_name: str,
        parameters: dict[str, PARAM_TYPE],
        session: zenoh.Session,
    ):
        self._server_name = server_name
        self._encoded_parameters: dict[str, bytes] = {
            param_name: _param_to_serialized_struct(param)
            for param_name, param in parameters.items()
        }
        self._get_qr = session.declare_queryable(
            f"{self._server_name}/get_parameter", self._on_get_query
        )
        self._set_qr = session.declare_queryable(
            f"{self._server_name}/set_parameter", self._on_set_query
        )

    def _on_get_query(self, query: zenoh.Query) -> None:
        with query:

            if not query.payload:
                query.reply_err(b"Query payload is required.")
                return

            param_name = query.payload.to_string()
            enc_param = self._encoded_parameters.get(param_name)

            if enc_param is None:
                query.reply_err(f"Parameter '{param_name}' not found.".encode("utf-8"))
                return

            query.reply(query.key_expr, enc_param)

    def _on_set_query(self, query: zenoh.Query) -> None:
        with query:

            if not query.payload:
                query.reply_err(b"Query payload is required.")
                return

            payload = bytes(query.payload)
            request = Struct()
            request.ParseFromString(payload)

            name = request.fields["name"].string_value
            if not name:
                query.reply_err(b"Parameter name is required.")
                return

            self._encoded_parameters[name] = payload
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
