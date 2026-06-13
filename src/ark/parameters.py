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


def get_parameter(
    server_name: str, param_name: str, session: zenoh.Session
) -> PARAM_TYPE:
    qr = session.declare_querier(server_name)
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


class ParameterServer:

    def __init__(
        self,
        server_name: str,
        parameters: dict[str, PARAM_TYPE],
        session: zenoh.Session,
    ):

        # Encode parameters
        self._encoded_parameters: dict[str, bytes] = {}
        for param_name, param in parameters.items():
            param_type = type(param)
            if param_type in SCALAR_ENCODERS:
                param_value = SCALAR_ENCODERS[param_type](param)
                param_type_name = param_type.__name__
            elif isinstance(param, tuple) and len(param) > 0:
                elem_type = type(param[0])

                if not all(isinstance(el, elem_type) for el in param):
                    raise ValueError(
                        f"All elements in the tuple for key '{param_name}' must be of the same type."
                    )

                if elem_type not in SCALAR_ENCODERS:
                    raise ValueError(
                        f"Unsupported element type in tuple for key '{param_name}': {elem_type.__name__}"
                    )

                encode = SCALAR_ENCODERS[elem_type]
                param_list_value = ListValue(values=[encode(el) for el in param])
                param_value = Value(list_value=param_list_value)
                param_type_name = f"tuple[{elem_type.__name__}]"
            else:
                raise ValueError(
                    f"Unsupported parameter type for key '{param_name}': {param_type.__name__}"
                )
            reply = Struct()
            reply.fields["type"].string_value = param_type_name
            reply.fields["value"].CopyFrom(param_value)
            self._encoded_parameters[param_name] = reply.SerializeToString()

        # Setup queryable
        self._qr = session.declare_queryable(server_name, self._on_query)

    def _on_query(self, query: zenoh.Query) -> None:
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

    def close(self):
        self._qr.undeclare()
