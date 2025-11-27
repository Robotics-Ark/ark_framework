DECODER_REGISTRY: dict[str, callable] = {}


def register_decoder(name: str):
    """
    Register a decoder function under a specified name.
    Args:
        name: The name under which to register the decoder function.

    Returns:
        A decorator function that registers the decoder when applied to a function.
    """

    def wrapper(fn):
        if name in DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' already registered")
        DECODER_REGISTRY[name] = fn
        return fn

    return wrapper


def get_decoder(name: str):
    """
    Retrieve a registered decoder function by name.
    Args:
        name: The name of the decoder to retrieve.

    Returns:
         The decoder function associated with the given name.
    """
    try:
        return DECODER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Decoder '{name}' not found. "
            f"Available decoders: {sorted(DECODER_REGISTRY.keys())}"
        )
