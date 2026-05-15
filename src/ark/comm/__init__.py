from .channel import Channel

__all__ = [
    "Channel",
    "PeriodicPublisher",
    "Publisher",
    "QuerySpace",
    "Queryable",
    "Querier",
    "SampleWindowListener",
    "StampedSample",
    "Subscriber",
    "TimeWindowListener",
]

_LAZY_IMPORTS = {
    "PeriodicPublisher": (".publisher", "PeriodicPublisher"),
    "Publisher": (".publisher", "Publisher"),
    "QuerySpace": (".end_point", "QuerySpace"),
    "Queryable": (".queryable", "Queryable"),
    "Querier": (".querier", "Querier"),
    "SampleWindowListener": (".subscriber", "SampleWindowListener"),
    "StampedSample": (".stamped_sample", "StampedSample"),
    "Subscriber": (".subscriber", "Subscriber"),
    "TimeWindowListener": (".subscriber", "TimeWindowListener"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
