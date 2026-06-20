from typing import Any


class Registry:

    def __init__(self, sub_cls: type = None):
        """A simple registry class to store and retrieve items by name.

        Parameters
        ----------
        sub_cls : type, optional
            If provided, all items registered must be subclasses of this type."""
        self._registry: dict[str, Any] = {}
        self._sub_cls = sub_cls

    def register_item(self, name: str, item: Any) -> None:
        """Register a new item in the registry."""
        if self._sub_cls is not None and not issubclass(item, self._sub_cls):
            raise TypeError(f"Item must be a subclass of {self._sub_cls.__name__}")
        self._registry[name] = item

    def register(self, name: str):
        """Register a new item in the registry using decorator syntax."""

        def register_item(item):
            self.register_item(name, item)
            return item

        return register_item

    def get(self, name: str) -> Any:
        """Get an item from the registry by name."""
        return self._registry[name]
