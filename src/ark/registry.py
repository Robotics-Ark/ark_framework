from copy import deepcopy
from typing import Any


class Registry:

    def __init__(self):
        self._registry: dict[str, Any] = {}
        self._counter = 0

    def register_item(self, name: str, item: Any) -> None:
        index = deepcopy(self._counter)
        self._registry[name] = (index, item)
        self._counter += 1

    def register(self, name: str):
        """Register a new item in the registry using decorator syntax."""

        def register_item(item):
            self.register_item(name, item)
            return item

        return register_item

    def get(self, name: str) -> Any:
        _, item = self._registry[name]
        return item

    def get_index(self, name: str) -> int:
        index, _ = self._registry[name]
        return index

    def get_name(self, index: int) -> str:
        for name, (idx, _) in self._registry.items():
            if idx == index:
                return name
        else:
            raise KeyError(f"Index {index} not found in registry.")
