from abc import ABC, abstractmethod


class BaseCliParser(ABC):

    def __init__(self):
        self._params = {}
        self._remaps = {}

    @abstractmethod
    def parse(self, args):
        """Parse command line arguments and return parameters and remappings."""
