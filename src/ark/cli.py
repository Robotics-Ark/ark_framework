from abc import ABC, abstractmethod


class BaseCliParser(ABC):

    @abstractmethod
    def parse(self, args):
        """Parse command line arguments and return parameters and remappings."""
