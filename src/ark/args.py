from abc import ABC, abstractmethod


class BaseArgsParser(ABC):

    @abstractmethod
    def parse(self, args):
        """Parse arguments."""
