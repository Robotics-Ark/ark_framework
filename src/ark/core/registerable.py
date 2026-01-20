from abc import ABC, abstractmethod


class Registerable(ABC):

    @abstractmethod
    def core_registration(self):
        """Register the object with ark core."""

    @abstractmethod
    def close(self):
        """Close the object and release any resources."""
