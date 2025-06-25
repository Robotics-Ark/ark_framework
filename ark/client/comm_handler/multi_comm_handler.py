
from abc import ABC, abstractmethod
from lcm import LCM
from typing import Optional
from ark.client.comm_handler.comm_handler import CommHandler


class MultiCommHandler(ABC):
    def __init__(self):
        self.channel_data = {}
        self._comm_handlers: list[CommHandler] = []
    
    def get_info(self) -> dict: 
        """!
        Should return a dictionary containing all information about the comms
        """
        info = []
        for ch in self._comm_handlers:
            ch_info = ch.get_info()
            info.append(ch_info)

        print(info)
        return info
    
    def suspend(self) -> None:
        """!
        Suspends the comms handler
        """
        for ch in self._comm_handlers:
            ch.suspend()
    
    def restart(self) -> None:
        """!
        Reactivates the comms handler
        """
        for ch in self._comm_handlers:
            ch.restart()