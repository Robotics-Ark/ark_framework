
import sys
from abc import ABC, abstractmethod
from typing import Any, Generator, Dict, Type
import traceback

import lcm
from lcm import LCM
import yaml
import os
import threading
import signal

from ark.client.comm_infrastructure.comm_endpoint import CommEndpoint
from ark.tools.log import log

class InstanceNode(CommEndpoint):
    """!
    Base class for nodes that interact with the LCM system. Handles the subscription,
    publishing, and communication processes for the node.

    The `BaseNode` class manages the LCM instance and communication handlers, and provides
    methods for creating publishers, subscribers, listeners, and steppers. It also provides
    functionality for handling command-line arguments and the graceful shutdown of the node.

    @param lcm: The LCM instance used for communication.
    @param channel_name: The name of the channel to subscribe to.
    @param channel_type: The type of the message expected for the channel.
    """

    def __init__(self, node_name: str, global_config=None) -> None:
        """!
        Initializes a BaseNode object with the specified node name and registry host and port.

        @param node_name: The name of the node.
        @param global_config: Contains IP Address and Port  
        """
        print(global_config)
        super().__init__(node_name, global_config)
        self.config  = self._load_config_section(global_config=global_config, name=node_name, type="other")
       
        self._done = False

        self.spin_thread = threading.Thread(target=self.spin, daemon=True)
        self.spin_thread.start()

    def spin(self) -> None:
        """!
        Runs the node’s main loop, handling LCM messages continuously until the node is finished.

        The loop calls `self._lcm.handle()` to process incoming messages. If an OSError is encountered,
        the loop will stop and the node will shut down.
        """
        while not self._done:
            try:
                self._lcm.handle_timeout(0)
            except OSError as e:
                log.warning(f"LCM threw OSError {e}")
                self._done = True