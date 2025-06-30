
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Generator, Dict, Type
import traceback

import lcm
from lcm import LCM
import yaml
import os

from ark.client.comm_infrastructure.comm_endpoint import CommEndpoint
from ark.tools.log import log

class HybridNode(CommEndpoint):
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
        super().__init__(node_name, global_config)
        

    def manual_spin(self) -> None:
        """!
        Process pending LCM messages once.

        This method calls ``handle_timeout`` a single time and updates the
        done flag if an error occurs.
        """
        try:
            self._lcm.handle_timeout(0)
        except OSError as e:
            log.warning(f"LCM threw OSError {e}")
            self._done = True
    


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


def main(node_cls: type[HybridNode], *args) -> None:
    """!
    Initializes and runs a node.

    This function creates an instance of the specified `node_cls`, spins the node to handle messages,
    and handles exceptions that occur during the node's execution.

    @param node_cls: The class of the node to run.
    @type node_cls: Type[BaseNode]
    """

    if "--help" in sys.argv or "-h" in sys.argv:
        print(node_cls.get_cli_doc())
        sys.exit(0)

    node = None
    log.ok(f"Initializing {node_cls.__name__} type node")
    try:
        node = node_cls(*args)
        log.ok(f"Initialized {node.node_name}")
        node.spin()
    except KeyboardInterrupt:
        log.warning(f"User killed node {node_cls.__name__}")
    except Exception:
        tb = traceback.format_exc()
        div = "=" * 30
        log.error(f"Exception thrown during node execution:\n{div}\n{tb}\n{div}")
    finally:
        if node is not None:
            node.shutdown()
            log.ok(f"Finished running node {node_cls.__name__}")
        else:
            log.warning(f"Node {node_cls.__name__} failed during initialization")
