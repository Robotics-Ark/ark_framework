from ark.client.comm_infrastructure.base_node import BaseNode,  main
from arktypes import string_t, flag_t
from ark.tools.log import log
from typing import Dict, Any, Optional
import time
import subprocess

class LoggerNode(BaseNode):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Logger")
        self.pub = self.create_publisher("chatter", string_t)
        self.create_service("logger/start", string_t, flag_t, self.start_logging)
        self.create_service("logger/stop", flag_t, flag_t, self.stop_logging)



if __name__ == "__main__":
    main(LoggerNode)
