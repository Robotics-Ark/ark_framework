from ark.client.comm_infrastructure.base_node import BaseNode,  main
from arktypes import string_t, flag_t
from ark.tools.log import log
from typing import Dict, Any, Optional
import time
import subprocess

class LoggerNode(BaseNode):

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.create_service("logger/start", string_t, flag_t, self.start_logging)
        self.create_service("logger/stop", flag_t, flag_t, self.stop_logging)

    def start_logging(self, channel: str, msg: string_t) -> flag_t:
        log.info("Starting logging to file: " + msg.data)
        self.proc = subprocess.Popen(
            ['lcm-logger', msg.data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        msg = flag_t()
        msg.flag = True
        return msg
    
    def stop_logging(self, channel: str, msg: flag_t) -> flag_t:
        log.info("Stopping logging")
        self.proc.kill()

        msg = flag_t()
        msg.flag = True
        return msg

if __name__ == "__main__":
    main(LoggerNode)
