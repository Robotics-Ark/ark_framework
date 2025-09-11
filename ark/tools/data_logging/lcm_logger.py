from typing import Optional, Any

from ark.client.comm_infrastructure.base_node import BaseNode, main
from ark.tools.log import log
from arktypes import string_t, flag_t, status_t
import subprocess


class LoggerNode(BaseNode):

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        """
        @brief Construct the logger node and register services.

        @param name   Node name.
        @param config Optional configuration dictionary (unused).
        """
        super().__init__(name)
        self.proc: Optional[subprocess.Popen] = None

        self.create_service(
            "logger/start", string_t, status_t, self.start_logging
        )
        self.create_service(
            "logger/stop", flag_t, status_t, self.stop_logging
        )

    def start_logging(self, channel: str, msg: string_t) -> status_t:
        """
        @brief Start an LCM logging session if none is running.

        @param channel Service channel name (unused).
        @param msg     Output file prefix/path (`string_t.data`) for `lcm-logger`.

        @return `status_t` 
        """
        out = status_t()

        if self.proc is not None:
            log.warning(
                "lcm-logger already running; refusing to start a second session."
            )
            out.success = False
            out.message = "lcm-logger already running"
            return out

        try:
            log.info("Started logging")
            self.proc = subprocess.Popen(
                ["lcm-logger", msg.data],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            out.success = True
            out.message = "lcm-logger started successfully"
        except Exception as e:
            log.error("Failed to start lcm-logger: %s", e)
            self.proc = None
            out.success = False
            out.message = str(e)

        return out

    def stop_logging(self, channel: str, msg: flag_t) -> status_t:
        """
        @brief Stop the current LCM logging session, if running.

        @param channel Service channel name.
        @param msg     Input `flag_t` (unused).
        """
        out = status_t()

        if self.proc is None:
            log.warning("No lcm-logger session is running.")
            out.success = False
            out.message = "No lcm-logger session is running."
            return out

        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
            log.info("Stopped logging")
            del self.proc
            self.proc = None
            out.success = True
            out.message = "lcm-logger stopped successfully"
        except Exception as e:
            log.error("Failed to stop lcm-logger: %s", e)
            out.success = False
            out.message = str(e)

        return out


if __name__ == "__main__":
    main(LoggerNode, "Logger")
