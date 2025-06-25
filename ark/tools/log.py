
import logging
from datetime import datetime
from typing import Optional


# Define custom colors for log levels using the bcolors class
class bcolors:
    """! This class contains color codes to be used in the log messages to provide
    visual cues for different log levels."""

    HEADER = "\033[95m"  # Purple
    OKBLUE = "\033[94m"  # Blue
    OKCYAN = "\033[96m"  # Cyan
    OKGREEN = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    FAIL = "\033[91m"  # Red
    ENDC = "\033[0m"  # Reset color
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    WHITE = "\033[97m"  # White
    GREY = "\033[90m"  # Grey


# Define a custom log level: OK (between INFO and WARNING)
OK_LEVEL_NUM = 25
logging.addLevelName(OK_LEVEL_NUM, "OK")


def ok(self: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    """! Custom log method for the OK log level.

    This method adds a custom logging level between INFO and WARNING. It is used
    to log messages that indicate normal operations, but with higher importance
    than INFO.

    @param message The log message.
    @param args Additional arguments for formatting the message.
    @param kwargs Additional keyword arguments.
    """
    if self.isEnabledFor(OK_LEVEL_NUM):
        self._log(OK_LEVEL_NUM, message, args, **kwargs)


logging.Logger.ok = ok  # Add the `ok` method to the Logger class

def apply_panda_style(text: str) -> str:
    styled_text = ""
    colors = [bcolors.WHITE, bcolors.GREY]
    for i, char in enumerate(text):
        styled_text += colors[i % 2] + char
    return styled_text + bcolors.ENDC

def log_panda(self: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    if self.isEnabledFor(logging.INFO):
        styled_message = apply_panda_style(message)
        self._log(logging.INFO, styled_message, args, **kwargs)

logging.Logger.panda = log_panda  # Add `log_panda` method to Logger class


class CustomFormatter(logging.Formatter):
    """! CustomFormatter for applying color coding to log levels and including timestamp."""

    COLORS: dict[str, str] = {
        "DEBUG": bcolors.OKBLUE,  # Blue for DEBUG level
        "INFO": bcolors.OKCYAN,  # Cyan for INFO level
        "OK": bcolors.OKGREEN + bcolors.BOLD,  # Bold Green for OK level
        "WARNING": bcolors.WARNING,  # Yellow for WARNING level
        "ERROR": bcolors.FAIL,  # Red for ERROR level
        "CRITICAL": bcolors.FAIL + bcolors.BOLD,  # Bold Red for CRITICAL level
        "NOTSET": bcolors.ENDC,  # Reset for NOTSET level
    }

    def __init__(
        self,
        fmt: str = "%(levelname)s [%(asctime)s] - %(message)s",
        datefmt: str = "%H:%M:%S.%f",
    ) -> None:
        """! Initializes the CustomFormatter with the specified format for the log messages.

        @param fmt The format string for log messages.
        @param datefmt The format string for the timestamp in the log message.
        """
        super().__init__(fmt, datefmt)

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """! Overrides the `formatTime` method to include milliseconds in the timestamp.

        @param record The log record.
        @param datefmt Optional format string for the timestamp.
        @return The formatted timestamp, including milliseconds.
        """
        if datefmt:
            return datetime.fromtimestamp(record.created).strftime(datefmt)
        else:
            # Format time as HH:MM:SS.mmm (milliseconds)
            return (
                datetime.fromtimestamp(record.created).strftime("%H:%M:%S.")
                + f"{int(record.msecs):02d}"
            )

    def format(self, record: logging.LogRecord) -> str:
        """! Formats the log message with the appropriate color based on the log level.

        @param record The log record.
        @return The formatted log message with color coding.
        """
        log_message = super().format(record)

        # Get the color for the log level, apply the color if it exists
        color = self.COLORS.get(record.levelname, bcolors.ENDC)

        # Apply color to the log message
        log_message = color + log_message + bcolors.ENDC
        return log_message


def setup_logger() -> logging.Logger:
    """! Configures and returns a global logger with a custom formatter for colorized output.

    This function sets up a logger that includes the custom colorized formatter, and
    configures it to output logs to the console. It uses the DEBUG level as the minimum level
    to capture logs.

    @return The configured logger.
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # Adjust to the level you need

    # Create a stream handler and set the formatter
    stream_handler = logging.StreamHandler()
    formatter = CustomFormatter(
        fmt="[%(levelname)s] [%(asctime)s] - %(message)s",
        datefmt="%H:%M:%S.%f",  # Use time format with milliseconds
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# Initialize the logger
log = setup_logger()


def query(msg: str) -> str:
    """! Prompts the user for input after printing a message to the console.

    This function logs the action of querying the user, prints the provided message
    to the console, and waits for the user's input.

    @param msg The message to display to the user.
    @return The user's input as a string.
    """
    log.info("querying user")
    print(msg)
    usrin = input(">> ")
    return usrin


# Attach the `query` function as a method to the `log` object for easier use.
log.query = query

# Ensure only the log object is exported
__all__ = ["log"]

if __name__ == "__main__":
    usrin = log.query("ready?")
    log.debug(f"user said '{usrin}'")
    log.ok("all good")
    log.info("hello world")
    log.warning("warn message")
    log.error("oh no")
    log.critical("bad times")
    log.panda("this is a panda log")
