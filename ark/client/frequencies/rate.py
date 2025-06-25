
import time
from ark.tools.log import log


class Rate:
    """!
    A convenience class for sleeping in a loop at a specified rate using `perf_counter_ns` for high precision.

    This class calculates the required sleep duration between loop iterations based on a specified rate (in Hz),
    and attempts to maintain that rate by sleeping for the appropriate amount of time. The rate is measured using
    nanoseconds for high precision.
    """

    def __init__(self, hz: float, reset: bool = False) -> None:
        """!
        Initializes the Rate object with a specified rate in Hertz (Hz) and an optional reset flag.

        @param hz: The target rate in Hertz (loops per second) to determine the sleep duration.
        @param reset: If True, resets the timer if the system time moves backward. Defaults to False.
        """
        self.last_time_ns: int = time.perf_counter_ns()
        self.sleep_dur_ns: int = int(
            1e9 / hz
        )  # Duration in nanoseconds for the given Hz rate
        self._reset: bool = reset

    def _remaining(self, curr_time_ns: int) -> int:
        """!
        Calculates the remaining time (in nanoseconds) before the next sleep interval.

        @param curr_time_ns: The current time in nanoseconds.
        @return: The remaining time to sleep in nanoseconds.
        @raises RuntimeError: If time moved backward and `reset` is False.
        """
        if self.last_time_ns > curr_time_ns:
            if self._reset:
                # Reset the last_time_ns if time moved backward and reset is True
                self.last_time_ns = curr_time_ns
            else:
                # Raise an error if time moved backwards and reset is False
                raise RuntimeError("Time moved backwards and reset is not allowed.")

        elapsed_ns = curr_time_ns - self.last_time_ns
        return self.sleep_dur_ns - elapsed_ns

    def remaining(self) -> float:
        """!
        Returns the time remaining (in seconds) before the next sleep interval.

        @return: The remaining sleep time in seconds.
        """
        curr_time_ns = time.perf_counter_ns()
        remaining_ns = self._remaining(curr_time_ns)
        return remaining_ns / 1e9  # Convert nanoseconds to seconds

    def sleep(self) -> None:
        """!
        Attempts to sleep at the specified rate.

        This method calculates the remaining time for the current cycle and sleeps for that duration to maintain
        the target rate. If the system time moved backward, a warning is printed. If the `reset` flag is set to False,
        a RuntimeError is raised.

        @raises RuntimeError: If the system time moved backward and `reset` is False.
        """
        curr_time_ns = time.perf_counter_ns()
        try:
            remaining_ns = self._remaining(curr_time_ns)
            if remaining_ns > 0:
                # Convert nanoseconds to seconds for time.sleep()
                time.sleep(remaining_ns / 1e9)
        except RuntimeError as e:
            # Handle time moving backward if reset is False
            log.warning(str(e))
            if not self._reset:
                raise

        # Update last_time_ns after sleeping
        self.last_time_ns = time.perf_counter_ns()

        # Check if the loop is too slow or if time jumped forward (greater than 2x sleep duration)
        if curr_time_ns - self.last_time_ns > self.sleep_dur_ns * 2:
            self.last_time_ns = curr_time_ns
