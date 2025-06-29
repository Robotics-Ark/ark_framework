
import threading
from ark.client.frequencies.rate import Rate
from ark.tools.log import log


class Stepper(threading.Thread):
    """!
    Convenience class for stepping a callback at a specified rate.

    This class runs a callback function at a specified rate (in Hertz) on a separate thread.
    The callback can be executed continuously or only once, depending on the `oneshot` parameter.
    The rate is controlled using the `Rate` class, which ensures the callback is called at a consistent interval.
    """

    def __init__(
        self,
        hz: float,
        callback: callable,
        oneshot: bool = False,
        reset: bool = True,
        callback_args: list = [],
    ) -> None:
        """!
        Initializes the Stepper thread.

        @param hz: The rate in Hertz (loops per second) at which to call the callback.
        @param callback: The callback function to be called at the specified rate.
        @param oneshot: If True, the callback is called only once; if False, it is called continuously until `shutdown` is called. Defaults to False.
        @param reset: If True, the timer is reset if the system time moves backward. Defaults to True.
        @param callback_args: Arguments to pass to the callback function when called. Defaults to an empty list.
        """
        super().__init__()
        self._hz: float = hz
        self._period_ns: float = 1e9 / float(hz)  # Period in nanoseconds
        self._callback: callable = callback
        self._oneshot: bool = oneshot
        self._reset: bool = reset
        self._shutdown: bool = False
        self.daemon: bool = True
        self._callback_args: list = callback_args
        self.start()
        log.ok("started stepper")

    def suspend(self) -> None:
        """!
        Signal the stepper thread to stop running.

        @return: ``None``
        """
        self._shutdown = True
        log.ok("stepper suspended")

    def run(self) -> None:
        """!
        Runs the callback at the specified rate until the thread is shut down.

        The `Rate` class is used to ensure the callback is executed at the correct intervals.
        If `oneshot` is set to True, the callback will only be executed once.
        """
        r = Rate(self._hz, reset=self._reset)
        while not self._shutdown:
            r.sleep()  # Sleep for the specified rate duration
            if self._shutdown:
                break

            # Call the callback with the provided arguments
            self._callback(*self._callback_args)

            if self._oneshot:
                self.suspend()
                break

    def restart(self):
        """!
        Restart the stepper if it was previously suspended.
        """
        if self._shutdown:
            self._shutdown = False
            self.start()
            log.ok("restarted stepper")
        else:
            log.error("stepper is already running")