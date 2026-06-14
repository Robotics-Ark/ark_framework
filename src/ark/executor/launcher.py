"""Process-based launcher for Ark node networks.

Usage::

    # Start in sim mode (overrides the YAML default)
    python -m ark.executor.launcher my_launch.yaml --sim

    # Start in real mode
    python -m ark.executor.launcher my_launch.yaml --real

    # Use the mode declared in the YAML
    python -m ark.executor.launcher my_launch.yaml

The launcher:
  1. Reads the YAML launch file.
  2. Broadcasts ``_ark/{env_name}/sim`` via Zenoh so every Clock/Node
     knows which mode is active (Step 9 groundwork).
  3. Starts the processes whose ``mode`` matches: "sim", "real", or "both".
  4. Monitors child processes and restarts crashed ones (optional).
  5. Shuts everything down cleanly on SIGINT / SIGTERM.

Dynamic sim→real switching
--------------------------
The launcher subscribes to ``_ark/{env_name}/switch`` for mode-change
requests.  When a ``b'sim'`` or ``b'real'`` payload arrives, the launcher:
  • Stops processes that belong to the outgoing mode.
  • Starts processes that belong to the incoming mode.
  • Republishes ``_ark/{env_name}/sim`` with the new value.

This lets an operator node (or test harness) hot-swap the backend without
restarting the whole network.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import zenoh

from ark.executor.config import LaunchConfig
from ark.executor.node import NodeRunner


_SWITCH_CHANNEL_TEMPLATE = "_ark/{env_name}/switch"
_SIM_CHANNEL_TEMPLATE = "_ark/{env_name}/sim"


class Launcher:
    """Manages the lifecycle of Ark node processes."""

    def __init__(self, config: LaunchConfig, sim: bool):
        self._config = config
        self._sim = sim
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        session_cfg = zenoh.Config()
        self._session = zenoh.open(session_cfg)

        env_name = config.env_name
        self._sim_pub = self._session.declare_publisher(
            _SIM_CHANNEL_TEMPLATE.format(env_name=env_name)
        )
        self._switch_sub = self._session.declare_subscriber(
            _SWITCH_CHANNEL_TEMPLATE.format(env_name=env_name),
            self._on_switch,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Launch all nodes for the current mode and broadcast sim flag."""
        self._broadcast_sim(self._sim)
        runners = self._config.node_runners(self._sim)
        with self._lock:
            for runner in runners:
                self._launch_runner(runner)

    def spin(self):
        """Block until stop() is called or a signal is received."""
        signal.signal(signal.SIGINT,  lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        self._stop_event.wait()

    def stop(self):
        """Terminate all child processes and release resources."""
        self._stop_event.set()
        with self._lock:
            for name, proc in list(self._procs.items()):
                _terminate(proc)
            self._procs.clear()
        self._switch_sub.undeclare()
        self._sim_pub.undeclare()
        self._session.close()

    # ------------------------------------------------------------------
    # Dynamic sim/real switching (Step 9)
    # ------------------------------------------------------------------

    def _on_switch(self, sample: zenoh.Sample):
        payload = bytes(sample.payload).decode(errors="replace").strip()
        if payload == "sim":
            new_sim = True
        elif payload == "real":
            new_sim = False
        else:
            return

        if new_sim == self._sim:
            return  # already in the requested mode

        old_sim = self._sim
        self._sim = new_sim

        # Identify runners for each mode so we can stop/start selectively
        old_runners = {
            r.node_name: r for r in self._config.node_runners(old_sim)
        }
        new_runners = {
            r.node_name: r for r in self._config.node_runners(new_sim)
        }

        with self._lock:
            # Stop processes that only belong to the outgoing mode
            to_stop = [n for n in self._procs if n not in new_runners]
            for name in to_stop:
                _terminate(self._procs.pop(name))

            # Start processes that only belong to the incoming mode
            for name, runner in new_runners.items():
                if name not in self._procs:
                    self._launch_runner(runner)

        self._broadcast_sim(new_sim)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _broadcast_sim(self, sim: bool):
        self._sim_pub.put(bytes([int(sim)]))

    def _launch_runner(self, runner: NodeRunner):
        proc = runner.launch()
        self._procs[runner.node_name] = proc

    def _monitor_loop(self):
        """Background thread: restart processes that crash unexpectedly."""
        while not self._stop_event.is_set():
            time.sleep(2.0)
            with self._lock:
                runners = {
                    r.node_name: r
                    for r in self._config.node_runners(self._sim)
                }
                for name, proc in list(self._procs.items()):
                    if proc.poll() is not None:  # process has exited
                        runner = runners.get(name)
                        if runner is not None:
                            self._procs[name] = runner.launch()


def _terminate(proc: subprocess.Popen, timeout: float = 5.0):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Launch an Ark node network from a YAML config file."
    )
    parser.add_argument("launch_file", type=Path, help="Path to the YAML launch file")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sim", dest="sim", action="store_true", default=None,
        help="Force sim mode (overrides YAML default)",
    )
    mode_group.add_argument(
        "--real", dest="real", action="store_true", default=None,
        help="Force real mode (overrides YAML default)",
    )
    args = parser.parse_args(argv)

    config = LaunchConfig.from_file(args.launch_file)

    if args.sim:
        sim = True
    elif args.real:
        sim = False
    else:
        sim = config.sim

    launcher = Launcher(config, sim)
    launcher.start()
    monitor_thread = threading.Thread(
        target=launcher._monitor_loop, daemon=True, name="launcher.monitor"
    )
    monitor_thread.start()
    launcher.spin()


if __name__ == "__main__":
    main()
