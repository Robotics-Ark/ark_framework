"""Generic simulator node entry point.

Loads the simulator from a Python config module that exposes a
``make_simulator(**kwargs) -> Simulator`` factory, then starts a SimulatorNode.

Usage::

    python -m ark.simulator.sim_node \\
        env_name:=go2_demo  node_name:=go2_sim \\
        config:=ark_unitree_go2.configs.mujoco

The ``config`` parameter must be a fully-qualified Python module name.
The module must expose::

    def make_simulator(**kwargs) -> Simulator: ...

Additional parameters (e.g. ``time_step_sec``) are forwarded to the factory
as keyword arguments; unknown kwargs are silently ignored by the factory.
"""

import importlib
import sys

from ark.node import NodeArgumentParser
from ark.time import SimulatedTime
from ark.simulator.node import SimulatorNode


def main():
    parser = NodeArgumentParser(sys.argv)
    env_name, node_name, parameters, channel_remaps, session = parser.parse()

    config_module_name = str(parameters.pop("config"))
    config_module = importlib.import_module(config_module_name)

    sim = config_module.make_simulator(**parameters)

    sim_time = SimulatedTime(env_name, sim.time_step_sec, session)
    node = SimulatorNode(
        env_name=env_name,
        node_name=node_name,
        simulator=sim,
        sim_time=sim_time,
        parameters=parameters,
        channel_remaps=channel_remaps,
        session=session,
    )
    try:
        node.spin()
    finally:
        node.close()


if __name__ == "__main__":
    main()
