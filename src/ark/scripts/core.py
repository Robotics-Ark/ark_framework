import yaml
import argparse
from pathlib import Path
from ark.core import Core
from ark.executor.host import load_hosts
from ark.logging import configure_logging


def parse_args():

    path = lambda p: Path(p).resolve()

    parser = argparse.ArgumentParser(description="Ark core")
    parser.add_argument(
        "--zenoh_config",
        type=path,
        default=None,
        help="Path to the zenoh configuration file.",
    )
    parser.add_argument(
        "--router_port",
        type=int,
        default=7447,
        help="Port for the zenoh router (only used when external hosts are present).",
    )
    parser.add_argument(
        "--sim_envs",
        type=lambda p: load_yaml(path(p)) if p else {},
        default=None,
        help="Path to the simulation environments configuration file.",
    )
    parser.add_argument(
        "--hosts",
        type=lambda p: load_hosts(load_yaml(path(p))),
        required=True,
        help="Path to the hosts configuration file.",
    )
    parser.add_argument(
        "--nodes",
        type=lambda p: load_yaml(path(p)) if p else None,
        default=None,
        help="Path to nodes.yaml — launch a node network in every env at startup.",
    )
    parser.add_argument(
        "--real_env",
        type=str,
        default=None,
        help="Name for a real-hardware env (sim=False). E.g. --real_env real.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    configure_logging(env="core")
    args = parse_args()

    core = None
    try:
        core = Core(args.hosts, args.sim_envs, args.zenoh_config, args.router_port, args.real_env)
        if args.nodes:
            core.launch_nodes(args.nodes)
        core.spin()
    except Exception as e:
        print(f"Core failed with exception: {e}")
    finally:
        if core is not None:
            core.close()


if __name__ == "__main__":
    main()
