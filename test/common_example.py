import argparse
import json

import zenoh


def add_config_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mode",
        "-m",
        dest="mode",
        choices=["peer", "client"],
        type=str,
        help="The zenoh session mode.",
    )
    parser.add_argument(
        "--connect",
        "-e",
        dest="connect",
        metavar="ENDPOINT",
        action="append",
        type=str,
        help="Endpoints to connect to.",
    )
    parser.add_argument(
        "--listen",
        "-l",
        dest="listen",
        metavar="ENDPOINT",
        action="append",
        type=str,
        help="Endpoints to listen on.",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        metavar="FILE",
        type=str,
        help="A configuration file.",
    )
    parser.add_argument(
        "--no-multicast-scouting",
        dest="no_multicast_scouting",
        default=False,
        action="store_true",
        help="Disable multicast scouting.",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg",
        metavar="CFG",
        default=[],
        action="append",
        type=str,
        help="Allows arbitrary configuration changes as column-separated KEY:VALUE pairs. Where KEY must be a valid config path and VALUE must be a valid JSON5 string that can be deserialized to the expected type for the KEY field. Example: --cfg='transport/unicast/max_links:2'.",
    )


def get_config_from_args(args) -> zenoh.Config:
    conf = (
        zenoh.Config.from_file(args.config)
        if args.config is not None
        else zenoh.Config()
    )
    if args.mode is not None:
        conf.insert_json5("mode", json.dumps(args.mode))
    if args.connect is not None:
        conf.insert_json5("connect/endpoints", json.dumps(args.connect))
    if args.listen is not None:
        conf.insert_json5("listen/endpoints", json.dumps(args.listen))
    if args.no_multicast_scouting:
        conf.insert_json5("scouting/multicast/enabled", json.dumps(False))

    for c in args.cfg:
        try:
            [key, value] = c.split(":", 1)
        except:
            print(f"`--cfg` argument: expected KEY:VALUE pair, got {c}")
            raise
        conf.insert_json5(key, value)

    return conf
