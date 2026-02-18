import argparse
import time
import zenoh
from ark.node import BaseNode
from ark_msgs import VariableInfo


class RegistryNode(BaseNode):

    def __init__(self, cfg):
        super().__init__("ark", "registry", cfg)
        self._var_registry: dict[str, VariableInfo] = {}
        self.create_subscriber("ark/vars/register", self._on_register)

    def _on_register(self, msg: VariableInfo):
        name = msg.output_name
        self._var_registry[name] = msg
        channel = f"ark/vars/{name}"
        if channel not in self._queriables:
            def _make_handler(n):
                def handler(_req):
                    return self._var_registry[n]
                return handler
            self.create_queryable(channel, _make_handler(name))
        print(f"Registered output variable '{name}' from node '{msg.node_name}' "
              f"with channels: {list(msg.grad_channels)}")

    def core_registration(self):
        pass

    def close(self):
        super().close()


def main():
    parser = argparse.ArgumentParser(
        prog="ark-core", description="Ark central registry"
    )
    parser.add_argument("--mode", "-m", dest="mode",
                        choices=["peer", "client"], type=str)
    parser.add_argument("--connect", "-e", dest="connect",
                        metavar="ENDPOINT", action="append", type=str)
    parser.add_argument("--listen", "-l", dest="listen",
                        metavar="ENDPOINT", action="append", type=str)
    args = parser.parse_args()

    cfg = zenoh.Config()
    if args.mode:
        import json
        cfg.insert_json5("mode", json.dumps(args.mode))
    if args.connect:
        import json
        cfg.insert_json5("connect/endpoints", json.dumps(args.connect))
    if args.listen:
        import json
        cfg.insert_json5("listen/endpoints", json.dumps(args.listen))

    node = RegistryNode(cfg)
    print("Ark registry running.")
    try:
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down registry.")
        node.close()


if __name__ == "__main__":
    main()
