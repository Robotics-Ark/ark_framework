from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import string_t
from pathlib import Path
import argparse

class TextRepeaterNode(BaseNode):
    def __init__(self, node_name: str, global_config):
        super().__init__(node_name, global_config)
        # Required keys in the global config
        text_path = self.config.get("text_path", "")
        text = self.config.get("text", "")
        channel = self.config.get("channel", "user_input")
        freq = self.config.get("freq", 1)

        # Exactly one of 'text' or 'text_path' should be provided; both default to "".
        if text and text_path:
            raise ValueError("Pass only one of 'text' or 'text_path' (not both).")

        if text_path:
            p = Path(text_path)
            if not p.is_file():
                raise FileNotFoundError(f"'text_path' does not exist or is not a file: {p}")
            self.text = p.read_text()
        else:
            self.text = text

        self.text_msg = string_t()
        self.text_msg.data = self.text

        # Publisher on requested channel
        self.pub = self.create_publisher(channel, string_t)

        # Stepper that publishes at the requested frequency
        self.publish_text = lambda: self.pub.publish(self.text_msg)
        self.create_stepper(freq, self.publish_text)


def get_args():
    parser = argparse.ArgumentParser(
        description="Publishes a text file as a string at a given frequency, using a global config.",
    )
    parser.add_argument(
        "--node-name",
        type=str,
        required=True,
        help="Name of this node.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to global config file (.json or .yaml/.yml) containing: text_path, channel, freq.",
    )
    args = parser.parse_args()
    return args.node_name, args.config


if __name__ == "__main__":
    node_name, global_config = get_args()
    # Pass exactly (node_name, global_config) to match TextRepeaterNode.__init__
    main(TextRepeaterNode, node_name, global_config)