from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import string_t
from pathlib import Path
import argparse


class TextRepeaterNode(BaseNode):

    def __init__(self, text_path: str, freq: int):
        super().__init__("text_repeater")
        self.text_path = Path(text_path)
        self.text = string_t()
        self.text.data = self.text_path.read_text()
        self.pub = self.create_publisher("text", string_t)
        self.publish_text = lambda: self.pub.publish(self.text)
        self.create_stepper(freq, self.publish_text)


def get_args():
    parser = argparse.ArgumentParser(
        description="Publishes a text file as a string at a given frequency.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the text file.",
    )
    parser.add_argument(
        "--freq",
        type=int,
        help="Frequency to publish text (Hz).",
        default=1,
    )
    args = parser.parse_args()
    return args.path, args.freq


if __name__ == "__main__":
    args = get_args()
    main(TextRepeaterNode, *args)
