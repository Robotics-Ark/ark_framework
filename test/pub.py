from ark.node import BaseNode
from itertools import count
from common import z_cfg


class PublisherNode(BaseNode):

    def __init__(self):
        super().__init__("env", "pub", z_cfg, sim=True)
        self.pub = self.create_publisher("chatter")
        self.rate = self.create_rate(1)  # 1 Hz

    def spin(self):
        for c in count():
            msg = f"Hello World {c}"
            self.pub.publish(msg.encode("utf-8"))
            print(f"Published: {msg}")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = PublisherNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down publisher node.")
        node.close()
