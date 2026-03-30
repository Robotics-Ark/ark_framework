from ark.node import Node
from ark_msgs import String
from itertools import count
from common import z_cfg


class PublisherNode(Node):

    def __init__(self):
        super().__init__("pub", z_cfg)
        self.pub = self.create_publisher("chatter")
        self.rate = self.create_rate(1)  # 1 Hz

    def spin(self):
        for c in count():
            msg = String(data=f"Hello World {c}")
            self.pub.publish(msg)
            print(f"Published: {msg}")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = PublisherNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down publisher node.")
        node.close()
