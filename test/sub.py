from common import z_cfg
from ark.node import Node


class SubscriberNode(Node):
    def __init__(self):
        super().__init__("sub", z_cfg)
        self.sub = self.create_subscriber("chatter", self.callback)

    def callback(self, msg):
        print(f"Recieved: {msg.data}")

    def spin(self):
        while True:
            pass


if __name__ == "__main__":
    try:
        node = SubscriberNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down subscriber node.")
        node.close()
