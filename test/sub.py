from common import z_cfg
from ark.node import BaseNode


class SubscriberNode(BaseNode):
    def __init__(self):
        super().__init__("env", "sub", z_cfg, sim=True, collect_data=True)
        self.sub = self.create_subscriber("chatter", self.callback)

    def callback(self, msg: bytes):
        print(f"Received: {msg.decode('utf-8')}")


if __name__ == "__main__":
    try:
        node = SubscriberNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down subscriber node.")
        node.close()
