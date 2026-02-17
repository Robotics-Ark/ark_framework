from ark.node import BaseNode
from ark_msgs import Value
import argparse
import common_example as common

HZ = 10


class ParamPublisherNode(BaseNode):

    def __init__(self, cfg):
        super().__init__("env", "param_pub", cfg, sim=True)
        self.pub_v = self.create_publisher("param/v")
        self.pub_m = self.create_publisher("param/m")
        self.pub_c = self.create_publisher("param/c")
        self.rate = self.create_rate(HZ)

    def spin(self):
        while True:
            self.pub_v.publish(Value(val=1.0))
            self.pub_m.publish(Value(val=0.5))
            self.pub_c.publish(Value(val=0.0))
            self.rate.sleep()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="param_publisher", description="Publishes parameter values"
        )
        common.add_config_arguments(parser)
        args = parser.parse_args()
        conf = common.get_config_from_args(args)

        node = ParamPublisherNode(conf)
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down param publisher.")
        node.close()
