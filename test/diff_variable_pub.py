from ark.node import BaseNode
from ark_msgs import Translation
import argparse
import common_example as common
import torch

HZ = 50
DT = 1.0 / HZ


class LineVariableNode(BaseNode):

    def __init__(self, cfg):
        super().__init__("env", "line_var_pub", cfg, sim=True)
        self.pos_pub = self.create_publisher("position")
        self.rate = self.create_rate(HZ)

        # Output variables auto-create grad queryables:
        # grad/v/x, grad/v/y, grad/m/x, grad/m/y, grad/c/x, grad/c/y
        self.v = self.create_variable("v", 0.0, mode="input")
        self.m = self.create_variable("m", 0.0, mode="input")
        self.c = self.create_variable("c", 0.0, mode="input")
        self.x = self.create_variable("x", 0.0, mode="output")
        self.y = self.create_variable("y", 0.0, mode="output")

        self.create_subscriber("param/v", lambda msg: self.v.tensor.data.fill_(msg.val))
        self.create_subscriber("param/m", lambda msg: self.m.tensor.data.fill_(msg.val))
        self.create_subscriber("param/c", lambda msg: self.c.tensor.data.fill_(msg.val))

    def spin(self):
        t = 0.0
        while True:
            t_val = torch.tensor(t, requires_grad=False)

            # Forward: x = v * t, y = m * x + c
            # Setting output tensors triggers eager backward and caches gradients
            self.x.tensor = self.v.tensor * t_val
            self.y.tensor = self.m.tensor * self.x.tensor + self.c.tensor

            self.pos_pub.publish(
                Translation(x=float(self.x.tensor.detach()), y=float(self.y.tensor.detach()), z=0.0)
            )

            t += DT
            self.rate.sleep()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="diff_variable_pub", description="Differentiable variable publisher"
        )
        common.add_config_arguments(parser)
        args = parser.parse_args()
        conf = common.get_config_from_args(args)

        node = LineVariableNode(conf)
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down diff variable publisher.")
        node.close()
