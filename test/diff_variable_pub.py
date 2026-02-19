from ark.node import BaseNode
from ark_msgs import Value
import argparse
import common_example as common
import torch

HZ = 50
DT = 1.0 / HZ


class LineVariableNode(BaseNode):

    def __init__(self, cfg):
        super().__init__("env", "line_var_pub", cfg, sim=True)
        self.x_pub = self.create_publisher("x")
        self.y_pub = self.create_publisher("y")

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

        self.x._replay_fn = self._replay_grad
        self.y._replay_fn = self._replay_grad

        self.create_stepper(HZ, self.step)

    def forward(self, ts, replay=False):
        """Compute outputs from inputs at a given timestamp.

        Builds the computation graph parameterised by ts so that
        gradients can later be evaluated at arbitrary times.
        When replay=True, uses historical input values at ts.
        """
        if replay:
            v, m, c = self.v.at(ts), self.m.at(ts), self.c.at(ts)
        else:
            v, m, c = self.v.tensor, self.m.tensor, self.c.tensor

        t_val = torch.tensor(ts / 1e9, requires_grad=False)
        x = v * t_val
        y = m * x + c
        return x, y

    def _replay_grad(self, ts, input_name, output_name):
        x, y = self.forward(ts, replay=True)
        outputs = {'x': x, 'y': y}
        inp_var = self._variables[input_name]
        (grad,) = torch.autograd.grad(outputs[output_name], inp_var._replay_tensor, retain_graph=True, allow_unused=True)
        return float(outputs[output_name].detach()), float(grad) if grad is not None else 0.0

    def step(self, ts):
        x, y = self.forward(ts)

        # Setting output tensors triggers eager backward and caches gradients
        self.x.tensor = x
        self.y.tensor = y

        # Snapshot input values at this timestamp
        self.v.snapshot(ts)
        self.m.snapshot(ts)
        self.c.snapshot(ts)

        self.x_pub.publish(Value(val=float(self.x.tensor.detach()), timestamp=ts))
        self.y_pub.publish(Value(val=float(self.y.tensor.detach()), timestamp=ts))


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
