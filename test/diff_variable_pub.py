import math
import time
from ark.node import BaseNode
from ark_msgs import Translation, Value
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

        # Create differentiable input variables — auto-creates grad queryables
        # grad/v/x, grad/v/y, grad/m/x, grad/m/y, grad/c/x, grad/c/y
        self.v = self.create_variable("v", 0.0, mode="input", out_fields=["x", "y"])
        self.m = self.create_variable("m", 0.0, mode="input", out_fields=["x", "y"])
        self.c = self.create_variable("c", 0.0, mode="input", out_fields=["x", "y"])

    def spin(self):
        t = 0.0
        while True:
            t_val = torch.tensor(t, requires_grad=False)

            # Forward: y = m * x + c, where x = v * t
            x = self.v.tensor * t_val
            y = self.m.tensor * x + self.c.tensor

            # Publish position
            self.pos_pub.publish(
                Translation(x=float(x.detach()), y=float(y.detach()), z=0.0)
            )

            # Backward: compute gradients
            if self.v.tensor.grad is not None:
                self.v.tensor.grad.zero_()
            if self.m.tensor.grad is not None:
                self.m.tensor.grad.zero_()
            if self.c.tensor.grad is not None:
                self.c.tensor.grad.zero_()

            x.backward(retain_graph=True)
            v_x = float(self.v.tensor.grad)
            self.v.tensor.grad.zero_()

            y.backward()
            v_y = float(self.v.tensor.grad)
            m_y = float(self.m.tensor.grad)
            c_y = float(self.c.tensor.grad)

            # Update variable gradients — served automatically by queryables
            self.v.update_gradients({"x": v_x, "y": v_y})
            self.m.update_gradients({"x": 0.0, "y": m_y})
            self.c.update_gradients({"x": 0.0, "y": c_y})

            print(f"t={t:.2f}  dx/dv={v_x:.3f}  dy/dm={m_y:.3f}")

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
