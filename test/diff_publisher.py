import math
import time
from ark.node import BaseNode
from ark_msgs import Translation, Value
from common import z_cfg
import torch

# Lissajous parameters
A, B = 1.0, 1.0
a, b = 3.0, 2.0
delta = math.pi / 2
HZ = 50
DT = 1.0 / HZ
class LissajousPublisherNode(BaseNode):
    def __init__(self):
        super().__init__("env", "diff_pub", z_cfg, sim=True)
        self.pos_pub = self.create_publisher("position")
        self.vel_pub = self.create_publisher("velocity")
        self.rate = self.create_rate(HZ)
    def spin(self):
        t = 0.0
        while True:
            x = A * math.sin(a * t + delta)
            y = B * math.sin(b * t)
            dx = A * a * math.cos(a * t + delta)
            dy = B * b * math.cos(b * t)
            self.pos_pub.publish(Translation(x=x, y=y, z=0.0))
            self.vel_pub.publish(dTranslation(x=dx, y=dy, z=0.0))
            t += DT
            self.rate.sleep()

class LinePublisherNode(BaseNode):

    def __init__(self):
        super().__init__("env", "line_pub", z_cfg, sim=True)
        self.pos_pub = self.create_publisher("position")
        self.rate = self.create_rate(HZ)
        self.v = torch.tensor(1.0, requires_grad=True)
        self.m = torch.tensor(0.5, requires_grad=True)
        self.c = torch.tensor(0.0, requires_grad=True)
        self.latest = {
            "x": 0.0,
            "y": 0.0,
            "v_x": 0.0,
            "v_y": 0.0,
            "m_x": 0.0,
            "m_y": 0.0,
            "c_x": 0.0,
            "c_y": 0.0,
        }
        self.create_queryable("grad/v/x", self._on_grad_v_x)
        self.create_queryable("grad/v/y", self._on_grad_v_y)
        self.create_queryable("grad/m/x", self._on_grad_m_x)
        self.create_queryable("grad/m/y", self._on_grad_m_y)
        self.create_queryable("grad/c/x", self._on_grad_c_x)
        self.create_queryable("grad/c/y", self._on_grad_c_y)

    def _on_grad_v_x(self, _req):
        return Value(val=self.latest["x"], grad=self.latest["v_x"])

    def _on_grad_v_y(self, _req):
        return Value(val=self.latest["y"], grad=self.latest["v_y"])

    def _on_grad_m_x(self, _req):
        return Value(val=self.latest["x"], grad=self.latest["m_x"])

    def _on_grad_m_y(self, _req):
        return Value(val=self.latest["y"], grad=self.latest["m_y"])

    def _on_grad_c_x(self, _req):
        return Value(val=self.latest["x"], grad=self.latest["c_x"])

    def _on_grad_c_y(self, _req):
        return Value(val=self.latest["y"], grad=self.latest["c_y"])

    def spin(self):
        t = 0.0
        while True:
            t_val = torch.tensor(t, requires_grad=False)
            x = self.v * t_val
            y = self.m * x + self.c
            self.pos_pub.publish(Translation(x=float(x.detach()),
                                             y=float(y.detach()), z=0.0))
            if self.v.grad is not None:
                self.v.grad.zero_()
            if self.m.grad is not None:
                self.m.grad.zero_()
            if self.c.grad is not None:
                self.c.grad.zero_()
            x.backward(retain_graph=True)
            self.latest["v_x"] = float(self.v.grad)
            # self.latest["m_x"] = float(self.m.grad)
            # self.latest["c_x"] = float(self.c.grad)
            self.v.grad.zero_()
            y.backward()
            self.latest["v_y"] = float(self.v.grad)
            self.latest["m_y"] = float(self.m.grad)
            self.latest["c_y"] = float(self.c.grad)
            self.latest["x"] = float(x.detach())
            self.latest["y"] = float(y.detach())
            t += DT
            self.rate.sleep()

if __name__ == "__main__":
    try:
        # node = LissajousPublisherNode()
        node = LinePublisherNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down diff publisher.")
        node.close()
