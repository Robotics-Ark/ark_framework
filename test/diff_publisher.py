import math
import time
from ark.node import BaseNode
from ark_msgs import Translation, dTranslation
from common import z_cfg
# Lissajous parameters
A, B = 1.0, 1.0
a, b = 3.0, 2.0
delta = math.pi / 2
HZ = 50
DT = 1.0 / HZ
class DiffPublisherNode(BaseNode):
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
if __name__ == "__main__":
    try:
        node = DiffPublisherNode()
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down diff publisher.")
        node.close()
