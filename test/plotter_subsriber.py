import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ark.node import BaseNode
from ark_msgs import Translation, dTranslation
from common import z_cfg
class SubscriberPlotterNode(BaseNode):
    def __init__(self):
        super().__init__("env", "plotter", z_cfg, sim=True)
        self.pos_x, self.pos_y = [], []
        self.vel_x, self.vel_y = [], []
        self.create_subscriber("position", self.on_position)
        self.create_subscriber("velocity", self.on_velocity)
    def on_position(self, msg: Translation):
        self.pos_x.append(msg.x)
        self.pos_y.append(msg.y)
    def on_velocity(self, msg: dTranslation):
        self.vel_x.append(msg.x)
        self.vel_y.append(msg.y)
def main():
    node = SubscriberPlotterNode()
    threading.Thread(target=node.spin, daemon=True).start()
    fig, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(10, 5))
    ax_pos.set_title("Position (Translation)")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")
    ax_pos.set_xlim(-1.5, 1.5)
    ax_pos.set_ylim(-1.5, 1.5)
    ax_pos.set_aspect("equal")
    (line_pos,) = ax_pos.plot([], [], "b-")
    ax_vel.set_title("Velocity (dTranslation)")
    ax_vel.set_xlabel("dx")
    ax_vel.set_ylabel("dy")
    ax_vel.set_xlim(-5, 5)
    ax_vel.set_ylim(-5, 5)
    ax_vel.set_aspect("equal")
    (line_vel,) = ax_vel.plot([], [], "r-")
    def update(frame):
        line_pos.set_data(node.pos_x, node.pos_y)
        line_vel.set_data(node.vel_x, node.vel_y)
        return line_pos, line_vel
    ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    node.close()
if __name__ == "__main__":
    main()
