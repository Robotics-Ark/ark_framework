import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ark.node import BaseNode
from ark_msgs import Translation, Value
from common import z_cfg

class AutodiffPlotterNode(BaseNode):
    def __init__(self):
        super().__init__("env", "autodiff_plotter", z_cfg, sim=True)
        self.pos_x, self.pos_y = [], []
        self.grad_vx, self.grad_my = [], []
        self.create_subscriber("position", self.on_position)
        self.grad_vx_querier = self.create_querier("grad/v/x")
        self.grad_my_querier = self.create_querier("grad/m/y")
    def on_position(self, msg: Translation):
        self.pos_x.append(msg.x)
        self.pos_y.append(msg.y)
    def fetch_grads(self):
        req = Translation(x=0.0, y=0.0, z=0.0)
        try:
            resp_vx = self.grad_vx_querier.query(req)
            print(f"Queried grad_vx: {resp_vx.grad}")
            if isinstance(resp_vx, Value):
                self.grad_vx.append(resp_vx.grad)
        except Exception:
            pass
        try:
            resp_my = self.grad_my_querier.query(req)
            if isinstance(resp_my, Value):
                self.grad_my.append(resp_my.grad)
        except Exception:
            pass
def main():
    node = AutodiffPlotterNode()
    threading.Thread(target=node.spin, daemon=True).start()
    fig, (ax_pos, ax_grad) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pos.set_title("Position (Translation)")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")
    ax_pos.set_xlim(-5, 5)
    ax_pos.set_ylim(-5, 5)
    ax_pos.set_aspect("equal")
    (line_pos,) = ax_pos.plot([], [], "b-")
    ax_grad.set_title("Gradients")
    ax_grad.set_xlabel("t")
    ax_grad.set_ylabel("grad")
    (line_grad_vx,) = ax_grad.plot([], [], "g-", label="dx/dv")
    (line_grad_my,) = ax_grad.plot([], [], "m-", label="dy/dm")
    ax_grad.legend()
    def update(frame):
        node.fetch_grads()
        line_pos.set_data(node.pos_x, node.pos_y)
        line_grad_vx.set_data(range(len(node.grad_vx)), node.grad_vx)
        line_grad_my.set_data(range(len(node.grad_my)), node.grad_my)
        return line_pos, line_grad_vx, line_grad_my
    ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    node.close()
if __name__ == "__main__":
    main()
