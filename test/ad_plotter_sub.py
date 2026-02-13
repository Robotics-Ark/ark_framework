import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ark.node import BaseNode
from ark_msgs import Translation, Value
# from common import connect_cfg, z_cfg
import argparse
import zenoh
import common_example as common


class AutodiffPlotterNode(BaseNode):
    def __init__(self, cfg, target):
        super().__init__("env", "autodiff_plotter", cfg, sim=True)
        self.pos_x, self.pos_y = [], []
        self.grad_vx, self.grad_my = [], []
        self.create_subscriber("position", self.on_position)
        # self.grad_vx_querier = self.create_querier("grad/v/x", target=target)
        # self.grad_my_querier = self.create_querier("grad/m/y", target=target)
        self.grad_vx_querier = self._session.declare_querier(
            "grad/v/x",
            target=target,
            timeout=10.0,
        )
        self.grad_my_querier = self._session.declare_querier(
            "grad/m/y",
            target=target,
            timeout=10.0
        )

    def on_position(self, msg: Translation):
        self.pos_x.append(msg.x)
        self.pos_y.append(msg.y)

    def fetch_grads(self):
        req = Translation(x=0.0, y=0.0, z=0.0)
        print("fetching grads")
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

    def fetch_grads_exp(self):
        try:
            resp_vx = self.grad_vx_querier.get()
            for resp in resp_vx:
                if resp.ok is None:
                    continue
                v_value_str = bytes(resp.ok.payload).decode("utf-8")
                v_value = float(v_value_str)
                print(f"Queried grad_vx: {v_value}")
                self.grad_vx.append(v_value)
        except Exception:
            pass
        try:
            resp_my = self.grad_my_querier.get()
            for resp in resp_my:
                if resp.ok is None:
                    continue
                m_value_str = bytes(resp.ok.payload).decode("utf-8")
                m_value = float(m_value_str)
                print(f"Queried grad_my: {m_value}")
                self.grad_my.append(m_value)
        except Exception:
            print("Failed to query grad_my")
            pass
def main():
    parser = argparse.ArgumentParser(description="Autodiff Plotter Node")
    common.add_config_arguments(parser)
    parser.add_argument(
        "--target",
        "-t",
        dest="target",
        choices=["ALL", "BEST_MATCHING", "ALL_COMPLETE", "NONE"],
        default="BEST_MATCHING",
        type=str,
        help="The target queryables of the query.",
    )
    parser.add_argument(
        "--timeout",
        "-o",
        dest="timeout",
        default=10.0,
        type=float,
        help="The query timeout",
    )
    parser.add_argument(
        "--iter", dest="iter", type=int, help="How many gets to perform"
    )
    parser.add_argument(
        "--add-matching-listener",
        default=False,
        action="store_true",
        help="Add matching listener",
    )
    
    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    target = {
        "ALL": zenoh.QueryTarget.ALL,
        "BEST_MATCHING": zenoh.QueryTarget.BEST_MATCHING,
        "ALL_COMPLETE": zenoh.QueryTarget.ALL_COMPLETE,
    }.get(args.target)

    node = AutodiffPlotterNode(conf, target)
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
    ax_grad.set_xlim(-5, 5)
    ax_grad.set_ylim(-5, 5)
    (line_grad_vx,) = ax_grad.plot([], [], "g-", label="dx/dv")
    (line_grad_my,) = ax_grad.plot([], [], "m-", label="dy/dm")
    ax_grad.legend()
    def update(frame):
        node.fetch_grads_exp()
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
