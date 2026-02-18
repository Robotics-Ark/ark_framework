import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ark.node import BaseNode
from ark_msgs import Value, VariableInfo

import argparse
import zenoh
import common_example as common


class AutodiffPlotterNode(BaseNode):
    def __init__(self, cfg, target):
        super().__init__("env", "autodiff_plotter", cfg, sim=True)
        self.pos_x, self.pos_y = [], []
        self._grad_queriers = {}  # channel -> Querier
        self._grad_data = {}      # channel -> [float]
        self._grad_times = []
        self.create_subscriber("x", self.on_x)
        self.create_subscriber("y", self.on_y)
        self._discover_grad_channels(["x", "y"], target)

    def _discover_grad_channels(self, output_names, target, timeout=5.0):
        for out in output_names:
            disc = self.create_querier(f"ark/vars/{out}", target=target)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    resp = disc.query(Value())
                    if isinstance(resp, VariableInfo):
                        for ch in resp.grad_channels:
                            self._grad_queriers[ch] = self.create_querier(ch, target=target)
                            self._grad_data[ch] = []
                        break
                except Exception:
                    time.sleep(0.2)

    def on_x(self, msg: Value):
        self.pos_x.append(msg.val)

    def on_y(self, msg: Value):
        self.pos_y.append(msg.val)

    def fetch_grads(self):
        req = Value()
        sim_t = self._clock.now() / 1e9
        for ch, querier in self._grad_queriers.items():
            try:
                resp = querier.query(req)
                if isinstance(resp, Value):
                    self._grad_data[ch].append(resp.grad)
            except Exception:
                pass
        self._grad_times.append(sim_t)


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
    ax_pos.set_title("Position")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")
    ax_pos.set_aspect("equal")
    (line_pos,) = ax_pos.plot([], [], "b-")
    ax_grad.set_title("Gradients")
    ax_grad.set_xlabel("sim time (s)")
    ax_grad.set_ylabel("grad")

    colors = plt.cm.tab10.colors
    grad_lines = {}
    for i, ch in enumerate(node._grad_queriers):
        (line,) = ax_grad.plot([], [], color=colors[i % 10], label=ch)
        grad_lines[ch] = line
    ax_grad.legend()

    def update(frame):
        node.fetch_grads()
        n = min(len(node.pos_x), len(node.pos_y))
        line_pos.set_data(node.pos_x[:n], node.pos_y[:n])
        ax_pos.relim()
        ax_pos.autoscale_view()
        times = node._grad_times
        for ch, line in grad_lines.items():
            data = node._grad_data[ch]
            line.set_data(times[: len(data)], data)
        ax_grad.relim()
        ax_grad.autoscale_view()
        return line_pos, *grad_lines.values()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    node.close()


if __name__ == "__main__":
    main()
