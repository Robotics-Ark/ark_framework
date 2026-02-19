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
        self.pos_x_ts, self.pos_y_ts = [], []
        self._grad_queriers = {}  # channel -> Querier
        self._grad_data = {}      # channel -> [float]
        self._grad_ts = {}        # channel -> [int]
        self.create_subscriber("x", self.on_x)
        self.create_subscriber("y", self.on_y)
        self._discover_grad_channels(["x", "y"], target)

    def _discover_grad_channels(self, output_names, target, timeout=5.0):
        for out in output_names:
            disc = self.create_querier(f"ark/vars/{out}", target=target)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    resp = disc.query(VariableInfo())
                    if isinstance(resp, VariableInfo):
                        for ch in resp.grad_channels:
                            self._grad_queriers[ch] = self.create_querier(ch, target=target)
                            self._grad_data[ch] = []
                            self._grad_ts[ch] = []
                        break
                except Exception:
                    time.sleep(0.2)

    def on_x(self, msg: Value):
        self.pos_x.append(msg.val)
        self.pos_x_ts.append(msg.timestamp)

    def on_y(self, msg: Value):
        self.pos_y.append(msg.val)
        self.pos_y_ts.append(msg.timestamp)

    def fetch_grads(self):
        req = Value()
        for ch, querier in self._grad_queriers.items():
            try:
                resp = querier.query(req)
                if isinstance(resp, Value):
                    self._grad_data[ch].append(resp.grad)
                    self._grad_ts[ch].append(resp.timestamp)
            except Exception:
                pass

    def fetch_grads_at(self, ts):
        req = Value(timestamp=ts)
        results = {}
        for ch, querier in self._grad_queriers.items():
            try:
                resp = querier.query(req)
                if isinstance(resp, Value):
                    results[ch] = (resp.val, resp.grad)
            except Exception:
                pass
        return results


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

    fig, (ax_pos, ax_grad, ax_replay) = plt.subplots(1, 3, figsize=(18, 5))
    ax_pos.set_title("Position")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")
    ax_pos.set_aspect("equal")
    (line_pos,) = ax_pos.plot([], [], "b-")
    ax_grad.set_title("Gradients (live)")
    ax_grad.set_xlabel("sim time (s)")
    ax_grad.set_ylabel("grad")

    colors = plt.cm.tab10.colors
    grad_lines = {}
    for i, ch in enumerate(node._grad_queriers):
        (line,) = ax_grad.plot([], [], color=colors[i % 10], label=ch)
        grad_lines[ch] = line
    ax_grad.legend()

    ax_replay.set_title("Gradients (replay)")
    ax_replay.set_xlabel("sim time (s)")
    ax_replay.set_ylabel("grad")
    replay_data = {ch: [] for ch in node._grad_queriers}
    replay_ts = {ch: [] for ch in node._grad_queriers}
    replay_lines = {}
    for i, ch in enumerate(node._grad_queriers):
        (line,) = ax_replay.plot([], [], color=colors[i % 10], label=ch)
        replay_lines[ch] = line
    ax_replay.legend()

    def update(frame):
        node.fetch_grads()

        # Replay: query gradient at a historical timestamp
        if len(node.pos_x_ts) > 10:
            historical_ts = node.pos_x_ts[-10]
            results = node.fetch_grads_at(historical_ts)
            for ch, (val, grad) in results.items():
                replay_data[ch].append(grad)
                replay_ts[ch].append(historical_ts)

        n = min(len(node.pos_x), len(node.pos_y))
        line_pos.set_data(node.pos_x[:n], node.pos_y[:n])
        ax_pos.relim()
        ax_pos.autoscale_view()
        for ch, line in grad_lines.items():
            data = node._grad_data[ch]
            times = [t / 1e9 for t in node._grad_ts[ch]]
            line.set_data(times[: len(data)], data)
        ax_grad.relim()
        ax_grad.autoscale_view()
        for ch, line in replay_lines.items():
            data = replay_data[ch]
            times = [t / 1e9 for t in replay_ts[ch]]
            line.set_data(times[: len(data)], data)
        ax_replay.relim()
        ax_replay.autoscale_view()
        return line_pos, *grad_lines.values(), *replay_lines.values()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    node.close()


if __name__ == "__main__":
    main()
