import time
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ark.node import BaseNode
from ark_msgs import Value, VariableInfo

import argparse
import zenoh
import common_example as common

HZ = 10


class GainOptimizerNode(BaseNode):

    def __init__(self, cfg, target, inertia):
        super().__init__("env", "gain_optimizer", cfg, sim=True)

        self.log_kp = math.log(30)
        self.lr = 0.01
        self.I = inertia

        # History for plotting
        self.kp_history = []
        self.kd_history = []
        self.loss_history = []
        self.grad_kp_history = []
        self.time_history = []

        # Publisher for gain updates
        self.pub_kp = self.create_publisher("param/kp")

        # Subscribe to loss for plotting
        self.create_subscriber("output/loss", self.on_loss)
        self._latest_loss = 0.0

        # Discover gradient channels for loss output
        self._grad_queriers = {}
        self._discover_grad_channels(["loss"], target)

        self.create_stepper(HZ, self.step)

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
                        print(f"Discovered grad channels: {resp.grad_channels}")
                        break
                except Exception:
                    time.sleep(0.2)

    def on_loss(self, msg: Value):
        self._latest_loss = msg.val

    def step(self, ts):
        # Query gradient of loss w.r.t. kp
        grad_kp = 0.0
        for ch, querier in self._grad_queriers.items():
            if "kp" not in ch:
                continue
            try:
                resp = querier.query(Value())
                if isinstance(resp, Value):
                    grad_kp = resp.grad
            except Exception:
                pass

        # Current Kp from log-space
        kp_val = math.exp(self.log_kp)

        # Gradient descent in log-space: d(loss)/d(log_kp) = d(loss)/d(kp) * kp
        self.log_kp -= self.lr * grad_kp * kp_val

        # Derive Kd for display
        kd_val = 2.0 * math.sqrt(kp_val * self.I)

        # Publish updated gain
        self.pub_kp.publish(Value(val=kp_val))

        # Record history for plotting
        self.kp_history.append(kp_val)
        self.kd_history.append(kd_val)
        self.loss_history.append(self._latest_loss)
        self.grad_kp_history.append(grad_kp)
        self.time_history.append(ts / 1e9)

        print(f"Kp={kp_val:.3f}  Kd={kd_val:.3f}  grad_kp={grad_kp:.6f}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="gain_optimizer", description="Gradient-descent gain optimizer"
        )
        common.add_config_arguments(parser)
        parser.add_argument(
            "--target", "-t",
            dest="target",
            choices=["ALL", "BEST_MATCHING", "ALL_COMPLETE"],
            default="BEST_MATCHING",
            type=str,
            help="The target queryables of the query.",
        )
        parser.add_argument(
            "--inertia", "-I",
            dest="inertia",
            default=3.333,
            type=float,
            help="Joint inertia for critical damping (match pendulum URDF).",
        )
        args = parser.parse_args()
        conf = common.get_config_from_args(args)

        target = {
            "ALL": zenoh.QueryTarget.ALL,
            "BEST_MATCHING": zenoh.QueryTarget.BEST_MATCHING,
            "ALL_COMPLETE": zenoh.QueryTarget.ALL_COMPLETE,
        }.get(args.target)

        node = GainOptimizerNode(conf, target, args.inertia)
        threading.Thread(target=node.spin, daemon=True).start()

        fig, (ax_gains, ax_loss, ax_grad) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        ax_gains.set_title("Gains")
        ax_gains.set_ylabel("Value")
        (line_kp,) = ax_gains.plot([], [], "b-", label="Kp")
        (line_kd,) = ax_gains.plot([], [], "r-", label="Kd")
        ax_gains.legend()

        ax_loss.set_title("Loss")
        ax_loss.set_ylabel("Loss")
        (line_loss,) = ax_loss.plot([], [], "g-")

        ax_grad.set_title("d(loss)/d(Kp)")
        ax_grad.set_xlabel("Sim time (s)")
        ax_grad.set_ylabel("Gradient")
        (line_grad,) = ax_grad.plot([], [], "m-")

        def update(frame):
            t = node.time_history
            line_kp.set_data(t, node.kp_history)
            line_kd.set_data(t, node.kd_history)
            line_loss.set_data(t, node.loss_history)
            line_grad.set_data(t, node.grad_kp_history)
            for ax in (ax_gains, ax_loss, ax_grad):
                ax.relim()
                ax.autoscale_view()
            return line_kp, line_kd, line_loss, line_grad

        ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        node.close()
    except KeyboardInterrupt:
        print("Shutting down gain optimizer.")
        node.close()
