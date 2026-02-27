from ark.node import BaseNode
from ark_msgs import Value
import argparse
import collections
import common_example as common
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import pybullet as p
import os

HZ = 240
DT = 1.0 / HZ


class PDBulletNode(BaseNode):

    def __init__(self, cfg, plot=False):
        super().__init__("env", "pd_bullet", cfg, sim=True)
        self._plot_enabled = plot

        # PyBullet setup
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(DT, physicsClientId=self.physics_client)
        urdf_path = os.path.join(os.path.dirname(__file__), "pendulum.urdf")
        self.robot_id = p.loadURDF(
            urdf_path, useFixedBase=True, physicsClientId=self.physics_client
        )
        self.joint_idx = 0

        # Disable default motor for torque control
        p.setJointMotorControl2(
            self.robot_id, self.joint_idx,
            p.VELOCITY_CONTROL, force=0,
            physicsClientId=self.physics_client,
        )

        # Initial displacement: 45 degrees
        p.resetJointState(
            self.robot_id, self.joint_idx,
            targetValue=0.785, targetVelocity=0.0,
            physicsClientId=self.physics_client,
        )

        # Query joint inertia from PyBullet for critical damping
        dyn_info = p.getDynamicsInfo(self.robot_id, 0, physicsClientId=self.physics_client)
        # dyn_info: (mass, lateral_friction, local_inertia_diagonal, ...)
        # local_inertia_diagonal is a 3-tuple (ixx, iyy, izz)
        # Joint rotates about X axis, so use ixx
        self.I = dyn_info[2][0]
        print(f"Joint inertia (about rotation axis): {self.I:.4f}")

        ############## Bullet setup complete, now define control variables and publishers ##############################

        # Target state
        self.q_target = 1.5708 # 90 degrees
        self.qd_target = 0.0

        # Input variable: only Kp (Kd derived from critical damping)
        self.kp = self.create_variable("kp", 30.0, mode="input")
        self.kp_gt = torch.tensor(300)

        # Output variable
        self.loss_var = self.create_variable("loss", 0.0, mode="output")

        # Replay function for temporal gradient queries
        self.loss_var._replay_fn = self._replay_grad
        self._state_history = {}

        # Publishers for visualization
        self.q_pub = self.create_publisher("q")
        self.torque_pub = self.create_publisher("torque_val")

        if self._plot_enabled:
            self._init_debug_plot()

        self.create_stepper(HZ, self.step)

    def forward(self, ts=None, replay=False):
        # Read state from PyBullet (constants, no grad)
        q_val, qd_val, _, _ = p.getJointState(
            self.robot_id, self.joint_idx, physicsClientId=self.physics_client
        )

        # Store state for replay
        if ts is not None and not replay:
            self._state_history[ts] = (q_val, qd_val)

        q = torch.tensor(q_val)
        qd = torch.tensor(qd_val)
        q_t = torch.tensor(self.q_target)
        qd_t = torch.tensor(self.qd_target)

        ## Retrieved robot state from simulator ##

        # Get Kp (differentiable)
        if replay:
            kp = self.kp.at(ts)
        else:
            kp = self.kp.tensor
        print(f"Current Kp: {float(kp.detach()):.4f}")

        # Critical damping: Kd = 2 * sqrt(Kp * I)
        kd = 2.0 * torch.sqrt(kp * self.I)
        kd_gt = 2.0 * torch.sqrt(self.kp_gt.detach() * self.I)

        error = q_t - q
        derror = qd_t - qd
        torque = kp * error + kd * derror
        torque_gt = self.kp_gt.detach() * error + kd_gt * derror

        # Loss: tracking error + control effort penalty
        w_effort = 0.01
        loss = error ** 2 + derror ** 2 + w_effort * (torque - torque_gt) ** 2

        return torque, loss, q_val, float(error.detach()), float(derror.detach())

    def _replay_grad(self, ts, input_name, output_name):
        # Used to replay the forward pass at historical timestamp ts for
        # gradient queries.
        q_val, qd_val = self._state_history[ts]

        q = torch.tensor(q_val)
        qd = torch.tensor(qd_val)
        q_t = torch.tensor(self.q_target)
        qd_t = torch.tensor(self.qd_target)

        kp = self.kp.at(ts)
        kd = 2.0 * torch.sqrt(kp * self.I)

        error = q_t - q
        derror = qd_t - qd
        torque = kp * error + kd * derror

        w_effort = 0.01
        loss = error ** 2 + w_effort * torque ** 2

        inp_var = self._variables[input_name]
        (grad,) = torch.autograd.grad(
            loss, inp_var._replay_tensor,
            retain_graph=True, allow_unused=True,
        )
        return float(loss.detach()), float(grad) if grad is not None else 0.0

    def _init_debug_plot(self):
        n = 200
        self._error_buf = collections.deque([0.0] * n, maxlen=n)
        self._derror_buf = collections.deque([0.0] * n, maxlen=n)

        self._plot_fig, (self._ax_e, self._ax_de) = plt.subplots(2, 1, figsize=(6, 4))
        self._plot_fig.suptitle("PD Controller")

        self._line_e, = self._ax_e.plot(list(self._error_buf), "r-")
        self._ax_e.set_ylabel("error (rad)")
        self._ax_e.set_ylim(-2.0, 2.0)
        self._ax_e.axhline(0, color="k", linewidth=0.5)

        self._line_de, = self._ax_de.plot(list(self._derror_buf), "g-")
        self._ax_de.set_ylabel("derror (rad/s)")
        self._ax_de.set_ylim(-10.0, 10.0)
        self._ax_de.axhline(0, color="k", linewidth=0.5)

        plt.tight_layout()

    def _animate(self, _frame):
        # Called on the main thread by FuncAnimation — safe to touch matplotlib
        self._line_e.set_ydata(list(self._error_buf))
        self._line_de.set_ydata(list(self._derror_buf))
        return self._line_e, self._line_de

    def _update_debug_plot(self, error, derror):
        # Called from the stepper thread — only touch thread-safe deques
        self._error_buf.append(error)
        self._derror_buf.append(derror)

    def step(self, ts):
        torque, loss, q_val, error, derror = self.forward(ts)

        if self._plot_enabled:
            self._update_debug_plot(error, derror)

        # Assign to output variable and compute gradients
        self.loss_var.tensor = loss
        self.loss_var.backward()

        # Snapshot input for replay
        self.kp.snapshot(ts)

        # Apply torque to PyBullet
        torque_scalar = float(torque.detach())
        p.setJointMotorControl2(
            self.robot_id, self.joint_idx,
            p.TORQUE_CONTROL, force=torque_scalar,
            physicsClientId=self.physics_client,
        )
        p.stepSimulation(physicsClientId=self.physics_client)

        # Publish for visualization
        self.q_pub.publish(Value(val=q_val, timestamp=ts))
        self.torque_pub.publish(Value(val=torque_scalar, timestamp=ts))

    def reset(self):
        p.resetJointState(
            self.robot_id, self.joint_idx,
            targetValue=0.785, targetVelocity=0.0,
            physicsClientId=self.physics_client,
        )
        self._state_history.clear()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="pd_bullet", description="PD Controller with PyBullet"
        )
        common.add_config_arguments(parser)
        parser.add_argument("--plot", action="store_true", help="Show live error/derror plot")
        args = parser.parse_args()
        conf = common.get_config_from_args(args)

        node = PDBulletNode(conf, plot=args.plot)

        if args.plot:
            # FuncAnimation drives redraws on the main thread; sim runs in background
            _ani = FuncAnimation(node._plot_fig, node._animate, interval=100, blit=False)
            plt.show()
        else:
            node.spin()
    except KeyboardInterrupt:
        print("Shutting down PD bullet node.")
        node.close()
