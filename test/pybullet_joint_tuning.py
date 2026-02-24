from ark.node import BaseNode
from ark_msgs import Value
import argparse
import common_example as common
import torch
import pybullet as p
import os

HZ = 240
DT = 1.0 / HZ


class PDBulletNode(BaseNode):

    def __init__(self, cfg):
        super().__init__("env", "pd_bullet", cfg, sim=True)

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

        # Target state
        self.q_target = 0.5236
        self.qd_target = 0.0

        # Input variable: only Kp (Kd derived from critical damping)
        self.kp = self.create_variable("kp", 0.0, mode="input")

        # Output variable
        self.loss_var = self.create_variable("loss", 0.0, mode="output")

        # Replay function for temporal gradient queries
        self.loss_var._replay_fn = self._replay_grad
        self._state_history = {}

        # Publishers for visualization
        self.q_pub = self.create_publisher("q")
        self.torque_pub = self.create_publisher("torque_val")
        self.loss_pub = self.create_publisher("loss_val")

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

        # Get Kp (differentiable)
        if replay:
            kp = self.kp.at(ts)
        else:
            kp = self.kp.tensor

        # Critical damping: Kd = 2 * sqrt(Kp * I)
        kd = 2.0 * torch.sqrt(kp * self.I)

        error = q_t - q
        derror = qd_t - qd
        torque = kp * error + kd * derror

        # Loss: tracking error + control effort penalty
        w_effort = 0.01
        loss = error ** 2 + w_effort * torque ** 2

        return torque, loss, q_val

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

    def step(self, ts):
        torque, loss, q_val = self.forward(ts)

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
        self.loss_pub.publish(Value(val=float(loss.detach()), timestamp=ts))

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
        args = parser.parse_args()
        conf = common.get_config_from_args(args)

        node = PDBulletNode(conf)
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down PD bullet node.")
        node.close()
