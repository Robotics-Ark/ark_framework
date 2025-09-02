import yaml
from functools import partial
from typing import Any, Dict, Optional

from ark.tools.log import log
from ark.client.comm_infrastructure.base_node import BaseNode
from arktypes import string_t  # keep for parity with your mapping tables

import rclpy
from rclpy.node import Node as RclpyNode
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


__doc__ = """ARK ⟷ ROS 2 translator/bridge"""


class ArkRos2Bridge(BaseNode):
    """
    Bridge Ark <-> ROS 2 using rclpy. Expects a mapping_table with:
      mapping_table = {
          "ros_to_ark": [
              {
                  "ros_channel": "/chatter",
                  "ros_type": std_msgs.msg.String,
                  "ark_channel": "ark/chatter",
                  "ark_type": string_t,
                  "translator_callback": callable,  # f(ros_msg, ros_ch, ros_type, ark_ch, ark_type) -> ark_msg
              },
              ...
          ],
          "ark_to_ros": [
              {
                  "ark_channel": "ark/cmd",
                  "ark_type": string_t,
                  "ros_channel": "/cmd",
                  "ros_type": std_msgs.msg.String,
                  "translator_callback": callable,  # f(t, ark_ch, ark_msg) -> ros_msg
              },
              ...
          ],
      }
    """

    def __init__(
        self,
        mapping_table: Dict[str, Any],
        node_name: str = "ark_ros2_bridge",
        global_config: Optional[Dict[str, Any]] = None,
        qos_profile: Optional[QoSProfile] = None,
    ):
        super().__init__(node_name, global_config=global_config)

        # ---- ROS 2 node setup ----
        if not rclpy.ok():
            rclpy.init(args=None)

        self._ros: RclpyNode = rclpy.create_node(node_name)

        # Default QoS (tuned for typical topic traffic; you can override via ctor)
        self._qos = qos_profile or QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Keep references so publishers/subscriptions don’t get GC’d
        self._ros_publishers = []
        self._ros_subscriptions = []

        # ---- Build mappings ----
        ros_to_ark_table = mapping_table.get("ros_to_ark", [])
        ark_to_ros_table = mapping_table.get("ark_to_ros", [])

        self.ros_to_ark_mapping = []
        for mapping in ros_to_ark_table:
            ros_channel = mapping["ros_channel"]
            ros_type = mapping["ros_type"]
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            translator_callback = mapping["translator_callback"]

            # ARK publisher
            ark_pub = self.create_publisher(ark_channel, ark_type)

            # Subscriber callback (ROS->ARK)
            sub_cb = partial(
                self._generic_ros_to_ark_translator_callback,
                translator_callback=translator_callback,
                ros_channel=ros_channel,
                ros_type=ros_type,
                ark_channel=ark_channel,
                ark_type=ark_type,
                ark_publisher=ark_pub,
            )

            # ROS 2 subscription
            sub = self._ros.create_subscription(ros_type, ros_channel, sub_cb, self._qos)
            self._ros_subscriptions.append(sub)

            self.ros_to_ark_mapping.append(
                {
                    "ros_channel": ros_channel,
                    "ros_type": ros_type,
                    "ark_channel": ark_channel,
                    "ark_type": ark_type,
                    "translator_callback": translator_callback,
                    "publisher": ark_pub,
                }
            )

        self.ark_to_ros_mapping = []
        for mapping in ark_to_ros_table:
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            ros_channel = mapping["ros_channel"]
            ros_type = mapping["ros_type"]
            translator_callback = mapping["translator_callback"]

            # ROS 2 publisher
            ros_pub = self._ros.create_publisher(ros_type, ros_channel, self._qos)
            self._ros_publishers.append(ros_pub)

            # ARK subscriber (ARK->ROS)
            ark_cb = partial(
                self._generic_ark_to_ros_translator_callback,
                translator_callback=translator_callback,
                ark_channel=ark_channel,
                ark_type=ark_type,
                ros_channel=ros_channel,
                ros_type=ros_type,
                ros_publisher=ros_pub,
            )
            self.create_subscriber(ark_channel, ark_type, ark_cb)

            self.ark_to_ros_mapping.append(
                {
                    "ros_channel": ros_channel,
                    "ros_type": ros_type,
                    "ark_channel": ark_channel,
                    "ark_type": ark_type,
                    "translator_callback": translator_callback,
                    "publisher": ros_pub,
                }
            )

    # ---------- Callbacks ----------

    def _generic_ros_to_ark_translator_callback(
        self,
        ros_msg: Any,
        *,
        translator_callback,
        ros_channel: str,
        ros_type: Any,
        ark_channel: str,
        ark_type: Any,
        ark_publisher,
    ) -> None:
        """
        Translate ROS 2 -> ARK.
        translator_callback: (ros_msg, ros_channel, ros_type, ark_channel, ark_type) -> ark_msg
        """
        try:
            ark_msg = translator_callback(ros_msg, ros_channel, ros_type, ark_channel, ark_type)
            ark_publisher.publish(ark_msg)
        except Exception as e:
            self._ros.get_logger().warn(
                f"[ROS2→ARK] Failed translating {ros_channel} -> {ark_channel}: {e}"
            )

    def _generic_ark_to_ros_translator_callback(
        self,
        t: int,
        _channel: str,
        ark_msg: Any,
        *,
        translator_callback,
        ark_channel: str,
        ark_type: Any,
        ros_channel: str,
        ros_type: Any,
        ros_publisher,
    ) -> None:
        """
        Translate ARK -> ROS 2.
        translator_callback: (t, ark_channel, ark_msg) -> ros_msg
        """
        try:
            ros_msg = translator_callback(t, ark_channel, ark_msg)
            ros_publisher.publish(ros_msg)
        except Exception as e:
            self._ros.get_logger().warn(
                f"[ARK→ROS2] Failed translating {ark_channel} -> {ros_channel}: {e}"
            )

    # ---------- Lifecycle ----------

    def spin(self) -> None:
        """
        Drive both Ark (LCM) and ROS 2 event loops without blocking either.
        """
        try:
            while not self._done and rclpy.ok():
                # Pump Ark
                try:
                    self._lcm.handle_timeout(0)
                except OSError as e:
                    log.warning(f"Ark threw OSError {e}")
                    break

                # Pump ROS 2 once (non-blocking)
                rclpy.spin_once(self._ros, timeout_sec=0.0)
        finally:
            self.shutdown()

    @staticmethod
    def get_cli_doc():
        return __doc__

    def shutdown(self) -> None:
        """
        Cleanly stop Ark and ROS 2 resources.
        """
        # Ark side
        for ch in self._comm_handlers:
            try:
                ch.shutdown()
            except Exception:
                pass
        for s in self._steppers:
            try:
                s.shutdown()
            except Exception:
                pass

        # ROS 2 side
        try:
            # Destroy pubs/subs explicitly (optional but tidy)
            for sub in self._ros_subscriptions:
                try:
                    self._ros.destroy_subscription(sub)
                except Exception:
                    pass
            for pub in self._ros_publishers:
                try:
                    self._ros.destroy_publisher(pub)
                except Exception:
                    pass
            try:
                self._ros.destroy_node()
            except Exception:
                pass
        finally:
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
