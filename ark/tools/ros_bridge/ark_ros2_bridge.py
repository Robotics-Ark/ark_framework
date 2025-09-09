from functools import partial
from typing import Any, Dict, Optional

from ark.tools.log import log
from ark.client.comm_infrastructure.base_node import BaseNode

import rclpy
from rclpy.node import Node as RclpyNode
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


__doc__ = """ARK ⟷ ROS 2 translator/bridge"""


class ArkRos2Bridge(BaseNode):
    """
    Bridge Ark ⟷ ROS2 using `rclpy`.

    The bridge is bidirectional and driven by a user-supplied `mapping_table`
    that declares which topics/channels to connect and how to translate messages.

    Mapping table schema:
        mapping_table = {
            "ros2_to_ark": [
                {
                    "ros2_channel": "/chatter",                 # str: ROS2 topic name
                    "ros2_type": std_msgs.msg.String,           # type: ROS2 msg class
                    "ark_channel": "ark/chatter",              # str: Ark channel name
                    "ark_type": string_t,                      # type: Ark message type (class/struct)
                    "translator_callback": callable,           # (ros2_msg, ros2_channel, ros2_type, ark_channel, ark_type) -> ark_msg
                },
                ...
            ],
            "ark_to_ros2": [
                {
                    "ark_channel": "ark/cmd",                  # str: Ark channel name
                    "ark_type": string_t,                      # type: Ark message type (class/struct)
                    "ros2_channel": "/cmd",                    # str: ROS2 topic name
                    "ros2_type": std_msgs.msg.String,          # type: ROS2 msg class
                    "translator_callback": callable,           # (t, ark_channel, ark_msg) -> ros2_msg
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
        """
        Initialize the bridge and wire up all declared mappings.

        mapping_table See class docs for full schema. Missing keys default to empty lists.
        node_name Name for both the Ark BaseNode and the underlying rclpy node.
        global_config Optional Ark node configuration passed to BaseNode.
        qos_profile Optional ROS2 QoS profile. If omitted, uses RELIABLE/KEEP_LAST(10)/VOLATILE.
        """
        super().__init__(node_name, global_config=global_config)

        # ---- ROS2 node setup ----
        if not rclpy.ok():
            rclpy.init(args=None)

        self._ros2_node: RclpyNode = rclpy.create_node(node_name)

        # Default QoS
        self._qos = qos_profile or QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Keep references so publishers/subscriptions don’t get GC’d
        self._ros2_publishers = []
        self._ros2_subscriptions = []

        # ---- Build mappings ----
        ros2_to_ark_table = mapping_table.get("ros2_to_ark", [])
        ark_to_ros2_table = mapping_table.get("ark_to_ros2", [])

        self.ros2_to_ark_mapping = []
        for mapping in ros2_to_ark_table:
            ros2_channel = mapping["ros2_channel"]
            ros2_type = mapping["ros2_type"]
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            translator_callback = mapping["translator_callback"]

            # ARK publisher
            ark_pub = self.create_publisher(ark_channel, ark_type)

            # Subscriber callback (ROS2->ARK)
            sub_cb = partial(
                self._generic_ros2_to_ark_translator_callback,
                translator_callback=translator_callback,
                ros2_channel=ros2_channel,
                ros2_type=ros2_type,
                ark_channel=ark_channel,
                ark_type=ark_type,
                ark_publisher=ark_pub,
            )

            # ROS2 subscription
            sub = self._ros2_node.create_subscription(ros2_type, ros2_channel, sub_cb, self._qos)
            self._ros2_subscriptions.append(sub)

            self.ros2_to_ark_mapping.append(
                {
                    "ros2_channel": ros2_channel,
                    "ros2_type": ros2_type,
                    "ark_channel": ark_channel,
                    "ark_type": ark_type,
                    "translator_callback": translator_callback,
                    "publisher": ark_pub,
                }
            )

        self.ark_to_ros2_mapping = []
        for mapping in ark_to_ros2_table:
            ark_channel = mapping["ark_channel"]
            ark_type = mapping["ark_type"]
            ros2_channel = mapping["ros2_channel"]
            ros2_type = mapping["ros2_type"]
            translator_callback = mapping["translator_callback"]

            # ROS2 publisher
            ros2_pub = self._ros2_node.create_publisher(ros2_type, ros2_channel, self._qos)
            self._ros2_publishers.append(ros2_pub)

            # ARK subscriber (ARK->ROS2)
            ark_cb = partial(
                self._generic_ark_to_ros2_translator_callback,
                translator_callback=translator_callback,
                ark_channel=ark_channel,
                ark_type=ark_type,
                ros2_channel=ros2_channel,
                ros2_type=ros2_type,
                ros2_publisher=ros2_pub,
            )
            self.create_subscriber(ark_channel, ark_type, ark_cb)

            self.ark_to_ros2_mapping.append(
                {
                    "ros2_channel": ros2_channel,
                    "ros2_type": ros2_type,
                    "ark_channel": ark_channel,
                    "ark_type": ark_type,
                    "translator_callback": translator_callback,
                    "publisher": ros2_pub,
                }
            )

    # ---------- Callbacks ----------

    def _generic_ros2_to_ark_translator_callback(
        self,
        ros2_msg: Any,
        *,
        translator_callback,
        ros2_channel: str,
        ros2_type: Any,
        ark_channel: str,
        ark_type: Any,
        ark_publisher,
    ) -> None:
        """
        Translate ROS2 -> ARK.
        translator_callback: (ros2_msg, ros2_channel, ros2_type, ark_channel, ark_type) -> ark_msg
        """
        try:
            ark_msg = translator_callback(ros2_msg, ros2_channel, ros2_type, ark_channel, ark_type)
            ark_publisher.publish(ark_msg)
        except Exception as e:
            self._ros2_node.get_logger().warn(
                f"[ROS2→ARK] Failed translating {ros2_channel} -> {ark_channel}: {e}"
            )

    def _generic_ark_to_ros2_translator_callback(
        self,
        t: int,
        _channel: str,
        ark_msg: Any,
        *,
        translator_callback,
        ark_channel: str,
        ark_type: Any,
        ros2_channel: str,
        ros2_type: Any,
        ros2_publisher,
    ) -> None:
        """
        Translate ARK -> ROS2.
        translator_callback: (t, ark_channel, ark_msg) -> ros2_msg
        """
        try:
            ros2_msg = translator_callback(t, ark_channel, ark_msg)
            ros2_publisher.publish(ros2_msg)
        except Exception as e:
            self._ros2_node.get_logger().warn(
                f"[ARK→ROS2] Failed translating {ark_channel} -> {ros2_channel}: {e}"
            )

    # ---------- Lifecycle ----------

    def spin(self) -> None:
        """
        Drive both Ark (LCM) and ROS2 event loops without blocking either.
        """
        try:
            while not self._done and rclpy.ok():
                # Pump Ark
                try:
                    self._lcm.handle_timeout(0)
                except OSError as e:
                    log.warning(f"Ark threw OSError {e}")
                    break

                # Pump ROS2 once (non-blocking)
                rclpy.spin_once(self._ros2_node, timeout_sec=0.0)
        finally:
            self.shutdown()

    @staticmethod
    def get_cli_doc():
        return __doc__

    def shutdown(self) -> None:
        """
        Cleanly stop Ark and ROS2 resources.
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

        # ROS2 side
        try:
            # Destroy pubs/subs explicitly (optional but tidy)
            for sub in self._ros2_subscriptions:
                try:
                    self._ros2_node.destroy_subscription(sub)
                except Exception:
                    pass
            for pub in self._ros2_publishers:
                try:
                    self._ros2_node.destroy_publisher(pub)
                except Exception:
                    pass
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass
        finally:
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
