# default_translators.py
from typing import Any, Callable, Sequence
from dataclasses import is_dataclass

# ROS2 messages
from std_msgs.msg import String, Int16, Int64, Float32, Int64MultiArray, Float32MultiArray, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import Point, Quaternion as RosQuaternion, Pose, Pose2D, Twist, Vector3
from sensor_msgs.msg import JointState


# Helpers
def _construct_ark(ark_type: Any) -> Any:
    """Instantiate an Ark (LCM) message type (callable or alias)."""
    try:
        return ark_type() if callable(ark_type) else ark_type
    except Exception:
        return ark_type()

def _mk_layout_1d(n: int) -> MultiArrayLayout:
    return MultiArrayLayout(dim=[MultiArrayDimension(label="data", size=n, stride=n)], data_offset=0)

def _mk_layout_2d(m: int, n: int) -> MultiArrayLayout:
    return MultiArrayLayout(
        dim=[
            MultiArrayDimension(label="rows", size=m, stride=m * n),
            MultiArrayDimension(label="cols", size=n, stride=n),
        ],
        data_offset=0,
    )

def _infer_2d_from_layout(layout: MultiArrayLayout, data_len: int) -> tuple[int, int] | None:
    if layout and layout.dim and len(layout.dim) >= 2:
        m = layout.dim[0].size
        n = layout.dim[1].size
        if m * n == data_len and m > 0 and n > 0:
            return m, n
    return None


# ROS2 -> Ark translator generator
def make_ros2_to_ark_translator(ark_type_expected: Any) -> Callable[[Any, str, Any, str, Any], Any]:
    """
    Returns a function with signature:
        (ros2_msg, ros2_channel, ros2_type, ark_channel, ark_type) -> ark_msg
    which converts common ROS2 messages to the given Ark type.
    """

    def _to_vec(arr: Sequence[float]) -> tuple[int, list[float]]:
        return len(arr), list(arr)

    def translator(ros2_msg: Any, ros2_channel: str, ros2_type: Any, ark_channel: str, ark_type: Any) -> Any:
        # Prefer the expected type passed at factory time, but fall back to runtime ark_type
        target_type = ark_type_expected or ark_type
        name = getattr(target_type, "__name__", str(target_type))

        # General Purpose
        if name == "flag_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Int16):
                msg.flag = int(ros2_msg.data)
            else:
                msg.flag = int(ros2_msg)  # last resort
            return msg

        if name == "int_64_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Int64):
                msg.data = int(ros2_msg.data)
            else:
                msg.data = int(ros2_msg)
            return msg

        if name == "float_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Float32):
                msg.data = float(ros2_msg.data)
            else:
                msg.data = float(ros2_msg)
            return msg

        if name == "string_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, String):
                msg.data = ros2_msg.data
            else:
                msg.data = str(ros2_msg)
            return msg

        if name == "float_vector_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Float32MultiArray):
                n, data = _to_vec(ros2_msg.data)
            else:
                n, data = _to_vec(ros2_msg)  # assume sequence
            msg.n = n
            msg.data = data
            return msg

        if name == "double_vector_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Float64MultiArray):
                n, data = _to_vec(ros2_msg.data)
            else:
                n, data = _to_vec(ros2_msg)
            msg.n = n
            msg.data = [float(x) for x in data]
            return msg

        if name == "int64_vector_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Int64MultiArray):
                n, data = _to_vec(ros2_msg.data)
            else:
                n, data = _to_vec(ros2_msg)
            msg.n = n
            msg.data = [int(x) for x in data]
            return msg

        if name == "float_array_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Float32MultiArray):
                data = list(ros2_msg.data)
                shape = _infer_2d_from_layout(ros2_msg.layout, len(data))
                if shape is None:
                    # Fallback: 1 x N
                    msg.m, msg.n = 1, len(data)
                    msg.data = [data]  # [1][N]
                else:
                    m, n = shape
                    msg.m, msg.n = m, n
                    # reshape row-major
                    msg.data = [data[i * n : (i + 1) * n] for i in range(m)]
            else:
                # Assume list of lists or flat list
                if ros2_msg and isinstance(ros2_msg[0], (list, tuple)):
                    m = len(ros2_msg)
                    n = len(ros2_msg[0])
                    msg.m, msg.n = m, n
                    msg.data = [list(row) for row in ros2_msg]
                else:
                    msg.m, msg.n = 1, len(ros2_msg)
                    msg.data = [list(ros2_msg)]
            return msg

        if name == "double_array_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Float64MultiArray):
                data = list(ros2_msg.data)
                shape = _infer_2d_from_layout(ros2_msg.layout, len(data))
                if shape is None:
                    msg.m, msg.n = 1, len(data)
                    msg.data = [data]
                else:
                    m, n = shape
                    msg.m, msg.n = m, n
                    msg.data = [data[i * n : (i + 1) * n] for i in range(m)]
            else:
                if ros2_msg and isinstance(ros2_msg[0], (list, tuple)):
                    m = len(ros2_msg)
                    n = len(ros2_msg[0])
                    msg.m, msg.n = m, n
                    msg.data = [list(map(float, row)) for row in ros2_msg]
                else:
                    msg.m, msg.n = 1, len(ros2_msg)
                    msg.data = [list(map(float, ros2_msg))]
            return msg

        # Robotics
        if name == "position_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Point):
                msg.x, msg.y, msg.z = float(ros2_msg.x), float(ros2_msg.y), float(ros2_msg.z)
            else:
                msg.x, msg.y, msg.z = map(float, ros2_msg)  # sequence [x,y,z]
            return msg

        if name == "quaternion_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, RosQuaternion):
                msg.x, msg.y, msg.z, msg.w = float(ros2_msg.x), float(ros2_msg.y), float(ros2_msg.z), float(ros2_msg.w)
            else:
                msg.x, msg.y, msg.z, msg.w = map(float, ros2_msg)  # sequence [x,y,z,w]
            return msg

        if name == "pose_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Pose):
                msg.position = [float(ros2_msg.position.x), float(ros2_msg.position.y), float(ros2_msg.position.z)]
                msg.orientation = [float(ros2_msg.orientation.x), float(ros2_msg.orientation.y),
                                   float(ros2_msg.orientation.z), float(ros2_msg.orientation.w)]
            else:
                # expect dict or tuple ((x,y,z),(qx,qy,qz,qw))
                pos, ori = ros2_msg
                msg.position = list(map(float, pos))
                msg.orientation = list(map(float, ori))
            return msg

        if name == "pose_2d_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Pose2D):
                msg.x, msg.y, msg.theta = float(ros2_msg.x), float(ros2_msg.y), float(ros2_msg.theta)
            else:
                msg.x, msg.y, msg.theta = map(float, ros2_msg)
            return msg

        if name == "velocity_2d_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Twist):
                msg.v_x = float(ros2_msg.linear.x)
                msg.v_y = float(ros2_msg.linear.y)
                msg.w   = float(ros2_msg.angular.z)
            else:
                msg.v_x, msg.v_y, msg.w = map(float, ros2_msg)
            return msg

        if name == "wheeled_velocity_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Twist):
                msg.linear  = float(ros2_msg.linear.x)
                msg.angular = float(ros2_msg.angular.z)
            else:
                msg.linear, msg.angular = map(float, ros2_msg)
            return msg

        if name == "twist_t":
            msg = _construct_ark(target_type)
            if isinstance(ros2_msg, Twist):
                msg.linear_velocity  = [float(ros2_msg.linear.x), float(ros2_msg.linear.y), float(ros2_msg.linear.z)]
                msg.angular_velocity = [float(ros2_msg.angular.x), float(ros2_msg.angular.y), float(ros2_msg.angular.z)]
            else:
                lin, ang = ros2_msg
                msg.linear_velocity  = list(map(float, lin))
                msg.angular_velocity = list(map(float, ang))
            return msg

        if name == "robot_init_t":
            # Expect a custom ROS2 message or a tuple: (name:str, pose:Pose, q_init:Sequence[float])
            msg = _construct_ark(target_type)
            if hasattr(ros2_msg, "name") and hasattr(ros2_msg, "pose") and hasattr(ros2_msg, "q_init"):
                msg.name = str(ros2_msg.name)
                p = ros2_msg.pose
                msg.position = [float(p.position.x), float(p.position.y), float(p.position.z)]
                msg.orientation = [float(p.orientation.x), float(p.orientation.y), float(p.orientation.z), float(p.orientation.w)]
                msg.n = len(ros2_msg.q_init)
                msg.q_init = [float(x) for x in ros2_msg.q_init]
            else:
                name, pose_msg, q_init = ros2_msg  # tuple fallback
                msg.name = str(name)
                msg.position = [float(pose_msg.position.x), float(pose_msg.position.y), float(pose_msg.position.z)]
                msg.orientation = [float(pose_msg.orientation.x), float(pose_msg.orientation.y),
                                   float(pose_msg.orientation.z), float(pose_msg.orientation.w)]
                msg.n = len(q_init)
                msg.q_init = [float(x) for x in q_init]
            return msg

        if name == "rigid_body_state_t":
            msg = _construct_ark(target_type)
            # Expect a custom message or a (name:str, pose:Pose, lin_vel:Vector3, ang_vel:Vector3)
            if hasattr(ros2_msg, "name") and hasattr(ros2_msg, "pose") and hasattr(ros2_msg, "lin_velocity") and hasattr(ros2_msg, "ang_velocity"):
                msg.name = str(ros2_msg.name)
                pose = ros2_msg.pose
                msg.position = [pose.position.x, pose.position.y, pose.position.z]
                msg.orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                lv = ros2_msg.lin_velocity
                av = ros2_msg.ang_velocity
                msg.lin_velocity = [lv.x, lv.y, lv.z]
                msg.ang_velocity = [av.x, av.y, av.z]
            else:
                name, pose, lin_v, ang_v = ros2_msg
                msg.name = str(name)
                msg.position = [pose.position.x, pose.position.y, pose.position.z]
                msg.orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                msg.lin_velocity = [lin_v.x, lin_v.y, lin_v.z]
                msg.ang_velocity = [ang_v.x, ang_v.y, ang_v.z]
            return msg

        # If we got here, we don't know how to translate this type
        raise TypeError(f"No default ROS2→Ark translator for Ark type '{name}'")

    return translator

# Ark -> ROS2 factory
def make_ark_to_ros2_translator(ros2_type_expected: Any) -> Callable[[int, str, Any], Any]:
    """
    Returns a function with signature:
        (t, ark_channel, ark_msg) -> ros2_msg
    which converts common Ark messages to the given ROS2 message type.
    """

    def translator(t: int, ark_channel: str, ark_msg: Any) -> Any:
        rt = ros2_type_expected

        # General Purpose
        if rt is Int16:
            out = Int16()
            out.data = int(getattr(ark_msg, "flag", 0) if hasattr(ark_msg, "flag") else getattr(ark_msg, "data", 0))
            return out

        if rt is Int64:
            out = Int64()
            out.data = int(getattr(ark_msg, "data", 0))
            return out

        if rt is Float32:
            out = Float32()
            out.data = float(getattr(ark_msg, "data", 0.0))
            return out

        if rt is String:
            out = String()
            out.data = str(getattr(ark_msg, "data", ""))
            return out

        if rt is Float32MultiArray:
            out = Float32MultiArray()
            # vector or 2D array -> flatten
            if hasattr(ark_msg, "n") and hasattr(ark_msg, "data") and isinstance(ark_msg.data, (list, tuple)) and (not hasattr(ark_msg, "m")):
                out.data = [float(x) for x in ark_msg.data]
                out.layout = _mk_layout_1d(len(out.data))
            elif hasattr(ark_msg, "m") and hasattr(ark_msg, "n"):
                m, n = int(ark_msg.m), int(ark_msg.n)
                flat = []
                for i in range(m):
                    flat.extend(list(map(float, ark_msg.data[i])))
                out.data = flat
                out.layout = _mk_layout_2d(m, n)
            else:
                out.data = list(map(float, getattr(ark_msg, "data", [])))
                out.layout = _mk_layout_1d(len(out.data))
            return out

        if rt is Float64MultiArray:
            out = Float64MultiArray()
            if hasattr(ark_msg, "n") and hasattr(ark_msg, "data") and (not hasattr(ark_msg, "m")):
                out.data = [float(x) for x in ark_msg.data]
                out.layout = _mk_layout_1d(len(out.data))
            elif hasattr(ark_msg, "m") and hasattr(ark_msg, "n"):
                m, n = int(ark_msg.m), int(ark_msg.n)
                flat = []
                for i in range(m):
                    flat.extend(list(map(float, ark_msg.data[i])))
                out.data = flat
                out.layout = _mk_layout_2d(m, n)
            else:
                out.data = list(map(float, getattr(ark_msg, "data", [])))
                out.layout = _mk_layout_1d(len(out.data))
            return out

        if rt is Int64MultiArray:
            out = Int64MultiArray()
            if hasattr(ark_msg, "n") and hasattr(ark_msg, "data") and (not hasattr(ark_msg, "m")):
                out.data = [int(x) for x in ark_msg.data]
                out.layout = _mk_layout_1d(len(out.data))
            else:
                out.data = list(map(int, getattr(ark_msg, "data", [])))
                out.layout = _mk_layout_1d(len(out.data))
            return out

        # Robotics
        if rt is Point:
            out = Point()
            out.x = float(getattr(ark_msg, "x", 0.0))
            out.y = float(getattr(ark_msg, "y", 0.0))
            out.z = float(getattr(ark_msg, "z", 0.0))
            return out

        if rt is RosQuaternion:
            out = RosQuaternion()
            out.x, out.y, out.z, out.w = map(float, getattr(ark_msg, "orientation", [
                getattr(ark_msg, "x", 0.0), getattr(ark_msg, "y", 0.0),
                getattr(ark_msg, "z", 0.0), getattr(ark_msg, "w", 1.0),
            ]))
            return out

        if rt is Pose:
            out = Pose()
            if hasattr(ark_msg, "position") and hasattr(ark_msg, "orientation"):
                px, py, pz = ark_msg.position
                qx, qy, qz, qw = ark_msg.orientation
            else:
                # position_t + quaternion_t shaped
                px, py, pz = getattr(ark_msg, "x", 0.0), getattr(ark_msg, "y", 0.0), getattr(ark_msg, "z", 0.0)
                qx, qy, qz, qw = getattr(ark_msg, "qx", 0.0), getattr(ark_msg, "qy", 0.0), getattr(ark_msg, "qz", 0.0), getattr(ark_msg, "qw", 1.0)
            out.position = Point(x=float(px), y=float(py), z=float(pz))
            out.orientation = RosQuaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
            return out

        if rt is Pose2D:
            out = Pose2D()
            out.x = float(getattr(ark_msg, "x", 0.0))
            out.y = float(getattr(ark_msg, "y", 0.0))
            out.theta = float(getattr(ark_msg, "theta", 0.0))
            return out

        if rt is Twist:
            out = Twist()
            # Map from velocity_2d_t, wheeled_velocity_t, or twist_t
            if hasattr(ark_msg, "linear_velocity") and hasattr(ark_msg, "angular_velocity"):
                lx, ly, lz = (ark_msg.linear_velocity + [0, 0, 0])[:3]
                ax, ay, az = (ark_msg.angular_velocity + [0, 0, 0])[:3]
                out.linear = Vector3(x=float(lx), y=float(ly), z=float(lz))
                out.angular = Vector3(x=float(ax), y=float(ay), z=float(az))
            elif hasattr(ark_msg, "v_x") and hasattr(ark_msg, "v_y") and hasattr(ark_msg, "w"):
                out.linear = Vector3(x=float(ark_msg.v_x), y=float(ark_msg.v_y), z=0.0)
                out.angular = Vector3(x=0.0, y=0.0, z=float(ark_msg.w))
            elif hasattr(ark_msg, "linear") and hasattr(ark_msg, "angular"):
                out.linear = Vector3(x=float(ark_msg.linear), y=0.0, z=0.0)
                out.angular = Vector3(x=0.0, y=0.0, z=float(ark_msg.angular))
            else:
                # default zeros
                out.linear = Vector3()
                out.angular = Vector3()
            return out


        raise TypeError(f"No default Ark→ROS2 translator for ROS2 type '{rt}'")

    return translator
