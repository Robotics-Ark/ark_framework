from ark.frames import FrameForest
from scipy.spatial.transform import RigidTransform, Rotation


def tf_xyz_rpy(
    x: float,
    y: float,
    z: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> RigidTransform:
    return RigidTransform.from_components(
        [x, y, z],
        Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True),
    )


def build_forest() -> FrameForest:
    forest = FrameForest()

    # Main robot tree rooted at world.
    forest.register_static_transform("world", "base", tf_xyz_rpy(0.0, 0.0, 0.6))
    forest.register_static_transform("base", "camera", tf_xyz_rpy(0.0, 0.0, 0.15))
    forest.register_static_transform(
        "base", "lidar", tf_xyz_rpy(0.18, 0.0, 0.28, 0.0, 0.0, 180.0)
    )
    forest.register_dynamic_transform(
        "base", "shoulder", tf_xyz_rpy(0.0, 0.0, 0.35, 0.0, 0.0, 10.0)
    )
    forest.register_static_transform(
        "shoulder", "shoulder_camera", tf_xyz_rpy(0.08, 0.12, 0.05, -20.0, 0.0, 35.0)
    )
    forest.register_dynamic_transform(
        "shoulder", "elbow", tf_xyz_rpy(0.32, 0.0, 0.05, 0.0, 15.0, 0.0)
    )
    forest.register_dynamic_transform(
        "elbow", "wrist", tf_xyz_rpy(0.28, 0.0, -0.02, 0.0, -25.0, 0.0)
    )
    forest.register_static_transform("wrist", "ft_sensor", tf_xyz_rpy(0.0, 0.0, 0.06))
    forest.register_static_transform(
        "wrist", "tool", tf_xyz_rpy(0.0, 0.0, 0.12, 0.0, 90.0, 0.0)
    )
    forest.register_static_transform(
        "tool", "imu", tf_xyz_rpy(0.06, 0.0, 0.04, -90.0, 0.0, 90.0)
    )
    forest.register_dynamic_transform(
        "tool", "left_finger", tf_xyz_rpy(0.04, 0.03, 0.0)
    )
    forest.register_dynamic_transform(
        "tool", "right_finger", tf_xyz_rpy(0.04, -0.03, 0.0)
    )

    # Separate map tree to show a disconnected root.
    forest.register_static_transform(
        "map", "marker_0", tf_xyz_rpy(1.2, 0.5, 0.0, 0.0, 0.0, 15.0)
    )
    forest.register_static_transform(
        "marker_0", "inspection_target", tf_xyz_rpy(0.15, 0.0, 0.2)
    )
    forest.register_static_transform(
        "map", "marker_1", tf_xyz_rpy(-0.8, -0.4, 0.0, 0.0, 0.0, -30.0)
    )
    forest.register_static_transform(
        "marker_1", "drop_zone", tf_xyz_rpy(0.25, 0.1, 0.0)
    )

    # Another fully disconnected tree.
    forest.register_static_transform(
        "calibration_rig", "checkerboard", tf_xyz_rpy(0.0, 0.0, 1.0, 0.0, 0.0, 90.0)
    )
    forest.register_static_transform(
        "calibration_rig",
        "reference_camera",
        tf_xyz_rpy(0.4, -0.2, 1.3, -25.0, 0.0, 120.0),
    )
    forest.register_static_transform(
        "reference_camera", "lens_tip", tf_xyz_rpy(0.0, 0.0, 0.08)
    )

    return forest


def main() -> None:
    forest = build_forest()
    forest.to_image("frame_example.pdf")
    print("Rendered image: frame_example.pdf")


if __name__ == "__main__":
    main()
