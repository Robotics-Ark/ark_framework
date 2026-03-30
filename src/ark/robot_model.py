from pathlib import Path


class RobotModel:

    @property
    def urdf_path(self) -> Path | None:
        """The path to the robot's URDF file."""
        return None  # overwrite this method in subclasses if URDF is supported

    def is_urdf_supported(self) -> bool:
        """Whether the robot supports URDF format."""
        return isinstance(self.urdf_path, Path)

    @property
    def mjcf_path(self) -> Path | None:
        """The path to the robot's MJCF file."""
        return None  # overwrite this method in subclasses if MJCF is supported

    def is_mjcf_supported(self) -> bool:
        """Whether the robot supports MJCF format."""
        return isinstance(self.mjcf_path, Path)
