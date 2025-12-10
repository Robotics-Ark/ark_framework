"""Newton camera driver with RaycastSensor for depth and ViewerGL for RGB."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

from ark.tools.log import log
from ark.system.driver.sensor_driver import CameraDriver

try:
    from newton.sensors import RaycastSensor
    RAYCAST_AVAILABLE = True
except ImportError:
    log.warning("Newton RaycastSensor not available - depth images will be placeholders")
    RAYCAST_AVAILABLE = False


class CameraType(Enum):
    """Supported camera models."""

    FIXED = "fixed"
    ATTACHED = "attached"


class NewtonCameraDriver(CameraDriver):
    """Camera driver using Newton RaycastSensor for depth and ViewerGL for RGB.

    Supports two camera modes:
    - FIXED: Static camera with position defined by yaw/pitch/roll around a target
    - ATTACHED: Camera attached to a robot link with offset transform

    Note: Segmentation is not supported by Newton - returns zeros with warning.
    """

    def __init__(
        self,
        component_name: str,
        component_config: Dict[str, Any],
        attached_body_id: Optional[int] = None,
    ) -> None:
        super().__init__(component_name, component_config, True)

        sim_cfg = component_config.get("sim_config", {})

        # Image dimensions
        self.width = int(component_config.get("width", sim_cfg.get("width", 640)))
        self.height = int(component_config.get("height", sim_cfg.get("height", 480)))

        # Camera intrinsics
        self.fov = float(sim_cfg.get("fov", 60.0))
        self.near = float(sim_cfg.get("near", sim_cfg.get("near_val", 0.1)))
        self.far = float(sim_cfg.get("far", sim_cfg.get("far_val", 100.0)))
        self.fov_radians = math.radians(self.fov)

        # Determine camera type
        try:
            camera_type_str = component_config.get("camera_type", "fixed")
            self.camera_type = CameraType(camera_type_str.lower())
        except ValueError:
            log.warning(f"Newton camera '{component_name}': Unknown camera_type '{camera_type_str}', defaulting to FIXED")
            self.camera_type = CameraType.FIXED

        # Stream configuration
        streams = component_config.get("streams", {})
        self.color_stream = streams.get("color", {}).get("enable", True)
        self.depth_stream = streams.get("depth", {}).get("enable", True)
        self.segmentation_stream = streams.get("segmentation", {}).get("enable", False)
        self._segmentation_warned = False

        # Fixed camera parameters
        if self.camera_type == CameraType.FIXED:
            fix_cfg = sim_cfg.get("fix", {})
            self.camera_target_position = fix_cfg.get("camera_target_position", [0, 0, 0])
            self.distance = fix_cfg.get("distance", 1.5)
            self.yaw = fix_cfg.get("yaw", 0.0)
            self.pitch = fix_cfg.get("pitch", -30.0)
            self.roll = fix_cfg.get("roll", 0.0)
            self.up_axis_index = fix_cfg.get("up_axis_index", 2)  # Z-up default

            # Compute initial camera pose from yaw/pitch/roll
            self._compute_fixed_camera_pose()

        # Attached camera parameters
        else:
            attach_cfg = sim_cfg.get("attach", {})
            self._parent_name = attach_cfg.get("parent_name")
            self._parent_link = attach_cfg.get("parent_link")
            self._offset_position = np.array(attach_cfg.get("position", attach_cfg.get("offset_translation", [0.0, 0.0, 0.0])), dtype=np.float32)
            offset_rot = attach_cfg.get("orientation", attach_cfg.get("offset_rotation", [0.0, 0.0, 0.0, 1.0]))
            if len(offset_rot) == 3:
                # Euler angles (degrees)
                self._offset_rotation = R.from_euler('xyz', offset_rot, degrees=True)
            else:
                # Quaternion (xyzw)
                self._offset_rotation = R.from_quat(offset_rot)
            self._rel_camera_target = attach_cfg.get("rel_camera_target", [1, 0, 0])

        # Runtime bindings (set in bind_runtime)
        self._model = None
        self._state_accessor: Callable[[], Any] = lambda: None
        self._viewer_manager = None
        self._raycast_sensor: Optional[Any] = None
        self._body_index: Optional[int] = attached_body_id

        # Current camera pose (updated for ATTACHED mode)
        self.current_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.current_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        log.info(
            f"Newton camera driver '{component_name}': Initialized "
            f"(size={self.width}x{self.height}, type={self.camera_type.value})"
        )

    def _compute_fixed_camera_pose(self) -> None:
        """Compute camera position and orientation from yaw/pitch/roll parameters."""
        # Convert angles to radians
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        # Compute camera position using spherical coordinates
        # Camera looks at target from distance, with yaw rotating around vertical axis
        # and pitch tilting up/down
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Position offset from target (spherical coords)
        offset_x = self.distance * cos_pitch * cos_yaw
        offset_y = self.distance * cos_pitch * sin_yaw
        offset_z = self.distance * sin_pitch

        target = np.array(self.camera_target_position, dtype=np.float32)
        self.current_position = target + np.array([offset_x, offset_y, offset_z], dtype=np.float32)

        # Direction is from camera to target (normalized)
        self.current_direction = target - self.current_position
        norm = np.linalg.norm(self.current_direction)
        if norm > 1e-6:
            self.current_direction = self.current_direction / norm
        else:
            self.current_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Up vector based on up_axis_index
        self.current_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Default Z-up

    def bind_runtime(
        self,
        model: Any,
        state_accessor: Callable[[], Any],
        viewer_manager: Optional[Any] = None,
    ) -> None:
        """Bind to Newton model after finalization.

        Args:
            model: Finalized newton.Model
            state_accessor: Callable returning current newton.State
            viewer_manager: Optional NewtonViewerManager for RGB capture
        """
        self._model = model
        self._state_accessor = state_accessor
        self._viewer_manager = viewer_manager

        # Resolve parent body index for ATTACHED mode
        if self.camera_type == CameraType.ATTACHED and self._body_index is None:
            if self._parent_name and hasattr(model, 'body_key'):
                key_to_index = {name: idx for idx, name in enumerate(model.body_key)}
                # Try parent_link first, then parent_name
                search_key = self._parent_link or self._parent_name
                self._body_index = key_to_index.get(search_key)

                if self._body_index is None:
                    log.warning(
                        f"Newton camera driver '{self.component_name}': "
                        f"Parent body '{search_key}' not found in model. "
                        f"Camera will not track body motion."
                    )

        # Create RaycastSensor for depth
        if RAYCAST_AVAILABLE and self._model is not None:
            try:
                self._raycast_sensor = RaycastSensor(
                    model=self._model,
                    camera_position=tuple(self.current_position),
                    camera_direction=tuple(self.current_direction),
                    camera_up=tuple(self.current_up),
                    fov_radians=self.fov_radians,
                    width=self.width,
                    height=self.height,
                    max_distance=self.far,
                )
                log.ok(f"Newton camera driver '{self.component_name}': RaycastSensor created")
            except Exception as e:
                log.warning(f"Newton camera driver '{self.component_name}': Failed to create RaycastSensor: {e}")
                self._raycast_sensor = None

    def _update_camera_pose(self) -> None:
        """Update camera pose for ATTACHED mode from body state."""
        if self.camera_type != CameraType.ATTACHED or self._body_index is None:
            return

        state = self._state_accessor()
        if state is None or state.body_q is None:
            return

        # Get body transform from state
        body_q = state.body_q.numpy()
        if self._body_index >= len(body_q):
            return

        # body_q is array of transforms, each is [px, py, pz, qx, qy, qz, qw]
        body_transform = body_q[self._body_index]
        body_pos = body_transform[:3]
        body_quat = body_transform[3:7]  # xyzw format

        # Apply offset transform
        body_rot = R.from_quat(body_quat)
        offset_pos_world = body_rot.apply(self._offset_position)
        self.current_position = (body_pos + offset_pos_world).astype(np.float32)

        # Compute camera direction (forward in local frame transformed to world)
        combined_rot = body_rot * self._offset_rotation
        local_forward = np.array(self._rel_camera_target, dtype=np.float32)
        self.current_direction = combined_rot.apply(local_forward).astype(np.float32)
        norm = np.linalg.norm(self.current_direction)
        if norm > 1e-6:
            self.current_direction = self.current_direction / norm

        # Up vector
        local_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.current_up = combined_rot.apply(local_up).astype(np.float32)

        # Update RaycastSensor pose
        if self._raycast_sensor is not None:
            self._raycast_sensor.update_camera_pose(
                position=tuple(self.current_position),
                direction=tuple(self.current_direction),
                up=tuple(self.current_up),
            )

    def get_images(self) -> Dict[str, np.ndarray]:
        """Get camera images.

        Returns dict with optional keys:
        - "color": RGB image (H, W, 3) uint8
        - "depth": Depth image (H, W) float32 in meters
        - "segmentation": Zeros (H, W) int32 (not supported by Newton)
        """
        images = {}

        # Update pose for attached cameras
        if self.camera_type == CameraType.ATTACHED:
            self._update_camera_pose()

        # Get depth from RaycastSensor
        if self.depth_stream:
            depth = self._get_depth_image()
            images["depth"] = depth

        # Get RGB from ViewerGL (if available)
        if self.color_stream:
            color = self._get_color_image()
            images["color"] = color

        # Segmentation not supported - return zeros with warning
        if self.segmentation_stream:
            if not self._segmentation_warned:
                log.warning(
                    f"Newton camera '{self.component_name}': Segmentation masks are not "
                    f"supported by Newton. Returning zeros. Use PyBullet backend if "
                    f"segmentation is required."
                )
                self._segmentation_warned = True
            images["segmentation"] = np.zeros((self.height, self.width), dtype=np.int32)

        return images

    def _get_depth_image(self) -> np.ndarray:
        """Get depth image from RaycastSensor."""
        if self._raycast_sensor is None:
            # Return placeholder (infinity = no depth data)
            return np.full((self.height, self.width), np.inf, dtype=np.float32)

        state = self._state_accessor()
        if state is None:
            return np.full((self.height, self.width), np.inf, dtype=np.float32)

        # Evaluate raycast
        self._raycast_sensor.eval(state)
        depth = self._raycast_sensor.get_depth_image_numpy()

        # Convert -1.0 (no hit) to infinity for consistency with PyBullet
        depth = depth.copy()
        depth[depth < 0] = np.inf

        return depth.astype(np.float32)

    def _get_color_image(self) -> np.ndarray:
        """Get RGB image from ViewerGL if available."""
        if self._viewer_manager is None or not getattr(self._viewer_manager, '_gui_enabled', False):
            # Return black placeholder in headless mode
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        viewer = getattr(self._viewer_manager, 'viewer', None)
        if viewer is None or not hasattr(viewer, 'get_frame'):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        try:
            # Get frame from ViewerGL (returns Warp array)
            frame = viewer.get_frame()
            if frame is not None:
                rgb = frame.numpy()
                # Resize if dimensions don't match
                if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                    # Simple resize by cropping/padding or using scipy
                    try:
                        from scipy.ndimage import zoom
                        scale_h = self.height / rgb.shape[0]
                        scale_w = self.width / rgb.shape[1]
                        rgb = zoom(rgb, (scale_h, scale_w, 1), order=1).astype(np.uint8)
                    except ImportError:
                        # Fall back to simple resize
                        rgb = rgb[:self.height, :self.width, :]
                return rgb
        except Exception as e:
            log.warning(f"Newton camera '{self.component_name}': Failed to get frame from viewer: {e}")

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def shutdown_driver(self) -> None:
        """Clean up resources."""
        self._raycast_sensor = None
        self._model = None
        self._viewer_manager = None
        super().shutdown_driver()
