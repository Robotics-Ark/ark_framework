"""Newton viewer manager for ARK framework.

This module encapsulates all GUI/visualization logic for the Newton backend,
keeping the physics backend clean and modular.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import newton
import warp as wp

from ark.tools.log import log


class NewtonViewerManager:
    """Manages Newton viewer lifecycle and rendering.

    This class handles all visualization concerns for the Newton backend,
    allowing the physics simulation code to remain minimal and focused.
    The viewer can be completely disabled by setting connection_mode to "DIRECT".
    """

    def __init__(self, sim_config: dict[str, Any], model: newton.Model) -> None:
        """Initialize Newton viewer based on configuration.

        Args:
            sim_config: Simulator configuration dictionary containing:
                - connection_mode: "GUI" for interactive viewer, "DIRECT" for headless
                - viewer_width: Window width in pixels (default: 1280)
                - viewer_height: Window height in pixels (default: 800)
                - viewer_vsync: Enable vsync (default: False)
                - show_contacts: Show contact forces (default: True)
                - mp4: Optional path to record MP4 video
            model: Finalized Newton physics model
        """
        from newton.viewer import ViewerGL, ViewerNull

        self.model = model
        self.sim_config = sim_config
        self.viewer: Optional[newton.viewer.ViewerBase] = None
        self._gui_enabled = False

        # Get configuration parameters
        connection_mode = sim_config.get("connection_mode", "DIRECT").upper()
        viewer_width = int(sim_config.get("viewer_width", 1280))
        viewer_height = int(sim_config.get("viewer_height", 800))
        viewer_vsync = bool(sim_config.get("viewer_vsync", False))
        show_contacts = bool(sim_config.get("show_contacts", True))

        # Check for GUI mode
        if connection_mode == "GUI":
            # Check if DISPLAY is available (for X11-based systems)
            display = os.environ.get("DISPLAY")
            headless = display in (None, "")

            if headless:
                log.warning(
                    "Newton viewer: DISPLAY environment variable not set, "
                    "GUI mode unavailable. Falling back to headless mode."
                )

            try:
                if headless:
                    # Skip ViewerGL if we know display is unavailable
                    raise RuntimeError("DISPLAY unavailable")

                # Create interactive OpenGL viewer
                self.viewer = ViewerGL(
                    width=viewer_width,
                    height=viewer_height,
                    vsync=viewer_vsync,
                    headless=False,
                )
                self.viewer.set_model(model)

                # Position camera to view scene from bird's eye angle
                # Without this, camera may not capture robot in view
                self.viewer.set_camera(
                    pos=wp.vec3(3.0, -3.0, 2.0),  # Back-left, elevated
                    pitch=-20.0,                   # Look down slightly
                    yaw=45.0                       # Diagonal view
                )

                self.viewer.show_contacts = show_contacts
                self.viewer.show_collision = True   # Show collision geometry (robot meshes)
                self.viewer.show_static = True      # Show static shapes (ground plane)
                self._gui_enabled = True

                log.ok(
                    f"Newton viewer: ViewerGL initialized successfully "
                    f"({viewer_width}x{viewer_height}, vsync={viewer_vsync}, "
                    f"contacts={show_contacts})"
                )

            except Exception as exc:  # noqa: BLE001
                # Fall back to null viewer if ViewerGL fails
                log.warning(
                    f"Newton viewer: ViewerGL initialization failed ({exc}). "
                    "Falling back to headless mode (ViewerNull)."
                )
                self.viewer = ViewerNull()
                self.viewer.set_model(model)
                self._gui_enabled = False
                log.info("Newton viewer: Running in headless mode (no visualization)")

        else:
            # DIRECT mode - headless operation
            self.viewer = ViewerNull()
            self.viewer.set_model(model)
            self._gui_enabled = False
            log.info("Newton viewer: Running in DIRECT mode (headless, no visualization)")

        # Check for MP4 recording
        mp4_path = sim_config.get("mp4")
        if mp4_path:
            log.info(f"Newton viewer: MP4 recording requested: {mp4_path}")
            log.warning("Newton viewer: MP4 recording not yet implemented")
            # TODO: Implement MP4 recording support

    def render(
        self,
        state: newton.State,
        contacts: Optional[newton.Contacts],
        sim_time: float,
    ) -> None:
        """Render current simulation state to the viewer.

        This method should be called once per simulation step (not per substep)
        to update the visualization with the current physics state.

        Args:
            state: Current Newton simulation state
            contacts: Current contact information (optional)
            sim_time: Current simulation time in seconds
        """
        if self.viewer is None:
            return

        try:
            # Begin frame with current simulation time
            self.viewer.begin_frame(sim_time)

            # Log current state (updates all bodies, joints, etc.)
            self.viewer.log_state(state)

            # Optionally render contact forces
            if contacts is not None and hasattr(self.viewer, "show_contacts"):
                if self.viewer.show_contacts:
                    self.viewer.log_contacts(contacts, state)

            # Finalize frame
            self.viewer.end_frame()

        except Exception as exc:  # noqa: BLE001
            # Log error but don't crash simulation if rendering fails
            log.warning(f"Newton viewer: Frame render failed: {exc}")

    def is_running(self) -> bool:
        """Check if the viewer window is still open.

        Returns:
            True if viewer is running (or in headless mode), False if window was closed.
        """
        if self.viewer is None:
            return True

        try:
            return self.viewer.is_running()
        except Exception:  # noqa: BLE001, S110
            # If checking fails, assume viewer is still running to avoid premature exit
            return True

    def shutdown(self) -> None:
        """Clean up viewer resources.

        This should be called when the simulation is shutting down to properly
        release OpenGL contexts and other viewer resources.
        """
        if self.viewer is None:
            return

        try:
            # ViewerGL has a close() method
            if hasattr(self.viewer, "close"):
                self.viewer.close()
            log.info("Newton viewer: Shutdown complete")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Newton viewer: Shutdown failed: {exc}")
        finally:
            self.viewer = None

    @property
    def gui_enabled(self) -> bool:
        """Check if interactive GUI is enabled.

        Returns:
            True if ViewerGL is active, False if running headless (ViewerNull).
        """
        return self._gui_enabled
