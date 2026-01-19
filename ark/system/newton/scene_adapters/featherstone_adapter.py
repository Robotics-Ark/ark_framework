"""Featherstone solver adapter with native ground plane support."""

from typing import Any, Dict

import newton

from ark.tools.log import log
from ark.system.newton.scene_adapters.base_adapter import SolverSceneAdapter


class FeatherstoneAdapter(SolverSceneAdapter):
    """Adapter for Newton's Featherstone solver.

    Featherstone uses generalized coordinates and supports the native
    ground plane API, so the adapter is straightforward.
    """

    @property
    def solver_name(self) -> str:
        """Return solver display name."""
        return "Featherstone"

    def adapt_ground_plane(self, descriptor) -> None:
        """Add ground plane using Featherstone's native support."""
        # Featherstone supports native ground plane
        self.builder.builder.add_ground_plane()
        log.ok("Featherstone adapter: Added native ground plane")

    def create_solver(
        self,
        model: "newton.Model",
        sim_cfg: Dict[str, Any],
    ) -> "newton.solvers.SolverBase":
        """Create Featherstone solver with default parameters."""
        # Featherstone solver uses robust defaults for articulated robots.
        log.info("Featherstone adapter: Creating solver")
        return newton.solvers.SolverFeatherstone(model)
