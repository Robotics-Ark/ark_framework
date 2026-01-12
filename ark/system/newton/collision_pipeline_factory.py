"""Factory for Newton collision pipelines.

Extracts collision pipeline configuration from YAML and creates
appropriate CollisionPipeline instances. This is Newton-specific
infrastructure - other backends handle collision implicitly.

Newton's collision system supports two modes:
1. Standard pipeline (default): Uses model.collide() with automatic configuration
2. Unified pipeline: For contact-rich scenes with configurable contact budget

The unified pipeline provides:
- Higher per-pair contact budget for stable friction
- SDF hydroelastic contacts for volumetric compliance
- Configurable broad-phase modes (NXN, SAP, EXPLICIT)
- Contact matching for temporal coherence

Configuration via YAML:
    newton_physics:
        collision_pipeline:
            type: unified  # or "standard"
            broad_phase_mode: sap
            rigid_contact_max_per_pair: 16
            reduce_contacts: true
            sdf_hydroelastic:
                enabled: true
"""

from typing import TYPE_CHECKING, Any, Optional

import warp as wp

from ark.tools.log import log

if TYPE_CHECKING:
    import newton


class CollisionPipelineFactory:
    """Factory for creating Newton collision pipelines from config.

    This factory centralizes all collision pipeline configuration, keeping
    the newton_backend.py focused on orchestration. The factory handles:

    1. Parsing collision_pipeline YAML configuration
    2. Warp compatibility checks (tiled BVH queries)
    3. Unified pipeline creation with SDF hydroelastic support
    4. Fallback to standard pipeline on errors

    Example:
        >>> pipeline = CollisionPipelineFactory.from_config(model, sim_cfg)
        >>> contacts = CollisionPipelineFactory.collide(model, state, pipeline)
    """

    @staticmethod
    def from_config(
        model: "newton.Model",
        sim_cfg: dict[str, Any],
    ) -> Optional["newton.CollisionPipeline"]:
        """Create collision pipeline from simulation config.

        Returns None for standard pipeline (model.collide() default).
        Returns CollisionPipelineUnified for contact-rich scenes.

        Args:
            model: Finalized Newton model
            sim_cfg: Simulation config with newton_physics.collision_pipeline

        Returns:
            CollisionPipelineUnified or None (standard pipeline)
        """
        import newton

        newton_cfg = sim_cfg.get("newton_physics", {})
        pipeline_cfg = newton_cfg.get("collision_pipeline", {})
        if not isinstance(pipeline_cfg, dict) or not pipeline_cfg:
            return None

        pipeline_type = str(pipeline_cfg.get("type", "standard") or "standard").lower()
        if pipeline_type in ("standard", "default", "none", "off"):
            return None
        if pipeline_type != "unified":
            log.warning(
                "CollisionPipelineFactory: Unknown type '%s' (expected 'unified' or 'standard'); "
                "falling back to standard pipeline.",
                pipeline_type,
            )
            return None

        # Warp compatibility: some environments don't ship the experimental tiled BVH query intrinsics
        # used by Newton's unified pipeline. If they're missing, disable tiled queries so Newton falls
        # back to the standard mesh_query_aabb path.
        CollisionPipelineFactory._ensure_warp_compatibility()

        # Parse broad phase mode
        broad_phase_mode = CollisionPipelineFactory._parse_broad_phase_mode(
            pipeline_cfg, model
        )

        # Parse contact configuration
        rigid_contact_max_per_pair = pipeline_cfg.get("rigid_contact_max_per_pair", None)
        if rigid_contact_max_per_pair is not None:
            try:
                rigid_contact_max_per_pair = int(rigid_contact_max_per_pair)
            except (TypeError, ValueError):
                log.warning(
                    "CollisionPipelineFactory: Invalid rigid_contact_max_per_pair=%r; ignoring.",
                    rigid_contact_max_per_pair,
                )
                rigid_contact_max_per_pair = None

        reduce_contacts = bool(pipeline_cfg.get("reduce_contacts", True))
        iterate_mesh_vertices = bool(pipeline_cfg.get("iterate_mesh_vertices", True))
        enable_contact_matching = bool(pipeline_cfg.get("enable_contact_matching", False))
        edge_sdf_iter = int(pipeline_cfg.get("edge_sdf_iter", 10))

        # Configure SDF hydroelastic contacts if enabled
        sdf_hydroelastic_config = CollisionPipelineFactory._create_hydroelastic_config(
            pipeline_cfg
        )

        # Create the unified pipeline
        try:
            pipeline_kwargs = {
                "rigid_contact_max_per_pair": rigid_contact_max_per_pair,
                "reduce_contacts": reduce_contacts,
                "iterate_mesh_vertices": iterate_mesh_vertices,
                "enable_contact_matching": enable_contact_matching,
                "edge_sdf_iter": edge_sdf_iter,
                "broad_phase_mode": broad_phase_mode,
            }
            if sdf_hydroelastic_config is not None:
                pipeline_kwargs["sdf_hydroelastic_config"] = sdf_hydroelastic_config

            pipeline = newton.CollisionPipelineUnified.from_model(
                model,
                **pipeline_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "CollisionPipelineFactory: Failed to create unified pipeline (%s). "
                "Falling back to standard pipeline.",
                exc,
            )
            return None

        # Extract broad_phase_key for logging
        broad_phase_cfg = pipeline_cfg.get("broad_phase_mode", pipeline_cfg.get("broad_phase", "nxn"))
        broad_phase_key = str(broad_phase_cfg or "nxn").lower()

        log.info(
            "CollisionPipelineFactory: Created unified pipeline "
            "(broad_phase=%s, contact_max_per_pair=%s, reduce=%s, hydroelastic=%s)",
            broad_phase_key,
            rigid_contact_max_per_pair,
            reduce_contacts,
            sdf_hydroelastic_config is not None,
        )
        return pipeline

    @staticmethod
    def collide(
        model: "newton.Model",
        state: "newton.State",
        pipeline: Optional["newton.CollisionPipeline"],
    ) -> tuple["newton.Contacts", bool]:
        """Generate contacts using appropriate pipeline.

        Falls back to standard pipeline if unified fails.

        Args:
            model: Newton model
            state: Current simulation state
            pipeline: Optional unified pipeline (None = standard)

        Returns:
            Tuple of (contacts, pipeline_ok):
            - contacts: Contact information for solver
            - pipeline_ok: True if pipeline worked, False if fell back to standard
        """
        if pipeline is None:
            return model.collide(state), True

        try:
            return model.collide(state, collision_pipeline=pipeline), True
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "CollisionPipelineFactory: Unified pipeline failed during collide (%s). "
                "Falling back to standard pipeline.",
                exc,
            )

            # Ensure Model.collide() doesn't reuse the prior pipeline instance.
            if hasattr(model, "_collision_pipeline"):
                try:
                    delattr(model, "_collision_pipeline")
                except Exception:  # noqa: BLE001
                    pass

            return model.collide(state), False

    @staticmethod
    def _ensure_warp_compatibility() -> None:
        """Disable tiled BVH queries if Warp doesn't support them.

        Some Warp environments don't ship the experimental tiled BVH query intrinsics
        used by Newton's unified pipeline. This method disables them so Newton falls
        back to the standard mesh_query_aabb path.
        """
        if hasattr(wp, "tile_mesh_query_aabb"):
            return  # Warp has tiled BVH support, nothing to do

        try:
            from newton._src.geometry import collision_core  # type: ignore[attr-defined]  # noqa: PLC0415

            collision_core.ENABLE_TILE_BVH_QUERY = False

            # narrow_phase imports ENABLE_TILE_BVH_QUERY by value; update best-effort for any runtime uses.
            try:
                from newton._src.geometry import narrow_phase as narrow_phase_mod  # type: ignore[attr-defined]  # noqa: PLC0415

                if hasattr(narrow_phase_mod, "ENABLE_TILE_BVH_QUERY"):
                    narrow_phase_mod.ENABLE_TILE_BVH_QUERY = False
            except Exception:  # noqa: BLE001
                pass

            log.warning(
                "CollisionPipelineFactory: Warp is missing wp.tile_mesh_query_aabb; "
                "disabled Newton tiled BVH queries for compatibility."
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "CollisionPipelineFactory: Failed to disable Newton tiled BVH queries (%s); "
                "unified collision pipeline may fail to compile.",
                exc,
            )

    @staticmethod
    def _parse_broad_phase_mode(
        pipeline_cfg: dict[str, Any],
        model: "newton.Model",
    ) -> "newton.BroadPhaseMode":
        """Parse broad phase mode from config with fallback handling.

        Args:
            pipeline_cfg: Collision pipeline configuration dict
            model: Newton model (for EXPLICIT mode validation)

        Returns:
            Appropriate BroadPhaseMode enum value
        """
        import newton

        broad_phase_raw = pipeline_cfg.get("broad_phase_mode", pipeline_cfg.get("broad_phase", "nxn"))
        broad_phase_key = str(broad_phase_raw or "nxn").lower()
        broad_phase_map = {
            "nxn": newton.BroadPhaseMode.NXN,
            "sap": newton.BroadPhaseMode.SAP,
            "explicit": newton.BroadPhaseMode.EXPLICIT,
        }
        broad_phase_mode = broad_phase_map.get(broad_phase_key, newton.BroadPhaseMode.NXN)

        if broad_phase_mode == newton.BroadPhaseMode.EXPLICIT and getattr(model, "shape_contact_pairs", None) is None:
            log.warning(
                "CollisionPipelineFactory: broad_phase_mode='explicit' requested but "
                "model.shape_contact_pairs is missing; falling back to 'sap'."
            )
            broad_phase_mode = newton.BroadPhaseMode.SAP

        return broad_phase_mode

    @staticmethod
    def _create_hydroelastic_config(
        pipeline_cfg: dict[str, Any],
    ) -> Optional[Any]:
        """Create SDF hydroelastic config if enabled.

        SDF hydroelastic contacts provide volumetric compliance for stable grasping.

        Args:
            pipeline_cfg: Collision pipeline configuration dict

        Returns:
            SDFHydroelasticConfig or None if disabled/unavailable
        """
        hydroelastic_cfg = pipeline_cfg.get("sdf_hydroelastic", {})
        if not hydroelastic_cfg.get("enabled", True):  # Default enabled when using unified pipeline
            return None

        try:
            from newton.geometry import SDFHydroelasticConfig
            config = SDFHydroelasticConfig(
                output_contact_surface=bool(hydroelastic_cfg.get("output_contact_surface", False)),
            )
            log.info("CollisionPipelineFactory: SDF hydroelastic config enabled")
            return config
        except ImportError:
            log.warning(
                "CollisionPipelineFactory: SDFHydroelasticConfig not available in this Newton version; "
                "hydroelastic contacts disabled."
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "CollisionPipelineFactory: Failed to create SDFHydroelasticConfig (%s); "
                "hydroelastic contacts disabled.",
                exc,
            )
        return None
