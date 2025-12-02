import os
from pathlib import Path

import pytest

# Allow overriding USD path; defaults to the requested asset.
URDF_ASSET_PATH = Path(
    os.environ.get(
        "ARK_USD_PATH",
        "/home/refinath/ark/ark_franka/franka_panda/panda_with_gripper.urdf",
    )
)

omni_kit = pytest.importorskip(
    "omni.isaac.kit",
    reason="Isaac Sim Python packages are required to start the simulator.",
)


@pytest.mark.skipif(
    not URDF_ASSET_PATH.exists(),
    reason=(
        f"USD asset not found at {URDF_ASSET_PATH}. "
        "Set ARK_USD_PATH to point at a local copy."
    ),
)
def test_isaac_sim_headless_loads_usd():
    """Boot Isaac Sim headless, reference the USD, and verify a prim appears on the stage."""
    app = omni_kit.SimulationApp({"headless": True})
    try:
        from omni.isaac.core import World
        from isaacsim.asset.importer.urdf import URDFCreateImportConfig
        import omni.kit.commands
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.articulations import Articulation
        import omni.kit.commands

        # config.set_search_path([root_dir])

        world = World(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
        prim_path = "/World/Robot"
        root_dir = os.path.dirname(URDF_ASSET_PATH)

        # Configure physics and import options via _config
        config = URDFCreateImportConfig()
        config.fix_base = False
        config.self_collision = True
        config.merge_fixed_joints = True

        articulation = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=URDF_ASSET_PATH,
            import_config=config,
            get_articulation_root=True,
        )
        breakpoint()

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        assert prim and prim.IsValid(), "Referenced USD prim failed to load in stage"

        for p in stage.Traverse():
            print(p.GetPath(), p.GetTypeName())

        component_name = "Robot"
        world.scene.add(Articulation(prim_path=prim_path, name=component_name))
        _articulation = world.scene.get_object(component_name)
        world.reset()
        _joint_names = list(_articulation.get_joint_names())
        _joint_name_to_index = {name: idx for idx, name in enumerate(_joint_names)}
        world.step(render=False)

    finally:
        pass
        # app.close()


if __name__ == "__main__":
    test_isaac_sim_headless_loads_usd()
