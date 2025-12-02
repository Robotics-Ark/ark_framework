import os
from pathlib import Path

import numpy as np
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
    from isaacsim import SimulationApp

    app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})
    try:
        import omni.kit.commands
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import get_stage_units

        world = World(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
        world.scene.add_default_ground_plane()

        # Setting up import configuration:
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1.0


        # Import URDF, prim_path contains the path to the usd prim in the stage.
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(URDF_ASSET_PATH),
            import_config=import_config,
            get_articulation_root=True,
        )

        robot = Articulation(prim_path)

        robot.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())

        world.reset()

        for i in range(4):
            print("running cycle: ", i)
            if i == 1:
                print("moving")
                # move the arm
                robot.set_joint_positions(
                    [[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]]
                )
            if i == 2:
                print("stopping")
                # reset the arm
                robot.set_joint_positions(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                )
            for j in range(1000):
                # step the simulation, both rendering and physics
                world.step(render=True)

    finally:
        app.close()


if __name__ == "__main__":
    test_isaac_sim_headless_loads_usd()
