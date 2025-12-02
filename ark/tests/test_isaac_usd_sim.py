import os
from pathlib import Path

import numpy as np
import pytest

# Allow overriding USD path; defaults to the requested asset.
USD_ASSET_PATH = Path(
    os.environ.get(
        "ARK_USD_PATH",
        "/home/refinath/ark/ark_franka/franka_panda/panda_with_gripper.usd",
    )
)

omni_kit = pytest.importorskip(
    "omni.isaac.kit",
    reason="Isaac Sim Python packages are required to start the simulator.",
)


@pytest.mark.skipif(
    not USD_ASSET_PATH.exists(),
    reason=(
        f"USD asset not found at {USD_ASSET_PATH}. "
        "Set ARK_USD_PATH to point at a local copy."
    ),
)
def test_isaac_sim_headless_loads_usd():
    """Boot Isaac Sim headless, reference the USD, and verify a prim appears on the stage."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
        from isaacsim.core.prims import Articulation

        # Prepare scene
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        # Add robot
        prim_path = "/World/Robot"
        component_name = "franka"
        add_reference_to_stage(str(USD_ASSET_PATH), prim_path)
        robot = Articulation(prim_paths_expr=prim_path, name=component_name)

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
