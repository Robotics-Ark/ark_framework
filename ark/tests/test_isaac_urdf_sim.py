import os
from pathlib import Path

import numpy as np
import pytest
from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver


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

    try:
        world = World(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
        world.scene.add_default_ground_plane()

        # Setting up import configuration:
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.import_inertia_tensor = True
        import_config.convex_decomp = False

        import_config.distance_scale = 1.0
        import_config.density = 0.0
        import_config.self_collision = False
        import_config.make_default_prim = True
        import_config.create_physics_scene = True

        # Import URDF, prim_path contains the path to the usd prim in the stage.
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(URDF_ASSET_PATH),
            import_config=import_config,
            get_articulation_root=True,
        )

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        # Enable physics
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        # Set gravity
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        # Set solver settings
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
        physxSceneAPI.CreateEnableCCDAttr(True)
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreateSolverTypeAttr("TGS")

        robot = Articulation(prim_path)
        world.scene.add(robot)

        rigid_bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(str(prim.GetPath()))

        print("Rigid bodies:", rigid_bodies)

        world.scene.add_default_ground_plane()

        omni.timeline.get_timeline_interface().play()
        app.update()
        robot.initialize()
        world.step(render=False)

        base_position = [0, 0, 0.2]
        base_orientation = [0, 0, 0, 1]
        q_int = [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0, 0, 0]
        robot.set_world_poses(
            positions=np.array([base_position]),
            orientations=np.array([base_orientation]),
        )
        robot.set_joint_positions([q_int])
        world.step(render=False)

        gripper = ParallelGripper(
            end_effector_prim_path=f"/panda/panda_hand",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )

        franka = world.scene.add(
            SingleManipulator(
                prim_path=prim_path,
                name="franka",
                end_effector_prim_path="/panda/panda_hand",
                gripper=gripper,
            )
        )
        franka.initialize()

        for _ in range(10):
            world.step(render=False)

        cube = world.scene.add(
            DynamicCuboid(
                name="cube",
                position=np.array([0.3, 0.3, 0.3]),
                prim_path="/World/Cube",
                scale=np.array([0.0515, 0.0515, 0.0515]),
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        )

        franka.gripper.set_default_state(franka.gripper.joint_opened_positions)

        controller = KinematicsSolver(franka)

        for _ in range(100):
            world.step(render=True)

        position = [0.35000014901161194, 0.28286391496658325, 0.3802158236503601]
        quaternion = [
            0.9999999403953552,
            1.9853024113558604e-08,
            1.1971462754445383e-07,
            8.532768447366834e-08,
        ]
        actions, succ = controller.compute_inverse_kinematics(
            target_position=np.asarray(position),
            target_orientation=np.asarray(quaternion),
        )
        if succ:
            franka.apply_action(actions)
        else:
            print("IK did not converge to a solution.  No action is being taken.")

        for _ in range(1000):
            world.step(render=True)

    finally:
        app.close()


if __name__ == "__main__":
    test_isaac_sim_headless_loads_usd()
