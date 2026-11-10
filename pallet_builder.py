"""Heavy mixed-case palletizing cell (Isaac Sim 6.0 standalone): UR20 arm, weight-adaptive vacuum gripper,
optimized pallet.

A Universal Robots UR20 (20 kg rated payload, 1.75 m reach - UR's palletizing arm) on a 0.7 m pedestal builds a
3 x 3 x 3 pallet cube from boxes of different sizes (1x1, 2x1, 2x2 cells of 32 cm), heights (18 / 22 / 26 cm),
weights (1.5 - 16 kg; 16 kg box + 4 kg gripper = the UR20's 20 kg payload) and stacking strengths:

  1. The boxes wait in a buffer. Their dimensions and weights are known from the infeed dimensioner and
     checkweigher, so before picking anything the planner (pallet_planner.py) works out the whole pallet.
  2. Boxes are sent to the conveyor in that order. The arm seals a 4-cup vacuum gripper on the box's top face at
     full vacuum and lifts 8 cm; the UR20's wrist force/torque sensor weighs it.
  3. The vacuum is then regulated to what that box needs:  vacuum = margin * SF * m * (g + a_max / 0.5) / cup area.
     Every physics step the real load on the cups is compared with what the vacuum can hold.
  4. The arm swings the box over the pallet, turns it to its planned orientation, lowers it, measures where the
     box really is and corrects, and releases it 2 mm above its resting place.
  5. Practice (--episodes N) learns per-spot placement corrections and the lowest safe vacuum margin.

The controller itself lives in palletizer_core.py (shared with the Isaac Lab version, isaaclab_pallet_cells.py).

Run from the Isaac Sim install folder:
    python.bat <path-to>/pallet_builder.py
See README.md for all options.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="UR20 mixed-case palletizer with weight-adaptive vacuum gripper")
parser.add_argument("--headless", action="store_true", help="Run without the GUI")
parser.add_argument("--test", action="store_true", help="Exit when all episodes are done")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Physics device")
parser.add_argument("--env", default="warehouse",
                    help="'warehouse' (NVIDIA Simple_Warehouse), 'grid' (plain floor), or a path/URL to your own .usd")
parser.add_argument("--env-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="Shift the environment if shelves overlap the robot cell")
parser.add_argument("--seed", type=int, default=7, help="Random seed for the box set (sizes, weights, strengths)")
parser.add_argument("--max-box-mass", type=float, default=16.0,
                    help="Heaviest box in kg (default 16: with the 4 kg gripper that is the UR20's 20 kg payload)")
parser.add_argument("--episodes", type=int, default=1, help="How many times to build the pallet (practice runs)")
parser.add_argument("--safety-factor", type=float, default=2.0, help="Safety factor on the vacuum holding force")
parser.add_argument("--no-optimize", action="store_true",
                    help="Build the pallet in arrival order without optimizing (for comparison)")
parser.add_argument("--plan-iters", type=int, default=12000, help="Simulated-annealing iterations for the planner")
parser.add_argument("--weight-source", choices=["wrist", "manifest"], default="wrist",
                    help="Vacuum from the wrist F/T measurement (default) or the checkweigher value")
parser.add_argument("--no-learning", action="store_true", help="Do not apply or update learned corrections")
parser.add_argument("--reset-learning", action="store_true", help="Forget previously learned corrections")
parser.add_argument("--verbose", action="store_true", help="Print every step of every box")
parser.add_argument("--speed", type=int, default=1,
                    help="Fast-forward: physics + control steps per rendered frame (e.g. 4 = up to 4x faster in the GUI)")
parser.add_argument("--max-boxes", type=int, default=0, help=argparse.SUPPRESS)   # testing: stop after N boxes
parser.add_argument("--selftest-stop", type=int, default=0, help=argparse.SUPPRESS)  # testing: Stop/Play at frame N, 2N
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import sys
from pathlib import Path

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pallet_planner as pp  # noqa: E402
import palletizer_core as core  # noqa: E402
from palletizer_core import (  # noqa: E402
    CONVEYOR_TOP, PALLET_XY, PEDESTAL_H, PHYSICS_DT, SEED_Q, TOOL_MASS, WRIST_GAINS,
)

LEARNING_FILE = HERE / "learned_pallet.json"
TRAINING_LOG = HERE / "training_log_pallet.csv"
ARM_DOFS = list(range(6))


# --- Isaac Sim (experimental core API) versions of the robot and box ---------------------------------------------


class UR20(core.UR20Base):
    def __init__(self, path: str = "/World/Mount/UR20") -> None:
        # Stand the robot on its pedestal: the asset goes under a raised parent transform (its base is fixed to the
        # world where it starts).
        mount = UsdGeom.Xform.Define(stage_utils.get_current_stage(), path.rsplit("/", 1)[0])
        UsdGeom.XformCommonAPI(mount).SetTranslate(Gf.Vec3d(0.0, 0.0, PEDESTAL_H))
        stage_utils.add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur20/ur20.usd", path)
        self.path = path
        self.art = Articulation(path)
        self.art.set_default_state(dof_positions=SEED_Q)
        self.flange_path = f"{path}/wrist_3_link"
        self.flange = RigidPrim(self.flange_path)
        self.ee_index = self.art.get_link_indices("wrist_3_link").list()[0]
        self._tool_added = False

    def configure(self) -> None:
        k, d = self.art.get_dof_gains()
        k, d = k.numpy()[0].copy(), d.numpy()[0].copy()
        k[3:], d[3:] = WRIST_GAINS
        self.art.set_dof_gains(stiffnesses=k.reshape(1, -1), dampings=d.reshape(1, -1), dof_indices=ARM_DOFS)
        try:   # the vacuum tool's mass on the flange
            m = self.art.get_link_masses(link_indices=[self.ee_index]).numpy()[0]
            if not self._tool_added:
                self.art.set_link_masses((m + TOOL_MASS).reshape(1, -1), link_indices=[self.ee_index])
                self._tool_added = True
        except Exception as e:
            print(f"Could not add the tool mass: {e}")

    # Note: .numpy() of a simulator buffer can be a view that later reads overwrite, so values are copied.
    def q(self) -> np.ndarray:
        return self.art.get_dof_positions().numpy()[0][:6].copy()

    def flange_pose(self):
        p, q = self.flange.get_world_poses()
        return p.numpy()[0].copy(), q.numpy()[0].copy()

    def jacobian(self) -> np.ndarray:
        return self.art.get_jacobian_matrices().numpy()[0, self.ee_index - 1, :, :6].copy()

    def set_targets(self, q) -> None:
        self.art.set_dof_position_targets(np.asarray(q, dtype=float).reshape(1, -1), dof_indices=ARM_DOFS)

    def flange_force(self) -> np.ndarray:
        f, _ = self.art.get_link_incoming_joint_force(link_indices=[self.ee_index])
        return f.numpy()[0, 0].copy()

    def joint_load(self) -> float:
        try:
            tau = np.abs(self.art.get_dof_projected_joint_forces().numpy()[0][:6])
            return float(np.max(tau / self.art.get_dof_max_efforts().numpy()[0][:6]))
        except Exception:
            return float("nan")


class Box(core.BoxBase):
    def __init__(self, spec: pp.BoxSpec, staging: np.ndarray) -> None:
        self.spec = spec
        self.path = f"/World/Boxes/Box_{spec.id:02d}"
        self.size = np.array([spec.length, spec.width, spec.height])
        self.staging = staging.copy()
        self.shape = Cube(paths=self.path, positions=staging, sizes=1.0, scales=self.size.tolist(),
                          colors=core.box_color(spec))
        self.geom = GeomPrim(paths=self.shape.paths, apply_collision_apis=True)
        self.body = RigidPrim(paths=self.shape.paths)
        self.body.set_masses([spec.mass])
        prim = stage_utils.get_current_stage().GetPrimAtPath(self.path)
        # A sleeping body ignores a joint added at runtime (the vacuum seal), so boxes never fall asleep.
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.0)
        self.reset_results()

    def pose(self):
        p, q = self.body.get_world_poses()
        return p.numpy()[0].copy(), q.numpy()[0].copy()

    def lin_vel(self) -> np.ndarray:
        return self.body.get_velocities()[0].numpy()[0].copy()

    def teleport(self, position, yaw: float = 0.0) -> None:
        self.body.set_world_poses(positions=np.asarray(position, dtype=float), orientations=core.yaw_quat(yaw))
        self.body.set_velocities(linear_velocities=[0.0, 0.0, 0.0], angular_velocities=[0.0, 0.0, 0.0])


# --- scene -----------------------------------------------------------------------------------------------------


def static_box(path: str, center, size, color: str, collide: bool = True) -> None:
    c = Cube(paths=path, positions=list(center), sizes=1.0, scales=list(size), colors=color)
    if collide:
        GeomPrim(paths=c.paths, apply_collision_apis=True)


def build_pallet() -> None:
    """A 1.0 x 1.0 m block pallet: bottom boards, 9 blocks, stringer boards, 7 top deck boards (0.144 m)."""
    x0, y0 = PALLET_XY
    wood, dark = "#b98d5c", "#9c7448"
    for k, y in enumerate((-0.45, 0.0, 0.45)):
        static_box(f"/World/Pallet/Bottom_{k}", (x0, y0 + y, 0.011), (1.0, 0.10, 0.022), dark)
        static_box(f"/World/Pallet/Stringer_{k}", (x0, y0 + y, 0.111), (1.0, 0.10, 0.022), dark)
        for m, x in enumerate((-0.45, 0.0, 0.45)):
            static_box(f"/World/Pallet/Block_{k}{m}", (x0 + x, y0 + y, 0.061), (0.10, 0.10, 0.078), dark)
    for k, x in enumerate(np.linspace(-0.45, 0.45, 7)):
        static_box(f"/World/Pallet/Deck_{k}", (x0 + x, y0, 0.133), (0.10, 1.0, 0.022), wood)


def build_scene(boxes_spec: list[pp.BoxSpec]):
    if args.env == "warehouse":
        env_path = get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    elif args.env == "grid":
        env_path = None
    else:
        env_path = args.env
    if env_path:
        print(f"Loading environment: {env_path}")
        env_prim = stage_utils.add_reference_to_stage(usd_path=env_path, path="/World/Environment")
        if any(args.env_offset):
            UsdGeom.XformCommonAPI(env_prim).SetTranslate(Gf.Vec3d(*args.env_offset))
    else:
        GroundPlane("/World/ground_plane")
        DomeLight("/World/DomeLight").set_intensities(1000)

    static_box("/World/Cell/FloorMat", (-0.5, 0.5, -0.004), (8.0, 5.6, 0.008), "#2e3338")
    static_box("/World/Cell/Pedestal", (0.0, 0.0, PEDESTAL_H / 2), (0.32, 0.32, PEDESTAL_H), "#3d4a5c")
    # Infeed conveyor along +y ending next to the robot.
    static_box("/World/Cell/Conveyor", (0.0, 2.0, CONVEYOR_TOP - 0.04), (0.8, 2.4, 0.08), "#4a4f55")
    for k, y in enumerate(np.arange(0.9, 3.15, 0.15)):
        static_box(f"/World/Cell/Roller_{k}", (0.0, y, CONVEYOR_TOP - 0.01), (0.78, 0.03, 0.02), "#8a9099", False)
    for k, (x, y) in enumerate([(-0.35, 0.9), (0.35, 0.9), (-0.35, 3.1), (0.35, 3.1)]):
        static_box(f"/World/Cell/ConveyorLeg_{k}", (x, y, (CONVEYOR_TOP - 0.08) / 2), (0.06, 0.06, CONVEYOR_TOP - 0.08),
                   "#2b2f33")
    static_box("/World/Cell/PickStop", (0.0, 0.52, CONVEYOR_TOP + 0.04), (0.8, 0.03, 0.08), "#e0b000")
    build_pallet()

    robot = UR20()
    robot.tool_visual = core.ToolVisual(robot, stage_utils.get_current_stage(), "/World/VacuumTool")

    cardboard = RigidBodyMaterial("/World/Physics/Cardboard", static_frictions=0.6, dynamic_frictions=0.5,
                                  restitutions=0.0)
    boxes = {}
    for k, spec in enumerate(boxes_spec):
        xy = core.staging_xy(k)
        b = Box(spec, np.array([xy[0], xy[1], spec.height / 2 + 0.002]))
        b.geom.apply_physics_materials(cardboard)
        boxes[spec.id] = b
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[3.4, -2.8, 2.7], target=[0.5, 0.3, 0.6], camera_prim_path="/OmniverseKit_Persp")
    except Exception:
        pass
    return robot, boxes


# --- main ------------------------------------------------------------------------------------------------------


def main() -> None:
    if args.max_box_mass + TOOL_MASS > core.PAYLOAD_RATED_KG:
        print(f"WARNING: --max-box-mass {args.max_box_mass} kg + {TOOL_MASS:.0f} kg gripper exceeds the UR20's "
              f"{core.PAYLOAD_RATED_KG:.0f} kg payload; boxes over {core.MAX_BOX_KG:.0f} kg will be set back down.")
    SimulationManager.set_physics_sim_device(args.device)
    SimulationManager.setup_simulation(dt=PHYSICS_DT)
    simulation_app.update()

    specs = pp.generate_boxes(args.seed, args.max_box_mass)
    plan = core.make_plan(specs, not args.no_optimize, args.plan_iters, args.seed)
    robot, boxes = build_scene(specs)
    learner = core.Learner(LEARNING_FILE, enabled=not args.no_learning, reset=args.reset_learning)
    settings = core.Settings(safety_factor=args.safety_factor, episodes=args.episodes, weight_source=args.weight_source,
                             optimized=not args.no_optimize, verbose=args.verbose, max_boxes=args.max_boxes,
                             training_log=TRAINING_LOG)
    simulation_app.update()

    print("Waiting for environment assets to load...")
    for i in range(3000):
        if not stage_utils.is_stage_loading():
            break
        if i % 120 == 0:
            loaded, total = omni.usd.get_context().get_stage_loading_status()[1:3]
            print(f"  loading assets: {loaded}/{total} files")
        simulation_app.update()
    for _ in range(60):
        simulation_app.update()
    print("Assets loaded.")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    task, done, frame = None, False, 0
    stage = stage_utils.get_current_stage()
    while simulation_app.is_running():
        frame += 1
        if args.selftest_stop and frame in (args.selftest_stop, args.selftest_stop * 2):
            print(f"[selftest] Stop at frame {frame}")
            timeline.stop()
        if args.selftest_stop and frame in (args.selftest_stop + 30, args.selftest_stop * 2 + 30):
            timeline.play()
        if timeline.is_stopped() and (task is not None or done):
            print("\nSimulation stopped - run cancelled. Press Play to start again (learning is kept).")
            task, done = None, False
            core.VacuumGripper.clear_all(stage)
        # Step the controller only while the timeline is actually playing (not paused or stopped), so its
        # motion sequence never runs ahead of the physics.
        if timeline.is_playing() and SimulationManager.is_simulating() and not done:
            if task is None:
                core.VacuumGripper.clear_all(stage)
                robot.art.reset_to_default_state()
                robot.set_targets(SEED_Q)
                task = core.Palletizer(robot, boxes, plan, learner, settings, stage).train()
            try:
                next(task)
                # Fast-forward: extra physics steps without rendering (the frame's own step happens in update()).
                for _ in range(max(1, args.speed) - 1):
                    SimulationManager.step(steps=1)
                    next(task)
            except StopIteration:
                done = True
                if args.test:
                    break
        if not args.headless:
            robot.tool_visual.update(live=timeline.is_playing() and SimulationManager.is_simulating())
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
