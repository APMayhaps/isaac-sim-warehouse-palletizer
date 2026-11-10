"""Several UR20 palletizing cells inside the NVIDIA warehouse, in Isaac Lab.

Each cell is a copy of the pallet_builder.py cell: a UR20 (20 kg rated payload) on a 0.7 m pedestal, a 4-cup
vacuum gripper (4 kg), an infeed conveyor, a 1.0 x 1.0 m wooden pallet and a buffer of mixed boxes up to 16 kg.
Isaac Lab clones the cell into a grid (one cell per Isaac Lab environment) and they all run at once; the
palletizing logic is the same code as the standalone version (palletizer_core.py), one controller per cell.

Run from the Isaac Lab folder (Isaac Lab release v3.0.0-beta2.patch1, for Isaac Sim 6.0):
    isaaclab.bat -p <path-to>/isaaclab_pallet_cells.py --viz kit
Without --viz it runs headless. See README.md for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR20 palletizing cells in the warehouse (Isaac Lab)")
parser.add_argument("--num_cells", type=int, default=4, help="Number of cells (4 fit between the warehouse racks)")
parser.add_argument("--cell_spacing", type=float, default=6.5, help="Distance between cells in metres")
parser.add_argument("--env", default="warehouse", help="'warehouse' (NVIDIA Simple_Warehouse) or 'grid' (plain floor)")
parser.add_argument("--env_offset", type=float, nargs=3, default=[-1.6, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="Shift the warehouse relative to the cells (default centres 4 cells between the racks)")
parser.add_argument("--seed", type=int, default=7, help="Box set (every cell gets the same set)")
parser.add_argument("--max_box_mass", type=float, default=16.0,
                    help="Heaviest box in kg (16 kg + 4 kg gripper = the UR20's 20 kg payload)")
parser.add_argument("--episodes", type=int, default=1, help="Pallets each cell builds (practice runs)")
parser.add_argument("--safety_factor", type=float, default=2.0, help="Safety factor on the vacuum holding force")
parser.add_argument("--no_optimize", action="store_true", help="Build in arrival order (no optimization)")
parser.add_argument("--plan_iters", type=int, default=12000, help="Pallet optimizer iterations")
parser.add_argument("--weight_source", choices=["wrist", "manifest"], default="wrist")
parser.add_argument("--no_learning", action="store_true", help="Do not apply or update learned corrections")
parser.add_argument("--reset_learning", action="store_true", help="Forget previously learned corrections")
parser.add_argument("--speed", type=int, default=1, help="Physics + control steps per rendered frame (fast-forward)")
parser.add_argument("--log_steps", action="store_true", help="Print every step of every box")
parser.add_argument("--test", action="store_true", help="Exit when every cell has finished")
parser.add_argument("--max_boxes", type=int, default=0, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")   # the vacuum seals are joints created while running: use CPU physics
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pallet_planner as pp  # noqa: E402
import palletizer_core as core  # noqa: E402
from palletizer_core import CONVEYOR_TOP, PALLET_XY, PEDESTAL_H, PHYSICS_DT, SEED_Q, TOOL_MASS, WRIST_GAINS  # noqa: E402

LEARNING_FILE = HERE / "learned_pallet_isaaclab.json"
TRAINING_LOG = HERE / "training_log_isaaclab.csv"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


# --- scene ---------------------------------------------------------------------------------------------------------


@configclass
class CellsSceneCfg(InteractiveSceneCfg):
    """The parts every cell shares; the per-cell static parts and boxes are added in build_scene_cfg()."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur20/ur20.usd"),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, PEDESTAL_H),
                                                   joint_pos=dict(zip(JOINTS, SEED_Q.tolist()))),
        actuators={
            # Shoulder and elbow keep the asset's servo gains and the UR20's torque limits (730/730/430 N m).
            "arm": ImplicitActuatorCfg(joint_names_expr=JOINTS[:3], stiffness=None, damping=None),
            # The asset's wrist servos are soft for a 20 kg payload: stiffen them (torque limits unchanged).
            "wrist": ImplicitActuatorCfg(joint_names_expr=JOINTS[3:],
                                         stiffness=dict(zip(JOINTS[3:], WRIST_GAINS[0])),
                                         damping=dict(zip(JOINTS[3:], WRIST_GAINS[1]))),
        },
    )


def _static(size, pos, color, collide=True) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="",  # set by the caller
        spawn=sim_utils.CuboidCfg(size=tuple(size), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                                  collision_props=sim_utils.CollisionPropertiesCfg() if collide else None),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos)),
    )


def build_scene_cfg(specs: list[pp.BoxSpec]) -> CellsSceneCfg:
    cfg = CellsSceneCfg(num_envs=args.num_cells, env_spacing=args.cell_spacing)
    if args.env == "warehouse":
        cfg.warehouse = AssetBaseCfg(
            prim_path="/World/Warehouse",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(args.env_offset)))
    else:
        cfg.ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))

    parts: dict[str, AssetBaseCfg] = {
        "floor_mat": _static((6.2, 6.3, 0.008), (-1.6, 0.15, -0.004), (0.18, 0.2, 0.22)),
        "pedestal": _static((0.32, 0.32, PEDESTAL_H), (0.0, 0.0, PEDESTAL_H / 2), (0.24, 0.29, 0.36)),
        "conveyor": _static((0.8, 2.4, 0.08), (0.0, 2.0, CONVEYOR_TOP - 0.04), (0.29, 0.31, 0.33)),
        "pick_stop": _static((0.8, 0.03, 0.08), (0.0, 0.52, CONVEYOR_TOP + 0.04), (0.88, 0.69, 0.0)),
    }
    for k, y in enumerate(np.arange(0.9, 3.15, 0.15)):
        parts[f"roller_{k}"] = _static((0.78, 0.03, 0.02), (0.0, y, CONVEYOR_TOP - 0.01), (0.54, 0.56, 0.6), False)
    for k, (x, y) in enumerate([(-0.35, 0.9), (0.35, 0.9), (-0.35, 3.1), (0.35, 3.1)]):
        parts[f"conveyor_leg_{k}"] = _static((0.06, 0.06, CONVEYOR_TOP - 0.08), (x, y, (CONVEYOR_TOP - 0.08) / 2),
                                             (0.17, 0.18, 0.2))
    # 1.0 x 1.0 m block pallet: bottom boards, 9 blocks, stringers, 7 top deck boards (0.144 m tall).
    wood, dark = (0.73, 0.55, 0.36), (0.61, 0.45, 0.28)
    x0, y0 = PALLET_XY
    for k, y in enumerate((-0.45, 0.0, 0.45)):
        parts[f"pallet_bottom_{k}"] = _static((1.0, 0.10, 0.022), (x0, y0 + y, 0.011), dark)
        parts[f"pallet_stringer_{k}"] = _static((1.0, 0.10, 0.022), (x0, y0 + y, 0.111), dark)
        for m, x in enumerate((-0.45, 0.0, 0.45)):
            parts[f"pallet_block_{k}{m}"] = _static((0.10, 0.10, 0.078), (x0 + x, y0 + y, 0.061), dark)
    for k, x in enumerate(np.linspace(-0.45, 0.45, 7)):
        parts[f"pallet_deck_{k}"] = _static((0.10, 1.0, 0.022), (x0 + x, y0, 0.133), wood)
    for name, part in parts.items():
        part.prim_path = "{ENV_REGEX_NS}/Cell/" + name
        setattr(cfg, name, part)

    # Boxes: the same set in every cell, waiting in the buffer behind the robot.
    cardboard = sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.0)
    for k, s in enumerate(specs):
        xy = core.staging_xy(k)
        setattr(cfg, f"box_{s.id:02d}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + f"Box_{s.id:02d}",
            spawn=sim_utils.CuboidCfg(
                size=(s.length, s.width, s.height),
                # a sleeping body ignores a joint added at runtime (the vacuum seal): boxes never sleep
                rigid_props=sim_utils.RigidBodyPropertiesCfg(sleep_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=s.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=cardboard,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=core.box_color_rgb(s))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(float(xy[0]), float(xy[1]), s.height / 2 + 0.002))))
    return cfg


# --- Isaac Lab adapters: batched tensors <-> one robot / box per cell ----------------------------------------------


def xyzw_to_wxyz(q):
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=float)


class Cells:
    """Reads every cell's state once per physics step into numpy arrays, and collects joint targets."""

    def __init__(self, scene: InteractiveScene, specs: list[pp.BoxSpec]) -> None:
        self.scene = scene
        self.robot = scene["robot"]
        self.n = scene.num_envs
        self.origins = scene.env_origins.cpu().numpy().copy()
        self.ee = self.robot.find_bodies("wrist_3_link")[0][0]
        self.joint_ids = self.robot.find_joints(JOINTS, preserve_order=True)[0]
        self.box_objs = {s.id: scene[f"box_{s.id:02d}"] for s in specs}
        self.targets = np.tile(SEED_Q, (self.n, 1))
        view = self.robot.root_physx_view
        self.tau_max = self._np(view.get_dof_max_forces())[:, self.joint_ids].copy()
        print(f"[INFO] UR20 joint torque limits (N m): {np.round(self.tau_max[0], 0).tolist()}")
        self.refresh()

    @staticmethod
    def _np(a) -> np.ndarray:
        if hasattr(a, "numpy"):
            return np.asarray(a.numpy())
        return np.asarray(a.cpu()) if hasattr(a, "cpu") else np.asarray(a)

    def refresh(self) -> None:
        d = self.robot.data
        self.q = d.joint_pos.torch[:, self.joint_ids].cpu().numpy().copy()
        pose = d.body_link_pose_w.torch[:, self.ee].cpu().numpy().copy()
        self.fl_p = pose[:, :3]
        self.fl_q = pose[:, [6, 3, 4, 5]]                                   # Isaac Lab xyzw -> wxyz
        # Fixed-base robot: the Jacobian has no row for the base link, so the flange is at index ee - 1.
        self.jac = d.body_link_jacobian_w.torch[:, self.ee - 1, :, :][:, :, self.joint_ids].cpu().numpy().copy()
        view = self.robot.root_physx_view
        self.force = self._np(view.get_link_incoming_joint_force())[:, self.ee, :3].copy()
        # Torques PhysX actually solved (Isaac Lab's applied_torque for implicit servos is a clipped PD estimate).
        self.tau = self._np(view.get_dof_projected_joint_forces())[:, self.joint_ids].copy()
        self.box_pose = {k: o.data.root_pose_w.torch.cpu().numpy().copy() for k, o in self.box_objs.items()}
        self.box_vel = {k: o.data.root_lin_vel_w.torch.cpu().numpy().copy() for k, o in self.box_objs.items()}

    def apply_targets(self) -> None:
        t = torch.as_tensor(self.targets, dtype=torch.float32, device=self.robot.device)
        self.robot.set_joint_position_target_index(target=t, joint_ids=self.joint_ids)

    def add_tool_mass(self) -> None:
        try:
            m = self.robot.data.body_mass.torch.clone().cpu()
            m[:, self.ee] += TOOL_MASS
            self.robot.set_masses_index(masses=m[:, [self.ee]].to(self.robot.device), body_ids=[self.ee])
        except Exception as e:
            print(f"Could not add the tool mass to the flange: {e}")


class CellRobot(core.UR20Base):
    def __init__(self, cells: Cells, i: int) -> None:
        self.c, self.i = cells, i
        self.flange_path = f"/World/envs/env_{i}/Robot/wrist_3_link"

    def configure(self) -> None:
        pass   # gains come from the actuator config; the tool mass is added once for all cells

    def q(self):
        return self.c.q[self.i].copy()

    def flange_pose(self):
        return self.c.fl_p[self.i].copy(), self.c.fl_q[self.i].copy()

    def jacobian(self):
        return self.c.jac[self.i].copy()

    def set_targets(self, q) -> None:
        self.c.targets[self.i] = np.asarray(q, dtype=float)

    def flange_force(self):
        return self.c.force[self.i].copy()

    def joint_load(self) -> float:
        return float(np.max(np.abs(self.c.tau[self.i]) / self.c.tau_max[self.i]))


class CellBox(core.BoxBase):
    def __init__(self, cells: Cells, i: int, spec: pp.BoxSpec, staging_world: np.ndarray) -> None:
        self.c, self.i, self.spec = cells, i, spec
        self.obj = cells.box_objs[spec.id]
        self.path = f"/World/envs/env_{i}/Box_{spec.id:02d}"
        self.size = np.array([spec.length, spec.width, spec.height])
        self.staging = staging_world
        self.reset_results()

    def pose(self):
        p = self.c.box_pose[self.spec.id][self.i]
        return p[:3].copy(), xyzw_to_wxyz(p[3:])

    def lin_vel(self):
        return self.c.box_vel[self.spec.id][self.i].copy()

    def teleport(self, position, yaw: float = 0.0) -> None:
        pos = np.asarray(position, dtype=float)
        pose = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])   # xyzw
        dev = self.obj.device
        self.obj.write_root_pose_to_sim_index(root_pose=torch.tensor([pose], dtype=torch.float32, device=dev),
                                              env_ids=[self.i])
        self.obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev), env_ids=[self.i])
        # Keep this step's snapshot consistent with the teleport.
        self.c.box_pose[self.spec.id][self.i] = pose
        self.c.box_vel[self.spec.id][self.i] = 0.0


# --- main ------------------------------------------------------------------------------------------------------


def main() -> None:
    if args.env == "warehouse" and args.num_cells > 4:
        print(f"WARNING: {args.num_cells} cells won't fit between the warehouse racks (4 do); "
              f"use --env grid for more.")
    if args.max_box_mass + TOOL_MASS > core.PAYLOAD_RATED_KG:
        print(f"WARNING: --max_box_mass {args.max_box_mass} kg + {TOOL_MASS:.0f} kg gripper exceeds the UR20's "
              f"{core.PAYLOAD_RATED_KG:.0f} kg payload; heavier boxes will be set back down.")

    specs = pp.generate_boxes(args.seed, args.max_box_mass)
    plan = core.make_plan(specs, not args.no_optimize, args.plan_iters, args.seed)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    scene = InteractiveScene(build_scene_cfg(specs))
    sim.reset()
    stage = omni.usd.get_context().get_stage()
    print(f"[INFO] {scene.num_envs} palletizing cells, physics on {sim.device}")

    robot = scene["robot"]
    joint_ids = robot.find_joints(JOINTS, preserve_order=True)[0]
    seed = torch.as_tensor(np.tile(SEED_Q, (scene.num_envs, 1)), dtype=torch.float32, device=robot.device)
    robot.write_joint_position_to_sim_index(position=seed, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(seed), joint_ids=joint_ids)
    robot.reset()

    cells = Cells(scene, specs)
    cells.add_tool_mass()
    learner = core.Learner(LEARNING_FILE, enabled=not args.no_learning, reset=args.reset_learning)
    settings = core.Settings(safety_factor=args.safety_factor, episodes=args.episodes,
                             weight_source=args.weight_source, optimized=not args.no_optimize,
                             verbose=args.log_steps, max_boxes=args.max_boxes, training_log=TRAINING_LOG)
    viz = bool(getattr(args, "viz", None)) and args.viz not in ("none",)
    tasks, tools = [], []
    for i in range(cells.n):
        o = cells.origins[i]
        robot_i = CellRobot(cells, i)
        boxes_i = {}
        for k, s in enumerate(specs):
            xy = core.staging_xy(k)
            boxes_i[s.id] = CellBox(cells, i, s, o + np.array([xy[0], xy[1], s.height / 2 + 0.002]))
        if viz:
            tools.append(core.ToolVisual(robot_i, stage, f"/World/VacuumTools/Tool_{i}"))
        pal = core.Palletizer(robot_i, boxes_i, plan, learner, settings, stage, origin=o, name=f"cell {i + 1}")
        tasks.append(pal.train())
    try:
        c = cells.origins.mean(axis=0)
        sim.set_camera_view(eye=(c[0] + 9.0, c[1] - 10.0, 7.5), target=(c[0] - 1.6, c[1] + 0.3, 0.6))
    except Exception:
        pass

    active = list(range(cells.n))
    dt = sim.get_physics_dt()
    while app.is_running():
        if sim.is_stopped():
            print("Simulation stopped - closing (restart the script to run again).")
            break
        if not sim.is_playing():          # paused: keep the window responsive, don't advance the controllers
            sim.render()
            continue
        for k in range(max(1, args.speed)):
            for i in list(active):
                try:
                    next(tasks[i])
                except StopIteration:
                    active.remove(i)
            cells.apply_targets()
            scene.write_data_to_sim()
            sim.step(render=(k == max(1, args.speed) - 1))
            scene.update(dt)
            cells.refresh()
        for t in tools:
            t.update()
        if not active:
            if args.test:
                break
            print("All cells finished. Close the window to exit.")
            active = None
            break
    while active is None and app.is_running() and not args.test:
        sim.render()
    core.VacuumGripper.clear_all(stage)
    print(f"Per-cell log: {TRAINING_LOG}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        app.close()
