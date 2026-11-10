"""Warehouse box palletizer with weight-adaptive grip force and learned placement (Isaac Sim 6.0).

Two grippers (--gripper):
  suction (default)  a vacuum cup on the hand. It seals on the box's top face at full vacuum for weighing, then
                     regulates the vacuum to what that box needs: vacuum = margin * SF * m * (g + a/0.5) / area.
                     The cup holds rigidly, so boxes are stacked to well under a millimetre, 7 mm apart.
  fingers            the Franka's two-finger hand with a weight-adaptive squeeze force (steps 2-3 below).

A Franka Panda arm works a pick cell inside a warehouse:

  1. Up to 81 boxes of unknown weight wait in an infeed staging area. They are fed one at a time onto
     a fixed pick spot.
  2. The arm grasps with a probe force and lifts the box 6 cm. If it slips, the force goes up and the
     grasp is retried. A simulated wrist force/torque sensor weighs the box while it is held still.
  3. The grip force for the carry is computed from the weight and the trajectory's peak acceleration:
         drive = preload + margin * SF * m * (g + a_max) / (efficiency * mu),   F_finger = drive / 2
     (preload and efficiency were calibrated in this simulator; margin is learned, see 6).
  4. Boxes are stacked by weight class into LIGHT / MEDIUM / HEAVY zones as 3x3x3 cubes
     (27 boxes per zone), one layer at a time.
  5. Precise placement: above the slot the arm measures where the box actually is and shifts the hand
     to line it up (fine alignment), lowers it, and releases with a small finger opening so it does not
     hit neighboring boxes.
  6. Practice (--episodes N): after every placement the arm measures how far the box landed from its
     slot and updates a per-slot correction (iterative learning). It also learns the lowest grip margin
     that does not slip. Both are saved to learned_corrections.json, so every run - and every new
     launch - starts from what was learned before. Per-episode precision is logged to training_log.csv.

Run from the Isaac Sim install folder:
    python.bat <path-to>/warehouse_box_sorter.py
See README.md for all options.
"""

from __future__ import annotations

# 1. Arguments + SimulationApp ---------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description="Weight-adaptive warehouse box palletizer")
parser.add_argument("--headless", action="store_true", help="Run without the GUI")
parser.add_argument("--test", action="store_true", help="Exit when all episodes are done")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Physics device")
parser.add_argument("--env", default="warehouse",
                    help="'warehouse' (NVIDIA Simple_Warehouse), 'grid' (plain floor), or a path/URL to your own .usd")
parser.add_argument("--env-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="Shift the environment if shelves overlap the robot cell")
parser.add_argument("--num-boxes", type=int, default=81, help="Number of boxes (1-81; 81 = three full 3x3x3 cubes)")
parser.add_argument("--episodes", type=int, default=1,
                    help="How many times to run the whole job (practice runs). ~30 s of sim per box per episode")
parser.add_argument("--seed", type=int, default=7, help="Random seed for box weights")
parser.add_argument("--gripper", choices=["suction", "fingers"], default="suction",
                    help="'suction' = vacuum cup with weight-adaptive vacuum (default), "
                         "'fingers' = the Franka's two-finger hand with weight-adaptive squeeze force")
parser.add_argument("--friction", type=float, default=1.0,
                    help="Box/finger-pad friction coefficient (mu). The force law is calibrated at 1.0")
parser.add_argument("--safety-factor", type=float, default=None,
                    help="Safety factor on the load (default 2.0 for suction, 1.25 for fingers)")
parser.add_argument("--probe-force", type=float, default=25.0, help="Initial probe grip force per finger (N)")
parser.add_argument("--weight-source", choices=["wrist", "manifest"], default="wrist",
                    help="'wrist' = measure with the F/T sensor, 'manifest' = read the box's labelled mass")
parser.add_argument("--no-learning", action="store_true", help="Do not apply or update learned corrections")
parser.add_argument("--reset-learning", action="store_true", help="Forget previously learned corrections")
parser.add_argument("--verbose", action="store_true", help="Print every step of every box")
parser.add_argument("--speed", type=int, default=1,
                    help="Fast-forward: physics + control steps per rendered frame (e.g. 4 = up to 4x faster in the GUI)")
parser.add_argument("--selftest-stop", type=int, default=0, help=argparse.SUPPRESS)  # press Stop/Play after N frames
parser.add_argument("--weak-vacuum-test", action="store_true", help=argparse.SUPPRESS)  # 5% vacuum: boxes must drop
args, _ = parser.parse_known_args()
SUCTION = args.gripper == "suction"
if args.safety_factor is None:
    args.safety_factor = 2.0 if SUCTION else 1.25

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.experimental.manipulators.examples", True
)

# Omniverse imports must come after SimulationApp is created.
import csv
import json
import time
from pathlib import Path

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka
from isaacsim.storage.native import get_assets_root_path

# 2. Constants -------------------------------------------------------------------------------------

PHYSICS_DT = 1.0 / 60.0
G = 9.81
HERE = Path(__file__).resolve().parent
LEARNING_FILE = HERE / ("learned_corrections_suction.json" if SUCTION else "learned_corrections.json")
TRAINING_LOG = HERE / ("training_log_suction.csv" if SUCTION else "training_log.csv")

# Suction gripper: a vacuum cup mounted on the hand between the (fully open) fingers. It holds the box by its top
# face with force = vacuum x cup area; the vacuum level is regulated per box from its measured weight (as real
# vacuum palletizers do to save compressed air), and the seal breaks if the load exceeds the holding force.
CUP_RADIUS = 0.02                       # m (40 mm cup)
CUP_AREA = np.pi * CUP_RADIUS ** 2      # 12.6 cm^2
CUP_REACH = 0.13                        # cup lip below the hand frame; the fingertips end ~1.3 cm higher
VACUUM_MIN_KPA, VACUUM_MAX_KPA = 8.0, 80.0   # regulator range (80 kPa -> ~100 N holding force)
PROBE_VACUUM_KPA = 80.0                 # full vacuum for the first (weighing) lift
SEAL_XY_TOL = 0.012                     # cup must land within this of the box's top-face center to seal
SEAL_LOST_TOL = 0.006                   # box moving this far relative to the cup = seal lost
CUP_SHEAR_RATIO = 0.5                   # sideways holding capacity relative to the pull capacity (lip friction)

# Gripper. panda_finger_joint2 is a mimic of joint1, so one drive moves both fingers and each finger
# squeezes with half the drive force.
FINGER_DOFS = [7, 8]
DRIVEN_FINGER_DOF = 7
FINGER_OPEN = 0.04            # m per finger, fully open (8 cm)
FINGER_RELEASE = 0.034        # m per finger when releasing in a stack (clears neighboring boxes)
FINGER_STIFFNESS = 2.0e4      # N/m - high so the drive saturates at the force limit
FINGER_DAMPING = 2.0e2
OPEN_FORCE = 40.0             # N per finger used while opening / moving empty
GRIP_FORCE_MIN = 3.0          # N per finger
GRIP_FORCE_MAX = 50.0         # N per finger. The Franka Hand can do 70, but in this simulator the fingers start
                              # sinking into the box above ~40-50 N and the contact solver can squeeze it out.
ARM_PAYLOAD = 3.0             # kg

# Grip force law, calibrated in this simulator (Franka fingers on a 5.5 cm box, mu = 1.0): the lowest drive
# force that lifts 0.25 / 1.0 / 2.5 kg without slip fits drive ~= 16 N + 40 N/kg * m.
GRIP_PRELOAD = 16.0
GRIP_EFFICIENCY = 0.245

# Geometry
BOX_SIZE = 0.055              # all boxes the same size so they stack into neat cubes
WORK_SURFACE_Z = 0.01         # top of the work-cell floor mat
# Hand height above box center when grasping. Fingers: the pads then sit at the box's center of mass (gripping
# below it makes a heavy box topple in the fingers like an inverted pendulum) and the fingertips stay ~1.5 cm
# clear of the surface below. Suction: the cup lip sits 1 mm above the box's top face.
HAND_TO_GRASP = BOX_SIZE / 2.0 + CUP_REACH + 0.001 if SUCTION else 0.105
TORSIONAL_PATCH = 0.015       # m contact patch radius: models finger-pad area so contacts resist twisting
# Release this far above the surface. Fingers: any contact while still gripping pushes back on the loaded hand
# and tilts it, which then shoves the box sideways when releasing. Suction holds rigidly, so 1.5 mm is enough.
PLACE_PRESS = -0.0015 if SUCTION else -0.004
MAX_RELEASE_TILT_DEG = 1.5    # wait until the hand is this level before opening the fingers
TRANSIT_Z = 0.42              # hand height while moving between stations (clears 3-high stacks)
WEIGH_LIFT = 0.06             # m the probe lift raises the box for weighing
PICK_SPOT = np.array([0.48, 0.0])
GRID, LAYERS = 3, 3           # 3x3 per layer, 3 layers = 27 boxes per zone
SLOT_PITCH = 0.062 if SUCTION else 0.075   # fingers need a 2 cm gap between boxes; suction packs 7 mm apart
LAYER_HEIGHT = BOX_SIZE + 0.001
ZONE_CAPACITY = GRID * GRID * LAYERS

# Weight classes, with gaps between the generated ranges so small weighing errors don't misclassify.
# Heavy tops out at 2.0 kg: in this simulator a two-finger friction grip at the hand's 70 N limit is not
# reliable for heavier boxes (the real Franka is rated for 3 kg).
CLASSES = {"light": (0.10, 0.60), "medium": (0.80, 1.30), "heavy": (1.50, 2.00)}
LIGHT_MAX, MEDIUM_MAX = 0.70, 1.40

ZONES = {
    "light": np.array([0.13, 0.47]),
    "medium": np.array([0.37, 0.29]),
    "heavy": np.array([0.29, -0.36]),   # heavy boxes closest to the base (less joint torque)
}

# Slip / drop thresholds (judged with the hand stationary; while moving the box readout trails by ~5 mm).
SLIP_PROBE_TOL = 0.004
SLIP_CARRY_TOL = 0.005
SLIP_MOVING_TOL = 0.02
DROP_TOL = 0.05

# Learning
# Bounds on the learned grip margin. Fingers: more force is not always better here (see GRIP_FORCE_MAX).
GRIP_MARGIN_MIN, GRIP_MARGIN_MAX = (0.6, 2.0) if SUCTION else (0.9, 1.3)
SLIP_MARGIN_STEP = 1.08 if SUCTION else 1.02  # a lost seal is a drop, so suction backs off faster
LEARN_GAIN = 0.6             # fraction of the measured placement error corrected next time
MAX_CORRECTION = 0.015        # m
ALIGN_PASSES = 2

IK_DAMPING = 0.05             # damped-least-squares damping (same as the stock Franka IK)
NULLSPACE_GAIN = 0.05         # per-step pull toward the comfortable posture, in the null space of the hand task
ARM_STIFFNESS_SCALE = 4.0     # the simulated arm joints are much softer than a real Franka's; stiffen them so a
                              # loaded arm doesn't sag (damping scaled by sqrt to keep the same damping ratio)

READY_Q = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741])  # Franka ready pose (hand down, ~0.3 m out)

COLORS = {"unknown": "peru", "light": "limegreen", "medium": "gold", "heavy": "orangered", "rejected": "darkred"}


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def peak_accel(distance: float, steps: int) -> float:
    """Peak acceleration of a smoothstep profile: 6 D / T^2."""
    T = steps * PHYSICS_DT
    return 6.0 * distance / (T * T)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the frame given by quaternion q (w, x, y, z)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ v


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def yaw_error_deg(q_wxyz: np.ndarray) -> float:
    """Yaw of a cube relative to the grid, folded into [0, 45] degrees (cubes are 90-degree symmetric)."""
    w, x, y, z = q_wxyz
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return float(abs((yaw + 45.0) % 90.0 - 45.0))


class BoxDropped(Exception):
    pass


# 3. Data ------------------------------------------------------------------------------------------


class Box:
    def __init__(self, index: int, path: str, staging: np.ndarray, mass: float) -> None:
        self.index = index
        self.path = path
        self.size = BOX_SIZE
        self.true_mass = mass
        self.staging = staging.copy()
        self.shape = Cube(paths=path, positions=staging, sizes=1.0, scales=[BOX_SIZE] * 3, colors=COLORS["unknown"])
        self.geom = GeomPrim(paths=self.shape.paths, apply_collision_apis=True)
        self.body = RigidPrim(paths=self.shape.paths)
        self.body.set_masses([mass])
        # By default PhysX contacts are points, so nothing resists the box twisting about the finger axis.
        # A torsional patch radius models the pad's contact area (torsional friction).
        try:
            from pxr import PhysxSchema

            prim = stage_utils.get_current_stage().GetPrimAtPath(path)
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateTorsionalPatchRadiusAttr(TORSIONAL_PATCH)
            api.CreateMinTorsionalPatchRadiusAttr(TORSIONAL_PATCH)
            if SUCTION:
                # A sleeping body ignores a joint added at runtime (the suction seal), and a velocity write does
                # not wake it, so boxes never fall asleep.
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.0)
        except Exception as e:
            print(f"Could not set torsional friction on {path}: {e}")
        self.reset_results()

    def reset_results(self) -> None:
        self.est_mass: float | None = None
        self.category = "unknown"
        self.grip_force = 0.0
        self.probe_attempts = 0
        self.slip_events = 0
        self.status = "waiting"
        self.place_error_mm: float | None = None
        self.yaw_error_deg: float | None = None
        self.seal_peak: float | None = None   # suction: highest fraction of the cup's capacity used while carrying

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.body.get_world_poses()
        return p.numpy()[0], q.numpy()[0]

    def position(self) -> np.ndarray:
        return self.pose()[0]

    def teleport(self, position) -> None:
        self.body.set_world_poses(positions=np.asarray(position, dtype=float), orientations=[1.0, 0.0, 0.0, 0.0])
        try:
            self.body.set_velocities(linear_velocities=[0.0, 0.0, 0.0], angular_velocities=[0.0, 0.0, 0.0])
        except Exception:
            pass

    def set_color(self, key: str) -> None:
        try:
            self.shape.set_display_colors(COLORS[key])
        except Exception:
            pass


class Learner:
    """Per-slot placement corrections and a grip-force margin, learned from practice and saved to disk."""

    def __init__(self, enabled: bool, reset: bool) -> None:
        self.enabled = enabled
        self.slots: dict[str, list[float]] = {}
        self.grip_margin = 1.0
        self.episodes_trained = 0
        if enabled and not reset and LEARNING_FILE.exists():
            try:
                data = json.loads(LEARNING_FILE.read_text())
                self.slots = {k: list(v) for k, v in data.get("slots", {}).items()}
                self.grip_margin = float(data.get("grip_margin", 1.0))
                self.episodes_trained = int(data.get("episodes_trained", 0))
                print(f"Loaded learned corrections: {len(self.slots)} slots, grip margin {self.grip_margin:.2f}, "
                      f"{self.episodes_trained} episodes of practice so far")
            except Exception as e:
                print(f"Could not read {LEARNING_FILE.name} ({e}); starting fresh")

    @staticmethod
    def key(zone: str, slot: int) -> str:
        return f"{zone}:{slot}"

    def correction(self, zone: str, slot: int) -> np.ndarray:
        if not self.enabled:
            return np.zeros(2)
        k = self.key(zone, slot)
        if k in self.slots:
            return np.array(self.slots[k])
        # Never placed here before: start from the average of what was learned in the same zone and layer.
        layer = slot // (GRID * GRID)
        same = [np.array(v) for kk, v in self.slots.items()
                if kk.startswith(zone + ":") and int(kk.split(":")[1]) // (GRID * GRID) == layer]
        return np.mean(same, axis=0) if same else np.zeros(2)

    def update(self, zone: str, slot: int, error_xy: np.ndarray) -> None:
        if not self.enabled:
            return
        new = self.correction(zone, slot) - LEARN_GAIN * error_xy
        self.slots[self.key(zone, slot)] = np.clip(new, -MAX_CORRECTION, MAX_CORRECTION).tolist()

    def on_slip(self) -> None:
        if self.enabled:
            self.grip_margin = min(GRIP_MARGIN_MAX, self.grip_margin * SLIP_MARGIN_STEP)

    def on_clean_episode(self) -> None:
        """No slips all episode: try a slightly gentler grip next time."""
        if self.enabled:
            self.grip_margin = max(GRIP_MARGIN_MIN, self.grip_margin * 0.98)

    def save(self) -> None:
        if not self.enabled:
            return
        LEARNING_FILE.write_text(json.dumps(
            {"episodes_trained": self.episodes_trained, "grip_margin": self.grip_margin, "slots": self.slots},
            indent=1))


# 4. Gripper force control + wrist sensor ----------------------------------------------------------


class AdaptiveGripper:
    """Force-limited parallel gripper: target fully closed, drive max force = desired squeeze."""

    def __init__(self, robot: Franka, mu: float, safety_factor: float) -> None:
        self.robot = robot
        self.mu = mu
        self.sf = safety_factor
        self.force = OPEN_FORCE

    def _set_drive_limit(self, per_finger: float) -> None:
        self.robot.set_dof_max_efforts([2.0 * per_finger], dof_indices=[DRIVEN_FINGER_DOF])

    def configure_drives(self) -> None:
        self.robot.set_dof_gains(stiffnesses=[FINGER_STIFFNESS], dampings=[FINGER_DAMPING],
                                 dof_indices=[DRIVEN_FINGER_DOF])
        self._set_drive_limit(OPEN_FORCE)

    def required_force(self, mass: float, accel: float, margin: float = 1.0) -> float:
        drive = GRIP_PRELOAD + margin * self.sf * mass * (G + accel) / (GRIP_EFFICIENCY * self.mu)
        return drive / 2.0

    def squeeze(self, force: float) -> float:
        self.force = float(np.clip(force, GRIP_FORCE_MIN, GRIP_FORCE_MAX))
        self._set_drive_limit(self.force)
        self.robot.set_dof_position_targets([0.0, 0.0], dof_indices=FINGER_DOFS)
        return self.force

    def open(self, width_per_finger: float = FINGER_OPEN) -> None:
        self.force = OPEN_FORCE
        self._set_drive_limit(OPEN_FORCE)
        self.robot.set_dof_position_targets([width_per_finger, width_per_finger], dof_indices=FINGER_DOFS)

    def width(self) -> float:
        q = self.robot.get_dof_positions().numpy()[0]
        return float(q[7] + q[8])


class SuctionGripper:
    """Vacuum suction cup on the hand.

    The seal is a fixed joint between the hand and the box's top face, created when the cup touches the box and
    removed on release. Every physics step the load on the cup (weight plus inertia) is compared with what the
    current vacuum can hold (vacuum x cup area, less for sideways load); if it is exceeded the seal is lost and
    the box drops. The vacuum is regulated per box from the measured weight:
        F_hold = margin * SF * m * (g + a_max / shear_ratio),   vacuum = F_hold / cup area
    """

    def __init__(self, robot: Franka, safety_factor: float) -> None:
        self.robot = robot
        self.sf = safety_factor
        self.stage = stage_utils.get_current_stage()
        self.hand_path = robot.end_effector_link.paths[0]
        self.joint = None
        self.vacuum = 0.0
        self._n = 0

    @staticmethod
    def hold_force(kpa: float) -> float:
        return kpa * 1000.0 * CUP_AREA

    def required_vacuum(self, mass: float, accel: float, margin: float = 1.0) -> float:
        # Worst case: the peak acceleration acting sideways, where the cup is weaker (CUP_SHEAR_RATIO).
        need = margin * self.sf * mass * (G + accel / CUP_SHEAR_RATIO)
        return float(np.clip(need / CUP_AREA / 1000.0, VACUUM_MIN_KPA, VACUUM_MAX_KPA))

    def attach(self, box: Box, kpa: float) -> None:
        """Seal onto the box where it is now (no snapping)."""
        from pxr import Gf, UsdPhysics

        self.release()
        _, pos, quat = self.robot.get_current_state()
        b_pos, b_quat = box.pose()
        p_rel = quat_rotate_inverse(quat[0], b_pos - pos[0])
        q_rel = quat_mul(quat_conj(quat[0]), b_quat)
        q_rel /= np.linalg.norm(q_rel)
        path = f"/World/Suction/Seal_{self._n}"
        self._n += 1
        joint = UsdPhysics.FixedJoint.Define(self.stage, path)
        joint.CreateBody0Rel().SetTargets([self.hand_path])
        joint.CreateBody1Rel().SetTargets([box.path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in p_rel]))
        joint.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in q_rel]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))   # the box frame (unscaled at its center)
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr(True)        # body0 is an articulation link
        self.joint = joint
        self.set_vacuum(kpa)

    def set_vacuum(self, kpa: float) -> float:
        self.vacuum = float(np.clip(kpa, 0.0, VACUUM_MAX_KPA))
        self._v_prev = None
        self._util = 0.0
        self.peak_util = 0.0
        return self.vacuum

    def seal_utilization(self, box: Box) -> float:
        """Fraction of the cup's capacity in use this step (> 1 means the seal fails).

        The cup must supply F = m (a + g) to the box. The axial (pull) part is limited by vacuum x area; the
        sideways (shear) part only by the lip's friction, about half as much. (PhysX's own joint break force
        can't be used for this: in a loop with the arm's articulation the solver reports constraint forces
        many times the actual load.)"""
        v = box.body.get_velocities()[0].numpy()[0]
        if self._v_prev is None:
            self._v_prev = v
            return self._util
        accel = (v - self._v_prev) / PHYSICS_DT
        self._v_prev = v
        f = box.true_mass * (accel + np.array([0.0, 0.0, G]))
        hold = self.hold_force(self.vacuum)
        u = max(f[2], 0.0) / hold + float(np.hypot(f[0], f[1])) / (CUP_SHEAR_RATIO * hold)
        self._util = 0.5 * self._util + 0.5 * u   # 2-step smoothing of the finite-difference acceleration
        self.peak_util = max(self.peak_util, self._util)
        return self._util

    def release(self) -> None:
        if self.joint is not None:
            self.stage.RemovePrim(self.joint.GetPath())
            self.joint = None
        self.vacuum = 0.0


class WristFTSensor:
    """Wrist force sensor from the incoming joint force of the panda_hand link."""

    def __init__(self, robot: Franka) -> None:
        self.robot = robot
        self.link = robot.get_link_indices("panda_hand").list()[0]
        self.tare_vec: np.ndarray | None = None
        self._samples: list[np.ndarray] = []

    def read(self) -> np.ndarray:
        forces, _ = self.robot.get_link_incoming_joint_force(link_indices=[self.link])
        return forces.numpy()[0, 0]

    def start(self) -> None:
        self._samples = []

    def sample(self) -> None:
        self._samples.append(self.read())

    def average(self) -> np.ndarray:
        return np.median(np.array(self._samples), axis=0)

    def mass_from_samples(self) -> tuple[float, bool]:
        """(mass, steady). Uses force magnitudes so a slight hand tilt under load doesn't bias the result."""
        if self.tare_vec is None or len(self._samples) < 4:
            return float("nan"), False
        s = np.array(self._samples)
        half = len(s) // 2
        tare = np.linalg.norm(self.tare_vec)

        def mass(block: np.ndarray) -> float:
            return (np.linalg.norm(np.median(block, axis=0)) - tare) / G

        m_all, m_a, m_b = mass(s), mass(s[:half]), mass(s[half:])
        return float(m_all), bool(abs(m_a - m_b) <= 0.1 * max(m_all, 0.05))


# 5. Controller (a generator: one yield == one physics step) ---------------------------------------


class Palletizer:
    def __init__(self, robot: Franka, boxes: list[Box], learner: Learner) -> None:
        self.robot = robot
        self.boxes = boxes
        self.learner = learner
        self.gripper = AdaptiveGripper(robot, args.friction, args.safety_factor)
        self.suction = SuctionGripper(robot, args.safety_factor) if SUCTION else None
        self.sensor = WristFTSensor(robot)
        self.down = robot.get_downward_orientation()
        self.zone_count = {k: 0 for k in ZONES}
        self.carrying: Box | None = None
        self._rel0 = np.zeros(3)
        self.episode_slips = 0
        self.reject_count = 0
        self.stage = ""
        self.q_cmd: np.ndarray | None = None

    def log(self, msg: str) -> None:
        if args.verbose:
            print(msg)

    # --- helpers ---------------------------------------------------------------------------------

    def hand_pos(self) -> np.ndarray:
        return self.robot.get_current_state()[1][0]

    def ik(self, position) -> None:
        """One differential-IK step toward `position` with the hand pointing straight down, plus null-space
        posture control.

        The hand pose uses 6 of the Franka's 7 joints. The stock IK leaves the 7th direction free, so on long
        moves the arm can drift into contorted postures with joints pinned at their limits, where it can no
        longer reach the target. The null-space term gently pulls the joints toward a comfortable posture
        (the ready pose turned to face the target) without moving the hand.
        """
        q_all, pos, quat = self.robot.get_current_state()
        q = q_all[0, :7]
        J = self.robot.get_jacobian_matrices().numpy()[0, self.robot.end_effector_link_index - 1, :, :7]
        goal = np.asarray(position, dtype=float)
        e_pos = goal - pos[0]
        q_err = quat_mul(np.asarray(self.down, dtype=float), quat_conj(quat[0]))
        e_rot = q_err[1:] * np.sign(q_err[0] if q_err[0] != 0 else 1.0)
        e = np.concatenate([e_pos, e_rot])
        JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJt)
        dq = J_pinv @ e
        posture = self.facing(goal[:2])
        dq += (np.eye(7) - J_pinv @ J) @ (NULLSPACE_GAIN * (posture - q))
        self.robot.set_dof_position_targets((q + dq).reshape(1, -1), dof_indices=list(range(7)))

    def hand_tilt_deg(self) -> float:
        """Angle between the hand's orientation and straight-down."""
        q = self.robot.get_current_state()[2][0]
        dot = abs(float(np.dot(q, np.asarray(self.down, dtype=float))))
        return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))

    def _rel(self, box: Box) -> np.ndarray:
        """Box position in the hand frame, so hand tilt is not mistaken for slip."""
        _, pos, quat = self.robot.get_current_state()
        return quat_rotate_inverse(quat[0], box.position() - pos[0])

    @staticmethod
    def _slip(delta: np.ndarray) -> float:
        """Slip magnitude ignoring the finger axis (box + fingers can float sideways together)."""
        return float(np.hypot(delta[0], delta[2]))

    def _watch_slip(self, moving: bool = False) -> None:
        box = self.carrying
        if box is None:
            return
        if self.suction is not None:
            # Seal fails when the load exceeds the vacuum's holding capacity (or the box moves relative to the cup).
            if (self.suction.seal_utilization(box) > 1.0
                    or np.linalg.norm(self._rel(box) - self._rel0) > SEAL_LOST_TOL):
                kpa = self.suction.vacuum
                self.suction.release()
                self.episode_slips += 1
                box.slip_events += 1
                self.learner.on_slip()
                print(f"    ! box {box.index}: seal lost during {self.stage} at {kpa:.1f} kPa "
                      f"(learned vacuum margin now {self.learner.grip_margin:.2f})")
                raise BoxDropped()
            return
        drift = self._slip(self._rel(box) - self._rel0)
        if drift > DROP_TOL:
            raise BoxDropped()
        tol = SLIP_MOVING_TOL if moving else SLIP_CARRY_TOL
        if drift > tol and self.gripper.force < GRIP_FORCE_MAX:
            old = self.gripper.force
            new = self.gripper.squeeze(old * 1.15)
            box.slip_events += 1
            box.grip_force = new
            self.episode_slips += 1
            self.learner.on_slip()
            self._rel0 = self._rel(box)
            print(f"    ! box {box.index}: slip {drift * 1000:.1f} mm during {self.stage} -> grip {old:.1f} N -> {new:.1f} N "
                  f"(learned grip margin now {self.learner.grip_margin:.2f})")

    def move(self, goal, steps: int, settle: int = 0, tol: float | None = None, max_wait: int = 120):
        """Move along a smooth path. With `tol`, keep holding after the path until the hand is really within
        `tol` metres of the goal (the arm's tracking lags the commanded path by up to a few centimetres)."""
        start = self.hand_pos()
        goal = np.asarray(goal, dtype=float)
        for k in range(1, steps + 1):
            target = start + (goal - start) * smoothstep(k / steps)
            self.ik(target)
            self._watch_slip(moving=True)
            yield
        if tol is not None:
            for _ in range(max_wait):
                if np.linalg.norm(self.hand_pos() - goal) < tol:
                    break
                self.ik(goal)
                self._watch_slip(moving=True)
                yield
        for i in range(settle):
            self.ik(goal)
            self._watch_slip(moving=i < settle - 3)
            yield

    def go_joints(self, q_goal: np.ndarray, steps: int = 70, settle: int = 15):
        """Joint-space move (slip-monitored if carrying)."""
        q0 = self.robot.get_dof_positions().numpy()[0][:7]
        self.q_cmd = None
        for k in range(1, steps + 1):
            q = q0 + (q_goal - q0) * smoothstep(k / steps)
            self.robot.set_dof_position_targets(q.reshape(1, -1), dof_indices=list(range(7)))
            self._watch_slip(moving=True)
            yield
        for i in range(settle):
            self.robot.set_dof_position_targets(q_goal.reshape(1, -1), dof_indices=list(range(7)))
            self._watch_slip(moving=i < settle - 3)
            yield
        self.q_cmd = None  # next IK step starts from the measured joints

    def go_ready(self, steps: int = 70):
        """Return to a fixed ready joint pose. The hand-only IK leaves the Franka's 7th (redundant) degree of
        freedom free, so it drifts over many moves; resetting each cycle keeps the arm well-conditioned."""
        yield from self.go_joints(READY_Q, steps)

    @staticmethod
    def facing(xy) -> np.ndarray:
        """The ready posture with the base turned to face `xy`. Joint 7 turns by the same angle so the hand
        keeps its orientation. Travelling between stations this way keeps the arm out of contorted postures
        (joint limits) that the hand-position IK can otherwise drift into on long Cartesian moves."""
        theta = float(np.arctan2(xy[1], xy[0]))
        q = READY_Q.copy()
        q[0] += theta
        q[6] += theta
        return q

    def hold(self, steps: int, goal=None):
        goal = self.hand_pos() if goal is None else np.asarray(goal, dtype=float)
        for _ in range(steps):
            self.ik(goal)
            self._watch_slip()
            yield

    def pick_xyz(self) -> np.ndarray:
        return np.array([PICK_SPOT[0], PICK_SPOT[1], WORK_SURFACE_Z + BOX_SIZE / 2.0 + 0.001])

    @staticmethod
    def slot_position(zone: str, n: int) -> np.ndarray:
        layer, cell = divmod(n, GRID * GRID)
        row, col = divmod(cell, GRID)
        xy = ZONES[zone] + SLOT_PITCH * np.array([col - 1, row - 1])
        z = WORK_SURFACE_Z + layer * LAYER_HEIGHT + BOX_SIZE / 2.0
        return np.array([xy[0], xy[1], z])

    @staticmethod
    def classify(mass: float) -> str:
        if mass < LIGHT_MAX:
            return "light"
        if mass < MEDIUM_MAX:
            return "medium"
        return "heavy"

    def choose_zone(self, category: str) -> str | None:
        """The box's own zone, or the nearest class with room if that cube is already full."""
        order = {"light": ["light", "medium", "heavy"], "medium": ["medium", "light", "heavy"],
                 "heavy": ["heavy", "medium", "light"]}[category]
        for z in order:
            if self.zone_count[z] < ZONE_CAPACITY:
                return z
        return None

    def reject(self, box: Box, status: str) -> None:
        """Move a box that could not be handled out of the pick spot to the reject lane."""
        box.status = status
        box.set_color("rejected")
        box.teleport([0.95 + 0.075 * self.reject_count, -0.6, WORK_SURFACE_Z + BOX_SIZE / 2 + 0.002])
        self.reject_count += 1

    # --- episode ---------------------------------------------------------------------------------

    def run_episode(self, episode: int):
        self.zone_count = {k: 0 for k in ZONES}
        self.episode_slips = 0
        self.reject_count = 0
        for b in self.boxes:
            b.reset_results()
            b.set_color("unknown")
            b.teleport(b.staging)
        self.gripper.configure_drives()
        if not getattr(self.robot, "_arm_stiffened", False):
            k, d = self.robot.get_dof_gains()
            k = k.numpy()[0][:7] * ARM_STIFFNESS_SCALE
            d = d.numpy()[0][:7] * np.sqrt(ARM_STIFFNESS_SCALE)
            self.robot.set_dof_gains(stiffnesses=k.reshape(1, -1), dampings=d.reshape(1, -1),
                                     dof_indices=list(range(7)))
            self.robot._arm_stiffened = True
        self.gripper.open()
        yield from self.go_ready()

        # Tare the wrist sensor once per episode, at the exact pose used for weighing (the pick spot is fixed).
        grasp = self.pick_xyz() + [0.0, 0.0, HAND_TO_GRASP]
        tare_pose = grasp + [0.0, 0.0, WEIGH_LIFT]
        yield from self.move([tare_pose[0], tare_pose[1], TRANSIT_Z], 80, settle=10)
        yield from self.move(tare_pose, 50, settle=30)
        self.sensor.start()
        for _ in range(30):
            self.ik(tare_pose)
            self.sensor.sample()
            yield
        self.sensor.tare_vec = self.sensor.average()
        yield from self.move([tare_pose[0], tare_pose[1], TRANSIT_Z], 30)

        t0 = time.time()
        for box in self.boxes:
            # Infeed: deliver the next box onto the pick spot while the arm returns to its ready pose.
            box.teleport(self.pick_xyz())
            self.gripper.open()
            yield from self.go_ready()
            try:
                yield from self.process(box)
            except BoxDropped:
                self.carrying = None
                print(f"    !! box {box.index} dropped during {self.stage}")
                self.let_go()
                yield from self.move(self.hand_pos() + [0, 0, 0.12], 40)
                self.reject(box, "dropped")
            done = sum(b.status != "waiting" for b in self.boxes)
            err = f"{box.place_error_mm:.1f} mm" if box.place_error_mm is not None else "-"
            print(f"  [ep {episode}] box {box.index:2d}  {done:2d}/{len(self.boxes)}  {box.status:<10} "
                  f"{box.category:<7} est {box.est_mass if box.est_mass else 0:.2f} kg (true {box.true_mass:.2f})  "
                  f"{self.grip_label(box)}  place error {err}")
        self.gripper.open()
        yield from self.move([0.35, 0.0, 0.45], 80, settle=20)
        self.summarize(episode, time.time() - t0)

    def let_go(self, finger_width: float = FINGER_OPEN) -> None:
        if self.suction is not None:
            self.suction.release()
        else:
            self.gripper.open(finger_width)

    def grip_label(self, box: Box) -> str:
        if self.suction is not None:
            if not box.grip_force:
                return "vacuum    -"
            peak = f", peak load {box.seal_peak * 100:3.0f}%" if box.seal_peak is not None else ""
            return f"vacuum {box.grip_force:4.1f} kPa ({SuctionGripper.hold_force(box.grip_force):5.1f} N hold{peak})"
        return f"grip {box.grip_force:4.1f} N"

    def _weigh_lift(self, grasp: np.ndarray, lift: np.ndarray):
        """Lift 6 cm and sample the wrist sensor once the lift has settled."""
        self.sensor.start()
        for k in range(75):
            frac = smoothstep(min(1.0, (k + 1) / 40))
            self.ik(grasp + (lift - grasp) * frac)
            if k >= 50:
                self.sensor.sample()
            yield

    def _suction_pick(self, box: Box, grasp: np.ndarray):
        """Seal at full vacuum and lift for weighing. Returns (grasp, lift) or None if the box was rejected."""
        for attempt in range(1, 4):
            box.probe_attempts = attempt
            if attempt > 1:
                grasp = box.position() + [0.0, 0.0, HAND_TO_GRASP]
                yield from self.move(grasp + [0.0, 0.0, 0.04], 25)
                yield from self.move(grasp, 25, settle=8, tol=0.002)
            off = float(np.linalg.norm(box.position()[:2] - self.hand_pos()[:2]))
            if off > SEAL_XY_TOL:
                self.log(f"    cup {off * 1000:.1f} mm off the box center - re-locating")
                continue
            self.suction.attach(box, PROBE_VACUUM_KPA)
            self.log(f"  sealed #{attempt} at {PROBE_VACUUM_KPA:.0f} kPa | cup offset {off * 1000:.1f} mm")
            yield from self.hold(10, goal=grasp)
            lift = grasp + [0.0, 0.0, WEIGH_LIFT]
            rel0 = self._rel(box)
            yield from self._weigh_lift(grasp, lift)
            if np.linalg.norm(self._rel(box) - rel0) <= SEAL_LOST_TOL:
                return grasp, lift
            self.log(f"    seal lost on the weighing lift (box moved {np.linalg.norm(self._rel(box) - rel0) * 1000:.0f} mm "
                     f"relative to the cup) - retrying")
            self.suction.release()
            yield from self.move(self.hand_pos() + [0.0, 0.0, 0.03], 25)
        self.reject(box, "unliftable")
        return None

    def _finger_pick(self, box: Box, grasp: np.ndarray):
        """Probe grasp with increasing force until the weighing lift doesn't slip. Returns (grasp, lift) or None."""
        probe = args.probe_force
        weighed = False
        lift = grasp + [0.0, 0.0, WEIGH_LIFT]
        for attempt in range(1, 6):
            box.probe_attempts = attempt
            if attempt > 1:
                pick = box.position()
                grasp = pick + [0.0, 0.0, HAND_TO_GRASP]
                lift = grasp + [0.0, 0.0, WEIGH_LIFT]
                yield from self.move(grasp + [0.0, 0.0, 0.04], 25)
                yield from self.move(grasp, 25, settle=8, tol=0.002)
            f = self.gripper.squeeze(probe)
            self.log(f"  probe grasp #{attempt}: {f:.1f} N per finger | box {np.round(box.position(), 3)} "
                     f"hand {np.round(self.hand_pos(), 3)} target {np.round(grasp, 3)}")
            yield from self.hold(25, goal=grasp)
            if self.gripper.width() < 0.5 * box.size:
                self.gripper.open()
                yield from self.move(grasp + [0.0, 0.0, 0.06], 30)
                if attempt == 5:
                    self.reject(box, "missed")
                    return None
                self.log("    fingers closed on nothing - re-locating box")
                continue
            rel0 = self._rel(box)
            yield from self._weigh_lift(grasp, lift)
            drift = self._slip(self._rel(box) - rel0)
            if drift <= SLIP_PROBE_TOL:
                weighed = True
                break
            self.log(f"    box slipped {drift * 1000:.1f} mm -> setting down, increasing force")
            yield from self.move(grasp, 25, settle=5)
            self.gripper.open()
            yield from self.hold(20, goal=grasp)
            yield from self.move(grasp + [0.0, 0.0, 0.04], 20)
            probe *= 1.6
        if not weighed:
            self.gripper.open()
            yield from self.move(grasp + [0.0, 0.0, 0.1], 30)
            self.reject(box, "unliftable")
            return None
        return grasp, lift

    def process(self, box: Box):
        pick = box.position()
        grasp = pick + [0.0, 0.0, HAND_TO_GRASP]
        self.let_go()
        yield from self.move([grasp[0], grasp[1], TRANSIT_Z], 70, settle=5)
        yield from self.move(grasp + [0.0, 0.0, 0.05], 30)
        yield from self.move(grasp, 30, settle=8, tol=0.002)

        # Grasp with a probe force / full vacuum and lift 6 cm to weigh.
        picked = yield from (self._suction_pick(box, grasp) if self.suction is not None
                             else self._finger_pick(box, grasp))
        if picked is None:
            return
        grasp, lift = picked

        measured, steady = self.sensor.mass_from_samples()
        for _ in range(3):
            if steady:
                break
            self.sensor.start()
            for _ in range(30):
                self.ik(lift)
                self.sensor.sample()
                yield
            measured, steady = self.sensor.mass_from_samples()
        if args.weight_source == "manifest" or not (steady and np.isfinite(measured) and 0.02 < measured < 10.0):
            box.est_mass = box.true_mass
        else:
            box.est_mass = measured

        if box.est_mass > ARM_PAYLOAD:
            yield from self.move(grasp, 30, settle=5)
            self.let_go()
            yield from self.hold(20, goal=grasp)
            yield from self.move(grasp + [0.0, 0.0, 0.1], 30)
            self.reject(box, "overweight")
            return

        category = self.classify(box.est_mass)
        zone = self.choose_zone(category)
        if zone is None:
            yield from self.move(grasp, 30, settle=5)
            self.let_go()
            yield from self.move(grasp + [0.0, 0.0, 0.1], 30)
            self.reject(box, "all full")
            return
        slot_n = self.zone_count[zone]
        slot = self.slot_position(zone, slot_n)
        correction = self.learner.correction(zone, slot_n)
        box.category = category
        box.set_color(category)

        # Where the box actually sits in the fingers (may differ from nominal after a probe slip).
        offset = box.position() - self.hand_pos()
        place = slot - offset - [0.0, 0.0, PLACE_PRESS]
        place[:2] += correction
        pre_place = place + [0.0, 0.0, 0.022]
        place_above = np.array([place[0], place[1], TRANSIT_Z])
        lift_top = np.array([lift[0], lift[1], TRANSIT_Z])
        # Heavier boxes are moved more slowly (like a real palletizer): durations scale with weight, which cuts
        # peak acceleration (and the swing that makes a heavy box pivot in a two-finger grip) by the square.
        # A suction cup holds rigidly, so it needs much less slow-down.
        slow = 1.0 + (0.25 if SUCTION else 0.55) * box.est_mass
        steps = {"lift": int(40 * slow), "transit": int(90 * slow), "lower": int(40 * slow)}
        a_max = max(
            peak_accel(TRANSIT_Z - lift[2], steps["lift"]),
            peak_accel(float(np.linalg.norm(place_above - lift_top)), steps["transit"]),
            peak_accel(TRANSIT_Z - pre_place[2], steps["lower"]),
        )
        if self.suction is not None:
            kpa = self.suction.required_vacuum(box.est_mass, a_max, self.learner.grip_margin)
            if args.weak_vacuum_test:
                kpa *= 0.05
            box.grip_force = self.suction.set_vacuum(kpa)
        else:
            box.grip_force = self.gripper.squeeze(
                self.gripper.required_force(box.est_mass, a_max, self.learner.grip_margin))
        self.log(f"  {box.est_mass:.3f} kg -> {category.upper()} -> {zone} slot {slot_n} | {self.grip_label(box)}")
        yield from self.hold(12)

        # Carry with slip monitoring.
        self.carrying = box
        self._rel0 = self._rel(box)
        self.stage = "lift"
        yield from self.move(lift_top, steps["lift"], settle=8)
        self.stage = "transit"
        # Travel between stations in joint space (base and wrist turn together, hand stays level), then a short
        # straight-line move to above the slot.
        yield from self.go_joints(self.facing(place_above[:2]), steps["transit"], settle=12)
        yield from self.move(place_above, int(35 * slow), settle=8)
        self.stage = "lower"
        yield from self.move(pre_place, steps["lower"], settle=12, tol=0.003)

        # Fine alignment: measure where the box really is and shift the hand to line it up with the target.
        self.stage = "align"
        target_xy = slot[:2] + correction
        for _ in range(ALIGN_PASSES):
            err = target_xy - box.position()[:2]
            pre_place[:2] += err
            yield from self.move(pre_place, int(15 * slow), settle=12, tol=0.002)
        place[:2] = pre_place[:2]
        self.stage = "place"
        if self.suction is not None:
            box.seal_peak = self.suction.peak_util
        self.carrying = None  # contact with the surface below moves the box in the fingers: not slip
        yield from self.move(place, int(25 * slow), settle=15, tol=0.002, max_wait=60)
        # Wait for the hand to be level so opening the fingers doesn't push the box sideways.
        for _ in range(90):
            if self.hand_tilt_deg() < MAX_RELEASE_TILT_DEG:
                break
            self.ik(place)
            yield

        before_release = box.position()[:2] - slot[:2]
        tilt_at_release = self.hand_tilt_deg()
        if args.verbose:
            qd = self.robot.get_dof_positions().numpy()[0][:7]
            q_tgt = self.robot.get_dof_position_targets().numpy()[0][:7]
            lims = self.robot.get_dof_limits()
            lo, hi = lims[0].numpy()[0][:7], lims[1].numpy()[0][:7]
            try:
                eff = self.robot.get_dof_projected_joint_forces().numpy()[0][:7]
            except Exception:
                eff = np.full(7, np.nan)
            maxe = self.robot.get_dof_max_efforts().numpy()[0][:7]
            print(f"    joints q {np.round(qd, 2)}\n           target-q {np.round(q_tgt - qd, 3)}\n"
                  f"           margin-to-limit {np.round(np.minimum(qd - lo, hi - qd), 2)}\n"
                  f"           torque {np.round(eff, 1)} / max {np.round(maxe, 0)}\n"
                  f"           hand pos {np.round(self.hand_pos(), 3)} target {np.round(place, 3)}")
        gap_below = (box.position()[2] - slot[2]) * 1000
        neighbours = [b for b in self.boxes if b.status == "stacked" and b.category == box.category]
        neigh_before = [b.position() for b in neighbours]

        # Release (fingers: small opening, neighboring boxes are 2 cm away; suction: vent the cup), then lift
        # straight up.
        self.let_go(FINGER_RELEASE)
        yield from self.hold(12 if SUCTION else 25, goal=place)
        yield from self.move(place + [0.0, 0.0, 0.08], 30)
        yield from self.move([place[0], place[1], TRANSIT_Z], 30)
        self.zone_count[zone] += 1
        box.status = "stacked"

        # Measure the placement and learn from it.
        pos, quat = box.pose()
        error_xy = pos[:2] - slot[:2]
        box.place_error_mm = float(np.linalg.norm(error_xy) * 1000)
        box.yaw_error_deg = yaw_error_deg(quat)
        knocked = max([np.linalg.norm(b.position() - p) for b, p in zip(neighbours, neigh_before)], default=0.0)
        hand_q = self.robot.get_current_state()[2][0]
        self.log(f"    place diag: slot {slot_n} err before release {np.round(before_release * 1000, 1)} mm, "
                 f"after {np.round(error_xy * 1000, 1)} mm, z err {(pos[2] - slot[2]) * 1000:.1f} mm, "
                 f"yaw {box.yaw_error_deg:.1f} deg, neighbours moved {knocked * 1000:.1f} mm, "
                 f"hand tilt at release {tilt_at_release:.1f} deg, box {gap_below:.1f} mm above rest")
        self.learner.update(zone, slot_n, error_xy)

    def summarize(self, episode: int, wall: float) -> None:
        stacked = [b for b in self.boxes if b.status == "stacked"]
        errs = np.array([b.place_error_mm for b in stacked]) if stacked else np.array([np.nan])
        yaws = np.array([b.yaw_error_deg for b in stacked]) if stacked else np.array([np.nan])
        weigh = [abs(b.est_mass - b.true_mass) / b.true_mass * 100 for b in self.boxes if b.est_mass]
        row = {
            "episode": episode,
            "practice_total": self.learner.episodes_trained,
            "boxes": len(self.boxes),
            "stacked": len(stacked),
            "mean_place_error_mm": round(float(np.nanmean(errs)), 2),
            "max_place_error_mm": round(float(np.nanmax(errs)), 2),
            "within_2mm_pct": round(float(np.mean(errs <= 2.0) * 100), 1) if stacked else 0.0,
            "mean_yaw_error_deg": round(float(np.nanmean(yaws)), 2),
            "mean_weigh_error_pct": round(float(np.mean(weigh)), 2) if weigh else float("nan"),
            "slips": self.episode_slips,
            "gripper": args.gripper,
            "mean_grip_setting": round(float(np.mean([b.grip_force for b in stacked])), 2) if stacked else 0.0,
            "grip_margin": round(self.learner.grip_margin, 3),
            "wall_seconds": round(wall, 1),
        }
        new_file = not TRAINING_LOG.exists()
        with TRAINING_LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print("\n" + "=" * 80)
        print(f"Episode {episode} (practice run #{self.learner.episodes_trained}): "
              f"{row['stacked']}/{row['boxes']} stacked | placement error mean {row['mean_place_error_mm']} mm, "
              f"max {row['max_place_error_mm']} mm, {row['within_2mm_pct']}% within 2 mm | "
              f"yaw {row['mean_yaw_error_deg']} deg | weighing error {row['mean_weigh_error_pct']}% | "
              f"slips {row['slips']} | grip margin {row['grip_margin']}")
        if SUCTION and stacked:
            print(f"Mean carry vacuum {row['mean_grip_setting']} kPa = "
                  f"{row['mean_grip_setting'] / VACUUM_MAX_KPA * 100:.0f}% of full vacuum "
                  f"(weight-adaptive regulation instead of always running at {VACUUM_MAX_KPA:.0f} kPa)")
        print("=" * 80 + "\n")
        return row

    def train(self):
        history = []
        for ep in range(1, max(1, args.episodes) + 1):
            print(f"\n######## Episode {ep}/{args.episodes} ########")
            yield from self.run_episode(ep)
            if self.episode_slips == 0:
                self.learner.on_clean_episode()
            self.learner.episodes_trained += 1
            self.learner.save()
            history.append(self.last_row())
        print("Training summary (this launch):")
        print(f"{'episode':>7} {'stacked':>8} {'mean mm':>8} {'max mm':>7} {'<=2mm %':>8} {'slips':>6} {'margin':>7}")
        for h in history:
            print(f"{h['episode']:>7} {h['stacked']:>8} {h['mean_place_error_mm']:>8} {h['max_place_error_mm']:>7} "
                  f"{h['within_2mm_pct']:>8} {h['slips']:>6} {h['grip_margin']:>7}")
        print(f"Learned corrections saved to {LEARNING_FILE}")
        print(f"Per-episode log appended to {TRAINING_LOG}")

    def last_row(self) -> dict:
        with TRAINING_LOG.open() as f:
            return list(csv.DictReader(f))[-1]


# 6. Scene -----------------------------------------------------------------------------------------


def add_suction_cup_visual(hand_path: str) -> None:
    """A vacuum tool between the open fingers: a stem and a black bellows cup whose lip is CUP_REACH below the
    hand frame (the hand's +z axis points along the fingers). Visual only - the seal itself is a joint."""
    try:
        from pxr import Gf, UsdGeom

        stage = stage_utils.get_current_stage()
        parts = [("Stem", 0.012, 0.050, 0.091, (0.55, 0.57, 0.60)),
                 ("Bellows", CUP_RADIUS, 0.014, CUP_REACH - 0.007, (0.05, 0.05, 0.06))]
        for name, radius, height, z, color in parts:
            cyl = UsdGeom.Cylinder.Define(stage, f"{hand_path}/SuctionCup_{name}")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr("Z")
            cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(0.0, 0.0, z))
    except Exception as e:
        print(f"Could not add the suction cup visual: {e}")


def build_scene() -> tuple[Franka, list[Box]]:
    assets = get_assets_root_path()
    if args.env == "warehouse":
        env_path = assets + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    elif args.env == "grid":
        env_path = None
    else:
        env_path = args.env
    if env_path:
        print(f"Loading environment: {env_path}")
        env_prim = stage_utils.add_reference_to_stage(usd_path=env_path, path="/World/Environment")
        if any(args.env_offset):
            from pxr import Gf, UsdGeom

            UsdGeom.XformCommonAPI(env_prim).SetTranslate(Gf.Vec3d(*args.env_offset))
    else:
        GroundPlane("/World/ground_plane")
        DomeLight("/World/DomeLight").set_intensities(1000)

    # Work cell floor mat (static collider) covering the robot, zones, and infeed staging area.
    mat = Cube(paths="/World/Cell/FloorMat", positions=[0.5, 0.0, WORK_SURFACE_Z - 0.01], sizes=1.0,
               scales=[2.4, 1.9, 0.02], colors="#2e3338")
    GeomPrim(paths=mat.paths, apply_collision_apis=True)
    Cube(paths="/World/Cell/PickPad", positions=[PICK_SPOT[0], PICK_SPOT[1], WORK_SURFACE_Z + 0.0005], sizes=1.0,
         scales=[0.12, 0.12, 0.001], colors="steelblue")
    Cube(paths="/World/Cell/Infeed", positions=[1.25, 0.0, WORK_SURFACE_Z + 0.0005], sizes=1.0,
         scales=[0.72, 0.72, 0.001], colors="slategray")
    for name, xy in ZONES.items():
        Cube(paths=f"/World/Cell/Zone_{name}", positions=[xy[0], xy[1], WORK_SURFACE_Z + 0.0005], sizes=1.0,
             scales=[0.25, 0.25, 0.001], colors=COLORS[name])

    robot = Franka(robot_path="/World/robot", create_robot=True)
    robot.set_world_poses(positions=[0.0, 0.0, 0.0], orientations=[1.0, 0.0, 0.0, 0.0])
    if SUCTION:
        add_suction_cup_visual(robot.end_effector_link.paths[0])

    cardboard = RigidBodyMaterial("/World/Physics/Cardboard", static_frictions=args.friction,
                                  dynamic_frictions=args.friction, restitutions=0.0)
    cardboard.set_combine_modes(frictions="max")

    # Balanced weight classes (at most 27 per class, so every cube can be filled), random order.
    rng = np.random.default_rng(args.seed)
    n = int(np.clip(args.num_boxes, 1, 3 * ZONE_CAPACITY))
    classes = [list(CLASSES)[i % 3] for i in range(n)]
    rng.shuffle(classes)
    boxes = []
    for i, c in enumerate(classes):
        lo, hi = CLASSES[c]
        mass = float(rng.uniform(lo, hi))
        row, col = divmod(i, 9)
        staging = np.array([0.95 + 0.075 * row, -0.30 + 0.075 * col, WORK_SURFACE_Z + BOX_SIZE / 2 + 0.001])
        b = Box(i, f"/World/Boxes/Box_{i:02d}", staging, mass)
        b.geom.apply_physics_materials(cardboard)
        boxes.append(b)
    counts = {c: classes.count(c) for c in CLASSES}
    print(f"Spawned {n} boxes (hidden weights): {counts}")

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[2.0, 1.6, 1.5], target=[0.45, 0.0, 0.1], camera_prim_path="/OmniverseKit_Persp")
    except Exception:
        pass
    return robot, boxes


# 7. Main loop -------------------------------------------------------------------------------------


def main() -> None:
    SimulationManager.set_physics_sim_device(args.device)
    SimulationManager.setup_simulation(dt=PHYSICS_DT)
    simulation_app.update()

    robot, boxes = build_scene()
    learner = Learner(enabled=not args.no_learning, reset=args.reset_learning)
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

    task = None
    done = False
    frame = 0
    stage = stage_utils.get_current_stage()

    def clear_seals() -> None:
        """Suction seals are joints created while running. Stop rewinds the robot and boxes but keeps prims added
        during the run, so a seal on a box being carried would yank them together on the next Play."""
        if stage.GetPrimAtPath("/World/Suction").IsValid():
            stage.RemovePrim("/World/Suction")

    while simulation_app.is_running():
        frame += 1
        if args.selftest_stop and frame == args.selftest_stop:
            timeline.stop()
            simulation_app.update()
        if args.selftest_stop and frame == args.selftest_stop + 30:
            timeline.play()
        # Stop rewinds the scene to its pre-Play state: cancel the run; Play starts fresh (learning is kept).
        if timeline.is_stopped() and (task is not None or done):
            print("\nSimulation stopped - run cancelled. Press Play to start again (learned corrections are kept).")
            task, done = None, False
            for b in boxes:
                b.set_color("unknown")
            clear_seals()
        # Step the controller only while the timeline is actually playing (not paused or stopped).
        if timeline.is_playing() and SimulationManager.is_simulating() and not done:
            if task is None:
                clear_seals()
                robot.reset_to_default_pose()
                task = Palletizer(robot, boxes, learner).train()
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
