"""Shared UR20 palletizing logic, used by both front ends:

  * pallet_builder.py         - one cell, Isaac Sim standalone (python.bat)
  * isaaclab_pallet_cells.py  - several cells in the warehouse, Isaac Lab (isaaclab.bat -p)

The front end supplies a robot and boxes with a small common interface (see UR20Base and BoxBase) and steps the
Palletizer generator once per physics step. Everything here works in one cell's frame: `origin` shifts the whole
cell, so several cells can run side by side.

Import this only after the simulator app has started (it uses pxr).
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics

import pallet_planner as pp

# --- constants (cell frame: robot base on the floor at the origin) -----------------------------------------------

PHYSICS_DT = 1.0 / 60.0
G = 9.81

# Payload. The UR20 is rated for 20 kg at the flange, and the vacuum gripper itself weighs 4 kg, so the heaviest box
# it may carry is 16 kg.
PAYLOAD_RATED_KG = 20.0
TOOL_MASS = 4.0
MAX_BOX_KG = PAYLOAD_RATED_KG - TOOL_MASS
WEIGH_TOLERANCE_KG = 0.2              # a box weighed at up to MAX_BOX_KG + 0.2 kg is still accepted

PEDESTAL_H = 0.70                     # robot base height (palletizing column)
PALLET_XY = np.array([0.88, 0.0])     # pallet centre (far corners ~1.24 m out: well inside reach)
DECK_Z = pp.PALLET_HEIGHT             # top of the pallet deck boards
CONVEYOR_TOP = 0.70
PICK_XY = np.array([0.0, 0.92])       # end of the infeed conveyor
TRANSIT_Z = 1.25                      # TCP height while travelling (clears a full 3-layer stack + the carried box)
READY_TCP = np.array([0.75, 0.0, TRANSIT_Z])
SEED_Q = np.array([0.0, -1.2, 1.6, -1.97, -1.5708, 0.0])   # UR "tool down" posture used to start the IK

# Vacuum gripper: 4 cups of 50 mm on a 22 cm plate under the flange.
TOOL_REACH = 0.14                     # flange to cup lips
CUP_RADIUS = 0.025
CUP_OFFSET = 0.07                     # cup centres at (+-7, +-7) cm
CUP_AREA = 4 * np.pi * CUP_RADIUS ** 2   # 78.5 cm^2 -> 628 N at 80 kPa
VACUUM_MIN_KPA, VACUUM_MAX_KPA = 8.0, 80.0
PROBE_VACUUM_KPA = 80.0
CUP_SHEAR_RATIO = 0.5
SEAL_XY_TOL = 0.02
SEAL_LOST_TOL = 0.008
WEIGH_LIFT = 0.08
PLACE_GAP = 0.002                     # release this far above the resting position
ALIGN_PASSES = 2

IK_DAMPING = 0.05
WRIST_GAINS = ([2.0e5, 1.0e5, 5.0e4], [1.0e3, 6.0e2, 3.0e2])   # the asset's wrist servos are soft; stiffen them
GRIP_MARGIN_MIN, GRIP_MARGIN_MAX = 0.6, 2.0
LEARN_GAIN = 0.6
MAX_CORRECTION = 0.015

Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])   # flange z pointing down, flange x along world x (w, x, y, z)


# --- small math helpers (quaternions are w, x, y, z) ----------------------------------------------------------------


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def peak_accel(distance: float, steps: int) -> float:
    T = steps * PHYSICS_DT
    return 6.0 * distance / (T * T)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_quat(yaw: float):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_of(q) -> float:
    R = quat_to_rot(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class BoxDropped(Exception):
    pass


def box_color_rgb(spec: pp.BoxSpec) -> tuple[float, float, float]:
    """Cardboard tan for light boxes to dark brown for the heaviest; pale for fragile goods."""
    if spec.fragile:
        return (0.85, 0.78, 0.66)
    t = min(1.0, spec.mass / MAX_BOX_KG)
    lo, hi = np.array([0.80, 0.64, 0.42]), np.array([0.42, 0.27, 0.14])
    return tuple(float(c) for c in lo + (hi - lo) * t)


def box_color(spec: pp.BoxSpec) -> str:
    r, g, b = box_color_rgb(spec)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class Settings:
    safety_factor: float = 2.0
    episodes: int = 1
    weight_source: str = "wrist"       # 'wrist' or 'manifest'
    optimized: bool = True
    verbose: bool = False
    max_boxes: int = 0                 # testing: stop each episode after N boxes (0 = all)
    training_log: Path | None = None


# --- interfaces the front ends implement --------------------------------------------------------------------------


class UR20Base:
    """Robot interface. Front ends implement: q(), flange_pose() -> (pos, quat wxyz), jacobian() -> 6x6 for the
    flange link (world frame), set_targets(q), flange_force() -> 3-vector, joint_load() -> fraction of the torque
    limit, configure(), and the attribute flange_path (USD path of wrist_3_link). Values must be copies."""

    flange_path: str = ""

    def tcp_pose(self):
        p, q = self.flange_pose()
        return p + quat_to_rot(q) @ np.array([0.0, 0.0, TOOL_REACH]), q

    def ik(self, tcp_goal, yaw_goal: float) -> float:
        """One damped-least-squares step toward a TCP position with the tool pointing down at `yaw_goal`.
        Returns the remaining position error."""
        q = self.q()
        p_f, q_f = self.flange_pose()
        r = quat_to_rot(q_f) @ np.array([0.0, 0.0, TOOL_REACH])
        J = self.jacobian()
        J_tcp = np.vstack([J[:3] - skew(r) @ J[3:], J[3:]])     # velocity of the TCP, not the flange origin
        e_pos = np.asarray(tcp_goal, dtype=float) - (p_f + r)
        q_err = quat_mul(quat_mul(yaw_quat(yaw_goal), Q_DOWN), quat_conj(q_f))
        e_rot = 2.0 * q_err[1:] * (1.0 if q_err[0] >= 0 else -1.0)
        n = np.linalg.norm(e_pos)
        e = np.concatenate([e_pos * min(1.0, 0.05 / max(n, 1e-9)), np.clip(e_rot, -0.15, 0.15)])
        dq = J_tcp.T @ np.linalg.solve(J_tcp @ J_tcp.T + IK_DAMPING ** 2 * np.eye(6), e)
        self.set_targets(q + dq)
        return float(n)


class BoxBase:
    """Box interface. Front ends set spec, path, size (L, W, H), staging (world) and implement pose() ->
    (pos, quat wxyz), teleport(pos, yaw) and lin_vel()."""

    def reset_results(self) -> None:
        self.status = "waiting"
        self.est_mass: float | None = None
        self.vacuum = 0.0
        self.seal_peak: float | None = None
        self.joint_peak: float | None = None
        self.place_error_mm: float | None = None
        self.yaw_error_deg: float | None = None

    def position(self):
        return self.pose()[0]


# --- learning -----------------------------------------------------------------------------------------------------


class Learner:
    """Per-spot placement corrections and the vacuum margin, learned from practice and saved to `path`."""

    def __init__(self, path: Path, enabled: bool, reset: bool) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.slots: dict[str, list[float]] = {}
        self.margin = 1.0
        self.episodes_trained = 0
        if enabled and not reset and self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.slots, self.margin = d.get("slots", {}), float(d.get("vacuum_margin", 1.0))
                self.episodes_trained = int(d.get("episodes_trained", 0))
                print(f"Loaded learning: {len(self.slots)} spots, vacuum margin {self.margin:.2f}, "
                      f"{self.episodes_trained} practice runs so far")
            except Exception as e:
                print(f"Could not read {self.path.name} ({e}); starting fresh")

    def correction(self, key: str) -> np.ndarray:
        return np.array(self.slots[key]) if self.enabled and key in self.slots else np.zeros(2)

    def update(self, key: str, error_xy: np.ndarray) -> None:
        if self.enabled:
            new = self.correction(key) - LEARN_GAIN * error_xy
            self.slots[key] = np.clip(new, -MAX_CORRECTION, MAX_CORRECTION).tolist()

    def on_seal_lost(self) -> None:
        if self.enabled:
            self.margin = min(GRIP_MARGIN_MAX, self.margin * 1.08)

    def on_clean_episode(self) -> None:
        if self.enabled:
            self.margin = max(GRIP_MARGIN_MIN, self.margin * 0.97)

    def save(self) -> None:
        if self.enabled:
            self.path.write_text(json.dumps({"episodes_trained": self.episodes_trained,
                                             "vacuum_margin": self.margin, "slots": self.slots}, indent=1))


# --- gripper and sensor ------------------------------------------------------------------------------------------


class VacuumGripper:
    """4-cup vacuum gripper. The seal is a fixed joint between the flange and the box top, created when the cups
    touch the box and removed on release; a per-step load check decides when the seal fails."""

    SEAL_ROOT = "/World/Vacuum"

    def __init__(self, robot: UR20Base, safety_factor: float, stage, tag: str = "") -> None:
        self.robot = robot
        self.sf = safety_factor
        self.stage = stage
        self.tag = tag
        self.joint = None
        self.vacuum = 0.0
        self._n = 0
        self._v_prev = None
        self._util = 0.0
        self.peak_util = 0.0

    @staticmethod
    def hold_force(kpa: float) -> float:
        return kpa * 1000.0 * CUP_AREA

    def required_vacuum(self, mass: float, accel: float, margin: float) -> float:
        need = margin * self.sf * mass * (G + accel / CUP_SHEAR_RATIO)
        return float(np.clip(need / CUP_AREA / 1000.0, VACUUM_MIN_KPA, VACUUM_MAX_KPA))

    def attach(self, box: BoxBase, kpa: float) -> None:
        self.release()
        p_f, q_f = self.robot.flange_pose()
        b_p, b_q = box.pose()
        p_rel = quat_to_rot(q_f).T @ (b_p - p_f)
        q_rel = quat_mul(quat_conj(q_f), b_q)
        q_rel /= np.linalg.norm(q_rel)
        joint = UsdPhysics.FixedJoint.Define(self.stage, f"{self.SEAL_ROOT}/Seal{self.tag}_{self._n}")
        self._n += 1
        joint.CreateBody0Rel().SetTargets([self.robot.flange_path])
        joint.CreateBody1Rel().SetTargets([box.path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in p_rel]))
        joint.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in q_rel]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr(True)
        self.joint = joint
        self.set_vacuum(kpa)

    def set_vacuum(self, kpa: float) -> float:
        self.vacuum = float(np.clip(kpa, 0.0, VACUUM_MAX_KPA))
        self._v_prev, self._util, self.peak_util = None, 0.0, 0.0
        return self.vacuum

    def seal_utilization(self, box: BoxBase) -> float:
        """Fraction of the cups' capacity in use (> 1: the seal fails). The cups must supply F = m (a + g)."""
        v = box.lin_vel()
        if self._v_prev is None:
            self._v_prev = v
            return self._util
        accel = (v - self._v_prev) / PHYSICS_DT
        self._v_prev = v
        f = box.spec.mass * (accel + np.array([0.0, 0.0, G]))
        hold = self.hold_force(self.vacuum)
        u = max(f[2], 0.0) / hold + float(np.hypot(f[0], f[1])) / (CUP_SHEAR_RATIO * hold)
        self._util = 0.5 * self._util + 0.5 * u
        self.peak_util = max(self.peak_util, self._util)
        return self._util

    def release(self) -> None:
        if self.joint is not None:
            self.stage.RemovePrim(self.joint.GetPath())
            self.joint = None
        self.vacuum = 0.0

    @classmethod
    def clear_all(cls, stage) -> None:
        """Remove every seal. Stop rewinds the robot and boxes but keeps prims added during the run, so a seal on a
        box being carried would yank the rewound robot and box together on the next Play."""
        if stage.GetPrimAtPath(cls.SEAL_ROOT).IsValid():
            stage.RemovePrim(cls.SEAL_ROOT)


class WristFT:
    """The UR20's flange force/torque sensor (incoming joint force of wrist_3_link)."""

    def __init__(self, robot: UR20Base) -> None:
        self.robot = robot
        self.tare: np.ndarray | None = None
        self.samples: list[np.ndarray] = []

    def read(self) -> np.ndarray:
        return self.robot.flange_force()

    def mass(self) -> tuple[float, bool]:
        if self.tare is None or len(self.samples) < 6:
            return float("nan"), False
        s = np.array(self.samples)
        t = np.linalg.norm(self.tare)
        m = lambda blk: (np.linalg.norm(np.median(blk, axis=0)) - t) / G   # noqa: E731
        half = len(s) // 2
        m_all, m_a, m_b = m(s), m(s[:half]), m(s[half:])
        return float(m_all), bool(abs(m_a - m_b) <= 0.02 * max(m_all, 0.5))


class ToolVisual:
    """The vacuum gripper's look. The UR20 asset's links are instanced (no child prims can be added), so the tool
    is its own prim that follows the flange every frame; its mass is added to the flange link separately."""

    def __init__(self, robot: UR20Base, stage, path: str) -> None:
        self.robot = robot
        self.stage = stage
        xf = UsdGeom.Xform.Define(stage, path)
        self.t_op = xf.AddTranslateOp()
        self.o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        parts = [("Stem", "cyl", 0.04, 0.06, (0, 0, 0.03), (0.6, 0.62, 0.65)),
                 ("Plate", "box", (0.22, 0.22, 0.02), None, (0, 0, 0.07), (0.2, 0.22, 0.25))]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append((f"Cup_{sx + 1}{sy + 1}", "cyl", CUP_RADIUS, 0.06,
                              (sx * CUP_OFFSET, sy * CUP_OFFSET, 0.11), (0.05, 0.05, 0.06)))
        for name, kind, a, b, pos, color in parts:
            if kind == "cyl":
                g = UsdGeom.Cylinder.Define(stage, f"{path}/Vacuum_{name}")
                g.CreateRadiusAttr(a)
                g.CreateHeightAttr(b)
                g.CreateAxisAttr("Z")
            else:
                g = UsdGeom.Cube.Define(stage, f"{path}/Vacuum_{name}")
                g.CreateSizeAttr(1.0)
                UsdGeom.XformCommonAPI(g).SetScale(Gf.Vec3f(*a))
            g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.XformCommonAPI(g).SetTranslate(Gf.Vec3d(*pos))

    def update(self, live: bool = True) -> None:
        """Follow the flange: from physics while running, from the (rewound) USD pose while stopped."""
        try:
            if live:
                p, q = self.robot.flange_pose()
            else:
                m = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robot.flange_path))
                t, r = m.ExtractTranslation(), m.ExtractRotationQuat()
                p, q = list(t), [r.GetReal(), *r.GetImaginary()]
            self.t_op.Set(Gf.Vec3d(*[float(v) for v in p]))
            self.o_op.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        except Exception:
            pass


# --- controller (a generator: one yield == one physics step) -------------------------------------------------------


class Palletizer:
    def __init__(self, robot: UR20Base, boxes: dict[int, BoxBase], plan: pp.Plan, learner: Learner,
                 settings: Settings, stage, origin=(0.0, 0.0, 0.0), name: str = "") -> None:
        self.robot = robot
        self.boxes = boxes
        self.plan = plan
        self.learner = learner
        self.cfg = settings
        self.o = np.asarray(origin, dtype=float)          # cell origin in the world (floor below the robot base)
        self.name = name
        self.prefix = f"[{name}] " if name else ""
        self.gripper = VacuumGripper(robot, settings.safety_factor, stage, tag=name.replace(" ", ""))
        self.sensor = WristFT(robot)
        self.carrying: BoxBase | None = None
        self.rel0 = np.zeros(3)
        self.cmd_yaw = 0.0
        self.stage = ""
        self.ready_q: np.ndarray | None = None
        self.seal_losses = 0
        self.reject_count = 0
        self.last_row: dict = {}
        self.pick = self.o + np.array([PICK_XY[0], PICK_XY[1], 0.0])
        self.ready_tcp = self.o + READY_TCP
        self.pallet_xy = self.o[:2] + PALLET_XY

    def log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(self.prefix + msg)

    # motion -------------------------------------------------------------------------------------------------------

    def tcp(self) -> np.ndarray:
        return self.robot.tcp_pose()[0]

    def tool_yaw(self) -> float:
        return yaw_of(quat_mul(self.robot.flange_pose()[1], quat_conj(Q_DOWN)))

    def _rel(self, box: BoxBase) -> np.ndarray:
        p_f, q_f = self.robot.flange_pose()
        return quat_to_rot(q_f).T @ (box.position() - p_f)

    def watch(self) -> None:
        box = self.carrying
        if box is None:
            return
        box.joint_peak = max(box.joint_peak or 0.0, self.robot.joint_load())
        util = self.gripper.seal_utilization(box)
        moved = float(np.linalg.norm(self._rel(box) - self.rel0))
        if util > 1.0 or moved > SEAL_LOST_TOL:
            kpa = self.gripper.vacuum
            self.log(f"    seal diag: load {util * 100:.0f}% of capacity, box moved {moved * 1000:.1f} mm in the cups, "
                     f"TCP {np.round(self.tcp() - self.o, 3)}, q {np.round(self.robot.q(), 2)}")
            self.gripper.release()
            self.seal_losses += 1
            self.learner.on_seal_lost()
            print(f"{self.prefix}    ! box #{box.spec.id:02d}: seal lost during {self.stage} at {kpa:.1f} kPa "
                  f"(learned vacuum margin now {self.learner.margin:.2f})")
            raise BoxDropped()

    def move(self, goal, steps: int, yaw: float | None = None, settle: int = 0, tol: float | None = None,
             max_wait: int = 150):
        start = self.tcp()
        goal = np.asarray(goal, dtype=float)
        yaw0 = self.cmd_yaw
        dyaw = 0.0 if yaw is None else wrap(yaw - yaw0)
        for k in range(1, steps + 1):
            s = smoothstep(k / steps)
            self.robot.ik(start + (goal - start) * s, yaw0 + dyaw * s)
            self.watch()
            yield
        self.cmd_yaw = yaw0 + dyaw
        if tol is not None:
            for _ in range(max_wait):
                if np.linalg.norm(self.tcp() - goal) < tol:
                    break
                self.robot.ik(goal, self.cmd_yaw)
                self.watch()
                yield
        for _ in range(settle):
            self.robot.ik(goal, self.cmd_yaw)
            self.watch()
            yield

    def go_joints(self, q_goal, steps: int, settle: int = 15):
        q0 = self.robot.q()
        for k in range(1, steps + 1):
            self.robot.set_targets(q0 + (q_goal - q0) * smoothstep(k / steps))
            self.watch()
            yield
        for _ in range(settle):
            self.robot.set_targets(q_goal)
            self.watch()
            yield
        self.cmd_yaw = self.tool_yaw()

    def hold(self, steps: int, goal=None):
        goal = self.tcp() if goal is None else np.asarray(goal, dtype=float)
        for _ in range(steps):
            self.robot.ik(goal, self.cmd_yaw)
            self.watch()
            yield

    def _angle(self, xy) -> float:
        return float(np.arctan2(xy[1] - self.o[1], xy[0] - self.o[0]))

    def facing(self, xy, yaw: float | None = None) -> np.ndarray:
        """The ready posture with the base turned toward world point `xy` (the arm swings in joint space, tool
        level); wrist 3 turns so the tool ends at `yaw` (or keeps the ready yaw)."""
        theta = self._angle(xy)
        q = self.ready_q.copy()
        q[0] += theta
        q[5] += self.yaw_per_base * theta              # keeps the tool at the ready yaw
        if yaw is not None:
            q[5] += wrap(yaw - self.ready_yaw) / self.yaw_per_q5
        return q

    def facing_tcp(self, xy) -> np.ndarray:
        """Where the TCP is (world xy) at facing(xy)."""
        theta = self._angle(xy)
        r = float(np.linalg.norm(READY_TCP[:2]))
        return self.o[:2] + np.array([r * np.cos(theta), r * np.sin(theta)])

    def calibrate(self):
        """Find the ready posture by IK from the UR tool-down seed, and how base and wrist-3 turns change the
        tool's yaw (from the Jacobian)."""
        yield from self.go_joints(SEED_Q, 60, settle=20)
        self.cmd_yaw = self.tool_yaw()
        err = float("nan")
        for _ in range(300):
            err = self.robot.ik(self.ready_tcp, self.cmd_yaw)
            yield
        self.ready_q = self.robot.q()
        self.ready_yaw = self.tool_yaw()
        J = self.robot.jacobian()
        self.yaw_per_q5 = float(J[5, 5])                       # d(tool yaw)/d(q5)
        self.yaw_per_base = -float(J[5, 0]) / self.yaw_per_q5  # q5 turn per base turn that keeps the yaw
        print(f"{self.prefix}Ready posture {np.round(self.ready_q, 3)} (TCP error {err * 1000:.1f} mm, tool yaw "
              f"{math.degrees(self.ready_yaw):.0f} deg)")

    # episode ------------------------------------------------------------------------------------------------------

    def world_target(self, pl: pp.Placement) -> np.ndarray:
        return np.array([self.pallet_xy[0] + pl.x, self.pallet_xy[1] + pl.y, self.o[2] + DECK_Z + pl.z])

    def run_episode(self, episode: int):
        self.seal_losses = 0
        self.reject_count = 0
        for b in self.boxes.values():
            b.reset_results()
            b.teleport(b.staging)
        self.robot.configure()
        if self.ready_q is None:
            yield from self.calibrate()
        yield from self.go_joints(self.facing(self.pick[:2]), 90)
        # Tare the wrist sensor (tool down, empty, still).
        self.sensor.samples = []
        tare_goal = self.tcp()
        for k in range(60):
            self.robot.ik(tare_goal, self.cmd_yaw)
            if k >= 30:
                self.sensor.samples.append(self.sensor.read())
            yield
        self.sensor.tare = np.median(np.array(self.sensor.samples), axis=0)

        t0 = time.time()
        placements = self.plan.placements
        if self.cfg.max_boxes:
            placements = placements[:self.cfg.max_boxes]
        for pl in placements:
            box = self.boxes[pl.box.id]
            h = box.size[2]
            box.teleport(self.pick + [0.0, 0.0, CONVEYOR_TOP + h / 2 + 0.002])   # the conveyor delivers it
            yield from self.go_joints(self.facing(self.pick[:2]), 60)
            try:
                yield from self.place_one(box, pl)
            except BoxDropped:
                self.carrying = None
                self.gripper.release()
                print(f"{self.prefix}    !! box #{box.spec.id:02d} dropped during {self.stage}")
                yield from self.move(self.tcp() + [0, 0, 0.15], 40)
                self.reject(box, "dropped")
            err = f"{box.place_error_mm:5.1f} mm" if box.place_error_mm is not None else "    -   "
            seal = f"peak seal load {box.seal_peak * 100:3.0f}%" if box.seal_peak is not None else ""
            print(f"{self.prefix}  [ep {episode}] {pl.order + 1:2d}/{len(placements)} box #{box.spec.id:02d} "
                  f"{box.size[0] * 100:4.1f}x{box.size[1] * 100:4.1f}x{h * 100:4.1f} cm  {box.status:<8} "
                  f"layer {pl.layer + 1}  weighed {box.est_mass or 0:5.2f} kg (true {box.spec.mass:5.2f})  "
                  f"vacuum {box.vacuum:4.1f} kPa  {seal}  max joint load {(box.joint_peak or 0) * 100:3.0f}%  "
                  f"place error {err}")
        self.gripper.release()
        yield from self.go_joints(self.facing(self.ready_tcp[:2]), 90)
        # Let the finished pallet stand for 2 s and check nothing moved.
        before = {i: b.position() for i, b in self.boxes.items() if b.status == "placed"}
        for _ in range(120):
            yield
        settle = max([np.linalg.norm(self.boxes[i].position() - p) for i, p in before.items()], default=0.0)
        self.summarize(episode, time.time() - t0, settle)

    def reject(self, box: BoxBase, status: str) -> None:
        box.status = status
        box.teleport(self.o + [-0.9, -1.2 - 0.7 * self.reject_count, box.size[2] / 2 + 0.002])
        self.reject_count += 1

    def place_one(self, box: BoxBase, pl: pp.Placement):
        h = box.size[2]
        top = box.position() + [0.0, 0.0, h / 2]
        transit_z = self.o[2] + TRANSIT_Z
        above = np.array([top[0], top[1], transit_z])
        self.stage = "approach"
        yield from self.move(above, 50, settle=5)
        yield from self.move(top + [0, 0, 0.06], 45)
        yield from self.move(top + [0, 0, 0.001], 35, settle=8, tol=0.002)

        # Seal at full vacuum and lift to weigh.
        off = float(np.linalg.norm(box.position()[:2] - self.tcp()[:2]))
        if off > SEAL_XY_TOL:
            self.reject(box, "missed")
            return
        self.gripper.attach(box, PROBE_VACUUM_KPA)
        yield from self.hold(10)
        grasp = self.tcp()
        lift = grasp + [0.0, 0.0, WEIGH_LIFT]
        self.sensor.samples = []
        for k in range(100):
            self.robot.ik(grasp + (lift - grasp) * smoothstep(min(1.0, (k + 1) / 45)), self.cmd_yaw)
            if k >= 70:
                self.sensor.samples.append(self.sensor.read())
            yield
        measured, steady = self.sensor.mass()
        for _ in range(3):
            if steady:
                break
            self.sensor.samples = []
            for _ in range(40):
                self.robot.ik(lift, self.cmd_yaw)
                self.sensor.samples.append(self.sensor.read())
                yield
            measured, steady = self.sensor.mass()
        box.est_mass = measured if (steady and np.isfinite(measured)) else box.spec.measured_mass
        if abs(box.est_mass - box.spec.measured_mass) > 0.05 * box.spec.measured_mass + 0.1:
            print(f"{self.prefix}    ? box #{box.spec.id:02d}: wrist weighs {box.est_mass:.2f} kg, checkweigher said "
                  f"{box.spec.measured_mass:.2f} kg")
        mass = box.est_mass if self.cfg.weight_source == "wrist" else box.spec.measured_mass
        if mass + TOOL_MASS > PAYLOAD_RATED_KG + WEIGH_TOLERANCE_KG:
            print(f"{self.prefix}    ! box #{box.spec.id:02d}: {mass:.1f} kg + {TOOL_MASS:.0f} kg gripper is over the "
                  f"UR20's {PAYLOAD_RATED_KG:.0f} kg payload - set it back down")
            yield from self.move(grasp, 40, settle=5)
            self.gripper.release()
            yield from self.move(grasp + [0, 0, 0.1], 30)
            self.reject(box, "too heavy")
            return

        # Where the box sits under the cups, and where it must go.
        offset = box.position() - self.tcp()
        target = self.world_target(pl)
        key = f"{pl.layer}:{pl.slot}"
        corr = self.learner.correction(key)
        delta = wrap(yaw_of(box.pose()[1]) - self.tool_yaw())          # box yaw relative to the tool
        sym = np.pi / 2 if pl.box.shape[0] == pl.box.shape[1] else np.pi
        base_yaw = pl.yaw - delta
        k = round((self.tool_yaw() - base_yaw) / sym)
        tool_yaw = wrap(base_yaw + k * sym)                             # least wrist turn for the planned orientation
        target_yaw_box = wrap(tool_yaw + delta)
        place = target - offset + [0.0, 0.0, PLACE_GAP]
        place[:2] += corr
        pre = place + [0.0, 0.0, 0.03]
        over = np.array([place[0], place[1], transit_z])

        # Move durations grow with distance and with weight (heavier boxes move more gently): this keeps the arm
        # tracking its path, which keeps accelerations - and the load on the cups - as planned.
        slow = 1.0 + 0.06 * mass

        def dur(dist: float, base: int = 30) -> int:
            return int((base + 130 * dist) * slow)

        swing_angle = abs(wrap(self._angle(over) - self._angle(lift)))
        over_dist = float(np.linalg.norm(over[:2] - self.facing_tcp(over[:2])))
        steps = {"lift": dur(transit_z - lift[2]), "swing": int((40 + 75 * swing_angle) * slow),
                 "over": dur(over_dist), "lower": dur(transit_z - pre[2], 40)}
        a_max = max(peak_accel(transit_z - lift[2], steps["lift"]),
                    peak_accel(swing_angle * float(np.linalg.norm(READY_TCP[:2])), steps["swing"]),
                    peak_accel(over_dist, steps["over"]),
                    peak_accel(transit_z - pre[2], steps["lower"]))
        box.vacuum = self.gripper.set_vacuum(self.gripper.required_vacuum(mass, a_max, self.learner.margin))
        self.log(f"  box #{box.spec.id:02d}: {mass:.2f} kg -> vacuum {box.vacuum:.1f} kPa "
                 f"({self.gripper.hold_force(box.vacuum):.0f} N hold) -> layer {pl.layer + 1} {pl.slot}, "
                 f"yaw {math.degrees(pl.yaw):.0f} deg")

        self.carrying = box
        self.rel0 = self._rel(box)
        self.stage = "lift"
        yield from self.move([lift[0], lift[1], transit_z], steps["lift"], settle=6)
        self.stage = "swing"
        yield from self.go_joints(self.facing(over[:2], tool_yaw), steps["swing"], settle=10)
        self.stage = "over pallet"
        yield from self.move(over, steps["over"], yaw=tool_yaw, settle=8)
        self.stage = "lower"
        yield from self.move(pre, steps["lower"], yaw=tool_yaw, settle=10, tol=0.003)
        self.stage = "align"
        for _ in range(ALIGN_PASSES):
            pre[:2] += (target[:2] + corr) - box.position()[:2]
            yield from self.move(pre, int(18 * slow), settle=10, tol=0.002)
        place[:2] = pre[:2]
        self.stage = "place"
        yield from self.move(place, int(30 * slow), settle=12, tol=0.0015, max_wait=80)
        box.seal_peak = self.gripper.peak_util
        self.carrying = None
        self.gripper.release()
        yield from self.hold(12, goal=place)
        yield from self.move(place + [0.0, 0.0, 0.10], 35)
        yield from self.move([place[0], place[1], transit_z], 35)
        box.status = "placed"

        pos, quat = box.pose()
        err = pos[:2] - target[:2]
        box.place_error_mm = float(np.linalg.norm(err) * 1000)
        yaw_err = abs(wrap(yaw_of(quat) - target_yaw_box))
        box.yaw_error_deg = math.degrees(min(yaw_err, abs(sym - yaw_err)))
        self.log(f"    placed: error {np.round(err * 1000, 1)} mm, yaw {box.yaw_error_deg:.2f} deg, "
                 f"height error {(pos[2] - target[2]) * 1000:.1f} mm")
        self.learner.update(key, err)

    # results ------------------------------------------------------------------------------------------------------

    def summarize(self, episode: int, wall: float, settle: float) -> dict:
        placed = [b for b in self.boxes.values() if b.status == "placed"]
        errs = np.array([b.place_error_mm for b in placed]) if placed else np.array([np.nan])
        yaws = np.array([b.yaw_error_deg for b in placed]) if placed else np.array([np.nan])
        weigh = [abs(b.est_mass - b.spec.mass) / b.spec.mass * 100 for b in self.boxes.values() if b.est_mass]
        m = sum(b.spec.mass for b in placed)
        com = (sum(b.spec.mass * b.position()[:2] for b in placed) / m - self.pallet_xy) if placed else np.zeros(2)
        heavy = max(placed, key=lambda b: b.spec.mass) if placed else None
        row = {
            "cell": self.name or "cell 1",
            "episode": episode,
            "practice_total": self.learner.episodes_trained,
            "optimized": self.cfg.optimized,
            "boxes": len(self.boxes),
            "placed": len(placed),
            "load_kg": round(m, 1),
            "heaviest_placed_kg": round(heavy.spec.mass, 1) if heavy else 0.0,
            "payload_rated_kg": PAYLOAD_RATED_KG,
            "mean_place_error_mm": round(float(np.nanmean(errs)), 2),
            "max_place_error_mm": round(float(np.nanmax(errs)), 2),
            "mean_yaw_error_deg": round(float(np.nanmean(yaws)), 2),
            "mean_weigh_error_pct": round(float(np.mean(weigh)), 2) if weigh else float("nan"),
            "seal_losses": self.seal_losses,
            "mean_vacuum_pct": round(float(np.mean([b.vacuum for b in placed])) / VACUUM_MAX_KPA * 100, 1)
            if placed else 0.0,
            "max_joint_load_pct": round(max([b.joint_peak or 0 for b in placed], default=0) * 100, 0),
            "load_com_offset_mm": round(float(np.linalg.norm(com)) * 1000, 1),
            "settled_mm": round(float(settle) * 1000, 2),
            "vacuum_margin": round(self.learner.margin, 3),
            "wall_seconds": round(wall, 1),
        }
        log = self.cfg.training_log
        if log is not None:
            new = not log.exists()
            with log.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        p = self.prefix
        print("\n" + "=" * 100)
        print(f"{p}Episode {episode}: {row['placed']}/{row['boxes']} boxes on the pallet ({row['load_kg']} kg, heaviest "
              f"{row['heaviest_placed_kg']} kg) | placement error mean {row['mean_place_error_mm']} mm, max "
              f"{row['max_place_error_mm']} mm, yaw {row['mean_yaw_error_deg']} deg")
        print(f"{p}  weighing error {row['mean_weigh_error_pct']}% | seal losses {row['seal_losses']} | mean vacuum "
              f"{row['mean_vacuum_pct']}% of full | peak joint torque {row['max_joint_load_pct']:.0f}% of limit")
        print(f"{p}  finished load's centre of mass {row['load_com_offset_mm']} mm from the pallet centre "
              f"(plan: {self.plan.metrics['com_offset_mm']} mm) | boxes moved {row['settled_mm']} mm while standing")
        print("=" * 100 + "\n")
        self.last_row = row
        return row

    def train(self):
        rows = []
        for ep in range(1, max(1, self.cfg.episodes) + 1):
            print(f"\n{self.prefix}######## Episode {ep}/{self.cfg.episodes} ########")
            yield from self.run_episode(ep)
            if self.seal_losses == 0:
                self.learner.on_clean_episode()
            self.learner.episodes_trained += 1
            self.learner.save()
            rows.append(self.last_row)
        print(f"{self.prefix}Summary (this launch):")
        for r in rows:
            print(f"{self.prefix}  episode {r['episode']}: {r['placed']}/{r['boxes']} placed, mean error "
                  f"{r['mean_place_error_mm']} mm, max {r['max_place_error_mm']} mm, seal losses {r['seal_losses']}, "
                  f"vacuum margin {r['vacuum_margin']}")


def make_plan(specs: list[pp.BoxSpec], optimize: bool, iters: int, seed: int) -> pp.Plan:
    """Print the box list and the unoptimized plan, then optimize (unless told not to) and sequence it."""
    naive = pp.naive_plan(specs)
    print(f"\n{len(specs)} boxes in the buffer (from the dimensioner + checkweigher; F = fragile):")
    for s in specs:
        print("  " + s.label())
    print()
    print(pp.render(naive, "Unoptimized pallet (layers and spots in arrival order):"))
    if not optimize:
        plan = naive
    else:
        t = time.time()
        plan = pp.optimize(specs, iters=iters, seed=seed)
        print()
        print(pp.render(plan, f"Optimized pallet ({time.time() - t:.1f} s of planning):"))
    pp.sequence(plan, (-PALLET_XY[0], -PALLET_XY[1]))
    print()
    return plan


def staging_xy(k: int) -> np.ndarray:
    """Where box k waits in the buffer behind the robot (cell frame)."""
    row, col = divmod(k, 4)
    return np.array([-1.3 - 0.75 * row, -1.1 + 0.75 * col])
