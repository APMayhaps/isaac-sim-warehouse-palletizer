"""Mixed-case pallet planner: builds a 3 x 3 x 3 pallet cube from boxes of different sizes, weights and strengths.

Pure Python (no Isaac Sim needed), so the plan can be inspected on its own:
    python pallet_planner.py --seed 7

The pallet is a 3 x 3 grid of 32 cm cells, three layers high (a 0.96 m cube footprint on a 1.0 x 1.0 m pallet).
Boxes come in three footprints that fit the grid - 1x1 cell, 2x1 cells and 2x2 cells - in three product-family
heights. Each box also has a stacking-strength label (the most weight it can carry on top); fragile boxes carry
little.

What the planner decides
  * which boxes go in which layer (every layer must be completely filled by boxes of one height, so the next
    layer has a flat, fully supported surface),
  * the order of the layers from bottom to top,
  * the pattern (tiling) of each layer and which box goes in which spot, including turning 2x1 boxes 90 degrees,
  * the order the robot places them in.

What it optimizes (lower cost is better)
  * weight low      - heavy boxes near the bottom (mass-weighted height of the load),
  * balance         - centre of mass of the whole load, and of every partial stack, over the pallet centre,
  * crush           - no box carries more than its stacking-strength label (hard constraint),
  * interlocking    - seams between boxes should not line up with the seams in the layer below (a bonded
                      pattern, like bricks, holds the load together; aligned seams make weak columns),
  * big boxes low   - larger footprints lower down for a stable base.

It starts from a greedy plan and improves it with simulated annealing (swap same-shape boxes, re-pattern a
layer, reorder layers, trade a 2x1 box for two 1x1 boxes between layers of the same height).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

CELL = 0.32                     # m, grid pitch
GRID = 3                        # 3 x 3 cells per layer
LAYERS = 3
GAP = 0.012                     # m, nominal clearance between neighbouring boxes
PALLET_SIZE = 1.0               # m, square pallet (1.0 x 1.0)
PALLET_HEIGHT = 0.144           # m, top of deck boards above the floor
HEIGHT_FAMILIES = (0.18, 0.22, 0.26)
MASS_RANGES = {(1, 1): (1.5, 9.0), (2, 1): (4.0, 14.0), (2, 2): (7.0, 16.0)}

W_MASS_Z, W_COM, W_STACK_COM, W_INTERLOCK, W_BIG_LOW, W_CRUSH = 4.0, 3.0, 1.0, 1.0, 0.3, 20.0


# --- boxes -----------------------------------------------------------------------------------------------------


@dataclass
class BoxSpec:
    id: int
    length: float          # m, along the infeed (long side)
    width: float           # m
    height: float          # m
    mass: float            # kg, true
    measured_mass: float   # kg, from the infeed checkweigher (what the planner sees)
    max_load: float        # kg it can carry on top (stacking-strength label)
    fragile: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        """Footprint in grid cells, long side first: (1, 1), (2, 1) or (2, 2)."""
        a = max(1, round((self.length + 2 * GAP) / CELL))
        b = max(1, round((self.width + 2 * GAP) / CELL))
        return (max(a, b), min(a, b))

    @property
    def cells(self) -> int:
        return self.shape[0] * self.shape[1]

    def label(self) -> str:
        return (f"#{self.id:02d} {self.length * 100:4.1f}x{self.width * 100:4.1f}x{self.height * 100:4.1f} cm "
                f"{self.measured_mass:5.1f} kg  max load {self.max_load:5.1f} kg{'  FRAGILE' if self.fragile else ''}")


def _random_tiling(rng: random.Random) -> list[tuple[int, int, int, int]]:
    grid = [[False] * GRID for _ in range(GRID)]
    slots, used_big = [], False
    for i in range(GRID):
        for j in range(GRID):
            if grid[i][j]:
                continue
            opts = []
            if not used_big and i + 1 < GRID and j + 1 < GRID and not (grid[i + 1][j] or grid[i][j + 1]
                                                                        or grid[i + 1][j + 1]):
                opts += [(2, 2)] * 3
            if i + 1 < GRID and not grid[i + 1][j]:
                opts += [(2, 1)] * 3
            if j + 1 < GRID and not grid[i][j + 1]:
                opts += [(1, 2)] * 3
            opts += [(1, 1)] * 2
            sx, sy = rng.choice(opts)
            used_big |= (sx, sy) == (2, 2)
            for a in range(sx):
                for b in range(sy):
                    grid[i + a][j + b] = True
            slots.append((i, j, sx, sy))
    return slots


def generate_boxes(seed: int = 7, max_mass: float = 16.0) -> list[BoxSpec]:
    """A random order of boxes that can exactly fill a 3 x 3 x 3 pallet cube.

    Each of the three layers' worth of boxes is one product family (same height) cut from a random 3 x 3 tiling,
    so a complete cube is possible - but the planner is not told how; it only sees the boxes. The heaviest box is
    always exactly `max_mass`."""
    rng = random.Random(seed)
    specs = []
    for _ in range(LAYERS):
        h = rng.choice(HEIGHT_FAMILIES)
        for (_, _, sx, sy) in _random_tiling(rng):
            shape = (max(sx, sy), min(sx, sy))
            length = shape[0] * CELL - GAP - rng.uniform(0.0, 0.012)
            width = shape[1] * CELL - GAP - rng.uniform(0.0, 0.012)
            lo, hi = MASS_RANGES[shape]
            mass = round(min(max_mass, rng.uniform(lo, hi)), 1)
            fragile = rng.random() < 0.18
            max_load = round(rng.uniform(8.0, 20.0) if fragile else rng.uniform(45.0, 140.0), 0)
            specs.append([length, width, h, mass, max_load, fragile])
    heaviest = max(range(len(specs)), key=lambda k: specs[k][3])
    specs[heaviest][3] = max_mass
    rng.shuffle(specs)
    boxes = []
    for k, (l, w, h, m, load, fragile) in enumerate(specs):
        measured = round(m * (1.0 + rng.gauss(0.0, 0.004)), 2)
        boxes.append(BoxSpec(k, l, w, h, m, measured, load, fragile))
    return boxes


# --- layer patterns --------------------------------------------------------------------------------------------

Slot = tuple[int, int, int, int]          # (i, j, sx, sy): anchor cell and size in cells along pallet x and y
_TILINGS: dict[tuple[int, int, int], list[tuple[Slot, ...]]] = {}


def comp_of(shapes: list[tuple[int, int]]) -> tuple[int, int, int]:
    return (sum(s == (2, 2) for s in shapes), sum(s == (2, 1) for s in shapes), sum(s == (1, 1) for s in shapes))


def tilings(comp: tuple[int, int, int]) -> list[tuple[Slot, ...]]:
    """All ways to fill the 3 x 3 grid with n4 2x2, n2 2x1 (either way round) and n1 1x1 pieces."""
    if comp in _TILINGS:
        return _TILINGS[comp]
    out: list[tuple[Slot, ...]] = []
    grid = [[False] * GRID for _ in range(GRID)]

    def rec(slots, n4, n2, n1):
        cell = next(((i, j) for i in range(GRID) for j in range(GRID) if not grid[i][j]), None)
        if cell is None:
            if n4 == n2 == n1 == 0:
                out.append(tuple(slots))
            return
        i, j = cell
        for (sx, sy), kind in (((2, 2), 4), ((2, 1), 2), ((1, 2), 2), ((1, 1), 1)):
            if (kind == 4 and n4 == 0) or (kind == 2 and n2 == 0) or (kind == 1 and n1 == 0):
                continue
            cells = [(i + a, j + b) for a in range(sx) for b in range(sy)]
            if all(ci < GRID and cj < GRID and not grid[ci][cj] for ci, cj in cells):
                for ci, cj in cells:
                    grid[ci][cj] = True
                rec(slots + [(i, j, sx, sy)], n4 - (kind == 4), n2 - (kind == 2), n1 - (kind == 1))
                for ci, cj in cells:
                    grid[ci][cj] = False

    rec([], *comp)
    _TILINGS[comp] = out
    return out


def slot_shape(slot: Slot) -> tuple[int, int]:
    return (max(slot[2], slot[3]), min(slot[2], slot[3]))


def slot_cells(slot: Slot) -> list[tuple[int, int]]:
    i, j, sx, sy = slot
    return [(i + a, j + b) for a in range(sx) for b in range(sy)]


def slot_center(slot: Slot) -> tuple[float, float]:
    """Slot centre in the pallet frame (origin at the pallet centre)."""
    i, j, sx, sy = slot
    half = GRID * CELL / 2.0
    return ((i + sx / 2.0) * CELL - half, (j + sy / 2.0) * CELL - half)


# --- plan state ------------------------------------------------------------------------------------------------


@dataclass
class Layer:
    height: float
    comp: tuple[int, int, int]
    tiling: tuple[Slot, ...]
    assign: list[int]                      # box id per slot

    def copy(self) -> "Layer":
        return Layer(self.height, self.comp, self.tiling, list(self.assign))


@dataclass
class Placement:
    order: int
    box: BoxSpec
    layer: int
    slot: Slot
    x: float                # pallet frame, m
    y: float
    z: float                # box centre height above the pallet deck, m
    yaw: float              # rad: 0 = long side along pallet x, pi/2 = along pallet y


@dataclass
class Plan:
    layers: list[Layer]
    boxes: dict[int, BoxSpec]
    cost: float = 0.0
    metrics: dict = field(default_factory=dict)
    placements: list[Placement] = field(default_factory=list)


def _assign_by_shape(tiling: tuple[Slot, ...], box_ids: list[int], boxes: dict[int, BoxSpec],
                     rng: random.Random | None = None) -> list[int]:
    pools: dict[tuple[int, int], list[int]] = {}
    for b in box_ids:
        pools.setdefault(boxes[b].shape, []).append(b)
    for pool in pools.values():
        if rng is not None:
            rng.shuffle(pool)
    return [pools[slot_shape(s)].pop() for s in tiling]


def evaluate(layers: list[Layer], boxes: dict[int, BoxSpec]) -> tuple[float, dict]:
    """Cost of a plan and its metrics."""
    total_h = sum(l.height for l in layers)
    z0 = 0.0
    rows = []                                   # (box, layer index, slot, x, y, zc)
    owner: list[dict[tuple[int, int], int]] = []
    for L, layer in enumerate(layers):
        own = {}
        for slot, b in zip(layer.tiling, layer.assign):
            x, y = slot_center(slot)
            rows.append((boxes[b], L, slot, x, y, z0 + layer.height / 2.0))
            for c in slot_cells(slot):
                own[c] = b
        owner.append(own)
        z0 += layer.height
    m_tot = sum(r[0].measured_mass for r in rows)

    mass_z = sum(r[0].measured_mass * r[5] for r in rows) / (m_tot * total_h)
    cx = sum(r[0].measured_mass * r[3] for r in rows) / m_tot
    cy = sum(r[0].measured_mass * r[4] for r in rows) / m_tot
    com = math.hypot(cx, cy)
    stack_com = 0.0
    for L in range(len(layers)):
        part = [r for r in rows if r[1] >= L]
        mp = sum(r[0].measured_mass for r in part)
        stack_com = max(stack_com, math.hypot(sum(r[0].measured_mass * r[3] for r in part) / mp,
                                              sum(r[0].measured_mass * r[4] for r in part) / mp))

    # Load path: each box's own weight plus what it carries goes down to the boxes under its footprint.
    carried = {r[0].id: 0.0 for r in rows}
    for L in range(len(layers) - 1, 0, -1):
        for slot, b in zip(layers[L].tiling, layers[L].assign):
            down = boxes[b].measured_mass + carried[b]
            cells = slot_cells(slot)
            for c in cells:
                carried[owner[L - 1][c]] += down / len(cells)
    util = {b: carried[b] / boxes[b].max_load for b in carried}
    crush_over = sum(max(0.0, u - 1.0) for u in util.values())

    def seams(own):
        s = set()
        for i in range(GRID):
            for j in range(GRID):
                if i + 1 < GRID and own[(i, j)] != own[(i + 1, j)]:
                    s.add(((i, j), (i + 1, j)))
                if j + 1 < GRID and own[(i, j)] != own[(i, j + 1)]:
                    s.add(((i, j), (i, j + 1)))
        return s

    aligned = total = 0
    for L in range(1, len(layers)):
        up, down = seams(owner[L]), seams(owner[L - 1])
        aligned += len(up & down)
        total += len(up)
    interlock = aligned / total if total else 0.0
    big_low = sum((r[0].cells / 4.0) * (r[1] / max(1, len(layers) - 1)) for r in rows) / len(rows)

    cost = (W_MASS_Z * mass_z + W_COM * com / CELL + W_STACK_COM * stack_com / CELL + W_INTERLOCK * interlock
            + W_BIG_LOW * big_low + W_CRUSH * crush_over)
    heaviest = sorted(rows, key=lambda r: -r[0].measured_mass)[:5]
    metrics = {
        "cost": round(cost, 4),
        "mass_height_pct": round(mass_z * 100, 1),
        "com_offset_mm": round(com * 1000, 1),
        "worst_stack_com_mm": round(stack_com * 1000, 1),
        "max_crush_pct": round(max(util.values()) * 100, 0),
        "overloaded_boxes": sum(u > 1.0 for u in util.values()),
        "aligned_seams_pct": round(interlock * 100, 0),
        "heaviest5_mean_layer": round(sum(r[1] for r in heaviest) / len(heaviest) + 1, 2),
        "load_kg": round(m_tot, 1),
        "stack_height_m": round(total_h, 3),
    }
    return cost, metrics


# --- building a plan ------------------------------------------------------------------------------------------


def _feasible_comps() -> list[tuple[int, int, int]]:
    comps = []
    for n4 in range(0, 3):
        for n2 in range(0, 5):
            n1 = 9 - 4 * n4 - 2 * n2
            if n1 >= 0 and tilings((n4, n2, n1)):
                comps.append((n4, n2, n1))
    return comps


def _split(counts: tuple[int, int, int], k: int) -> list[tuple[int, int, int]] | None:
    """Split a height group's piece counts into k fillable layers."""
    if k == 0:
        return [] if counts == (0, 0, 0) else None
    for c in _feasible_comps():
        rest = (counts[0] - c[0], counts[1] - c[1], counts[2] - c[2])
        if min(rest) < 0:
            continue
        tail = _split(rest, k - 1)
        if tail is not None:
            return [c] + tail
    return None


def _group_layers(boxes: list[BoxSpec], heavy_first: bool) -> list[Layer]:
    groups: dict[float, list[BoxSpec]] = {}
    for b in boxes:
        groups.setdefault(round(b.height, 2), []).append(b)
    by_id = {b.id: b for b in boxes}
    layers = []
    for h, members in groups.items():
        cells = sum(b.cells for b in members)
        if cells % (GRID * GRID):
            raise ValueError(f"boxes of height {h} m cover {cells} cells - not whole layers")
        comps = _split(comp_of([b.shape for b in members]), cells // (GRID * GRID))
        if comps is None:
            raise ValueError(f"boxes of height {h} m can't be arranged into full 3 x 3 layers")
        pools = {}
        for b in sorted(members, key=lambda b: -b.measured_mass) if heavy_first else members:
            pools.setdefault(b.shape, []).append(b.id)
        for c in comps:
            ids = [pools[(2, 2)].pop(0) for _ in range(c[0])] + [pools[(2, 1)].pop(0) for _ in range(c[1])] \
                + [pools[(1, 1)].pop(0) for _ in range(c[2])]
            t = tilings(c)[0]
            layers.append(Layer(h, c, t, _assign_by_shape(t, ids, by_id)))
    return layers


def naive_plan(boxes: list[BoxSpec]) -> Plan:
    """What you get without optimizing: layers in the order the boxes arrived, the first pattern that fits,
    boxes dropped into it as they come."""
    order = {}
    for b in boxes:
        order.setdefault(round(b.height, 2), b.id)
    layers = sorted(_group_layers(boxes, heavy_first=False), key=lambda l: order[round(l.height, 2)])
    by_id = {b.id: b for b in boxes}
    cost, metrics = evaluate(layers, by_id)
    return Plan(layers, by_id, cost, metrics)


def optimize(boxes: list[BoxSpec], iters: int = 12000, seed: int = 0) -> Plan:
    rng = random.Random(seed)
    by_id = {b.id: b for b in boxes}
    layers = sorted(_group_layers(boxes, heavy_first=True),
                    key=lambda l: -sum(by_id[b].measured_mass for b in l.assign))
    cost, _ = evaluate(layers, by_id)
    best, best_cost = [l.copy() for l in layers], cost
    t0, t1 = 0.3, 0.002
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / iters)
        new = [l.copy() for l in layers]
        move = rng.random()
        if move < 0.45:                                    # swap two boxes of the same shape and height
            L1, L2 = rng.randrange(len(new)), rng.randrange(len(new))
            if new[L1].height != new[L2].height:
                continue
            s1, s2 = rng.randrange(len(new[L1].assign)), rng.randrange(len(new[L2].assign))
            if by_id[new[L1].assign[s1]].shape != by_id[new[L2].assign[s2]].shape:
                continue
            new[L1].assign[s1], new[L2].assign[s2] = new[L2].assign[s2], new[L1].assign[s1]
        elif move < 0.75:                                  # new pattern for one layer
            L = rng.randrange(len(new))
            t = rng.choice(tilings(new[L].comp))
            new[L].tiling, new[L].assign = t, _assign_by_shape(t, new[L].assign, by_id, rng)
        elif move < 0.9:                                   # reorder layers
            L1, L2 = rng.sample(range(len(new)), 2) if len(new) > 1 else (0, 0)
            new[L1], new[L2] = new[L2], new[L1]
        else:                                              # trade a 2x1 for two 1x1 between same-height layers
            L1, L2 = rng.sample(range(len(new)), 2) if len(new) > 1 else (0, 0)
            if L1 == L2 or new[L1].height != new[L2].height:
                continue
            dom = [b for b in new[L1].assign if by_id[b].shape == (2, 1)]
            sing = [b for b in new[L2].assign if by_id[b].shape == (1, 1)]
            if not dom or len(sing) < 2:
                continue
            d, s = rng.choice(dom), rng.sample(sing, 2)
            ids1 = [b for b in new[L1].assign if b != d] + s
            ids2 = [b for b in new[L2].assign if b not in s] + [d]
            c1, c2 = comp_of([by_id[b].shape for b in ids1]), comp_of([by_id[b].shape for b in ids2])
            if not tilings(c1) or not tilings(c2):
                continue
            for L, ids, c in ((L1, ids1, c1), (L2, ids2, c2)):
                t = rng.choice(tilings(c))
                new[L] = Layer(new[L].height, c, t, _assign_by_shape(t, ids, by_id, rng))
        new_cost, _ = evaluate(new, by_id)
        if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / temp):
            layers, cost = new, new_cost
            if cost < best_cost:
                best, best_cost = [l.copy() for l in layers], cost
    cost, metrics = evaluate(best, by_id)
    return Plan(best, by_id, cost, metrics)


def sequence(plan: Plan, robot_xy: tuple[float, float]) -> list[Placement]:
    """Placement order: bottom layer first; within a layer, the spot farthest from the robot first, so the arm
    never lowers a box past one it has already placed."""
    out = []
    z0 = 0.0
    for L, layer in enumerate(plan.layers):
        spots = []
        for slot, b in zip(layer.tiling, layer.assign):
            x, y = slot_center(slot)
            yaw = math.pi / 2 if (slot[3] > slot[2]) else 0.0
            spots.append((math.hypot(x - robot_xy[0], y - robot_xy[1]), slot, b, x, y, yaw))
        for _, slot, b, x, y, yaw in sorted(spots, key=lambda s: -s[0]):
            out.append(Placement(len(out), plan.boxes[b], L, slot, x, y, z0 + layer.height / 2.0, yaw))
        z0 += layer.height
    plan.placements = out
    return out


def render(plan: Plan, title: str) -> str:
    lines = [title]
    for L in range(len(plan.layers) - 1, -1, -1):
        layer = plan.layers[L]
        own = {}
        for slot, b in zip(layer.tiling, layer.assign):
            for c in slot_cells(slot):
                own[c] = b
        mass = sum(plan.boxes[b].measured_mass for b in layer.assign)
        lines.append(f"  layer {L + 1} ({layer.height * 100:.0f} cm boxes, {mass:5.1f} kg)   "
                     f"robot side is on the left, pallet +x to the right")
        for j in range(GRID - 1, -1, -1):
            row = "   "
            for i in range(GRID):
                b = plan.boxes[own[(i, j)]]
                row += f"| #{b.id:02d} {b.measured_mass:4.1f}{'F' if b.fragile else ' '}"
            lines.append(row + "|")
    m = plan.metrics
    lines.append(f"  cost {m['cost']:.3f} | heaviest 5 boxes on layer {m['heaviest5_mean_layer']} on average | "
                 f"load centre {m['com_offset_mm']} mm off centre (worst partial stack {m['worst_stack_com_mm']} mm) | "
                 f"max crush {m['max_crush_pct']:.0f}% of rating, {m['overloaded_boxes']} overloaded | "
                 f"{m['aligned_seams_pct']:.0f}% of seams aligned with the layer below")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan a mixed-case 3x3x3 pallet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-box-mass", type=float, default=16.0)
    a = ap.parse_args()
    bx = generate_boxes(a.seed, a.max_box_mass)
    print(f"{len(bx)} boxes on the infeed (F = fragile):")
    for b in bx:
        print("  " + b.label())
    print()
    print(render(naive_plan(bx), "Unoptimized (layers and spots in arrival order):"))
    print()
    p = optimize(bx, seed=a.seed)
    print(render(p, "Optimized:"))
    for pl in sequence(p, (-0.95, 0.0)):
        print(f"  {pl.order + 1:2d}. box #{pl.box.id:02d} -> layer {pl.layer + 1}, cells {slot_cells(pl.slot)}, "
              f"yaw {math.degrees(pl.yaw):.0f} deg")
