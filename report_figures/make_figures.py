"""Section 4 (Design Justification & Proof) figures for the Team Chudlings case-comp report.

Draws six diagrams of the planned, weight-adaptive palletizing cell and saves each as PNG (300 dpi, for the report)
and SVG (editable text, opens in Figma / draw.io / Inkscape / Word):

  4.1 system architecture        4.2 per-box control flowchart    4.3 one-cycle swimlane
  4.4 vacuum circuit             4.5 weight data flow             4.6 cell layout (top + side view)

Numbers come from the report and from the Isaac Sim simulation (palletizer_core.py / pallet_planner.py).
Requires matplotlib:  python -m pip install matplotlib   then   python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

# Fill / edge colours per kind of element.
PAL = {
    "sw": ("#e9f1fb", "#2f5d9e"),       # software / control
    "sense": ("#e5f4f1", "#1d7466"),    # sensors / vision
    "mech": ("#f5efe6", "#8a6a3f"),     # mechanical
    "vac": ("#eaf4e3", "#3d7a2a"),      # vacuum / pneumatic
    "safe": ("#fcebe9", "#b03a2e"),     # safety
    "data": ("#f1ecf8", "#6a4c9c"),     # infeed / data sources
    "neutral": ("#f4f5f7", "#5b6470"),
    "term": ("#2f3b4c", "#2f3b4c"),     # start / end
    "fault": ("#fff4e5", "#c26a00"),    # fault handling
}
TXT = "#1d232b"
GREY = "#5b6470"
C_DATA, C_AIR, C_VAC, C_SAFE, C_MECH, C_FAULT = "#2f5d9e", "#3d7a2a", "#1d7466", "#b03a2e", "#8a6a3f", "#c26a00"
WHITE_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
UPI = 1.0  # drawing units per inch of the current figure (set by new_fig)


def new_fig(w_units: float, h_units: float, width_in: float, y0: float = 0.0):
    global UPI
    UPI = w_units / width_in
    fig = plt.figure(figsize=(width_in, width_in * h_units / w_units))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_units)
    ax.set_ylim(y0, y0 + h_units)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", name)


def _lines(s: str | None) -> int:
    return 0 if not s else s.count("\n") + 1


def text_block(ax, cx, cy, title, body=None, fs=8.0, tfs=None, color=TXT, title_color=None):
    """Bold title over a regular body, centred on (cx, cy)."""
    tfs = tfs or fs + 0.6
    ls = 1.18
    th = _lines(title) * tfs * ls / 72 * UPI
    bh = _lines(body) * fs * ls / 72 * UPI
    gap = 0.25 * fs / 72 * UPI if body else 0
    top = cy + (th + gap + bh) / 2
    ax.text(cx, top - th / 2, title, ha="center", va="center", fontsize=tfs, fontweight="bold",
            color=title_color or color, linespacing=ls)
    if body:
        ax.text(cx, top - th - gap - bh / 2, body, ha="center", va="center", fontsize=fs, color=color, linespacing=ls)


def box(ax, x, y, w, h, title, body=None, kind="sw", fs=8.0, tfs=None, r=0.12, lw=1.3):
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    text_block(ax, x + w / 2, y + h / 2, title, body, fs, tfs, title_color=ec)


def group(ax, x, y, w, h, label, kind="neutral", fs=8.0):
    _, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="none", ec=ec,
                                lw=1.1, ls=(0, (4, 3))))
    ax.text(x + 0.18, y + h - 0.12, label, ha="left", va="top", fontsize=fs, fontweight="bold", color=ec, zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def diamond(ax, cx, cy, w, h, text, fs=7.4, kind="sw"):
    fc, ec = PAL[kind]
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=TXT, linespacing=1.15)


def arrow(ax, pts, color=C_DATA, label=None, seg=None, lw=1.3, ls="-", both=False, fs=7.0, lpos=None,
          lha="center", head=True):
    """Polyline arrow through `pts` (head at the last point). Label at the middle of segment `seg`, or at lpos."""
    pts = [tuple(p) for p in pts]
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    style = "-|>" if head else "-"
    if both and len(pts) == 2:
        style = "<|-|>"
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))
    if both and len(pts) > 2:
        ax.add_patch(FancyArrowPatch(pts[1], pts[0], arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=1))
    if label:
        if lpos is None:
            k = seg if seg is not None else (len(pts) - 2) // 2
            (x1, y1), (x2, y2) = pts[k], pts[k + 1]
            lpos = ((x1 + x2) / 2, (y1 + y2) / 2)
        ax.text(*lpos, label, ha=lha, va="center", fontsize=fs, color=color, linespacing=1.1, zorder=3,
                bbox=WHITE_BOX)


def legend(ax, x, y, items, fs=7.2, step=3.4):
    for i, (color, ls, text) in enumerate(items):
        xx = x + i * step
        ax.add_patch(FancyArrowPatch((xx, y), (xx + 0.8, y), arrowstyle="-|>", mutation_scale=9, color=color,
                                     lw=1.3, linestyle=ls))
        ax.text(xx + 0.95, y, text, ha="left", va="center", fontsize=fs, color=TXT)


# --- 4.1 System architecture -----------------------------------------------------------------------------------------


def fig_architecture():
    fig, ax = new_fig(20, 13.4, 11.0, y0=-0.9)
    group(ax, 0.3, 9.6, 7.4, 2.75, "INFEED", "data")
    group(ax, 8.1, 9.6, 4.4, 2.75, "VISION", "sense")
    group(ax, 12.9, 9.6, 6.8, 2.75, "SAFETY", "safe")
    group(ax, 0.3, 4.3, 12.2, 4.95, "CELL CONTROLLER (industrial PC): software", "sw")
    group(ax, 12.9, 4.3, 6.8, 4.95, "ROBOT", "mech")
    group(ax, 0.3, 0.3, 12.2, 3.55, "VACUUM SYSTEM", "vac")
    group(ax, 12.9, 0.3, 6.8, 3.55, "END OF ARM", "mech")

    box(ax, 0.6, 9.85, 2.2, 1.65, "Dimensioner", "box size:\n3 footprints,\n3 heights", "sense")
    box(ax, 3.0, 9.85, 2.1, 1.65, "Checkweigher", "listed weight\n(±5 %)", "sense")
    box(ax, 5.3, 9.85, 2.2, 1.65, "Sequencing\nbuffer", "+ conveyor; releases\nboxes in plan order", "mech", fs=7.4)
    box(ax, 8.4, 9.85, 1.9, 1.65, "Camera A", "over the\npick station", "sense")
    box(ax, 10.4, 9.85, 1.9, 1.65, "Camera B", "over the\npallet", "sense")
    box(ax, 13.2, 9.85, 2.4, 1.65, "Safety\ncontroller", "ISO 13849-1\nsafety PLC", "safe")
    box(ax, 15.8, 9.85, 1.9, 1.65, "Fence +\ngate interlock", "ISO 10218-2", "safe")
    box(ax, 17.9, 9.85, 1.6, 1.65, "E-stops", "at gate and\noperator panel", "safe")

    box(ax, 0.6, 7.2, 3.3, 1.5, "(1) Pallet planner", "layers, spots, turns, order\n(before the first pick)")
    box(ax, 4.4, 7.2, 3.6, 1.5, "(5) Supervisor", "runs the sequence, checks\nweight, handles faults, logs")
    box(ax, 8.5, 7.2, 3.7, 1.5, "(4) Vision software", "box pose and size,\nplacement check")
    box(ax, 0.6, 4.65, 3.6, 1.6, "(2) Grip controller", "weight → vacuum setpoint,\nseal confirmation")
    box(ax, 6.0, 4.65, 3.6, 1.6, "(3) Motion controller", "IK paths; move time\nscaled by weight")

    box(ax, 13.2, 7.2, 3.0, 1.5, "UR20 controller", "joint servo loop,\nprotective stop", "mech")
    box(ax, 16.5, 7.2, 3.0, 1.5, "UR20 arm", "6-axis, 20 kg payload,\n1.75 m reach, 0.7 m pedestal", "mech", fs=7.4)
    box(ax, 16.5, 4.65, 3.0, 1.6, "Wrist F/T sensor", "weighs each box\nafter an 8 cm lift", "sense")

    box(ax, 0.6, 0.65, 2.2, 2.0, "Compressed\nair", "≈6 bar supply", "vac")
    box(ax, 3.2, 0.65, 2.3, 2.0, "Vacuum\ngenerator", "venturi ejector", "vac")
    box(ax, 5.9, 0.65, 2.6, 2.0, "Proportional\nregulator", "8–80 kPa\nsetpoint", "vac")
    box(ax, 8.9, 0.65, 3.3, 2.0, "Pressure sensor\n+ check valve", "confirms the seal;\nholds vacuum on a stop", "vac")
    box(ax, 13.2, 0.65, 3.4, 2.6, "Vacuum gripper (4 kg)", "4 × Ø50 mm cups\n78.5 cm² total\n628 N at 80 kPa", "mech")
    box(ax, 17.0, 0.65, 2.5, 2.6, "Box", "1.5–16 kg\nbox + tool\n≤ 20 kg payload", "neutral")

    # data / signals
    arrow(ax, [(1.7, 9.85), (1.7, 8.7)], label="size", lpos=(1.7, 9.43))
    arrow(ax, [(3.45, 9.85), (3.45, 8.7)], label="weight", lpos=(3.45, 9.43))
    arrow(ax, [(6.6, 8.7), (6.6, 9.85)], label="release next box", lpos=(6.6, 9.43))
    arrow(ax, [(3.9, 8.1), (4.4, 8.1)], label="plan", lpos=(4.15, 8.5))
    arrow(ax, [(9.35, 9.85), (9.35, 8.7)], color=C_VAC, label="images", lpos=(9.35, 9.43))
    arrow(ax, [(11.35, 9.85), (11.35, 8.7)], color=C_VAC, label="images", lpos=(11.35, 9.43))
    arrow(ax, [(8.5, 7.8), (8.0, 7.8)], label="checks", lpos=(8.25, 8.5))
    arrow(ax, [(9.2, 7.2), (9.2, 6.25)], label="box pose", lpos=(9.2, 6.6))
    arrow(ax, [(5.0, 7.2), (5.0, 6.75), (3.4, 6.75), (3.4, 6.25)], label="weight, peak accel.", lpos=(4.2, 6.75))
    arrow(ax, [(7.2, 7.2), (7.2, 6.25)], label="target spot", lpos=(7.2, 6.6))
    arrow(ax, [(16.5, 5.9), (12.75, 5.9), (12.75, 6.95), (7.8, 6.95), (7.8, 7.2)], label="measured weight",
          lpos=(11.1, 6.95))
    arrow(ax, [(9.6, 5.25), (12.3, 5.25), (12.3, 7.95), (13.2, 7.95)], label="joint targets / state",
          lpos=(11.0, 5.25), both=True)
    arrow(ax, [(16.2, 7.95), (16.5, 7.95)], color=C_MECH)
    # mechanical chain
    arrow(ax, [(18.0, 7.2), (18.0, 6.25)], color=C_MECH, lw=2.0, head=False)
    arrow(ax, [(18.0, 4.65), (18.0, 3.95), (14.9, 3.95), (14.9, 3.25)], color=C_MECH, lw=2.0, label="flange",
          lpos=(16.5, 3.95), head=False)
    arrow(ax, [(16.6, 1.95), (17.0, 1.95)], color=C_VAC, lw=2.0, head=False)
    ax.text(16.8, 2.4, "seal", ha="center", fontsize=7, color=C_VAC)
    # vacuum
    arrow(ax, [(2.4, 4.65), (2.4, 4.0), (7.2, 4.0), (7.2, 2.65)], label="vacuum setpoint", lpos=(4.8, 4.0))
    arrow(ax, [(10.55, 2.65), (10.55, 4.47), (4.6, 4.47), (4.6, 5.1), (4.2, 5.1)], color=C_VAC,
          label="seal confirmed", lpos=(8.6, 4.47))
    for x1, x2 in [(2.8, 3.2), (5.5, 5.9), (8.5, 8.9)]:
        arrow(ax, [(x1, 1.65), (x2, 1.65)], color=C_AIR, lw=2.2)
    arrow(ax, [(12.2, 1.65), (13.2, 1.65)], color=C_AIR, lw=2.2, label="vacuum\nline", lpos=(12.7, 2.15))
    # safety
    arrow(ax, [(15.8, 10.7), (15.6, 10.7)], color=C_SAFE, ls="--")
    arrow(ax, [(18.7, 11.5), (18.7, 12.05), (14.4, 12.05), (14.4, 11.5)], color=C_SAFE, ls="--")
    arrow(ax, [(14.4, 9.85), (14.4, 8.7)], color=C_SAFE, ls="--", label="protective\nstop", lpos=(14.4, 9.43))

    legend(ax, 0.6, -0.45, [(C_DATA, "-", "data / commands"), (C_VAC, "-", "sensor feedback"),
                            (C_AIR, "-", "compressed air / vacuum"), (C_SAFE, "--", "safety signals"),
                            (C_MECH, "-", "mechanical link")], step=3.75)
    save(fig, "fig4-1_system_architecture")


# --- 4.2 Per-box control flowchart ---------------------------------------------------------------------------------


def fig_flowchart():
    W = 20.0
    nodes = [
        ("term", "Next box in the plan", None),
        ("proc", "Buffer releases the box; the conveyor brings it to the pick stop", None),
        ("dec", "Vision finds it and\nthe size matches?",
         ("No", "Retry vision twice; if still no, send it to the\nmanual lane and flag it")),
        ("proc", "Gripper descends onto the top face; seal at full vacuum (80 kPa)", None),
        ("dec", "Seal confirmed\nwithin 0.5 s?", ("No", "Re-seat up to 2 times; if still no,\nreject the box")),
        ("proc", "Lift 8 cm and hold still; the wrist F/T sensor weighs the box", None),
        ("dec", "Differs from the\ncheckweigher by > 5 %?",
         ("Yes", "Flag the mismatch; continue with the\nmeasured weight", True)),
        ("dec", "Box + 4 kg gripper\nover 20 kg?", ("Yes", "Set the box back down and reject it\n(over the UR20 payload)")),
        ("proc", "Set vacuum:  F = margin × SF × m × (g + a / 0.5)\nP = F ÷ 78.5 cm², limited to 8–80 kPa", None),
        ("proc", "Carry to the pallet; move time grows with weight and distance", None),
        ("dec", "Vacuum stays above\nthe hold level?", ("No", "Raise to 80 kPa, protective stop,\nalert the operator")),
        ("proc", "Lower to 3 cm above the spot; vision measures the box\nand the arm corrects its position (2 passes)", None),
        ("proc", "Release 2 mm above the resting place; retract", None),
        ("dec", "Vision: placement\nerror ≤ 5 mm?", ("No", "Re-place once; if still off,\nflag and log it", True)),
        ("proc", "Log weight, vacuum and placement error", None),
        ("dec", "More boxes\nin the plan?", ("loop", None)),
        ("term", "Pallet stands 2 s (stability check);\nrelease the finished pallet", None),
    ]
    heights = {"term": 1.05, "proc": 1.15, "dec": 1.55}
    gap = 0.5
    H = sum(heights[k] for k, _, _ in nodes) + gap * (len(nodes) - 1) + 1.0
    fig, ax = new_fig(W, H, 7.0)
    cx, pw, dw = 6.6, 10.6, 5.8
    fx, fw = 13.3, 6.4
    y = H - 0.5
    centers = []
    for kind, text, fault in nodes:
        h = heights[kind]
        cy = y - h / 2
        centers.append((kind, cy, h))
        if kind == "term":
            fc, ec = PAL["term"]
            ax.add_patch(FancyBboxPatch((cx - pw / 2 + 0.5, cy - h / 2), pw - 1.0, h,
                                        boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=fc, ec=ec))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=7.6, color="white", fontweight="bold",
                    linespacing=1.15)
        elif kind == "proc":
            fc, ec = PAL["sw"]
            ax.add_patch(FancyBboxPatch((cx - pw / 2, cy - h / 2), pw, h, boxstyle="round,pad=0,rounding_size=0.1",
                                        fc=fc, ec=ec, lw=1.2))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=7.3, color=TXT, linespacing=1.2)
        else:
            diamond(ax, cx, cy, dw, h, text, fs=7.1)
            if fault and fault[0] != "loop":
                word, ftext = fault[0], fault[1]
                rejoin = len(fault) > 2
                fc, ec = PAL["fault"]
                ax.add_patch(FancyBboxPatch((fx, cy - 0.55), fw, 1.1, boxstyle="round,pad=0,rounding_size=0.1",
                                            fc=fc, ec=ec, lw=1.2))
                ax.text(fx + fw / 2, cy, ftext, ha="center", va="center", fontsize=6.9, color=TXT)
                arrow(ax, [(cx + dw / 2, cy), (fx, cy)], color=C_FAULT, label=word,
                      lpos=((cx + dw / 2 + fx) / 2, cy + 0.3), fs=7)
                if rejoin:   # continues down the main flow
                    nxt = y - h - gap
                    arrow(ax, [(fx + fw / 2, cy - 0.55), (fx + fw / 2, nxt + 0.25), (cx + 0.12, nxt + 0.25)],
                          color=C_FAULT, ls="--", lw=1.0)
        y -= h + gap
    for i in range(len(centers) - 1):
        kind, cy, h = centers[i]
        _, ny, nh = centers[i + 1]
        arrow(ax, [(cx, cy - h / 2), (cx, ny + nh / 2)], color=TXT, lw=1.1)
        if kind == "dec":
            word = nodes[i][2][0]
            down = {"No": "Yes", "Yes": "No", "loop": "No"}[word]
            ax.text(cx - 0.15, cy - h / 2 - 0.22, down, ha="right", va="center", fontsize=7, color=TXT)
    li = [i for i, n in enumerate(nodes) if n[2] and n[2][0] == "loop"][0]
    _, ly, _ = centers[li]
    _, ty, _ = centers[1]
    lx = 0.45
    arrow(ax, [(cx - dw / 2, ly), (lx, ly), (lx, ty), (cx - pw / 2, ty)], color=C_DATA, label="Yes",
          lpos=(lx + 0.9, ly + 0.3), lw=1.2)
    ax.text(fx + fw / 2, H - 0.45, "Fault handling", ha="center", va="center", fontsize=8.2, fontweight="bold",
            color=C_FAULT)
    ax.text(fx + fw / 2, H - 1.25, "Rejected or flagged boxes go to the\nreject lane; the cycle then moves on\n"
            "to the next box in the plan.", ha="center", va="center", fontsize=6.7, color=GREY, style="italic")
    save(fig, "fig4-2_per_box_flowchart")


# --- 4.3 One-cycle swimlane ----------------------------------------------------------------------------------------


def fig_swimlane():
    lanes = ["Infeed + buffer", "Supervisor", "Vision", "Motion controller\n+ UR20", "Grip controller\n+ vacuum",
             "Wrist F/T sensor"]
    kinds = ["data", "sw", "sense", "mech", "vac", "sense"]
    lane_h, label_w, col_w, bw, bh = 1.55, 3.4, 2.1, 1.92, 1.05
    steps = [  # (column, lane, text)
        (0, 1, "Request box k\n(plan order)"),
        (1, 0, "Release box;\nstop at pick"),
        (2, 2, "Locate box\n(pose, size)"),
        (3, 3, "Descend onto\ntop face"),
        (4, 4, "Seal at 80 kPa;\nconfirm"),
        (5, 3, "Lift 8 cm,\nhold still"),
        (6, 5, "Weigh box\n(median)"),
        (7, 1, "Check ±5 %\nand ≤ 20 kg"),
        (8, 4, "Set vacuum\n8–80 kPa"),
        (9, 3, "Carry to\nthe pallet"),
        (10, 2, "Measure box\nabove spot"),
        (11, 3, "Correct ×2;\nlower to 2 mm"),
        (12, 4, "Vent:\nrelease"),
        (13, 2, "Verify\nplacement"),
        (14, 1, "Log;\nnext box"),
    ]
    ncols = 15
    W = label_w + ncols * col_w + 0.4
    H = len(lanes) * lane_h + 1.2
    fig, ax = new_fig(W, H, 14.0, y0=-0.9)
    top = H - 0.9 - 0.3
    for i, (name, k) in enumerate(zip(lanes, kinds)):
        y0 = top - (i + 1) * lane_h
        fc, ec = PAL[k]
        ax.add_patch(Rectangle((0.1, y0), W - 0.2, lane_h, fc=fc if i % 2 == 0 else "white", ec="#c9ced6", lw=0.8))
        ax.add_patch(Rectangle((0.1, y0), label_w - 0.15, lane_h, fc=fc, ec=ec, lw=1.0))
        ax.text(0.1 + (label_w - 0.15) / 2, y0 + lane_h / 2, name, ha="center", va="center", fontsize=8,
                fontweight="bold", color=ec)

    def xy(col, lane):
        return label_w + col * col_w + col_w / 2, top - lane * lane_h - lane_h / 2

    pos = []
    for col, lane, text in steps:
        x, y = xy(col, lane)
        _, ec = PAL[kinds[lane]]
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec=ec, lw=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=6.9, color=TXT, linespacing=1.15)
        pos.append((x, y))
    for (x1, y1), (x2, y2) in zip(pos[:-1], pos[1:]):
        xm = (x1 + bw / 2 + x2 - bw / 2) / 2
        if abs(y1 - y2) < 1e-6:
            arrow(ax, [(x1 + bw / 2, y1), (x2 - bw / 2, y2)], color=TXT, lw=1.0)
        else:
            arrow(ax, [(x1 + bw / 2, y1), (xm, y1), (xm, y2), (x2 - bw / 2, y2)], color=TXT, lw=1.0)
    xs, ys = xy(8, 4)
    xe, _ = xy(12, 4)
    ax.plot([xs + bw / 2 + 0.05, xe - bw / 2 - 0.05], [ys - 0.62, ys - 0.62], color=C_VAC, lw=2.2, ls=(0, (3, 2)))
    ax.text((xs + xe) / 2, ys - 0.62, "vacuum monitored during the carry", ha="center", va="center", fontsize=6.8,
            color=C_VAC, bbox=WHITE_BOX)
    phases = [(0, 1, "Feed"), (2, 4, "Pick"), (5, 8, "Weigh + set grip"), (9, 12, "Carry + place"),
              (13, 14, "Verify")]
    for c0, c1, name in phases:
        x0 = label_w + c0 * col_w + 0.08
        x1 = label_w + (c1 + 1) * col_w - 0.08
        ax.plot([x0, x1], [-0.35, -0.35], color=GREY, lw=1.0)
        ax.plot([x0, x0], [-0.45, -0.25], color=GREY, lw=1.0)
        ax.plot([x1, x1], [-0.45, -0.25], color=GREY, lw=1.0)
        ax.text((x0 + x1) / 2, -0.62, name, ha="center", va="center", fontsize=7.6, color=GREY, fontweight="bold")
    ax.text(0.1 + (label_w - 0.15) / 2, -0.5, "time →", ha="center", va="center", fontsize=7.6, color=GREY)
    save(fig, "fig4-3_cycle_swimlane")


# --- 4.4 Vacuum circuit ------------------------------------------------------------------------------------------


def fig_vacuum():
    fig, ax = new_fig(20, 9.4, 10.5)
    y = 5.2
    row = [(0.3, 2.3, "Compressed air", "≈6 bar shop supply"), (3.1, 2.2, "Filter +\nregulator", None),
           (5.8, 2.3, "Solenoid valve", "‘vacuum on’"), (8.6, 2.5, "Vacuum generator", "venturi ejector"),
           (11.6, 2.6, "Proportional\nregulator", "8–80 kPa"), (14.7, 1.9, "Check\nvalve", "holds vacuum\nif air is lost"),
           (17.1, 2.6, "Manifold", "to the 4 cups")]
    for x, w, t, b in row:
        box(ax, x, y - 0.9, w, 1.8, t, b, "vac", fs=6.9 if w < 2.0 else 7.2)
    for x1, x2 in [(2.6, 3.1), (5.3, 5.8), (8.1, 8.6)]:
        arrow(ax, [(x1, y), (x2, y)], color=C_AIR, lw=2.4)
    for x1, x2 in [(11.1, 11.6), (14.2, 14.7), (16.6, 17.1)]:
        arrow(ax, [(x1, y), (x2, y)], color=C_VAC, lw=2.4)
    ax.text(5.5, y + 1.2, "compressed air", ha="center", fontsize=7.2, color=C_AIR)
    ax.text(14.3, y + 1.2, "vacuum", ha="center", fontsize=7.2, color=C_VAC)
    # blow-off branch
    box(ax, 6.0, 6.95, 3.2, 1.2, "Blow-off valve", "short pulse to release", "vac", fs=7.0)
    arrow(ax, [(4.2, y + 0.9), (4.2, 7.55), (6.0, 7.55)], color=C_AIR, lw=1.6)
    arrow(ax, [(9.2, 7.55), (18.4, 7.55), (18.4, y + 0.9)], color=C_AIR, lw=1.6, ls="--", label="release pulse",
          lpos=(13.5, 7.55))
    # gripper + box
    gx, gy = 15.0, 1.55
    ax.add_patch(Rectangle((gx - 0.2, gy + 0.95), 4.4, 0.32, fc="#c9ced6", ec=GREY, lw=1.0))
    for i in range(4):
        cxp = gx + 0.35 + i * 1.08
        ax.add_patch(Polygon([(cxp - 0.25, gy + 0.95), (cxp + 0.25, gy + 0.95), (cxp + 0.36, gy + 0.45),
                              (cxp - 0.36, gy + 0.45)], closed=True, fc="#2b2f33", ec="#2b2f33"))
    ax.add_patch(Rectangle((gx - 0.6, gy - 1.2), 5.2, 1.62, fc="#d8b98a", ec=C_MECH, lw=1.0))
    ax.text(gx + 2.0, gy - 0.4, "Box  (1.5–16 kg)", ha="center", va="center", fontsize=7.6, color=TXT)
    ax.text(gx + 1.4, gy + 1.5, "4 × Ø50 mm cups on a 22 cm plate", ha="center", va="center", fontsize=6.9,
            color=TXT)
    arrow(ax, [(18.4, y - 0.9), (18.4, gy + 1.27)], color=C_VAC, lw=2.4)
    # sensor + controller
    box(ax, 11.4, 2.15, 2.3, 1.3, "Vacuum\npressure sensor", None, "sense", fs=6.8)
    arrow(ax, [(17.4, y - 0.9), (17.4, 3.95), (12.2, 3.95), (12.2, 3.45)], color=C_VAC, lw=1.4)
    box(ax, 6.9, 1.2, 3.6, 1.8, "Grip controller", "setpoint from the box's\nweight; seal confirmed\nwhen pressure is reached",
        "sw", fs=6.8)
    arrow(ax, [(11.4, 2.8), (10.5, 2.8)], color=C_DATA, ls="--")
    ax.text(10.95, 2.52, "pressure", ha="center", fontsize=6.2, color=C_DATA)
    arrow(ax, [(10.5, 1.6), (13.9, 1.6), (13.9, y - 0.9)], color=C_DATA, ls="--", label="setpoint",
          lpos=(12.2, 1.6), fs=6.8)
    arrow(ax, [(7.6, 3.0), (7.6, 3.6), (6.95, 3.6), (6.95, y - 0.9)], color=C_DATA, ls="--", label="on / off",
          lpos=(7.3, 3.6), fs=6.8)
    # calculation panel
    px, py = 0.4, 0.3
    ax.add_patch(FancyBboxPatch((px, py), 5.9, 3.55, boxstyle="round,pad=0,rounding_size=0.12", fc="#fafbfc",
                                ec="#9aa3ad", lw=1.0))
    ax.text(px + 0.2, py + 3.3, "Holding capacity", fontsize=8, fontweight="bold", color=TXT, va="top")
    ax.text(px + 0.2, py + 2.85, "A = 4 × π × (2.5 cm)² = 78.5 cm²\nF = P × A = 80 kPa × 78.5 cm² = 628 N",
            fontsize=7, color=TXT, va="top", linespacing=1.35)
    rows = [("Box", "Vacuum", "Holds"), ("3.3 kg", "9.9 kPa", "78 N"), ("7.3 kg", "20.8 kPa", "163 N"),
            ("16.0 kg", "42.9 kPa", "337 N"), ("full", "80 kPa", "628 N")]
    for r, rr in enumerate(rows):
        for c, cell in enumerate(rr):
            ax.text(px + 0.4 + c * 1.85, py + 1.75 - r * 0.33, cell, fontsize=7, color=TXT,
                    fontweight="bold" if r == 0 else "normal", va="center")
    ax.text(px + 0.2, py + 0.1, "vacuum per box from the Isaac Sim runs", fontsize=6.4, color=GREY, va="bottom",
            style="italic")
    save(fig, "fig4-4_vacuum_circuit")


# --- 4.5 Weight data flow ------------------------------------------------------------------------------------------


def fig_weight_flow():
    fig, ax = new_fig(20, 9.6, 10.5)
    box(ax, 0.4, 7.15, 3.6, 1.7, "Checkweigher", "listed weight, at the\ninfeed (before planning)", "data", fs=7.2)
    box(ax, 0.4, 1.2, 3.6, 1.7, "Wrist F/T sensor", "measured weight,\nafter an 8 cm lift", "sense", fs=7.2)
    diamond(ax, 2.2, 4.55, 3.4, 1.7, "cross-check:\ndiffer by > 5 %?", fs=7)
    arrow(ax, [(2.2, 7.15), (2.2, 5.4)], color=C_DATA)
    arrow(ax, [(2.2, 2.9), (2.2, 3.7)], color=C_VAC)
    ax.text(3.0, 3.55, "yes: flag the mismatch", fontsize=6.8, color=C_FAULT, ha="left")
    ax.add_patch(Circle((6.6, 4.55), 1.2, fc="#2f3b4c", ec="#2f3b4c"))
    ax.text(6.6, 4.85, "box weight", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(6.6, 4.45, "m", ha="center", va="center", fontsize=11, color="white", style="italic")
    ax.text(6.6, 4.05, "(measured)", ha="center", va="center", fontsize=6.8, color="white")
    arrow(ax, [(3.9, 4.55), (5.4, 4.55)], color=TXT)

    box(ax, 10.0, 6.9, 4.4, 2.2, "Pallet planner", "where each box goes:\nheavy boxes low, centred load,\nno box over its crush rating",
        "sw", fs=7.0)
    box(ax, 10.0, 3.45, 4.4, 2.2, "Motion timing", "move time × (1 + 0.06 m)\nlonger, gentler moves\nfor heavier boxes",
        "sw", fs=7.0)
    box(ax, 10.0, 0.3, 4.4, 2.2, "Grip controller", "P = margin × SF × m × (g + a / 0.5) ÷ A\nvacuum matched to\nthis box and this move",
        "sw", fs=6.8)
    arrow(ax, [(4.0, 8.0), (10.0, 8.0)], color=TXT, label="listed weight (before the first pick)",
          lpos=(7.0, 8.0), fs=6.8)
    arrow(ax, [(7.8, 4.55), (10.0, 4.55)], color=TXT, label="m", lpos=(8.9, 4.55), fs=7.2)
    arrow(ax, [(7.45, 3.7), (8.7, 1.4), (10.0, 1.4)], color=TXT, label="m", lpos=(8.35, 2.2), fs=7.2)
    arrow(ax, [(12.2, 6.9), (12.2, 5.65)], color=C_DATA, label="spot → path length", lpos=(13.3, 6.27), fs=6.8)
    arrow(ax, [(12.2, 3.45), (12.2, 2.5)], color=C_DATA, label="peak acceleration a", lpos=(13.35, 2.97), fs=6.8)
    for yy, text in [(8.0, "pallet plan\n(layer, spot, turn, order)"), (4.55, "arm path and\nmove durations"),
                     (1.4, "vacuum setpoint\n(8–80 kPa)")]:
        arrow(ax, [(14.4, yy), (15.6, yy)], color=TXT)
        ax.text(15.75, yy, text, ha="left", va="center", fontsize=7.2, color=TXT)
    ax.text(0.4, 0.35, "The box's weight drives all three decisions,\nso the plan, the motion and the grip stay consistent.",
            ha="left", va="bottom", fontsize=6.8, color=GREY, style="italic")
    save(fig, "fig4-5_weight_data_flow")


# --- 4.6 Cell layout ----------------------------------------------------------------------------------------------


def dim(ax, p1, p2, text, off=(0, 0), fs=7, color="#3b4350"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>", mutation_scale=7, color=color, lw=0.9, shrinkA=0,
                                 shrinkB=0))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], text, ha="center", va="center",
                fontsize=fs, color=color, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def fig_layout():
    fig = plt.figure(figsize=(12.0, 6.6))
    a = fig.add_axes([0.02, 0.04, 0.52, 0.93])
    b = fig.add_axes([0.58, 0.12, 0.40, 0.78])
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.axis("off")
    # ---- top view
    a.set_xlim(-2.5, 2.8)
    a.set_ylim(-2.45, 4.55)
    a.add_patch(Rectangle((-2.0, -2.0), 4.0, 4.0, fc="#fbfbfc", ec=C_SAFE, lw=1.6))
    a.plot([2.0, 2.0], [-0.65, 0.65], color="white", lw=3)
    a.plot([2.0, 2.0], [-0.65, 0.65], color=C_SAFE, lw=1.6, ls=(0, (2, 2)))
    a.text(2.08, 0.0, "interlocked gate\n(pallet exit)", ha="left", va="center", fontsize=7, color=C_SAFE)
    a.plot([-0.45, 0.45], [2.0, 2.0], color="white", lw=3)
    a.text(-1.95, -1.88, "safety fence (proposed, 4.0 × 4.0 m)", fontsize=7, color=C_SAFE, va="bottom")
    a.add_patch(Circle((0, 0), 1.75, fc="none", ec="#7a8592", lw=1.0, ls=(0, (5, 3))))
    a.text(-1.25, -1.3, "UR20 reach\nR 1.75 m", fontsize=7, color=GREY, ha="center", va="center", bbox=WHITE_BOX)
    a.add_patch(Rectangle((-0.16, -0.16), 0.32, 0.32, fc="#3d4a5c", ec="#2b2f33"))
    a.text(-0.24, -0.24, "UR20 on 0.7 m\npedestal", ha="right", va="top", fontsize=7, color=TXT)
    a.add_patch(Rectangle((0.38, -0.5), 1.0, 1.0, fc="#e8d3b0", ec=C_MECH, lw=1.2))
    for k in range(4):
        a.plot([0.40 + 0.32 * k, 0.40 + 0.32 * k], [-0.48, 0.48], color=C_MECH, lw=0.6)
        a.plot([0.40, 1.36], [-0.48 + 0.32 * k, -0.48 + 0.32 * k], color=C_MECH, lw=0.6)
    a.text(0.88, 0.0, "pallet: 3 × 3 grid\nof 32 cm cells", ha="center", va="center", fontsize=6.4, color=TXT,
           bbox=dict(boxstyle="round,pad=0.12", fc="#f6ead6", ec="none"))
    # conveyor from the sequencing buffer to the pick stop
    a.add_patch(Rectangle((-0.4, 0.55), 0.8, 2.05, fc="#dfe2e6", ec=GREY, lw=1.0))
    for yy in [0.75 + 0.15 * k for k in range(12)]:
        a.plot([-0.38, 0.38], [yy, yy], color="#b4bac2", lw=0.5)
    a.plot([-0.4, 0.4], [0.53, 0.53], color="#e0b000", lw=3)
    a.text(-0.45, 0.53, "pick stop", fontsize=6.4, color="#9a7a00", ha="right", va="center")
    a.plot(0.0, 0.92, marker="+", color=C_DATA, ms=10, mew=1.6)
    a.text(0.5, 0.92, "pick point", fontsize=7, color=C_DATA, va="center")
    dim(a, (-0.4, 1.65), (0.4, 1.65), "0.8 m", off=(0, 0.14), fs=6.6)
    # upstream infeed (outside the fence): dimensioner -> checkweigher -> sequencing buffer
    a.add_patch(Rectangle((-0.9, 2.6), 1.8, 0.62, fc=PAL["mech"][0], ec=PAL["mech"][1], lw=1.1))
    a.text(0.0, 2.91, "sequencing buffer", ha="center", va="center", fontsize=6.8, color=PAL["mech"][1])
    a.add_patch(Rectangle((-0.42, 3.32), 0.84, 0.42, fc=PAL["data"][0], ec=PAL["data"][1], lw=1.1))
    a.text(0.5, 3.53, "checkweigher", fontsize=7, color=PAL["data"][1], va="center")
    a.add_patch(Rectangle((-0.42, 3.84), 0.84, 0.42, fc=PAL["data"][0], ec=PAL["data"][1], lw=1.1))
    a.text(0.5, 4.05, "dimensioner", fontsize=7, color=PAL["data"][1], va="center")
    for y1, y2 in [(3.84, 3.74), (3.32, 3.22)]:
        a.annotate("", xy=(0.0, y2), xytext=(0.0, y1), arrowprops=dict(arrowstyle="-|>", color=TXT, lw=0.9))
    a.annotate("", xy=(0.0, 4.26), xytext=(0.0, 4.48), arrowprops=dict(arrowstyle="-|>", color=TXT))
    a.text(0.12, 4.42, "boxes from the infeed", fontsize=7, color=TXT, va="center")
    # cameras
    a.add_patch(Circle((0.0, 0.92), 0.42, fc="none", ec=C_VAC, lw=0.9, ls=(0, (2, 2))))
    a.add_patch(Circle((0.88, 0.0), 0.78, fc="none", ec=C_VAC, lw=0.9, ls=(0, (2, 2))))
    a.text(-0.5, 1.22, "Camera A view", fontsize=6.6, color=C_VAC, ha="right", va="center")
    a.text(1.7, 0.78, "Camera B\nview", fontsize=6.6, color=C_VAC, ha="left")
    for ex, ey in [(2.0, -1.0), (-2.0, 2.0), (2.0, 2.0)]:
        a.add_patch(Circle((ex, ey), 0.08, fc=C_SAFE, ec="white", lw=0.8, zorder=5))
    a.text(2.12, -1.0, "E-stop", fontsize=7, color=C_SAFE, va="center")
    dim(a, (0.0, -0.75), (0.88, -0.75), "0.88 m", off=(0, -0.14))
    dim(a, (0.38, -1.05), (1.38, -1.05), "1.00 m", off=(0, -0.14))
    dim(a, (-2.0, -2.25), (2.0, -2.25), "4.0 m", off=(0, -0.13))
    a.text(-2.45, 4.5, "Top view (to scale, metres)", fontsize=9, fontweight="bold", color=TXT, va="top")
    # ---- side view (section through the robot and the pallet)
    b.set_xlim(-0.9, 2.1)
    b.set_ylim(-0.12, 1.95)
    b.plot([-0.9, 2.1], [0, 0], color=GREY, lw=1.2)
    b.add_patch(Rectangle((-0.16, 0), 0.32, 0.70, fc="#3d4a5c", ec="#2b2f33"))
    b.add_patch(Circle((0, 0.936), 0.07, fc="#c9ced6", ec=GREY))
    b.text(-0.12, 0.936, "shoulder", ha="right", va="center", fontsize=6.6, color=TXT)
    b.add_patch(Rectangle((0.38, 0), 1.0, 0.144, fc="#c9a46b", ec=C_MECH))
    layers = [(0.144, 0.22), (0.364, 0.22), (0.584, 0.26)]
    for (z0, h), col in zip(layers, ["#d8b98a", "#e2c79d", "#ead6b4"]):
        b.add_patch(Rectangle((0.40, z0), 0.96, h, fc=col, ec=C_MECH, lw=0.8))
    b.plot([0.3, 1.5], [0.924, 0.924], color=C_SAFE, lw=0.9, ls=(0, (3, 2)))
    b.text(1.52, 0.924, "tallest possible stack\n0.92 m (3 × 26 cm)", fontsize=6.4, color=C_SAFE, va="center")
    b.plot([-0.4, 1.6], [1.25, 1.25], color=C_DATA, lw=0.9, ls=(0, (3, 2)))
    b.text(1.62, 1.25, "transit height of\nthe cups: 1.25 m", fontsize=6.4, color=C_DATA, va="center")
    b.add_patch(Rectangle((0.73, 0.99), 0.30, 0.26, fc="#d8b98a", ec=C_MECH, lw=0.9))
    b.add_patch(Rectangle((0.75, 1.25), 0.26, 0.05, fc="#2b2f33"))
    b.add_patch(Rectangle((0.84, 1.30), 0.08, 0.16, fc="#9aa3ad"))
    b.text(0.62, 1.15, "carried box\n(up to 26 cm)", ha="right", va="center", fontsize=6.4, color=TXT)
    dim(b, (0.68, 0.924), (0.68, 0.99), "")
    b.text(0.62, 0.957, "≥ 6 cm clearance", ha="right", va="center", fontsize=6.2, color="#3b4350")
    dim(b, (-0.4, 0), (-0.4, 0.70), "0.70 m\npedestal", off=(-0.22, 0), fs=6.4)
    dim(b, (1.45, 0), (1.45, 0.144), "")
    b.text(1.5, 0.07, "pallet 0.144 m", fontsize=6.4, color="#3b4350", va="center")
    for (z0, h), lab in zip(layers, ["layer 1: 22 cm", "layer 2: 22 cm", "layer 3: 26 cm"]):
        b.text(0.88, z0 + h / 2, lab, ha="center", va="center", fontsize=6.4, color=TXT)
    b.text(-0.88, 1.93, "Side view (box set 7; arm not drawn)", fontsize=9, fontweight="bold", color=TXT, va="top")
    save(fig, "fig4-6_cell_layout")


if __name__ == "__main__":
    fig_architecture()
    fig_flowchart()
    fig_swimlane()
    fig_vacuum()
    fig_weight_flow()
    fig_layout()
