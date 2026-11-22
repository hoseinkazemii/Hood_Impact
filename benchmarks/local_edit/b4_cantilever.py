"""B4: notched S355 cantilever, tip pushed down 10 mm into a plastic hinge (CPS4I, plane stress).

One structured 0.5 mm grid on the 200 x 20 mm beam (10 mm thick). A notch is
made by pulling each node column down from the top edge (or up from the bottom
edge) to a 60-degree V with a 1 mm root radius, and the column's other nodes
are spaced evenly between the two edges. Every design therefore keeps the same
nodes and elements, and the change nodes are the columns under the edited notches.

Edits live in the root half, where a notch sets the collapse load: with the tip
load the moment grows towards the root, so a notch at x governs when
(remaining depth)^2 / (L - x) is the smallest along the beam (hand model below).
Root radius is fixed: it hardly changes the collapse load.

Main output: tip force vs tip deflection. Fields: U, S, LE, PEEQ at 21 frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import quad_quality, save_design, write_mesh, write_nset, write_s355

BENCHMARK = "B4_cantilever"
L, H, B = 200.0, 20.0, 10.0
NX, NY = 400, 40
TIP_DISPLACEMENT = 10.0  # 15 mm distorted the 0.5 mm elements at a 7 mm notch root
ROOT_RADIUS = 1.0
HALF_ANGLE = np.radians(30.0)  # 60-degree V
YIELD = 355.0

BASE = [dict(x=50.0, depth=4.0, side="top"), dict(x=95.0, depth=5.0, side="bottom")]
DESIGNS = [  # name, edit, notches
    ("base", "family base: notches at 50 mm (4 mm, top) and 95 mm (5 mm, bottom)", BASE),
    ("e1_deepen", "governing notch at 50 mm deepened 4 -> 7 mm",
     [dict(x=50.0, depth=7.0, side="top"), BASE[1]]),
    ("e2_move", "governing notch moved from 50 to 25 mm from the root",
     [dict(x=25.0, depth=4.0, side="top"), BASE[1]]),
    ("e3_add_pair", "two new 5 mm top notches at 25 and 35 mm",
     BASE + [dict(x=25.0, depth=5.0, side="top"), dict(x=35.0, depth=5.0, side="top")]),
    ("e4_shallow", "governing notch at 50 mm made shallow, 4 -> 1.5 mm",
     [dict(x=50.0, depth=1.5, side="top"), BASE[1]]),
    ("e5_group", "notch at 95 mm deepened 5 -> 9 mm plus a new 6 mm top notch at 110 mm",
     [BASE[0], dict(x=95.0, depth=9.0, side="bottom"), dict(x=110.0, depth=6.0, side="top")]),
]


def grid():
    x, y = np.meshgrid(np.linspace(0, L, NX + 1), np.linspace(0, H, NY + 1))
    xyz = np.column_stack([x.ravel(), y.ravel()])
    n = lambda i, j: j * (NX + 1) + i
    quads = np.array([[n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)]
                      for j in range(NY) for i in range(NX)])
    return xyz, quads


def notch_depth(x, notch):
    """Depth of a V notch with a rounded root at horizontal positions x."""
    u, a, r, b = np.abs(x - notch["x"]), notch["depth"], ROOT_RADIUS, HALF_ANGLE
    u_t, d_t = r * np.cos(b), a - r + r * np.sin(b)
    depth = np.where(u <= u_t, a - r + np.sqrt(np.maximum(r * r - u * u, 0.0)), d_t - (u - u_t) / np.tan(b))
    return np.clip(depth, 0.0, None)


def edge_depths(x, notches):
    top, bottom = np.zeros_like(x), np.zeros_like(x)
    for notch in notches:
        if notch["side"] == "top":
            top = np.maximum(top, notch_depth(x, notch))
        else:
            bottom = np.maximum(bottom, notch_depth(x, notch))
    return top, bottom


def morph(xyz, notches):
    top, bottom = edge_depths(xyz[:, 0], notches)
    out = xyz.copy()
    out[:, 1] = bottom + xyz[:, 1] / H * (H - top - bottom)
    return out


def collapse_load(notches):
    """Plane-stress limit analysis: min over sections of Mp(x) / (L - x)."""
    x = np.linspace(0.0, L - 20.0, 3601)
    top, bottom = edge_depths(x, notches)
    load = YIELD * B * (H - top - bottom) ** 2 / 4.0 / (L - x)
    i = int(np.argmin(load))
    return float(load[i]), float(x[i])


def write_inp(path, xyz, quads, title):
    root = np.where(xyz[:, 0] < 1e-6)[0]
    tip = np.where(xyz[:, 0] > L - 1e-6)[0]
    with open(path, "w") as f:
        f.write(f"*HEADING\n{title}\n")
        write_mesh(f, xyz, quads, "CPS4I", "BEAM")
        write_nset(f, "ROOT", root)
        write_nset(f, "TIP", tip)
        f.write(f"*SOLID SECTION, ELSET=BEAM, MATERIAL=S355\n{B},\n")
        write_s355(f)
        f.write("*BOUNDARY\nROOT, 1, 2, 0.\n")
        f.write("*STEP, NLGEOM=YES, INC=2000\n*STATIC\n0.01, 1., 1e-06, 0.02\n")
        f.write(f"*BOUNDARY\nTIP, 2, 2, {-TIP_DISPLACEMENT}\n")
        f.write("*OUTPUT, FIELD, TIME INTERVAL=0.05\n*NODE OUTPUT\nU,\n*ELEMENT OUTPUT, DIRECTIONS=YES\nS, LE, PEEQ\n")
        f.write("*OUTPUT, HISTORY, FREQUENCY=1\n*NODE OUTPUT, NSET=TIP\nU2, RF2\n*END STEP\n")


def generate(out_root):
    out_root = Path(out_root) / BENCHMARK
    template, quads = grid()
    base_xyz = morph(template, BASE)
    base_load, _ = collapse_load(BASE)
    records = []
    for name, edit, notches in DESIGNS:
        xyz = morph(template, notches)
        quality = quad_quality(xyz, quads)
        if quality < 0.2:
            raise ValueError(f"{name}: element quality {quality:.2f}")
        load, hinge = collapse_load(notches)
        proxy = {"model": "plane-stress limit analysis", "collapse_load_N": round(load, 1),
                 "hinge_x_mm": round(hinge, 1), "change_vs_base_percent": round(100 * (load / base_load - 1), 1)}
        folder = out_root / name
        record = save_design(folder, name, benchmark=BENCHMARK, family="A", base="base", edit=edit,
                             params={"notches": notches, "root_radius_mm": ROOT_RADIUS, "notch_angle_deg": 60},
                             xyz=xyz, base_xyz=base_xyz, elements=quads, etype="CPS4I", proxy=proxy)
        record["min_scaled_jacobian"] = round(quality, 3)
        write_inp(folder / f"{name}.inp", xyz, quads, f"{BENCHMARK} {name}: {edit}")
        records.append(record)
    return records
