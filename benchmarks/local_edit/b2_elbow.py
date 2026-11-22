"""B2: 6-inch X65 pipe elbow with one external corrosion defect, burst by injecting water (C3D8I).

A 90-degree long-radius elbow (OD 168.3 mm, t 7.11 mm, bend radius 229 mm)
with 270 mm straight tangents, both ends anchored. A straight pipe would not
do: under internal pressure a defect can slide along it or around it without
changing the burst pressure. In a bend the hoop stress is ~27 % higher on the
inside (intrados) and ~13 % lower on the outside (extrados), so where the defect
sits matters as much as how deep it is.

Structured hex mesh (72 around x 116 along x 3 through the wall). A defect is
made by moving nodes radially inward: the outer surface by the defect depth,
the through-thickness nodes proportionally, the inner surface not at all. The
mesh never changes, so the change nodes are the defect footprint(s).

Loading: water is pumped into the closed bore (surface-based fluid cavity,
mass flow), so the pressure-volume curve passes its peak (burst pressure)
without arc-length control.

Main output: cavity pressure vs injected volume. Fields: U, S, LE, PEEQ at 21 frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import boundary_faces, hex_quality, save_design, write_mesh, write_nset, write_surface

BENCHMARK = "B2_elbow"
OD, T = 168.3, 7.11
RO, RI = OD / 2, OD / 2 - T
RB = 229.0  # bend radius (6-inch long-radius elbow)
ARC = RB * np.pi / 2
N_CIRC, N_WALL, N_BEND, N_FINE, N_GRADED = 72, 3, 60, 20, 8
FINE_LENGTH, GRADED_LENGTH = 120.0, 150.0
TANGENT = FINE_LENGTH + GRADED_LENGTH
SMYS, UTS = 448.0, 531.0
X65_PLASTIC = [(448.0, 0.0), (470.0, 0.005), (495.0, 0.015), (520.0, 0.03), (545.0, 0.05),
               (570.0, 0.08), (595.0, 0.12), (625.0, 0.18), (660.0, 0.30)]
WATER_DENSITY, WATER_BULK = 1.0e-9, 20000.0  # tonne/mm^3; MPa (stiffened so the pump drives the pipe)
INJECTED_VOLUME = 1.0e6  # mm^3 over the step; pilot peaks came at 1.5-3.8e5
# Increment sizes as fractions of the step (x INJECTED_VOLUME = mm^3). Past the burst peak the bulge
# localises and the solver cuts back to ever-smaller increments for hours; the 200 mm^3 floor makes the
# job abort there instead ("aborted" past the peak is the normal end of a B2 run).
STATIC_INCREMENTS = "0.01, 1., 2e-04, 0.01"  # initial, period, minimum, maximum
FIELD_INTERVAL = 0.02  # a field frame every 2e4 mm^3 of injected water

S_MID_BEND = TANGENT + ARC / 2
BASE = dict(s=S_MID_BEND, theta=60.0, length=100.0, width=40.0, depth=0.45, angle=0.0)
DESIGNS = [  # name, edit, defects (s along centreline mm, theta from intrados deg, length/width mm at OD, depth/t, angle deg)
    ("base", "family base: 100 x 40 mm defect, 45 % deep, mid-bend, 60 deg from the intrados", [BASE]),
    ("e1_deepen", "defect deepened 45 -> 60 % of the wall", [{**BASE, "depth": 0.60}]),
    ("e2_to_intrados", "defect moved round to the intrados (60 -> 0 deg)", [{**BASE, "theta": 0.0}]),
    ("e3_to_tangent", "defect moved out of the bend, 65 mm into the straight tangent",
     [{**BASE, "s": TANGENT + ARC + 65.0}]),
    ("e4_lengthen", "defect lengthened 100 -> 200 mm along the pipe", [{**BASE, "length": 200.0}]),
    ("e5_rotate", "defect turned 90 deg (long axis around the pipe)", [{**BASE, "angle": 90.0}]),
]


def stations():
    """Centreline positions of the node rings, from anchored end 1 to anchored end 2."""
    lo, hi = 1.0, 2.0
    for _ in range(100):  # graded sizes h, h q, ..., h q^(n-1) ending at the fine size
        q = (lo + hi) / 2
        total = sum((FINE_LENGTH / N_FINE) * q ** k for k in range(1, N_GRADED + 1))
        lo, hi = (q, hi) if total < GRADED_LENGTH else (lo, q)
    graded = np.cumsum([(FINE_LENGTH / N_FINE) * q ** k for k in range(N_GRADED, 0, -1)])
    s = [0.0] + list(graded)
    s += list(GRADED_LENGTH + np.linspace(0, FINE_LENGTH, N_FINE + 1)[1:])
    s += list(TANGENT + np.linspace(0, ARC, N_BEND + 1)[1:])
    s += list(TANGENT + ARC + np.linspace(0, FINE_LENGTH, N_FINE + 1)[1:])
    s += list(TANGENT + ARC + FINE_LENGTH + (graded[-1] - graded[::-1])[1:]) + [2 * TANGENT + ARC]
    return np.array(s)


def frame(s):
    """Centreline point P and in-plane unit vector e1 (towards the bend centre) at centreline position s."""
    s = np.asarray(s, dtype=float)
    psi = np.clip((s - TANGENT) / RB, 0.0, np.pi / 2)
    p = np.stack([RB * np.cos(psi), RB * np.sin(psi), np.zeros_like(s)], -1)
    e1 = np.stack([-np.cos(psi), -np.sin(psi), np.zeros_like(s)], -1)
    before, after = s < TANGENT, s > TANGENT + ARC
    p[before] = np.stack([np.full(before.sum(), RB), s[before] - TANGENT, np.zeros(before.sum())], -1)
    p[after] = np.stack([-(s[after] - TANGENT - ARC), np.full(after.sum(), RB), np.zeros(after.sum())], -1)
    return p, e1


def axial_metric(s, theta):
    """Surface length per unit centreline length at the outer wall (1 on the tangents)."""
    in_bend = (s > TANGENT) & (s < TANGENT + ARC)
    return np.where(in_bend, (RB - RO * np.cos(np.radians(theta))) / RB, 1.0)


def template():
    """(s, theta, wall index) of every node, plus the C3D8 connectivity."""
    s = stations()
    theta = np.arange(N_CIRC) * 360.0 / N_CIRC
    ss, tt, kk = np.meshgrid(s, theta, np.arange(N_WALL + 1), indexing="ij")
    node = lambda i, j, k: (i * N_CIRC + j % N_CIRC) * (N_WALL + 1) + k
    hexes = []
    for i in range(len(s) - 1):
        for j in range(N_CIRC):
            for k in range(N_WALL):
                bottom = [node(i, j, k), node(i, j + 1, k), node(i, j + 1, k + 1), node(i, j, k + 1)]
                top = [node(i + 1, j, k), node(i + 1, j + 1, k), node(i + 1, j + 1, k + 1), node(i + 1, j, k + 1)]
                hexes.append(bottom + top)
    return ss.ravel(), tt.ravel(), kk.ravel(), np.array(hexes)


def taper(x, half):
    """1 inside, cosine down to 0 over the outer 30 % of the half-size."""
    edge = 0.3 * half
    inner = half - edge
    return np.where(x <= inner, 1.0, np.where(x >= half, 0.0, 0.5 * (1 + np.cos(np.pi * (x - inner) / edge))))


def defect_depth(s, theta, defect):
    """Metal loss (mm) at the outer wall of node positions (s, theta)."""
    du = (s - defect["s"]) * axial_metric((s + defect["s"]) / 2, defect["theta"])
    dv = np.radians((theta - defect["theta"] + 180.0) % 360.0 - 180.0) * RO
    a = np.radians(defect["angle"])
    along = du * np.cos(a) + dv * np.sin(a)
    across = -du * np.sin(a) + dv * np.cos(a)
    return defect["depth"] * T * taper(np.abs(along), defect["length"] / 2) * taper(np.abs(across), defect["width"] / 2)


def coordinates(s, theta, k, defects):
    loss = np.zeros_like(s)
    for defect in defects:
        loss = np.maximum(loss, defect_depth(s, theta, defect))
    rho = RI + T * k / N_WALL - loss * k / N_WALL
    p, e1 = frame(s)
    th = np.radians(theta)[:, None]
    return p + rho[:, None] * (np.cos(th) * e1 + np.sin(th) * np.array([0.0, 0.0, 1.0]))


def lorenz(defect):
    """Hoop-stress factor at the defect centre relative to a straight pipe."""
    if not TANGENT < defect["s"] < TANGENT + ARC:
        return 1.0
    r, c = (RO + RI) / 2, np.cos(np.radians(defect["theta"]))
    return (2 * RB - r * c) / (2 * (RB - r * c))


def burst_proxy(defect):
    """DNV-RP-F101 single-defect factor over the local Lorenz factor; the intact intrados caps it."""
    length = abs(defect["length"] * np.cos(np.radians(defect["angle"]))) + \
        abs(defect["width"] * np.sin(np.radians(defect["angle"])))
    q = np.sqrt(1 + 0.31 * (length / np.sqrt(OD * T)) ** 2)
    d = defect["depth"]
    straight = 2 * T * UTS / (OD - T)
    intrados = lorenz({"s": S_MID_BEND, "theta": 0.0})
    return straight * min((1 - d) / (1 - d / q) / lorenz(defect), 1 / intrados)


def write_inp(path, xyz, hexes, k, s, title):
    faces = boundary_faces(hexes)
    inner = [f for f in faces if np.all(k[f[2]] == 0)]
    ends = np.where((s < 1e-9) | (s > 2 * TANGENT + ARC - 1e-6))[0]
    ref = len(xyz) + 1
    with open(path, "w") as out:
        out.write(f"*HEADING\n{title}\n")
        write_mesh(out, xyz, hexes, "C3D8I", "PIPE")
        out.write(f"*NODE\n{ref}, {-TANGENT:.4f}, {-TANGENT:.4f}, 0.\n*NSET, NSET=CAVREF\n{ref},\n")
        write_nset(out, "ENDS", ends)
        write_surface(out, "BORE", inner)
        out.write("*SOLID SECTION, ELSET=PIPE, MATERIAL=X65\n")
        out.write("*MATERIAL, NAME=X65\n*ELASTIC\n207000., 0.3\n*PLASTIC\n")
        out.write("".join(f"{a}, {b}\n" for a, b in X65_PLASTIC))
        out.write(f"*FLUID BEHAVIOR, NAME=WATER\n*FLUID DENSITY\n{WATER_DENSITY},\n*FLUID BULK MODULUS\n{WATER_BULK},\n")
        # The data line (out-of-plane thickness) is required; 1.0 is the default and unused in 3D.
        out.write("*FLUID CAVITY, NAME=CAVITY, BEHAVIOR=WATER, REF NODE=CAVREF, SURFACE=BORE\n1.0\n")
        out.write("*BOUNDARY\nENDS, ENCASTRE\n")
        out.write(f"*STEP, NLGEOM=YES, INC=5000\n*STATIC\n{STATIC_INCREMENTS}\n")
        out.write(f"*FLUID FLUX\nCAVREF, {INJECTED_VOLUME * WATER_DENSITY:.6e}\n")
        out.write(f"*OUTPUT, FIELD, TIME INTERVAL={FIELD_INTERVAL}\n*NODE OUTPUT\nU,\n"
                  "*ELEMENT OUTPUT, DIRECTIONS=YES\nS, LE, PEEQ\n")
        out.write("*OUTPUT, HISTORY, FREQUENCY=1\n*NODE OUTPUT, NSET=CAVREF\nPCAV, CVOL\n*END STEP\n")


def generate(out_root):
    out_root = Path(out_root) / BENCHMARK
    s, theta, k, hexes = template()
    base_xyz = coordinates(s, theta, k, [BASE])
    if hex_quality(base_xyz, hexes) < 0:
        hexes = hexes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    base_burst = burst_proxy(BASE)
    records = []
    for name, edit, defects in DESIGNS:
        xyz = coordinates(s, theta, k, defects)
        quality = hex_quality(xyz, hexes)
        if quality < 0.2:
            raise ValueError(f"{name}: element quality {quality:.2f}")
        burst = burst_proxy(defects[0])
        proxy = {"model": "DNV-RP-F101 single defect / Lorenz factor, capped by the intact intrados",
                 "burst_pressure_MPa": round(burst, 2),
                 "change_vs_base_percent": round(100 * (burst / base_burst - 1), 1)}
        folder = out_root / name
        record = save_design(folder, name, benchmark=BENCHMARK, family="A", base="base", edit=edit,
                             params={"defects": defects, "s_coordinate": "centreline mm from anchored end 1",
                                     "theta": "deg from the intrados", "depth": "fraction of wall"},
                             xyz=xyz, base_xyz=base_xyz, elements=hexes, etype="C3D8I", proxy=proxy)
        record["min_scaled_jacobian"] = round(quality, 3)
        write_inp(folder / f"{name}.inp", xyz, hexes, k, s, f"{BENCHMARK} {name}: {edit}")
        np.savez_compressed(folder / f"{name}_surface_coords.npz", s=s, theta=theta, wall_index=k)
        records.append(record)
    return records
