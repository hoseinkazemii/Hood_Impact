"""B3: 2D aluminium heat-sink section with three chips, power-on transient to 600 s (DC2D4).

Base 100 x 3 mm under 12 fins (1.5 mm thick, 30 mm tall, pitch 8.33 mm), 50 mm
deep (section thickness). Three chips (10 / 15 / 10 W, 10 mm wide) heat the
bottom face; every other face loses heat to 25 C air with h = 40 W/m^2K.

Structured mesh: node columns follow the fin edges, and each fin has the same
number of node rows whatever its height. A height edit only rescales the rows
of the edited fins, so the change nodes are exactly those fins' nodes.

Edits are height patterns on 2-4 neighbouring fins (cut, stub, raise, ramp).
Thickness, fin shifts and base thinning were dropped: the hand model gave them
<= 3 % change in chip temperature.

Main output: the three chip temperature histories. Field: NT at every 5 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import boundary_faces, quad_quality, save_design, write_mesh, write_nset, write_surface

BENCHMARK = "B3_heatsink"
W, TB, DEPTH = 100.0, 3.0, 50.0
N_FINS, TF, HF = 12, 1.5, 30.0
NX_EDGE, NX_FIN, NX_GAP, NY_BASE, NY_FIN = 5, 4, 10, 8, 40
CHIPS = [(20.0, 10.0), (50.0, 15.0), (80.0, 10.0)]  # centre x (mm), power (W)
CHIP_WIDTH = 10.0
FILM, AMBIENT = 0.04, 25.0  # mW/(mm^2 K) = 40 W/(m^2 K); deg C
CONDUCTIVITY, DENSITY, SPECIFIC_HEAT = 200.0, 2.7e-9, 9.0e8  # Al 6063
END_TIME, OUTPUT_EVERY = 600.0, 5.0

PITCH = W / N_FINS
BASE = [HF] * N_FINS


def heights(changes):
    h = list(BASE)
    for fin, value in changes.items():  # fins numbered 1..12 from the left
        h[fin - 1] = value
    return h


DESIGNS = [  # name, edit, fin heights
    ("base", "family base: 12 fins, all 30 mm", BASE),
    ("e1_keepout_centre", "fins 5-7 cut to 10 mm over the centre chip", heights({5: 10, 6: 10, 7: 10})),
    ("e2_cut_edge", "fins 1-3 cut to 10 mm next to chip 1", heights({1: 10, 2: 10, 3: 10})),
    ("e3_stubs", "fins 10-11 cut to 3 mm stubs near chip 3", heights({10: 3, 11: 3})),
    ("e4_raise_centre", "fins 5-7 raised to 45 mm", heights({5: 45, 6: 45, 7: 45})),
    ("e5_ramp", "fins 1-4 profiled to 24 / 18 / 12 / 6 mm", heights({1: 24, 2: 18, 3: 12, 4: 6})),
]


def build(fin_heights):
    """Node coordinates, quads, the base-column index of every fin's first column, and the fin node ids."""
    edges = [0.0]
    counts = []
    for i in range(N_FINS):
        left = (i + 0.5) * PITCH - TF / 2
        edges += [left, left + TF]
        counts += [NX_EDGE if i == 0 else NX_GAP, NX_FIN]
    edges.append(W)
    counts.append(NX_EDGE)
    xs = [edges[0]]
    fin_column = []
    for k, n in enumerate(counts):
        if k % 2 == 1:
            fin_column.append(len(xs) - 1)
        xs += list(np.linspace(edges[k], edges[k + 1], n + 1)[1:])
    xs = np.array(xs)
    ncol = len(xs)
    base = np.array([(x, TB * j / NY_BASE) for j in range(NY_BASE + 1) for x in xs])
    node = lambda c, j: j * ncol + c
    quads = [[node(c, j), node(c + 1, j), node(c + 1, j + 1), node(c, j + 1)]
             for j in range(NY_BASE) for c in range(ncol - 1)]
    points, fin_nodes = [base], []
    start = len(base)
    for f, c0 in enumerate(fin_column):
        rows = {0: [node(c0 + i, NY_BASE) for i in range(NX_FIN + 1)]}
        block = []
        for j in range(1, NY_FIN + 1):
            rows[j] = list(range(start, start + NX_FIN + 1))
            block += [(xs[c0 + i], TB + fin_heights[f] * j / NY_FIN) for i in range(NX_FIN + 1)]
            start += NX_FIN + 1
        points.append(np.array(block))
        fin_nodes.append(np.concatenate([rows[j] for j in range(1, NY_FIN + 1)]))
        quads += [[rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]]
                  for j in range(NY_FIN) for i in range(NX_FIN)]
    return np.vstack(points), np.array(quads), fin_nodes


def chip_temperatures_1d(fin_heights):
    """Hand model: 1D base strip with fin conductances, steady chip temperature rise (K)."""
    k, h, n = CONDUCTIVITY, FILM * 1e3, 2000  # W/(m K), W/(m^2 K)
    dx = W / 1000 / n
    x = (np.arange(n) + 0.5) * dx
    g = np.full(n, h)
    for i, height in enumerate(fin_heights):
        t, m = TF / 1000, np.sqrt(2 * h / (k * TF / 1000))
        mh = m * height / 1000
        conductance = np.sqrt(2 * h * k * t) * np.tanh(mh) + h * t * (1 - np.tanh(mh) ** 2)
        g[np.abs(x - (i + 0.5) * PITCH / 1000) <= t / 2] = conductance / t
    q = np.zeros(n)
    for xc, power in CHIPS:
        q[np.abs(x - xc / 1000) <= CHIP_WIDTH / 2000] += power / (DEPTH / 1000) / (CHIP_WIDTH / 1000)
    kt = k * TB / 1000 / dx ** 2
    a = np.diag(g + 2 * kt)
    a[0, 0] -= kt
    a[-1, -1] -= kt
    a[np.arange(n - 1), np.arange(1, n)] = -kt
    a[np.arange(1, n), np.arange(n - 1)] = -kt
    t = np.linalg.solve(a, q)
    return [float(t[np.abs(x - xc / 1000) <= CHIP_WIDTH / 2000].mean()) for xc, _ in CHIPS]


def write_inp(path, xyz, quads, title):
    faces = boundary_faces(quads)
    bottom = [f for f in faces if np.all(xyz[f[2], 1] < 1e-9)]
    convective = [f for f in faces if not np.all(xyz[f[2], 1] < 1e-9)]
    with open(path, "w") as out:
        out.write(f"*HEADING\n{title}\n")
        write_mesh(out, xyz, quads, "DC2D4", "SINK")
        out.write(f"*NSET, NSET=ALLN, GENERATE\n1, {len(xyz)}, 1\n")
        flux = []
        for c, (xc, power) in enumerate(CHIPS, 1):
            chip = [f for f in bottom if abs(xyz[f[2], 0].mean() - xc) <= CHIP_WIDTH / 2]
            length = sum(abs(xyz[f[2][1], 0] - xyz[f[2][0], 0]) for f in chip)
            flux.append(power * 1e3 / (length * DEPTH))  # mW/mm^2, exact chip power
            write_surface(out, f"CHIP{c}", chip)
            write_nset(out, f"CHIP{c}_NODES", sorted({int(n) for f in chip for n in f[2]}))
        write_surface(out, "CONV", convective)
        out.write(f"*SOLID SECTION, ELSET=SINK, MATERIAL=AL6063\n{DEPTH},\n")
        out.write(f"*MATERIAL, NAME=AL6063\n*CONDUCTIVITY\n{CONDUCTIVITY},\n*DENSITY\n{DENSITY},\n"
                  f"*SPECIFIC HEAT\n{SPECIFIC_HEAT},\n")
        out.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALLN, {AMBIENT}\n")
        out.write(f"*STEP, INC=5000\n*HEAT TRANSFER, END=PERIOD, DELTMX=1.0\n0.5, {END_TIME}, 1e-05, 5.\n")
        out.write("*DSFLUX\n" + "".join(f"CHIP{c}, S, {q:.6f}\n" for c, q in enumerate(flux, 1)))
        out.write(f"*SFILM\nCONV, F, {AMBIENT}, {FILM}\n")
        out.write(f"*OUTPUT, FIELD, TIME INTERVAL={OUTPUT_EVERY}\n*NODE OUTPUT\nNT,\n")
        out.write("*OUTPUT, HISTORY, FREQUENCY=1\n")
        for c in range(1, len(CHIPS) + 1):
            out.write(f"*NODE OUTPUT, NSET=CHIP{c}_NODES\nNT11,\n")
        out.write("*END STEP\n")


def generate(out_root):
    out_root = Path(out_root) / BENCHMARK
    base_xyz, quads, fin_nodes = build(BASE)
    base_temps = chip_temperatures_1d(BASE)
    records = []
    for name, edit, fin_heights in DESIGNS:
        xyz, _, _ = build(fin_heights)
        quality = quad_quality(xyz, quads)
        if quality < 0.2:
            raise ValueError(f"{name}: element quality {quality:.2f}")
        temps = chip_temperatures_1d(fin_heights)
        proxy = {"model": "1D base strip with fin conductances, steady state",
                 "chip_rise_K": [round(t, 2) for t in temps],
                 "change_vs_base_percent": [round(100 * (t / b - 1), 1) for t, b in zip(temps, base_temps)]}
        folder = out_root / name
        record = save_design(folder, name, benchmark=BENCHMARK, family="A", base="base", edit=edit,
                             params={"fin_heights_mm": fin_heights, "fin_thickness_mm": TF,
                                     "chips": [{"x_mm": x, "power_W": p} for x, p in CHIPS]},
                             xyz=xyz, base_xyz=base_xyz, elements=quads, etype="DC2D4", proxy=proxy,
                             node_groups={f"fin_{i + 1}": n for i, n in enumerate(fin_nodes)})
        record["min_scaled_jacobian"] = round(quality, 3)
        write_inp(folder / f"{name}.inp", xyz, quads, f"{BENCHMARK} {name}: {edit}")
        records.append(record)
    return records
