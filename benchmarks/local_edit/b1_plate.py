"""B1: S355 plate (200 x 60 x 3 mm) with five staggered bolt holes, pulled 1.5 mm (CPS4, plane stress).

Like a bolted tension member: two staggered rows plus a centre hole, all 11 mm
(M10 clearance). Every hole sits on a possible net-section failure path, so
moving, resizing or turning a hole into a slot changes the strength (hand model:
the steel-design net width w - sum d + sum s^2 / 4g over the worst path).

One gmsh quad mesh for the family (0.7 mm in the hole band, 2.5 mm outside).
An edit moves the boundary nodes of the edited holes onto the new hole shape
and carries the surrounding nodes along as a stiffened elastic pseudo-solid
(the usual deforming-mesh method) inside a disc around the hole, in 20 steps. Nothing else
moves, so the change nodes are that disc.

Main output: grip force vs elongation. Fields: U, S, LE, PEEQ at 21 frames.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from common import quad_quality, save_design, write_mesh, write_nset, write_s355

BENCHMARK = "B1_plate"
L, W, T = 200.0, 60.0, 3.0
ELONGATION = 1.5
BAND = (55.0, 145.0)
SIZE_FINE, SIZE_COARSE = 0.7, 2.5
YIELD = 355.0
MORPH_MARGIN = 12.0  # mm of mesh around an edited hole that follows it


def hole(x, y, d=11.0, length=None, angle=0.0):
    """Circle (length = d) or slot: width d, overall length along the axis at angle (deg from the load)."""
    return dict(x=float(x), y=float(y), d=float(d), length=float(length or d), angle=float(angle))


BASE = [hole(85, 15), hole(85, 45), hole(100, 30), hole(115, 15), hole(115, 45)]


def edited(**changes):
    holes = [dict(h) for h in BASE]
    for index, change in changes.items():
        holes[int(index[1:])].update({k: float(v) for k, v in change.items()})
    return holes


DESIGNS = [  # name, edit, holes (h0..h4 = row 1 low/high, centre, row 3 low/high)
    ("base", "family base: five 11 mm holes, rows at x = 85 / 100 / 115 mm", BASE),
    ("e1_shift_centre", "centre hole moved 6 mm towards row 1 (x 100 -> 94)", edited(h2=dict(x=94))),
    ("e2_slot_across", "centre hole elongated to an 11 x 19 mm slot across the load",
     edited(h2=dict(length=19, angle=90))),
    ("e3_enlarge_row1", "both row-1 holes enlarged 11 -> 14 mm", edited(h0=dict(d=14, length=14), h1=dict(d=14, length=14))),
    ("e4_slot_diagonal", "lower row-3 hole elongated to an 11 x 21 mm slot at 45 deg",
     edited(h3=dict(length=21, angle=45))),
    ("e5_shrink_centre", "centre hole reduced 11 -> 7 mm (M6 clearance)", edited(h2=dict(d=7, length=7))),
]


def base_mesh():
    """Quad mesh of the plate with the BASE holes: xyz (N, 2), quads (E, 4) counter-clockwise."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("b1")
    occ = gmsh.model.occ
    plate = occ.addRectangle(0, 0, 0, L, W)
    disks = [occ.addDisk(h["x"], h["y"], 0, h["d"] / 2, h["d"] / 2) for h in BASE]
    occ.cut([(2, plate)], [(2, d) for d in disks])
    occ.synchronize()
    field = gmsh.model.mesh.field
    box = field.add("Box")
    for key, value in (("VIn", 2 * SIZE_FINE), ("VOut", 2 * SIZE_COARSE), ("XMin", BAND[0]), ("XMax", BAND[1]),
                       ("YMin", -1.0), ("YMax", W + 1.0), ("Thickness", 20.0)):
        field.setNumber(box, key, value)
    field.setAsBackgroundMesh(box)
    for key, value in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                       ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 8), ("Mesh.RecombineAll", 1),
                       ("Mesh.RecombinationAlgorithm", 1), ("Mesh.SubdivisionAlgorithm", 1)):
        gmsh.option.setNumber(key, value)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    tags, coords, _ = gmsh.model.mesh.getNodes()
    types, _, connectivity = gmsh.model.mesh.getElements(2)
    gmsh.finalize()
    if list(types) != [3]:
        raise RuntimeError(f"gmsh left non-quad elements: types {list(types)}")
    index = np.full(int(tags.max()) + 1, -1)
    index[tags.astype(int)] = np.arange(len(tags))
    xyz = coords.reshape(-1, 3)[:, :2].copy()
    quads = index[connectivity[0].astype(int)].reshape(-1, 4)
    used = np.unique(quads)
    remap = np.full(len(xyz), -1)
    remap[used] = np.arange(len(used))
    xyz, quads = xyz[used], remap[quads]
    p = xyz[quads]
    area = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], axis=1)
    quads[area < 0] = quads[area < 0][:, ::-1]
    return xyz, quads


def slot_ray(h, phi):
    """Distance from the hole centre to its boundary along direction phi (radians, global)."""
    r, e = h["d"] / 2, (h["length"] - h["d"]) / 2
    psi = phi - np.radians(h["angle"])
    c, s = np.abs(np.cos(psi)), np.abs(np.sin(psi))
    side = np.where(s > 1e-12, r / np.maximum(s, 1e-12), np.inf)
    cap = e * c + np.sqrt(np.maximum(r * r - (e * s) ** 2, 0.0))
    return np.where(side * c <= e, side, cap)


GAUSS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) / np.sqrt(3.0)
POISSON = 0.3
ELASTIC = np.array([[1 - POISSON, POISSON, 0], [POISSON, 1 - POISSON, 0], [0, 0, (1 - 2 * POISSON) / 2]])


def mesh_stiffness(x, quads, stiffening):
    """Plane-strain stiffness of the mesh as a pseudo-solid; small elements are made stiffer."""
    p = x[quads]
    area = 0.5 * np.abs(np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], 1))
    modulus = (area.min() / area) ** stiffening
    ke = np.zeros((len(quads), 8, 8))
    for xi, eta in GAUSS:
        dxi = 0.25 * np.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])
        jac = np.stack([dxi @ p, deta @ p], axis=1)  # (E, 2, 2)
        det = np.linalg.det(jac)
        grad = np.linalg.solve(jac, np.broadcast_to(np.stack([dxi, deta]), (len(quads), 2, 4)))  # (E, 2, 4)
        b = np.zeros((len(quads), 3, 8))
        b[:, 0, 0::2], b[:, 1, 1::2] = grad[:, 0], grad[:, 1]
        b[:, 2, 0::2], b[:, 2, 1::2] = grad[:, 1], grad[:, 0]
        ke += np.einsum("eki,kl,elj->eij", b, ELASTIC, b) * (np.abs(det) * modulus)[:, None, None]
    dofs = np.stack([2 * quads, 2 * quads + 1], axis=2).reshape(-1, 8)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    return sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(2 * len(x), 2 * len(x))).tocsr()


def elastic_morph(xyz, quads, targets, free, steps=20, stiffening=1.5):
    """Move ``targets`` (node -> final xy) in steps; ``free`` nodes follow as a stiffened pseudo-solid.

    The usual deforming-mesh method: each step solves linear elasticity on the
    current mesh with the target nodes displaced and every non-free node held,
    element stiffness growing as (smallest area / area)^stiffening so the fine
    elements next to a hole keep their shape and the bigger ones take the strain.
    """
    moved = np.array(sorted(targets))
    start, end = xyz[moved].copy(), np.array([targets[i] for i in moved])
    free = np.asarray(sorted(free))
    free_dofs = np.stack([2 * free, 2 * free + 1], 1).ravel()
    moved_dofs = np.stack([2 * moved, 2 * moved + 1], 1).ravel()
    x = xyz.copy()
    for step in range(1, steps + 1):
        disp = start + (end - start) * step / steps - x[moved]
        k = mesh_stiffness(x, quads, stiffening)
        rhs = -k[free_dofs][:, moved_dofs] @ disp.ravel()
        x[free] += spla.spsolve(k[free_dofs][:, free_dofs].tocsc(), rhs).reshape(-1, 2)
        x[moved] += disp
    return x


def morph(xyz, quads, holes):
    """Base mesh -> mesh with ``holes`` (same nodes): boundary nodes onto the new shapes, harmonic around them."""
    on_hole = [np.where(np.abs(np.hypot(xyz[:, 0] - h["x"], xyz[:, 1] - h["y"]) - h["d"] / 2) < 1e-3)[0]
               for h in BASE]
    outer = np.where((xyz[:, 0] < 1e-6) | (xyz[:, 0] > L - 1e-6) | (xyz[:, 1] < 1e-6) | (xyz[:, 1] > W - 1e-6))[0]
    boundary = set(outer).union(*map(set, on_hole))
    targets, free = {}, set()
    for old, new, nodes in zip(BASE, holes, on_hole):
        if old == new:
            continue
        phi = np.arctan2(xyz[nodes, 1] - old["y"], xyz[nodes, 0] - old["x"])
        rad = slot_ray(new, phi)
        for n, p, r in zip(nodes, phi, rad):
            targets[int(n)] = np.array([new["x"] + r * np.cos(p), new["y"] + r * np.sin(p)])
        reach = max(old["length"], new["length"]) / 2 + MORPH_MARGIN
        a, b = np.array([old["x"], old["y"]]), np.array([new["x"], new["y"]])
        ab = b - a
        t = np.clip(((xyz - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        dist = np.linalg.norm(xyz - (a + t[:, None] * ab), axis=1)
        free |= set(np.where(dist <= reach)[0].tolist())
    if not targets:
        return xyz.copy()
    return elastic_morph(xyz, quads, targets, free - boundary)


def net_width(holes):
    """Steel-design net width (mm): worst chain across the plate of w - sum d + sum s^2 / 4g."""
    across = [(h["x"], h["y"], h["d"] + (h["length"] - h["d"]) * abs(np.sin(np.radians(h["angle"])))) for h in holes]
    across.sort(key=lambda h: h[1])
    best = W
    for r in range(1, len(across) + 1):
        for chain in itertools.combinations(across, r):
            if len({round(c[1], 6) for c in chain}) < r:
                continue
            width = W - sum(c[2] for c in chain)
            width += sum((q[0] - p[0]) ** 2 / (4 * abs(q[1] - p[1])) for p, q in zip(chain, chain[1:]))
            best = min(best, width)
    return best


def write_inp(path, xyz, quads, title):
    left = np.where(xyz[:, 0] < 1e-6)[0]
    right = np.where(xyz[:, 0] > L - 1e-6)[0]
    pin = left[np.argmin(np.abs(xyz[left, 1] - W / 2))]
    with open(path, "w") as f:
        f.write(f"*HEADING\n{title}\n")
        write_mesh(f, xyz, quads, "CPS4", "PLATE")
        write_nset(f, "LEFT", left)
        write_nset(f, "RIGHT", right)
        write_nset(f, "PIN", [pin])
        f.write(f"*SOLID SECTION, ELSET=PLATE, MATERIAL=S355\n{T},\n")
        write_s355(f)
        f.write("*BOUNDARY\nLEFT, 1, 1, 0.\nPIN, 2, 2, 0.\n")
        f.write("*STEP, NLGEOM=YES, INC=2000\n*STATIC\n0.01, 1., 1e-06, 0.02\n")
        f.write(f"*BOUNDARY\nRIGHT, 1, 1, {ELONGATION}\n")
        f.write("*OUTPUT, FIELD, TIME INTERVAL=0.05\n*NODE OUTPUT\nU,\n*ELEMENT OUTPUT, DIRECTIONS=YES\nS, LE, PEEQ\n")
        f.write("*OUTPUT, HISTORY, FREQUENCY=1\n*NODE OUTPUT, NSET=RIGHT\nU1, RF1\n*END STEP\n")


def generate(out_root):
    out_root = Path(out_root) / BENCHMARK
    base_xyz, quads = base_mesh()
    base_width = net_width(BASE)
    records = []
    for name, edit, holes in DESIGNS:
        xyz = morph(base_xyz, quads, holes)
        quality = quad_quality(xyz, quads)
        if quality < 0.15:
            raise ValueError(f"{name}: element quality {quality:.2f}")
        width = net_width(holes)
        proxy = {"model": "net-section yield, steel-design net width with stagger",
                 "net_width_mm": round(width, 2), "yield_load_kN": round(YIELD * T * width / 1e3, 2),
                 "change_vs_base_percent": round(100 * (width / base_width - 1), 1)}
        folder = out_root / name
        record = save_design(folder, name, benchmark=BENCHMARK, family="A", base="base", edit=edit,
                             params={"holes": holes, "hole_fields": "centre x, y; width d; overall length; "
                                                                  "axis angle deg from the load"},
                             xyz=xyz, base_xyz=base_xyz, elements=quads, etype="CPS4", proxy=proxy)
        record["min_scaled_jacobian"] = round(quality, 3)
        write_inp(folder / f"{name}.inp", xyz, quads, f"{BENCHMARK} {name}: {edit}")
        records.append(record)
    return records
