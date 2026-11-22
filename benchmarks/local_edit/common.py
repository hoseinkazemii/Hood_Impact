"""Shared pieces of the local-edit benchmark generators (B1-B4).

Every family uses one mesh. A design is the family's base mesh with some nodes
moved (morphing), never a new mesh, so node IDs match across the family as in
the hood decks, and a design's change nodes are exactly the nodes its edit
moved: ``change_node_ids``, ``change_base_xyz`` and ``change_displacement`` in
<design>_mesh.npz, the same base-position + displacement rows that
design_change_nodes.py builds for the hood.

Units: mm, N, MPa, tonne, s (thermal: mJ, so W/(m K) = mJ/(s mm K)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path(r"D:\Papers\Hood_Impact\LocalEditBenchmarks")
CHANGE_TOL_MM = 1e-4

# Corner order of each face, S1.. in Abaqus numbering.
FACES = {
    4: [(0, 1), (1, 2), (2, 3), (3, 0)],  # CPS4 / CPS4I / DC2D4
    8: [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)],  # C3D8(I)
}

# S355 structural steel, true stress (MPa) vs plastic strain.
S355_PLASTIC = [(355.0, 0.0), (375.0, 0.01), (410.0, 0.03), (450.0, 0.06), (500.0, 0.12),
                (540.0, 0.20), (580.0, 0.35)]


def boundary_faces(elements):
    """[(element index, face number, node indices)] of faces that belong to one element only."""
    faces = FACES[elements.shape[1]]
    seen = {}
    for e, conn in enumerate(elements):
        for f, corners in enumerate(faces):
            nodes = conn[list(corners)]
            seen.setdefault(tuple(sorted(nodes.tolist())), []).append((e, f + 1, nodes))
    return [owners[0] for owners in seen.values() if len(owners) == 1]


def _lines(f, values, per_line=16):
    values = list(values)
    for i in range(0, len(values), per_line):
        f.write(", ".join(str(v) for v in values[i:i + per_line]) + "\n")


def write_mesh(f, xyz, elements, etype, elset):
    """*NODE (labels 1..N) and *ELEMENT (labels 1..E) blocks."""
    f.write("*NODE\n")
    for i, p in enumerate(xyz, 1):
        f.write(f"{i}, " + ", ".join(f"{c:.8f}" for c in p) + "\n")
    f.write(f"*ELEMENT, TYPE={etype}, ELSET={elset}\n")
    for e, conn in enumerate(elements, 1):
        f.write(f"{e}, " + ", ".join(str(n + 1) for n in conn) + "\n")


def write_nset(f, name, index):
    f.write(f"*NSET, NSET={name}\n")
    _lines(f, np.asarray(index, dtype=int) + 1)


def write_surface(f, name, faces):
    """Element-based surface from [(element index, face number, ...)]."""
    by_face = {}
    for face in faces:
        by_face.setdefault(face[1], []).append(face[0] + 1)
    for k, labels in sorted(by_face.items()):
        f.write(f"*ELSET, ELSET={name}_S{k}\n")
        _lines(f, labels)
    f.write(f"*SURFACE, TYPE=ELEMENT, NAME={name}\n")
    for k in sorted(by_face):
        f.write(f"{name}_S{k}, S{k}\n")


def write_s355(f, name="S355"):
    f.write(f"*MATERIAL, NAME={name}\n*ELASTIC\n210000., 0.3\n*PLASTIC\n")
    for stress, strain in S355_PLASTIC:
        f.write(f"{stress}, {strain}\n")


def quad_quality(xyz, quads):
    """Smallest scaled corner Jacobian of 2D quads (1 = square, <= 0 = inverted)."""
    p = xyz[quads][:, :, :2]
    a = np.roll(p, -1, axis=1) - p
    b = np.roll(p, 1, axis=1) - p
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return float((cross / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).min())


def hex_quality(xyz, hexes):
    """Smallest scaled corner Jacobian of 8-node hexes."""
    corner_edges = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
    worst = np.inf
    p = xyz[hexes]
    for c, i, j, k in corner_edges:
        a, b, d = p[:, i] - p[:, c], p[:, j] - p[:, c], p[:, k] - p[:, c]
        det = np.einsum("ij,ij->i", np.cross(a, b), d)
        norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(d, axis=1)
        worst = min(worst, float((det / norm).min()))
    return worst


def save_design(out_dir, name, *, benchmark, family, base, edit, params, xyz, base_xyz, elements, etype,
                proxy=None, node_groups=None):
    """Write <name>_mesh.npz (mesh + change nodes) and <name>.json (edit record); return the record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.arange(1, len(xyz) + 1)
    move = xyz - base_xyz
    moved = np.linalg.norm(move, axis=1) > CHANGE_TOL_MM
    groups = {f"group_{k}": np.asarray(v, dtype=int) + 1 for k, v in (node_groups or {}).items()}
    np.savez_compressed(out_dir / f"{name}_mesh.npz", node_ids=ids, xyz=xyz, elements=elements + 1,
                        element_type=etype, change_node_ids=ids[moved], change_base_xyz=base_xyz[moved],
                        change_displacement=move[moved], **groups)
    record = {
        "benchmark": benchmark, "family": family, "design": name, "base_design": base, "edit": edit,
        "params": params, "nodes": int(len(xyz)), "elements": int(len(elements)), "element_type": etype,
        "change_nodes": int(moved.sum()),
        "change_nodes_over_0.1mm": int((np.linalg.norm(move, axis=1) > 0.1).sum()),  # the hood's threshold
        "max_node_move_mm": float(np.linalg.norm(move, axis=1).max()),
        "proxy": proxy or {},
        "units": "mm, N, MPa, tonne, s",
    }
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record
