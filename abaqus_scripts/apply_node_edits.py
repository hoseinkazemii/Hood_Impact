"""
apply_node_edits.py
===================
Patch edited node coordinates into an ORIGINAL flat HoodImpact deck.

Input is a CSV "label,x,y,z" (for example the full-part export written by
cae_inner_panel_dip.py inside Abaqus/CAE). Only the node lines whose
coordinates moved by more than --tol mm are rewritten; every other byte of the
deck stays identical, so node and element IDs, fasteners, couplings, contact
pairs, BCs and output requests are exactly those of the delivered deck.

Refuses the edit (unless --force) when it
  * moves a headform (impactor) node,
  * moves a node that belongs to any *NSET (BCs, hinge/latch spiders,
    fastener reference nodes, probes),
  * moves a node within --fastener-zone mm of a fastener connector node
    (the fastener *COUPLINGs grab surface nodes within INFLUENCE RADIUS=5),
  * brings a moved node closer than --min-clearance mm to the outer skin
    (vertical gap) when it was farther than that before.

Writes the new deck to --out and, next to it, <stem>_changed_nodes.csv with
label, elsets, dx, dy, dz of every moved node (the design-change record).

    python abaqus_scripts/apply_node_edits.py \
        --base Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_1.inp \
        --coords edited_nodes.csv \
        --out abaqus_scripts/work_design_edits/dip15/HoodImpact_1.inp
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from inp_geom import Deck, OuterSurface, HOOD_OUTER_ELSET  # noqa: E402
from generate_impact_decks import format_node_line  # noqa: E402

SHELL_ELSETS = ["Hood_Outer-1-2", "Hood_Inner-1-2", "Hinges-1-2"]


def connector_node_ids(deck):
    """Node ids used by *ELEMENT, TYPE=CONN3D2 (the point fasteners)."""
    ids, on = set(), False
    for line in deck.lines:
        s = line.strip()
        if s.startswith("*"):
            on = s.upper().replace(" ", "").startswith("*ELEMENT,TYPE=CONN3D2")
            continue
        if on and s:
            ids.update(int(v) for v in s.split(",")[1:] if v.strip())
    return ids


def attachment_node_sets(deck):
    """Node groups that tie the hood to BCs, couplings, probes:
    every *NSET referenced somewhere besides its own definition, plus the
    nodes of *SURFACE, TYPE=NODE blocks (hinge/latch coupling spiders).
    Unreferenced export leftovers (e.g. the 37k-node 'Spatial Group.3') are
    skipped."""
    groups = {}
    for name, members in deck.nsets.items():
        quoted = '"%s"' % name
        used = any(quoted in ln for ln in deck.lines
                   if not ln.lstrip().upper().startswith("*NSET"))
        if used:
            groups["NSET " + name] = set(members)
    cur = None
    for line in deck.lines:
        s = line.strip()
        if s.startswith("*"):
            u = s.upper().replace(" ", "")
            cur = None
            if u.startswith("*SURFACE") and "TYPE=NODE" in u:
                cur = "SURFACE " + s.split('NAME="')[1].split('"')[0]
                groups[cur] = set()
            continue
        if cur and s:
            head = s.split(",")[0].strip()
            if head.isdigit():
                groups[cur].add(int(head))
    return groups


def min_edge(deck, eids, coords):
    """Smallest edge length over the given shell elements."""
    best = np.inf
    for e in eids:
        c = np.array([coords[n] for n in deck.elems[e]], float)
        best = min(best, np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).min())
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="original flat deck")
    ap.add_argument("--coords", required=True, help="CSV label,x,y,z")
    ap.add_argument("--out", required=True, help="path of the patched deck")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="mm; smaller moves count as unchanged (CAE round-off)")
    ap.add_argument("--fastener-zone", type=float, default=10.0)
    ap.add_argument("--min-clearance", type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="write despite failed checks")
    args = ap.parse_args()

    deck = Deck(args.base, keep_lines=True)
    new = {}
    with open(args.coords, newline="") as f:
        for row in csv.DictReader(f):
            new[int(row["label"])] = (float(row["x"]), float(row["y"]), float(row["z"]))
    unknown = sorted(set(new) - set(deck.nodes))
    if unknown:
        # CAE adds a stray node at the origin on import (44167 for design 1)
        print("ignoring %d labels not in the base deck: %s" % (len(unknown), unknown[:5]))
        for nid in unknown:
            del new[nid]
    hood_nodes = set()
    for name in SHELL_ELSETS:
        hood_nodes |= deck.elset_nodes(name)
    missing = hood_nodes - set(new)
    if missing and len(new) > len(hood_nodes) // 2:
        raise SystemExit("%d hood nodes are absent from the coordinate file, e.g. %s -- "
                         "deleting/adding nodes or elements is not supported (move nodes only)"
                         % (len(missing), sorted(missing)[:5]))

    moved = {}
    for nid, xyz in new.items():
        d = np.subtract(xyz, deck.nodes[nid])
        if np.abs(d).max() > args.tol:
            # keep the deck's exact value in components that did not move
            # (CAE stores coordinates in single precision)
            d[np.abs(d) <= args.tol] = 0.0
            moved[nid] = d
    if not moved:
        raise SystemExit("no node moved by more than %g mm -- nothing to write" % args.tol)

    node_sets = {}
    for name in SHELL_ELSETS:
        for n in deck.elset_nodes(name):
            node_sets.setdefault(n, []).append(name.replace("-1-2", ""))
    ids = np.array(sorted(moved))
    disp = np.array([moved[i] for i in ids])
    old = np.array([deck.nodes[i] for i in ids])
    upd = old + disp
    by_set = {}
    for i in ids:
        for s in node_sets.get(i, ["(no shell elset)"]):
            by_set[s] = by_set.get(s, 0) + 1
    print("moved nodes: %d  %s" % (len(ids), by_set))
    print("max |displacement| %.3f mm   dz range [%.3f, %.3f] mm"
          % (np.linalg.norm(disp, axis=1).max(), disp[:, 2].min(), disp[:, 2].max()))
    print("plan bbox x [%.1f, %.1f]  y [%.1f, %.1f]"
          % (old[:, 0].min(), old[:, 0].max(), old[:, 1].min(), old[:, 1].max()))

    problems = []
    imp = deck.impactor_node_ids() & set(moved)
    if imp:
        problems.append("%d headform nodes moved" % len(imp))
    in_group = {}
    for name, members in attachment_node_sets(deck).items():
        hit = members & set(moved)
        if hit:
            in_group[name] = len(hit)
    if in_group:
        problems.append("moved nodes belong to BC/coupling/probe groups %s" % in_group)
    if HOOD_OUTER_ELSET.replace("-1-2", "") in by_set:
        problems.append("outer-skin nodes moved (headform placement assumes the "
                        "delivered skin)")

    conn = np.array([deck.nodes[i] for i in sorted(connector_node_ids(deck))], float)
    if len(conn):
        from scipy.spatial import cKDTree
        tree = cKDTree(conn)
        near = min(tree.query(old)[0].min(), tree.query(upd)[0].min())
        print("closest moved node to a fastener: %.1f mm" % near)
        if near < args.fastener_zone:
            problems.append("edit reaches within %.1f mm of a fastener" % near)

    skin = OuterSurface(deck)
    gap_old = skin.surf_z(old[:, 0], old[:, 1]) - old[:, 2]
    gap_new = skin.surf_z(upd[:, 0], upd[:, 1]) - upd[:, 2]
    print("vertical gap to outer skin over moved nodes: before min %.2f / median %.2f, "
          "after min %.2f / median %.2f mm"
          % (gap_old.min(), np.median(gap_old), gap_new.min(), np.median(gap_new)))
    tight = (gap_new < args.min_clearance) & (gap_new < gap_old - 1e-9)
    if tight.any():
        problems.append("%d moved nodes end up < %.1f mm below the skin"
                        % (tight.sum(), args.min_clearance))

    hood_elems = [e for name in SHELL_ELSETS for e in deck.elsets[name]]
    touched = [e for e in hood_elems if any(n in moved for n in deck.elems[e])]
    coords_new = dict(deck.nodes)
    for i, row in zip(ids, upd):
        coords_new[i] = tuple(row)
    print("edited elements: %d   min edge before %.3f mm, after %.3f mm"
          % (len(touched), min_edge(deck, touched, deck.nodes),
             min_edge(deck, touched, coords_new)))
    print("whole hood min edge (sets the stable time increment): before %.3f, after %.3f mm"
          % (min_edge(deck, hood_elems, deck.nodes), min_edge(deck, hood_elems, coords_new)))

    if problems:
        print("CHECKS FAILED:\n  - " + "\n  - ".join(problems))
        if not args.force:
            raise SystemExit("not writing the deck (use --force to override)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    replaced = {}
    for i, row in zip(ids, upd):
        idx = deck.node_line_idx[i]
        line = deck.lines[idx]
        term = line[len(line.rstrip("\r\n")):]
        replaced[idx] = format_node_line(i, row[0], row[1], row[2], term)
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        for k, line in enumerate(deck.lines):
            f.write(replaced.get(k, line))
    os.replace(tmp, args.out)

    rec = os.path.splitext(args.out)[0] + "_changed_nodes.csv"
    with open(rec, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "elsets", "dx", "dy", "dz"])
        for i, d in zip(ids, disp):
            w.writerow([i, "|".join(node_sets.get(i, [])), "%.6f" % d[0],
                        "%.6f" % d[1], "%.6f" % d[2]])
    print("wrote %s (%d node lines replaced) and %s" % (args.out, len(replaced), rec))


if __name__ == "__main__":
    main()
