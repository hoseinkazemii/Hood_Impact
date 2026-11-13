"""
make_local_edit_decks.py
========================
Write the Abaqus decks of the local-edit dataset planned by local_edit_doe.py.

Every deck is design 6 (HoodImpact_31.inp) with exactly three kinds of change,
and every other byte identical to the delivered deck (mesh connectivity,
materials, fasteners, couplings, contact, mass scaling, output requests):

  1. the design's inner-panel node lines (z only), from local_edit_doe.csv
     through the same notch math as cae_local_edit_designs.py; the base design
     D6 changes none;
  2. the headform node lines, rigidly translated to the location with the
     contact surfaces 1.0 mm apart (manifest columns headform_d*_mm);
  3. the *DYNAMIC card: DIRECT USER CONTROL at a fixed 4.5e-7 s increment.

Before writing, every design is re-checked like apply_node_edits.py (no
headform, BC/coupling or skin nodes moved, >= 10 mm from fasteners, skin
clearance, every edge >= 3.0 mm: the smooth notches keep >= 5.25 mm, the
creases go down to 3.0 mm, which the fixed 4.5e-7 s increment solves without
mass scaling). Decks are written atomically, so an interrupted run never
leaves a half deck that looks complete. With --skip-existing only the missing
decks are written, so new designs can be added to a running campaign.

Also writes, once per design, design_changes/<design>_changed_nodes.csv
(label, x, y, z of design 6, dz): the design geometry for the ML side.

Usage (from the Code directory):
    python abaqus_scripts/make_local_edit_decks.py                    # all decks
    python abaqus_scripts/make_local_edit_decks.py --runs 1-11        # the pilot
    python abaqus_scripts/make_local_edit_decks.py --changes-only     # only design_changes/
    python abaqus_scripts/make_local_edit_decks.py --verify 25        # re-read every 25th deck
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import local_edit_doe as doe  # noqa: E402
from inp_geom import Deck, HEADFORM_RADIUS, HOOD_OUTER_ELSET  # noqa: E402
from make_solver_factorial_decks import fixed_increment_lines, format_node_line, write_lines  # noqa: E402


def parse_runs(spec, n):
    if not spec:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def node_line(deck, nid, xyz):
    line = deck.lines[deck.node_line_idx[nid]]
    return format_node_line(nid, xyz[0], xyz[1], xyz[2], line[len(line.rstrip("\r\n")):])


def verify_deck(path, base, row, dz, imp_ids):
    """Re-read a written deck and check all three changes and nothing else."""
    d = Deck(path)
    shift = np.array([float(row["headform_d%s_mm" % c]) for c in "xyz"])
    for nid, xyz in base.nodes.items():
        want = np.array(xyz, float)
        if nid in imp_ids:
            want = want + shift
        elif nid in dz:
            want[2] += dz[nid]
        if np.abs(np.array(d.nodes[nid]) - want).max() > 1e-9:
            raise AssertionError("run %s: node %d at %s, expected %s" % (row["run"], nid, d.nodes[nid], want))
    if set(d.nodes) != set(base.nodes) or d.elems != base.elems:
        raise AssertionError("run %s: node or element set changed" % row["run"])
    outer = np.array([d.nodes[i] for i in sorted(d.elset_nodes(HOOD_OUTER_ELSET))], float)
    ref = np.array(d.nodes[d.rigid_ref], float)
    pen = HEADFORM_RADIUS - np.linalg.norm(outer - ref, axis=1).min()
    if abs(pen - float(row["sphere_pen_mm"])) > 1e-3:
        raise AssertionError("run %s: sphere penetration %.4f, manifest %s" % (row["run"], pen, row["sphere_pen_mm"]))
    if abs(ref[0] - float(row["x_mm"])) > 1e-6 or abs(ref[1] - float(row["y_mm"])) > 1e-6:
        raise AssertionError("run %s: headform at %s, location %s %s" % (row["run"], ref, row["x_mm"], row["y_mm"]))
    with open(path) as f:
        text = f.read()
    if "*DYNAMIC, EXPLICIT, DIRECT USER CONTROL" not in text or "%r, " % doe.FIXED_DT not in text:
        raise AssertionError("run %s: fixed-increment *DYNAMIC card missing" % row["run"])
    return pen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=doe.OUT_DIR, help="holds the csvs; decks go to <data-dir>/inp_files")
    ap.add_argument("--runs", default="", help="e.g. 1-11,20 (default: all)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--changes-only", action="store_true")
    ap.add_argument("--verify", type=int, default=0, help="re-read every Nth written deck (0: none)")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, doe.MANIFEST_CSV)) as f:
        manifest = list(csv.DictReader(f))
    if [int(r["run"]) for r in manifest] != list(range(1, len(manifest) + 1)):
        raise SystemExit("manifest runs must be numbered 1..N in order")
    designs = doe.read_doe(os.path.join(args.data_dir, doe.DOE_CSV))
    hood = doe.Hood(keep_lines=True)
    base = hood.deck
    checks = doe.EditChecks(hood)
    imp_ids = base.impactor_node_ids()
    dt_lines = fixed_increment_lines(base.lines, doe.FIXED_DT)
    if {r["time_increment_s"] for r in manifest} != {"%r" % doe.FIXED_DT}:
        raise SystemExit("manifest time increment differs from %r" % doe.FIXED_DT)

    # design edits: node-line replacements, checked once per design
    edit_lines, edit_dz = {doe.BASE_DESIGN: {}}, {doe.BASE_DESIGN: {}}
    change_dir = os.path.join(args.data_dir, "design_changes")
    os.makedirs(change_dir, exist_ok=True)
    for name, notches in designs.items():
        dz, info = doe.design_offsets(hood, notches)
        res = checks(dz, info)
        if res["problems"]:
            raise SystemExit("%s fails the edit checks: %s" % (name, res["problems"]))
        edit_dz[name] = dz
        edit_lines[name] = {base.node_line_idx[n]: node_line(base, n, np.add(base.nodes[n], (0.0, 0.0, v)))
                            for n, v in dz.items()}
        with open(os.path.join(change_dir, "%s_changed_nodes.csv" % name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "x", "y", "z_design6", "dz"])
            for n in sorted(dz):
                x, y, z = base.nodes[n]
                w.writerow([n, repr(float(x)), repr(float(y)), repr(float(z)), "%.2f" % dz[n]])
        print("%-4s %4d nodes moved, dz %.2f..%.2f mm, min edge %.3f mm, fastener %.0f mm"
              % (name, res["n_nodes"], res["dz_min"], res["dz_max"], res["min_edge"], res["fastener_mm"]), flush=True)
    missing = {r["design"] for r in manifest} - set(edit_lines)
    if missing:
        raise SystemExit("manifest designs missing from the doe csv: %s" % sorted(missing))
    if args.changes_only:
        print("wrote %d design change files to %s" % (len(designs), change_dir))
        return

    inp_dir = os.path.join(args.data_dir, "inp_files")
    os.makedirs(inp_dir, exist_ok=True)
    runs = parse_runs(args.runs, len(manifest))
    written = verified = 0
    shift_cache = {}
    for k, run in enumerate(runs):
        row = manifest[run - 1]
        out = os.path.join(inp_dir, "HoodImpact_%d.inp" % run)
        if args.skip_existing and os.path.exists(out):
            continue
        loc = row["location"]
        if loc not in shift_cache:
            shift = np.array([float(row["headform_d%s_mm" % c]) for c in "xyz"])
            shift_cache[loc] = {base.node_line_idx[n]: node_line(base, n, np.add(base.nodes[n], shift)) for n in imp_ids}
        replaced = dict(dt_lines)
        replaced.update(shift_cache[loc])
        overlap = set(edit_lines[row["design"]]) & set(shift_cache[loc])
        if overlap:
            raise SystemExit("run %d: a design edit touches a headform node line" % run)
        replaced.update(edit_lines[row["design"]])
        write_lines(base.lines, replaced, out)
        written += 1
        if args.verify and (k % args.verify == 0 or row["pilot"] == "1"):
            verify_deck(out, base, row, edit_dz[row["design"]], imp_ids)
            verified += 1
        if written % 100 == 0:
            print("  %d decks written (run %d)" % (written, run), flush=True)
    print("wrote %d decks to %s (%d re-read and verified)" % (written, inp_dir, verified))


if __name__ == "__main__":
    main()
