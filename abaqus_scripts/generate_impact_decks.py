"""
generate_impact_decks.py
========================
Build a EuroNCAP dataset's impact decks. For each of the 12 hood designs, take
the design's first delivered deck (runs 1, 6, 11, ..., 56) as the geometry and
physics template, then reposition the rigid headform to every location in the
requested EuroNCAP CSV. The IndustryLike coordinates are never used as target
locations.

Repositioning is a RIGID TRANSLATION of the impactor nodes only (the ~286
shell elements of ELSET "Rigid Body1_2-1-3" plus the rigid-body reference
node). Everything else in the deck -- hood mesh, materials, fasteners,
contact pairs, the initial-velocity and output requests (which reference the
ref node by set name) -- stays byte-identical, so the new decks solve exactly
like the delivered ones.

Vertical placement reproduces the partner's measured convention: the sphere
bottom sits ~2.0 mm above the hood outer skin at the point directly below the
headform CENTER (verified across all 60 delivered decks: center-point
clearance = 2.006 mm exactly for L1/L2/L3 in every design, 2.0-2.9 mm at
L4/L5). So dz = surf_z(new center) - surf_z(old center), which keeps each
design's own center clearance unchanged. The delivered decks themselves start
with REAL sphere-into-hood overclosures on locally raised features (true 3D
sphere-node penetration up to +1.63 mm in every design's L1 deck; L4 sits at
exact touch), which Abaqus/Explicit general contact resolves with its
standard strain-free initial adjustment. The generator therefore allows the
same regime but no worse: after the center-convention placement it computes
the true penetration (82.5 - min distance from any hood-outer node to the
sphere center) and, where it would exceed the delivered maximum (PEN_CAP =
1.65 mm), lifts the headform just enough to respect the cap. Both the
penetration and the applied lift are recorded in the manifest.

The location count is inferred from the CSV. Numbering is
``run = locations_per_design*design + loc``, so the original 50-location CSV
still produces runs 1..600 while the 92-location complement produces
runs 1..1104 in its separate dataset directory.

Usage (from the Code/ directory):
    # Original 600 decks -- intended to run ON DELTA
    python abaqus_scripts/generate_impact_decks.py

    # Remaining 1,104 decks (92 locations x 12 designs)
    python abaqus_scripts/generate_impact_decks.py \
        --locations Data/HoodImpact_1104_EuroNCAP_Remaining/impact_locations_remaining_92.csv \
        --out-dir Data/HoodImpact_1104_EuroNCAP_Remaining --skip-existing

    # local validation: manifest only, no decks written
    python abaqus_scripts/generate_impact_decks.py --dry-run

    # local validation: two decks of design 0
    python abaqus_scripts/generate_impact_decks.py --designs 0 --locs 1,2
"""
import os
import csv
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CODE_ROOT = os.path.dirname(HERE)

from inp_geom import Deck, OuterSurface, VELOCITY_NSET, IMPACTOR_ELSET

DEFAULT_INP60 = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "inp_files")
DEFAULT_COORDS60 = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "ImpactCoords_60.csv")
DEFAULT_OUT = os.path.join(CODE_ROOT, "Data", "HoodImpact_600_EuroNCAP")
LOCATIONS_CSV = os.path.join(DEFAULT_OUT, "impact_locations_50.csv")

N_DESIGNS = 12

MANIFEST_FIELDS = ["run", "design", "loc", "base_run", "X1", "X2", "X3",
                   "dz_mm", "center_gap_mm", "sphere_pen_mm", "lift_mm",
                   "surf_z_mm"]

# worst true sphere-node penetration measured in the DELIVERED decks is
# +1.63 mm (all L1 decks); never place a headform deeper than that
PEN_CAP_MM = 1.65
HEADFORM_R = 82.5


def parse_span(spec, lo, hi):
    """'0-11' / '1,3,5' / '2' -> sorted list of ints, validated to [lo, hi]."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [v for v in out if not (lo <= v <= hi)]
    if bad:
        raise SystemExit("values out of range [%d, %d]: %s" % (lo, hi, bad))
    return sorted(out)


def load_locations(path):
    locs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            locs.append((int(row["loc"]), float(row["X1"]), float(row["X2"])))
    if [l for l, _, _ in locs] != list(range(1, len(locs) + 1)):
        raise SystemExit("unexpected loc numbering in %s" % path)
    if not locs:
        raise SystemExit("no impact locations found in %s" % path)
    return locs


def load_coords60(path):
    with open(path, newline="") as f:
        return [(float(r["X1"]), float(r["X2"]), float(r["X3"]))
                for r in csv.DictReader(f)]


def check_design_deck(deck, base_run, coords60):
    """Sanity-assert the deck anatomy before we touch it."""
    ref = deck.rigid_ref
    if ref is None:
        raise SystemExit("run %d: no *RIGID BODY for %s" % (base_run, IMPACTOR_ELSET))
    vel = set(deck.nsets.get(VELOCITY_NSET, []))
    if vel != {ref}:
        raise SystemExit("run %d: NSET '%s' = %s, expected {%d}"
                         % (base_run, VELOCITY_NSET, sorted(vel), ref))

    imp_ids = deck.impactor_node_ids()
    # the impactor must share no nodes with any other shell element
    imp_eids = set(deck.elsets[IMPACTOR_ELSET])
    hood_nodes = set()
    for eid, conn in deck.elems.items():
        if eid not in imp_eids:
            hood_nodes.update(conn)
    shared = hood_nodes & imp_ids
    if shared:
        raise SystemExit("run %d: impactor shares %d nodes with the hood mesh!"
                         % (base_run, len(shared)))

    # ref node must sit at the delivered impact coordinate
    got = np.array(deck.nodes[ref])
    want = np.array(coords60[base_run - 1])
    if np.abs(got - want).max() > 1e-3:
        raise SystemExit("run %d: ref node %d at %s but ImpactCoords_60 says %s"
                         % (base_run, ref, got, want))
    return imp_ids


def format_node_line(nid, x, y, z, terminator):
    # repr() gives the shortest exact round-trip float, matching the source
    # style; cast defensively -- repr(np.float64) would corrupt the deck
    return "%d, %r, %r, %r%s" % (nid, float(x), float(y), float(z), terminator)


def write_deck(deck, imp_ids, shift, out_path):
    """Write one deck atomically so an interrupted run cannot look complete."""
    dx, dy, dz = shift
    lines = deck.lines
    replaced = {}
    for nid in imp_ids:
        idx = deck.node_line_idx[nid]
        x, y, z = deck.nodes[nid]
        term = lines[idx][len(lines[idx].rstrip("\r\n")):]
        replaced[idx] = format_node_line(nid, x + dx, y + dy, z + dz, term)
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "w", newline="") as f:
            for i, line in enumerate(lines):
                f.write(replaced.get(i, line))
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", default="0-11", help="e.g. '0-11' or '0,3'")
    ap.add_argument("--locs", default=None,
                    help="optional location subset, e.g. '1-50' or '1,2'; "
                         "default: every row in --locations")
    ap.add_argument("--inp60-dir", default=DEFAULT_INP60)
    ap.add_argument("--coords60", default=DEFAULT_COORDS60)
    ap.add_argument("--locations", default=LOCATIONS_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute manifest + checks only, write no decks")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not rewrite decks already present; useful for "
                         "resume-safe Delta submissions")
    args = ap.parse_args()

    designs = parse_span(args.designs, 0, N_DESIGNS - 1)
    all_locations = load_locations(args.locations)
    n_locs = len(all_locations)
    if args.locs is None:
        locations = all_locations
    else:
        locs_wanted = set(parse_span(args.locs, 1, n_locs))
        locations = [l for l in all_locations if l[0] in locs_wanted]
    coords60 = load_coords60(args.coords60)

    os.makedirs(args.out_dir, exist_ok=True)
    inp_out = os.path.join(args.out_dir, "inp_files")
    if not args.dry_run:
        os.makedirs(inp_out, exist_ok=True)

    manifest = []
    for d in designs:
        base_run = 5 * d + 1
        base_path = os.path.join(args.inp60_dir, "HoodImpact_%d.inp" % base_run)
        print("design %2d: parsing %s ..." % (d, base_path))
        deck = Deck(base_path, keep_lines=not args.dry_run)
        imp_ids = check_design_deck(deck, base_run, coords60)
        surface = OuterSurface(deck)

        ids, P = deck.coords(imp_ids)          # impactor nodes (N,3)
        ref_xyz = np.array(deck.nodes[deck.rigid_ref])
        surf_c0 = float(surface.surf_z(ref_xyz[0], ref_xyz[1]))
        gap_c0 = ref_xyz[2] - HEADFORM_R - surf_c0  # partner's center clearance
        print("   impactor: %d nodes, ref %d at (%.3f, %.3f, %.3f); "
              "center clearance = %.3f mm"
              % (len(ids), deck.rigid_ref, ref_xyz[0], ref_xyz[1], ref_xyz[2],
                 gap_c0))
        if not (1.0 <= gap_c0 <= 4.0):
            raise SystemExit("design %d: implausible center clearance %.3f mm "
                             "(expected ~2.0)" % (d, gap_c0))

        # hood-outer node cloud for the true sphere-penetration check
        hood_ids = sorted(deck.elset_nodes("Hood_Outer-1-2"))
        H = np.array([deck.nodes[i] for i in hood_ids], float)

        n_lifted = 0
        n_written = 0
        n_skipped = 0
        for loc, xt, yt in locations:
            run = n_locs * d + loc
            dx, dy = xt - ref_xyz[0], yt - ref_xyz[1]
            surf_c = float(surface.surf_z(xt, yt))
            dz = surf_c - surf_c0              # keep center clearance constant

            # footprint must stay on the interpolable surface
            surf_new = surface.surf_z(P[:, 0] + dx, P[:, 1] + dy)
            if np.isnan(surf_new).any():
                raise SystemExit("run %d: headform footprint leaves the hood "
                                 "surface -- bad location (%.1f, %.1f)" % (run, xt, yt))

            # true penetration: hood node closest to the sphere center; lift
            # the headform if deeper than the worst delivered deck
            near = np.linalg.norm(
                H[:, :2] - np.array([xt, yt]), axis=1) < HEADFORM_R + 30.0
            Hn = H[near]
            lift = 0.0
            for _ in range(8):
                c = np.array([xt, yt, ref_xyz[2] + dz + lift])
                pen = HEADFORM_R - float(np.linalg.norm(Hn - c, axis=1).min())
                if pen <= PEN_CAP_MM + 1e-3:
                    break
                lift += pen - PEN_CAP_MM
            if lift > 0:
                n_lifted += 1
            dz += lift
            zc = ref_xyz[2] + dz

            if not args.dry_run:
                out_path = os.path.join(inp_out, "HoodImpact_%d.inp" % run)
                existing_complete = (
                    os.path.exists(out_path)
                    and os.path.getsize(out_path) >= 1000000)
                if args.skip_existing and existing_complete:
                    n_skipped += 1
                else:
                    if args.skip_existing and os.path.exists(out_path):
                        print("   run %d: replacing incomplete existing deck"
                              % run)
                    write_deck(deck, imp_ids, (dx, dy, dz), out_path)
                    n_written += 1

            manifest.append({
                "run": run, "design": d, "loc": loc, "base_run": base_run,
                "X1": "%.4f" % xt, "X2": "%.4f" % yt, "X3": "%.4f" % zc,
                "dz_mm": "%.4f" % dz, "center_gap_mm": "%.4f" % gap_c0,
                "sphere_pen_mm": "%.4f" % pen, "lift_mm": "%.4f" % lift,
                "surf_z_mm": "%.4f" % surf_c,
            })
        if n_lifted:
            print("   %d/%d locations lifted to respect the %.2f mm "
                  "penetration cap" % (n_lifted, len(locations), PEN_CAP_MM))
        if args.dry_run:
            deck_summary = "%d decks, dry-run" % len(locations)
        elif args.skip_existing:
            deck_summary = "%d written, %d already present" % (
                n_written, n_skipped)
        else:
            deck_summary = "%d decks" % n_written
        print("   design %2d done (%s)" % (d, deck_summary))

    manifest.sort(key=lambda r: r["run"])
    dataset_size = N_DESIGNS * n_locs
    full_dataset = (
        designs == list(range(N_DESIGNS)) and locations == all_locations)
    metadata_suffix = "" if full_dataset else "_partial"
    man_csv = os.path.join(
        args.out_dir, "manifest_%d%s.csv" % (dataset_size, metadata_suffix))
    with open(man_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)
    print("wrote %s (%d rows)" % (man_csv, len(manifest)))

    # delivered-format coordinates file (X1,X2,X3 per run, ordered by run)
    if len(manifest) and not args.dry_run:
        cc_csv = os.path.join(
            args.out_dir,
            "ImpactCoords_%d%s.csv" % (dataset_size, metadata_suffix))
        with open(cc_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["X1", "X2", "X3"])
            for r in manifest:
                w.writerow([r["X1"], r["X2"], r["X3"]])
        print("wrote %s" % cc_csv)


if __name__ == "__main__":
    main()
