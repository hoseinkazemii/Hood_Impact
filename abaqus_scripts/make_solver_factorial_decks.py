"""
make_solver_factorial_decks.py
==============================
Build the decks for the solver-factorial experiment: how much of the design-4
vs design-5 acceleration difference comes from the solver set-up rather than
from the hood geometry?

Why
---
Designs 4 and 5 differ at ~120 of 38,691 hood nodes, yet their headform
accelerations differ by ~11.7 g RMS, and every surrogate has zero skill on that
difference. Two non-geometric mechanisms are suspected (see
runs/mesh_impact_history/within_family_dt_artifact/):

* Time step. The decks use ``*VARIABLE MASS SCALING, DT=5e-07, TYPE=BELOW MIN``
  with automatic incrementation, so an inner-panel edit that creates a smaller
  element shrinks the global stable increment. Design 5 takes ~40.3k increments,
  design 4 ~26.6k.
* Start state. The headform is placed by the partner's centre-clearance rule,
  which leaves most runs starting in contact (1228/1704 once the 0.75 + 0.5 mm
  shell offsets are counted) and some at the 1.65 mm penetration cap. The
  far-field, early-onset part of the 4-5 difference concentrates there.

Both designs share the same outer skin and the same headform placement at every
location, so any difference between them is the inner-panel edit plus whatever
the solver does with it.

Design
------
For each selected location and each of designs 4 and 5 this writes one deck per
cell of a 2 x 2 factorial; the native/native cell is the delivered dataset
itself and is not re-solved unless asked for:

    fixed_dt              *DYNAMIC, EXPLICIT, DIRECT USER CONTROL with one
                          increment for both designs (default 4.5e-7 s, below
                          both designs' stable limits and below the 5e-7 s
                          mass-scaling target)
    gap_start             headform translated vertically so its contact surface
                          starts a fixed clearance (default 1.0 mm) above the
                          hood outer contact surface; native incrementation
    fixed_dt_gap_start    both changes (the "clean" cell)
    fixed_dt_gap_start_jitter
                          design 4 only: the clean cell with every hood node
                          displaced by a uniform random 1e-3 mm, i.e. the
                          chaotic floor under the SAME solver settings as the
                          clean difference (the noise-floor experiment measures
                          it under the native settings)
    native                optional byte-identical re-solve of the delivered deck

Everything else in a deck is byte-identical to the delivered one: the hood mesh,
materials, contact, mass scaling and output requests. Only the *DYNAMIC card
and/or the headform node lines change.

Reading the result (abaqus_scripts/analyze_solver_factorial.py)
----------------------------------------------------------------
Compare the design 4 vs 5 difference across cells, by distance from the impact
point to the edit, against the jitter floor of the noise-floor experiment:

    native >> fixed_dt_gap_start far from the edit         -> solver artifact confirmed
    fixed_dt_gap_start near the edit >> clean jitter floor -> genuine geometry signal

Usage
-----
    python abaqus_scripts/make_solver_factorial_decks.py --dry-run
    python abaqus_scripts/make_solver_factorial_decks.py
    python abaqus_scripts/make_solver_factorial_decks.py --locations 134,137 --cells fixed_dt_gap_start

Then solve with abaqus_scripts/delta/hood_design_difference_audit.sbatch.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from inp_geom import Deck, HEADFORM_RADIUS, HOOD_OUTER_ELSET

DEFAULT_DATA = os.path.join(CODE_ROOT, "Data", "HoodImpact_1704_EuroNCAP")
DEFAULT_OUT = os.path.join(CODE_ROOT, "Data", "HoodImpact_SolverFactorial")
MANIFEST = "manifest_solver_factorial.csv"
SAMPLES_PER_DESIGN = 142
DESIGNS = (4, 5)

# One location per lateral mirror pair (the hood and this edit are nearly
# mirror-symmetric). XY distance from the impact point to the 4-5 edit:
#   near (< 100 mm): 134, 80, 133, 140, 142, 135, 35, 62  -- where genuine
#                    geometry signal should live; mixed start states
#   mid (100-250):   115, 106, 128
#   far (> 250):     137, 127, 104 (large, early-onset differences: suspected
#                    artifact) and 88 (quiet control, 0.47 g)
# 128, 127, 115, 104, 137, 135 and 88 are also in the noise-floor experiment.
DEFAULT_LOCATIONS = (134, 80, 133, 140, 142, 135, 35, 62, 115, 106, 128, 137, 127, 104, 88)
# "Clean" first: it is the cell that answers the question, so a walltime kill
# still leaves it complete.
CELLS = ("fixed_dt_gap_start", "fixed_dt_gap_start_jitter", "fixed_dt", "gap_start", "native")
DEFAULT_CELLS = CELLS[:4]
JITTER_CELL = "fixed_dt_gap_start_jitter"
DEFAULT_JITTER_MM = 1e-3

DEFAULT_FIXED_DT = 4.5e-7          # s; below 5e-7 mass-scaling target
DEFAULT_CLEARANCE_MM = 1.0         # clear gap between contact surfaces
# Abaqus general contact uses the shell mid-surface offset by half the
# thickness: hood outer 1.5 mm (0.75) + headform "Layer2" steel 1.0 mm (0.5).
CONTACT_OFFSET_MM = 1.25
NEAR_RADIUS_MM = HEADFORM_RADIUS + 30.0


def source_run(design, location):
    """1-indexed run id of (design, location) in the 1704-sample dataset."""
    return design * SAMPLES_PER_DESIGN + location


def parse_list(text, cast=int):
    return [cast(part) for part in text.split(",") if part.strip()]


def dynamic_card_index(lines):
    """Index of the single *DYNAMIC, EXPLICIT card and of its data line."""
    cards = [i for i, line in enumerate(lines)
             if line.lstrip().upper().startswith("*DYNAMIC")]
    if len(cards) != 1:
        raise ValueError("expected exactly one *DYNAMIC card, found %d" % len(cards))
    card = cards[0]
    if "EXPLICIT" not in lines[card].upper():
        raise ValueError("the *DYNAMIC card is not an explicit step: %r" % lines[card])
    data = card + 1
    while data < len(lines) and lines[data].lstrip().startswith("**"):
        data += 1
    if data >= len(lines) or lines[data].lstrip().startswith("*"):
        raise ValueError("the *DYNAMIC card has no data line")
    return card, data


def fixed_increment_lines(lines, time_increment):
    """Replacement {index: line} for a fixed user-controlled increment.

    Automatic incrementation reads ``, period``; DIRECT USER CONTROL reads
    ``increment, period`` and cannot be combined with SCALE FACTOR, so the
    whole card is rewritten. The step period is kept from the original.
    """
    if not np.isfinite(time_increment) or time_increment <= 0:
        raise ValueError("time increment must be finite and positive")
    card, data = dynamic_card_index(lines)
    fields = [field.strip() for field in lines[data].rstrip("\r\n").split(",")]
    if len(fields) < 2 or not fields[1]:
        raise ValueError("cannot read the step period from %r" % lines[data])
    period = float(fields[1])
    if time_increment >= period:
        raise ValueError("time increment must be smaller than the step period")
    terminator = lines[card][len(lines[card].rstrip("\r\n")):]
    return {
        card: "*DYNAMIC, EXPLICIT, DIRECT USER CONTROL" + terminator,
        data: "%r, %r%s" % (float(time_increment), period,
                            lines[data][len(lines[data].rstrip("\r\n")):]),
    }


def sphere_penetration(hood_outer, centre):
    """True sphere-node penetration (mm); negative means a gap to the mid-surface."""
    return HEADFORM_RADIUS - float(np.linalg.norm(hood_outer - centre, axis=1).min())


def gap_start_lift(hood_outer, centre, target_penetration, tolerance=1e-4, max_iter=60):
    """Vertical shift (mm, + is up) giving the requested sphere penetration.

    The closest hood node can sit off the vertical through the centre, so a
    shift changes the penetration by cos(angle) times the shift. The fixed-point
    update converges geometrically; it can also lower a headform that starts
    farther away than the target.
    """
    near = np.linalg.norm(hood_outer[:, :2] - centre[:2], axis=1) < NEAR_RADIUS_MM
    nodes = hood_outer[near]
    if not len(nodes):
        raise ValueError("no hood outer node under the headform at %s" % (centre,))
    lift = 0.0
    for _ in range(max_iter):
        penetration = sphere_penetration(nodes, centre + np.array([0.0, 0.0, lift]))
        error = penetration - target_penetration
        if abs(error) <= tolerance:
            return lift, penetration
        lift += error
    raise RuntimeError("headform placement did not converge (error %.2e mm)" % error)


def format_node_line(nid, x, y, z, terminator):
    # repr() gives the shortest exact round-trip float, matching the deck style;
    # cast defensively -- repr(np.float64) would corrupt the deck.
    return "%d, %r, %r, %r%s" % (nid, float(x), float(y), float(z), terminator)


def jitter_lines(deck, amplitude, generator):
    """Replacement {index: line} displacing every hood node by U(-a, a) per axis."""
    held = deck.impactor_node_ids()
    movable = sorted(set(deck.nodes) - held)
    offsets = generator.uniform(-amplitude, amplitude, size=(len(movable), 3))
    replaced = {}
    for nid, (dx, dy, dz) in zip(movable, offsets):
        index = deck.node_line_idx[nid]
        x, y, z = deck.nodes[nid]
        line = deck.lines[index]
        replaced[index] = format_node_line(nid, x + dx, y + dy, z + dz, line[len(line.rstrip("\r\n")):])
    return replaced


def headform_lift_lines(deck, lift):
    """Replacement {index: line} translating every headform node by ``lift`` in z."""
    replaced = {}
    for nid in deck.impactor_node_ids():
        index = deck.node_line_idx[nid]
        x, y, z = deck.nodes[nid]
        line = deck.lines[index]
        replaced[index] = format_node_line(nid, x, y, z + lift, line[len(line.rstrip("\r\n")):])
    return replaced


def write_lines(lines, replaced, out_path):
    """Write the deck atomically so an interrupted run cannot look complete."""
    temporary = out_path + ".tmp"
    try:
        with open(temporary, "w", newline="") as handle:
            for index, line in enumerate(lines):
                handle.write(replaced.get(index, line))
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--locations", default=",".join(map(str, DEFAULT_LOCATIONS)),
                        help="comma-separated 1704 location ids (1-142)")
    parser.add_argument("--designs", default=",".join(map(str, DESIGNS)))
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                        help="subset of %s" % ",".join(CELLS))
    parser.add_argument("--fixed-dt", type=float, default=DEFAULT_FIXED_DT,
                        help="seconds; must stay below both designs' stable increment")
    parser.add_argument("--clearance-mm", type=float, default=DEFAULT_CLEARANCE_MM,
                        help="clear gap between headform and hood contact surfaces at t=0")
    parser.add_argument("--jitter-mm", type=float, default=DEFAULT_JITTER_MM,
                        help="hood-node jitter amplitude of the %s cell" % JITTER_CELL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the placement and write nothing")
    args = parser.parse_args()

    locations = parse_list(args.locations)
    designs = parse_list(args.designs)
    cells = [cell.strip() for cell in args.cells.split(",") if cell.strip()]
    bad = [cell for cell in cells if cell not in CELLS]
    if bad or not cells:
        raise SystemExit("unknown cells %s; choose from %s" % (bad, ", ".join(CELLS)))
    if any(not 1 <= loc <= SAMPLES_PER_DESIGN for loc in locations) or len(set(locations)) != len(locations):
        raise SystemExit("locations must be distinct ids in 1-%d" % SAMPLES_PER_DESIGN)
    if args.clearance_mm < 0 or not np.isfinite(args.clearance_mm):
        raise SystemExit("clearance must be finite and nonnegative")
    target_penetration = -(CONTACT_OFFSET_MM + args.clearance_mm)
    inp_dir = os.path.join(args.data_dir, "inp_files")
    if not os.path.isdir(inp_dir):
        raise SystemExit("missing source decks: %s" % inp_dir)

    out_inp = os.path.join(args.out_dir, "inp_files")
    if not args.dry_run:
        os.makedirs(out_inp, exist_ok=True)
    print("Cells: %s | designs %s | %d locations | fixed dt %.3g s | gap start: %.2f mm clear "
          "(sphere penetration %.2f mm)" % (", ".join(cells), designs, len(locations),
                                           args.fixed_dt, args.clearance_mm, target_penetration))
    # Run numbers are cell-major so the first cell finishes first on the
    # cluster; each source deck is still parsed only once.
    if not np.isfinite(args.jitter_mm) or args.jitter_mm <= 0:
        raise SystemExit("jitter amplitude must be finite and positive")
    generator = np.random.default_rng(args.seed)

    def cell_designs(cell):
        # The jitter floor is one design's sensitivity to a meaningless change.
        return designs[:1] if cell == JITTER_CELL else designs

    numbering = {}
    for cell in cells:
        for location in locations:
            for design in cell_designs(cell):
                numbering[(cell, location, design)] = len(numbering) + 1
    records = []
    placement = {}
    for location in locations:
        for design in designs:
            run = source_run(design, location)
            deck_path = os.path.join(inp_dir, "HoodImpact_%d.inp" % run)
            if not os.path.exists(deck_path):
                raise SystemExit("missing source deck: %s" % deck_path)
            deck = Deck(deck_path, keep_lines=True)
            ref = np.array(deck.nodes[deck.rigid_ref], float)
            outer_ids = sorted(deck.elset_nodes(HOOD_OUTER_ELSET))
            outer = np.array([deck.nodes[i] for i in outer_ids], float)
            native_pen = sphere_penetration(outer, ref)
            gap_lift, gap_pen = gap_start_lift(outer, ref, target_penetration)
            placement[(location, design)] = (native_pen, gap_lift)
            for cell in cells:
                if design not in cell_designs(cell):
                    continue
                fixed = cell.startswith("fixed_dt")
                gap = "gap_start" in cell
                jitter = args.jitter_mm if cell == JITTER_CELL else 0.0
                lift, penetration = (gap_lift, gap_pen) if gap else (0.0, native_pen)
                replaced = {}
                if fixed:
                    replaced.update(fixed_increment_lines(deck.lines, args.fixed_dt))
                if jitter:
                    replaced.update(jitter_lines(deck, jitter, generator))
                if gap:
                    replaced.update(headform_lift_lines(deck, lift))
                synthetic = numbering[(cell, location, design)]
                if not args.dry_run:
                    target = os.path.join(out_inp, "HoodImpact_%d.inp" % synthetic)
                    if cell == "native":
                        shutil.copyfile(deck_path, target)
                    else:
                        write_lines(deck.lines, replaced, target)
                records.append({
                    "synthetic_run": synthetic, "cell": cell, "fixed_dt": int(fixed),
                    "gap_start": int(gap), "time_increment_s": "%r" % args.fixed_dt if fixed else "",
                    "jitter_mm": jitter, "seed": args.seed if jitter else "",
                    "design": design, "location": location, "source_run": run,
                    "native_sphere_pen_mm": "%.4f" % native_pen,
                    "sphere_pen_mm": "%.4f" % penetration, "lift_mm": "%.4f" % lift,
                    "contact_clearance_mm": "%.4f" % (-penetration - CONTACT_OFFSET_MM),
                })
        spread = np.ptp([placement[(location, design)][1] for design in designs])
        if spread > 1e-3:
            print("  WARNING location %d: gap-start lift differs between designs by %.4f mm; "
                  "the outer skins differ under the headform" % (location, spread))
        print("  location %-4d native pen %7.3f mm | gap-start lift %7.3f mm"
              % (location, placement[(location, designs[0])][0], placement[(location, designs[0])][1]),
              flush=True)

    records.sort(key=lambda record: record["synthetic_run"])
    for cell in cells:
        print("  %-20s %d decks" % (cell, sum(record["cell"] == cell for record in records)))
    if args.dry_run:
        print("\nDry run: %d decks would be written." % len(records))
        return
    synthetic = len(records)
    manifest_path = os.path.join(args.out_dir, MANIFEST)
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n%d decks in %s\nmanifest: %s" % (synthetic, out_inp, manifest_path))
    print("Solve with abaqus_scripts/delta/hood_design_difference_audit.sbatch, then run\n"
          "  python abaqus_scripts/analyze_solver_factorial.py")


if __name__ == "__main__":
    main()
