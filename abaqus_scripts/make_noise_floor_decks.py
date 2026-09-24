"""
make_noise_floor_decks.py
=========================
Build the decks for the noise-floor experiment: how much of the measured
design-4 vs design-5 response difference survives a perturbation that cannot
matter physically?

Why
---
Across the 142 matched impact locations, designs 4 and 5 differ by ~11.7 g RMS
in headform acceleration, yet their hoods differ at only 102 of 38691 nodes
(0.26 %), and the largest response differences sit 200-370 mm away from any
moved node. Every surrogate trained so far has zero skill on that difference,
including on training designs it has already seen. Before more modelling
effort, we need to know how much of the 11.7 g an explicit crash solve will
produce from a perturbation of no physical consequence.

Design
------
For each selected impact location on design 4 this writes:

    control              byte-identical copy of the delivered deck
    jitter_<amplitude>   every HOOD node displaced by an independent uniform
                         random offset of that amplitude in mm

The rigid headform nodes, its reference node and the initial-velocity nset are
never moved: shifting them would change the impact position or the impactor
shape, which is a physical change, not a numerical one. Only hood structure
moves.

Amplitudes default to 1e-3 mm and 1e-1 mm. Both are far below anything
physical: the shell elements are ~8 mm and the real design-4/5 change reaches
9.9 mm, so 1e-3 mm is four orders of magnitude smaller than the effect we want
explained. Two amplitudes, rather than one, show whether divergence grows with
the perturbation or has already saturated.

Reading the result
------------------
    control vs delivered archive   platform / determinism floor (expect ~0.5 g)
    control vs jitter              chaotic amplification floor  <- the answer
    design 4 vs design 5           the difference we want predicted (~11.7 g)

If the jitter floor approaches the design difference, the difference is mostly
irreducible and no surrogate can be expected to predict it. If it stays near
the determinism floor, the difference is real signal and the modelling problem
stands.

Usage
-----
    python abaqus_scripts/make_noise_floor_decks.py            # top 8 + 2 controls
    python abaqus_scripts/make_noise_floor_decks.py --locations 115,128,142
    python abaqus_scripts/make_noise_floor_decks.py --amplitudes 1e-3 --dry-run

Then solve the generated directory with the existing batch driver; see
delta/hood_noise_floor.sbatch. Afterwards run analyze_noise_floor.py.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from inp_geom import Deck

DEFAULT_DATA = os.path.join(CODE_ROOT, "Data", "HoodImpact_1704_EuroNCAP")
DEFAULT_OUT = os.path.join(CODE_ROOT, "Data", "HoodImpact_NoiseFloor")
SAMPLES_PER_DESIGN = 142
BASE_DESIGN = 4
PAIR_DESIGN = 5
MANIFEST = "manifest_noise_floor.csv"


def source_run(design, location):
    """1-indexed run id of (design, location) in the 1704-sample dataset."""
    return design * SAMPLES_PER_DESIGN + location


def load_history(acc_dir, run):
    path = os.path.join(acc_dir, "HoodImpact_%d_SAE1000_interp1000.csv" % run)
    frame = pd.read_csv(path)
    return frame["Time"].to_numpy(float), frame["A(in g)"].to_numpy(float)


def rank_locations(acc_dir):
    """Per-location RMS acceleration difference between designs 4 and 5."""
    rows = []
    reference_time = None
    for location in range(1, SAMPLES_PER_DESIGN + 1):
        t4, a4 = load_history(acc_dir, source_run(BASE_DESIGN, location))
        t5, a5 = load_history(acc_dir, source_run(PAIR_DESIGN, location))
        if reference_time is None:
            reference_time = t4
        if not (np.array_equal(t4, t5) and np.array_equal(t4, reference_time)):
            raise SystemExit("Time grids differ at location %d; refusing to "
                             "compare histories by index." % location)
        rows.append((location, float(np.sqrt(np.mean((a4 - a5) ** 2)))))
    return pd.DataFrame(rows, columns=["location", "design_difference_rms_g"])


def select_locations(ranked, num_high, num_low):
    """The locations that most need explaining, plus quiet-location controls.

    Both ends matter. A jitter floor measured only where designs already differ
    could be mistaken for a property of those locations rather than of the
    solve, so quiet locations provide the contrast.
    """
    ordered = ranked.sort_values("design_difference_rms_g", ascending=False)
    high = ordered.head(num_high)["location"].tolist()
    low = ordered.tail(num_low)["location"].tolist()
    overlap = sorted(set(high) & set(low))
    if overlap:
        raise SystemExit("num_high + num_low exceeds the available locations")
    return [(int(loc), "high") for loc in high] + [(int(loc), "low") for loc in low]


def format_node_line(nid, x, y, z, terminator):
    # repr() gives the shortest exact round-trip float, matching the deck's own
    # style; cast defensively -- repr(np.float64) would corrupt the deck.
    return "%d, %r, %r, %r%s" % (nid, float(x), float(y), float(z), terminator)


def write_jittered(deck, movable, offsets, out_path):
    """Rewrite only the movable node lines; every other byte is preserved."""
    lines = deck.lines
    replaced = {}
    for nid, (dx, dy, dz) in zip(movable, offsets):
        index = deck.node_line_idx[nid]
        x, y, z = deck.nodes[nid]
        terminator = lines[index][len(lines[index].rstrip("\r\n")):]
        replaced[index] = format_node_line(nid, x + dx, y + dy, z + dz, terminator)
    temporary = out_path + ".tmp"
    try:
        with open(temporary, "w", newline="") as handle:
            for index, line in enumerate(lines):
                handle.write(replaced.get(index, line))
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def variant_names(amplitudes):
    return ["control"] + ["jitter_%g" % amplitude for amplitude in amplitudes]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--locations", default=None,
                        help="explicit location list, e.g. '115,128,142'; "
                             "default selects by measured design difference")
    parser.add_argument("--num-high", type=int, default=8,
                        help="locations with the largest design-4/5 difference")
    parser.add_argument("--num-low", type=int, default=2,
                        help="quiet-location controls")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1e-3, 1e-1],
                        help="jitter amplitudes in mm (uniform, per node, per axis)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--design", type=int, default=BASE_DESIGN,
                        help="design whose decks are perturbed")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the selection and write nothing")
    args = parser.parse_args()

    if any(not np.isfinite(a) or a <= 0 for a in args.amplitudes):
        raise SystemExit("amplitudes must be finite and positive")
    if len(set(args.amplitudes)) != len(args.amplitudes):
        raise SystemExit("amplitudes must be distinct")
    inp_dir = os.path.join(args.data_dir, "inp_files")
    acc_dir = os.path.join(args.data_dir, "output_history_acc")
    for path in (inp_dir, acc_dir):
        if not os.path.isdir(path):
            raise SystemExit("missing dataset directory: %s" % path)

    if args.locations:
        chosen = [(int(part), "manual") for part in args.locations.split(",") if part.strip()]
        bad = [loc for loc, _ in chosen if not 1 <= loc <= SAMPLES_PER_DESIGN]
        if bad:
            raise SystemExit("locations out of range 1-%d: %s" % (SAMPLES_PER_DESIGN, bad))
        ranked = rank_locations(acc_dir)
    else:
        ranked = rank_locations(acc_dir)
        chosen = select_locations(ranked, args.num_high, args.num_low)
    difference = dict(zip(ranked["location"], ranked["design_difference_rms_g"]))

    print("Design %d vs %d across all %d locations: mean %.2f g, median %.2f g, max %.2f g"
          % (BASE_DESIGN, PAIR_DESIGN, len(ranked),
             ranked["design_difference_rms_g"].mean(),
             ranked["design_difference_rms_g"].median(),
             ranked["design_difference_rms_g"].max()))
    print("\nSelected %d locations x %d variants = %d solves"
          % (len(chosen), len(args.amplitudes) + 1,
             len(chosen) * (len(args.amplitudes) + 1)))
    print("  %-10s %-6s %s" % ("location", "band", "design 4 vs 5 RMS (g)"))
    for location, band in chosen:
        print("  %-10d %-6s %.2f" % (location, band, difference[location]))
    if args.dry_run:
        print("\nDry run: no decks written.")
        return

    out_inp = os.path.join(args.out_dir, "inp_files")
    os.makedirs(out_inp, exist_ok=True)
    generator = np.random.default_rng(args.seed)
    records = []
    synthetic = 0
    for location, band in chosen:
        run = source_run(args.design, location)
        deck_path = os.path.join(inp_dir, "HoodImpact_%d.inp" % run)
        if not os.path.exists(deck_path):
            raise SystemExit("missing source deck: %s" % deck_path)
        deck = Deck(deck_path, keep_lines=True)
        held = deck.impactor_node_ids()
        movable = sorted(set(deck.nodes) - held)
        if not movable:
            raise SystemExit("run %d: no hood nodes left after holding the headform" % run)

        for variant in variant_names(args.amplitudes):
            synthetic += 1
            target = os.path.join(out_inp, "HoodImpact_%d.inp" % synthetic)
            if variant == "control":
                # A byte-identical copy guarantees the control is the delivered
                # deck, so control-vs-archive isolates the environment alone.
                shutil.copyfile(deck_path, target)
                amplitude = 0.0
            else:
                amplitude = float(variant.split("_", 1)[1])
                offsets = generator.uniform(-amplitude, amplitude, size=(len(movable), 3))
                write_jittered(deck, movable, offsets, target)
            records.append({
                "synthetic_run": synthetic, "variant": variant,
                "jitter_mm": amplitude, "design": args.design, "location": location,
                "band": band, "source_run": run,
                "pair_run": source_run(PAIR_DESIGN, location),
                "design_difference_rms_g": difference[location],
                "moved_nodes": 0 if variant == "control" else len(movable),
                "held_nodes": len(held), "seed": args.seed,
            })
        print("  wrote %s for location %d (run %d): %d hood nodes movable, %d held"
              % (", ".join(variant_names(args.amplitudes)), location, run,
                 len(movable), len(held)))

    manifest_path = os.path.join(args.out_dir, MANIFEST)
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n%d decks in %s" % (synthetic, out_inp))
    print("manifest: %s" % manifest_path)
    print("\nSolve them with:\n"
          "  sbatch abaqus_scripts/delta/hood_noise_floor.sbatch\n"
          "then analyse with:\n"
          "  python abaqus_scripts/analyze_noise_floor.py")


if __name__ == "__main__":
    main()
