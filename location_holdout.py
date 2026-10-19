"""Restrict which impact locations each design contributes; designs stay split as before.

Two kinds of split file, both stored by location ID with the location's XY so the
loaded impact coordinates prove the IDs line up:

``checkerboard``
    Disjoint locations for training/validation designs and test designs. The
    Euro-NCAP grid is a lattice of wrap-around-distance rows (``wadh_mm``, 100 mm
    apart) and lateral columns (``col_k``); even row + column parity is used only
    on the training and validation designs, odd only on the test designs. Every
    test location therefore is new to the network, with its lattice neighbours
    seen in training. The hood is nearly mirror-symmetric (HIC at col_k and
    -col_k agree within ~2%), and mirror images share parity, so no test
    location's mirror twin is trained on either.

``per_design_random``
    Each design keeps its own random subset of the locations, drawn
    independently per design from one seeded generator. The design-wise
    train/validation/test split is unchanged, and the same location may appear
    in several splits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_DIR = Path(__file__).resolve().parent / "location_splits"
DEFAULT_SPLIT_FILE = SPLIT_DIR / "euroncap1704_checkerboard.json"


def coordinates(grid):
    locations = grid["loc"].to_numpy(int)
    if sorted(locations) != list(range(1, len(locations) + 1)):
        raise ValueError("Location IDs must be 1..N without gaps")
    return {str(row.loc): [row.X1, row.X2] for row in grid.itertuples()}


def checkerboard(grid):
    """Return the split report for a table with loc, X1, X2, wadh_mm and col_k."""
    rows = grid["wadh_mm"].to_numpy(float) / 100
    if not np.allclose(rows, np.round(rows)):
        raise ValueError("wadh_mm must lie on the 100 mm Euro-NCAP lattice")
    odd = (np.round(rows).astype(int) + grid["col_k"].to_numpy(int)) % 2 == 1
    locations = grid["loc"].to_numpy(int)
    coordinate_table = coordinates(grid)
    return {
        "scheme": "checkerboard",
        "definition": "parity of wadh_mm/100 + col_k; even -> training/validation designs, odd -> test designs",
        "train_validation_locations": sorted(locations[~odd].tolist()),
        "test_locations": sorted(locations[odd].tolist()),
        "coordinates_mm": coordinate_table,
    }


def per_design_random(grid, num_designs=12, fraction=0.7, seed=42):
    """Each design keeps round(fraction * N) locations, drawn independently per design."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    coordinate_table = coordinates(grid)
    locations = np.arange(1, len(coordinate_table) + 1)
    count = int(round(fraction * len(locations)))
    rng = np.random.default_rng(seed)
    return {
        "scheme": "per_design_random",
        "definition": (f"each design keeps {count} of {len(locations)} locations, drawn without "
                       f"replacement and independently per design (numpy default_rng({seed}), designs in order)"),
        "fraction": fraction,
        "seed": seed,
        "design_locations": {str(design): sorted(rng.choice(locations, count, replace=False).tolist())
                             for design in range(num_designs)},
        "coordinates_mm": coordinate_table,
    }


def load_split(path=DEFAULT_SPLIT_FILE):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {int(key) for key in report["coordinates_mm"]}
    if report.get("scheme") == "per_design_random":
        for design, kept in report["design_locations"].items():
            if not kept or len(set(kept)) != len(kept) or not set(kept) <= known:
                raise ValueError(f"{path}: design {design} needs distinct, known location IDs")
    else:
        seen, unseen = set(report["train_validation_locations"]), set(report["test_locations"])
        if not seen or not unseen or seen & unseen:
            raise ValueError(f"{path}: location lists must be nonempty and disjoint")
        if seen | unseen != known:
            raise ValueError(f"{path}: every location needs exactly one coordinate entry")
    return {**report, "file": str(Path(path).resolve())}


def describe(report):
    if report.get("scheme") == "per_design_random":
        counts = sorted({len(kept) for kept in report["design_locations"].values()})
        return (f"per-design random: {'/'.join(map(str, counts))} of {len(report['coordinates_mm'])} "
                f"locations per design, seed {report['seed']}")
    return (f"{report['scheme']}: {len(report['train_validation_locations'])} training/validation "
            f"locations, {len(report['test_locations'])} test-only locations")


def apply_split(data, report, test_designs, samples_per_design=142, tolerance_mm=1e-3):
    """Keep, for every run, only the locations the split assigns to its design."""
    if report.get("scheme") == "per_design_random":
        per_design = {int(design): set(kept) for design, kept in report["design_locations"].items()}
    else:
        seen, unseen = set(report["train_validation_locations"]), set(report["test_locations"])
    test_designs = set(test_designs)
    indices = []
    for i, run in enumerate(data["run_numbers"]):
        design, location = divmod(int(run) - 1, samples_per_design)
        location += 1
        expected = report["coordinates_mm"].get(str(location))
        if expected is None:
            raise ValueError(f"Run {run}: location {location} is not in the location split")
        # The split is stored by location ID; the loaded XY proves the IDs line up.
        if np.abs(np.asarray(data["indentor_positions"][i], float) - expected).max() > tolerance_mm:
            raise ValueError(f"Run {run}: impact XY {data['indentor_positions'][i]} differs from "
                             f"location {location} in the split file {expected}")
        if report.get("scheme") == "per_design_random":
            if design not in per_design:
                raise ValueError(f"Run {run}: design {design} has no location list in the split file")
            kept = per_design[design]
        else:
            kept = unseen if design in test_designs else seen
        if location in kept:
            indices.append(i)
    return {key: value[indices] if isinstance(value, np.ndarray)
            else [value[i] for i in indices] for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path,
                        default=Path("Data/HoodImpact_1704_EuroNCAP/impact_locations_142.csv"))
    parser.add_argument("--scheme", choices=("checkerboard", "per_design_random"), default="checkerboard")
    parser.add_argument("--fraction", type=float, default=0.7, help="per_design_random only")
    parser.add_argument("--seed", type=int, default=42, help="per_design_random only")
    parser.add_argument("--num-designs", type=int, default=12, help="per_design_random only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grid = pd.read_csv(args.grid)
    if args.scheme == "checkerboard":
        report = checkerboard(grid)
        output = args.output or DEFAULT_SPLIT_FILE
    else:
        report = per_design_random(grid, args.num_designs, args.fraction, args.seed)
        output = args.output or SPLIT_DIR / f"euroncap1704_random{round(100 * args.fraction)}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{describe(report)} -> {output}")


if __name__ == "__main__":
    main()
