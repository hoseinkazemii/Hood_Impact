"""Disjoint impact locations for training/validation designs and test designs.

The Euro-NCAP grid is a lattice of wrap-around-distance rows (``wadh_mm``, 100 mm
apart) and lateral columns (``col_k``). The checkerboard split assigns each
location by the parity of row + column: even locations are used only on the
training and validation designs, odd locations only on the test designs. Every
test location therefore is new to the network, with its lattice neighbours seen
in training. The hood is nearly mirror-symmetric (HIC at col_k and -col_k agree
within ~2%), and mirror images share parity, so no test location's mirror twin
is trained on either.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPLIT_FILE = Path(__file__).resolve().parent / "location_splits" / "euroncap1704_checkerboard.json"


def checkerboard(grid):
    """Return the split report for a table with loc, X1, X2, wadh_mm and col_k."""
    rows = grid["wadh_mm"].to_numpy(float) / 100
    if not np.allclose(rows, np.round(rows)):
        raise ValueError("wadh_mm must lie on the 100 mm Euro-NCAP lattice")
    odd = (np.round(rows).astype(int) + grid["col_k"].to_numpy(int)) % 2 == 1
    locations = grid["loc"].to_numpy(int)
    if sorted(locations) != list(range(1, len(locations) + 1)):
        raise ValueError("Location IDs must be 1..N without gaps")
    return {
        "scheme": "checkerboard",
        "definition": "parity of wadh_mm/100 + col_k; even -> training/validation designs, odd -> test designs",
        "train_validation_locations": sorted(locations[~odd].tolist()),
        "test_locations": sorted(locations[odd].tolist()),
        "coordinates_mm": {str(row.loc): [row.X1, row.X2] for row in grid.itertuples()},
    }


def load_split(path=DEFAULT_SPLIT_FILE):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    seen, unseen = set(report["train_validation_locations"]), set(report["test_locations"])
    if not seen or not unseen or seen & unseen:
        raise ValueError(f"{path}: location lists must be nonempty and disjoint")
    if seen | unseen != {int(key) for key in report["coordinates_mm"]}:
        raise ValueError(f"{path}: every location needs exactly one coordinate entry")
    return {**report, "file": str(Path(path).resolve())}


def apply_split(data, report, test_designs, samples_per_design=142, tolerance_mm=1e-3):
    """Keep seen locations on non-test designs and unseen locations on test designs."""
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
        if location in (unseen if design in test_designs else seen):
            indices.append(i)
    return {key: value[indices] if isinstance(value, np.ndarray)
            else [value[i] for i in indices] for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path,
                        default=Path("Data/HoodImpact_1704_EuroNCAP/impact_locations_142.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_FILE)
    args = parser.parse_args()
    report = checkerboard(pd.read_csv(args.grid))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(report['train_validation_locations'])} training/validation locations, "
          f"{len(report['test_locations'])} test locations -> {args.output}")


if __name__ == "__main__":
    main()
