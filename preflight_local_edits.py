"""Check a local-edit training job before the long data load: files, meshes, split and CUDA.

Takes the training command's own arguments, so a job checks exactly what it
will train (run_mesh_change_local_edits_delta.sbatch passes the same list):

    python preflight_local_edits.py --data-format localedits --inp-dir ... [training options]

Checks every curve file of the edited runs and of design 6 (the reference), the
change tables against the design-6 deck, the meshes built for the held-out
designs and two training designs against their own decks, the time grid, and a
CUDA forward/backward of the configured network. With neighborhood layers the
probe writes one hood's kNN graph into the neighbor-graph cache, which training
reuses; nothing else is written.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_mesh_impact_history import (
    DESIGN_CHANGE_DEFAULTS, Config, DataPreprocessor, parse_args, validate_data, validate_location_shortcuts,
)
from preflight_mesh_impact_history_1704 import EXPERIMENT_MODEL_KEYS, validate_cuda_model, validate_runtime
from local_edits_data import (
    BASE_DESIGN, LOCATIONS, LocalEditGeometry, base_inner_panel, design_name, sample_number,
)
from local_edit_metrics import design_classes

EDITED_LOCATIONS = 20
DECK_CHECK_TOLERANCE_MM = 1e-3
# The trainer's localedits defaults (utils.utils.Config), relative to the repository.
DEFAULT_ROOT = Path("Data/HoodImpact_LocalEdits")


def check_files(manifest, acceleration_dir):
    base = manifest[manifest["design"] == BASE_DESIGN]
    if sorted(base["location"]) != list(range(1, LOCATIONS + 1)):
        raise ValueError(f"Design 6 must have one run at each of the {LOCATIONS} locations")
    edited = manifest[manifest["design"] != BASE_DESIGN]
    counts = edited.groupby("design").size()
    if (counts != EDITED_LOCATIONS).any():
        raise ValueError(f"Edited designs must have {EDITED_LOCATIONS} runs each: {counts[counts != EDITED_LOCATIONS].to_dict()}")
    missing = [int(run) for run in manifest["run"]
               if not (Path(acceleration_dir) / f"HoodImpact_{int(run)}_SAE1000_interp1000.csv").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} curve files are missing from {acceleration_dir}, e.g. runs {missing[:5]}")
    print(f"Files: {len(counts)} edited designs x {EDITED_LOCATIONS} runs + design 6 at {LOCATIONS} locations; "
          f"all {len(manifest)} curve files present")
    return sorted(edited["design_id"].unique())


def check_split(args, designs, classes):
    test, val = set(args.test_designs), set(args.val_designs)
    unknown = (test | val) - set(designs)
    if unknown:
        raise ValueError(f"Held-out designs {sorted(unknown)} are not edited designs (1..72; design 6 is never held out)")
    if not test or not val or test & val:
        raise ValueError("Test and validation designs must be nonempty and disjoint")
    for label, group in (("test", test), ("validation", val)):
        for design in sorted(group):
            name = design_name(design)
            info = classes.loc[name]
            print(f"Split: {label} {name} | {info['class']} at {info['site']} | {EDITED_LOCATIONS} runs")
    training = [d for d in designs if d not in test | val]
    print(f"Split: training {len(training)} designs, {len(training) * EDITED_LOCATIONS} runs | design 6 not trained on")
    return training


def check_decks(geometry, manifest, inp_dir, designs):
    for design in designs:
        runs = manifest.loc[manifest["design_id"] == design, "run"]
        deck = Path(inp_dir) / f"HoodImpact_{int(runs.min())}.inp"
        name = design_name(design)
        if not deck.is_file():
            print(f"Mesh check: {name} skipped ({deck.name} not present)")
            continue
        ids, xyz = base_inner_panel(deck)
        if not np.array_equal(ids, geometry.base_ids):
            raise ValueError(f"{name}: {deck.name} has other inner-panel nodes than design 6")
        error = float(np.abs(xyz - geometry.mesh(design)).max())
        if error > DECK_CHECK_TOLERANCE_MM:
            raise ValueError(f"{name}: built mesh differs from {deck.name} by {error:.3g} mm")
        moved = len(geometry.changes(design)["rows"])
        print(f"Mesh check: {name} = design 6 + {moved} moved nodes matches {deck.name} (max error {error:.1e} mm)")


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.data_format != "localedits":
            raise ValueError("This preflight is for --data-format localedits")
        if args.mesh_nodes != "inner_panel":
            raise ValueError("localedits builds only inner-panel meshes")
        if args.design_change_nodes and args.design_change_features != "xyz":
            raise ValueError("localedits change tables carry no panel gaps; use --design-change-features xyz")
        inp_dir = args.inp_dir or str(DEFAULT_ROOT / "inp_files")
        manifest_path = args.impact_coords_path or str(DEFAULT_ROOT / "manifest_local_edits.csv")
        acceleration_dir = args.acceleration_dir or str(DEFAULT_ROOT / "output_history_acc")
        cuda = (args.device or "").startswith("cuda")
        validate_runtime(cuda)
        validate_location_shortcuts(args.impactor_nodes, args.pool_impactor_nodes, args.mesh_nodes)

        geometry, manifest, base_deck = LocalEditGeometry.from_manifest(manifest_path, inp_dir)
        print(f"Base hood: inner panel of {base_deck} ({len(geometry.base_ids)} nodes)")
        designs = check_files(manifest, acceleration_dir)
        for design in designs:
            geometry.changes(design)  # reads each table and checks it against the base deck
        print(f"Change tables: {len(designs)} read; every listed node is on the inner panel and z_design6 "
              f"matches the base deck")
        classes = design_classes(Path(manifest_path).parent)
        training = check_split(args, designs, classes)
        check_decks(geometry, manifest, inp_dir,
                    [*sorted(args.test_designs), *sorted(args.val_designs), training[0], training[-1]])

        stride = args.time_subsample_stride or Config.time_subsample_stride
        processor = DataPreprocessor(SimpleNamespace(
            data_format="localedits", acceleration_dir=acceleration_dir, time_subsample_stride=stride,
            max_train_time=args.max_train_time, prediction_target="acceleration", mesh_nodes="inner_panel"))
        design = sorted(args.test_designs)[0]
        row = manifest[manifest["design_id"] == design].iloc[0]
        base_row = manifest[(manifest["design"] == BASE_DESIGN) & (manifest["location"] == row["location"])].iloc[0]
        times, accel = processor.load_acceleration_history(int(row["run"]))
        base_times, _ = processor.load_acceleration_history(int(base_row["run"]))
        if not np.array_equal(times, base_times):
            raise ValueError("Edited and design-6 curves are not on the same sampled time grid")
        data = {"run_numbers": [sample_number(design, int(row["location"]))],
                "mesh_geometries": [geometry.mesh(design)],
                "indentor_positions": [np.array([row["x_mm"], row["y_mm"]], dtype=np.float32)],
                "time_arrays": [times], "accelerations": [accel]}
        validate_data(data)
        print(f"Curves: {len(times)} sampled points from {1e3 * times[0]:.2f} to {1e3 * times[-1]:.2f} ms "
              f"(stride {stride}); design 6 on the same grid")

        if cuda:
            from mesh_neighborhood import default_neighbor_cache_directory
            cache = args.neighbor_cache_dir or default_neighbor_cache_directory("localedits", inp_dir)
            validate_cuda_model(
                data, decoder=args.decoder, neighbor_cache_dir=cache,
                design_changes={design: geometry.changes(design)} if args.design_change_nodes else None,
                design_change_options={key: getattr(args, key) for key in DESIGN_CHANGE_DEFAULTS},
                **{key: getattr(args, key) for key in EXPERIMENT_MODEL_KEYS})
        else:
            print("CUDA probe: skipped (--device is not cuda)")
        print("Preflight passed.")
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
