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

import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Import the training entry point so missing training dependencies fail here,
# before the scheduler starts the full dataset load.
from train_mesh_impact_history import MODEL_DEFAULTS, Config, DataPreprocessor, parse_args, validate_data
from local_edits_data import (
    BASE_DESIGN, BASE_DESIGN_ID, LOCATIONS, LocalEditGeometry, base_inner_panel, dataset_rows, design_name,
    sample_number,
)
from local_edit_metrics import design_classes
from mesh_design_change import DesignChangeHistoryNet, DesignChangeTable, change_displacement_rms

EDITED_LOCATIONS = 18
DECK_CHECK_TOLERANCE_MM = 1e-3
# The trainer's localedits defaults (utils.utils.Config), relative to the repository.
DEFAULT_ROOT = Path("Data/HoodImpact_LocalEdits")


def validate_runtime(require_cuda):
    version = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not version or tuple(map(int, version.groups())) < (2, 8):
        raise ValueError(f"PyTorch >= 2.8 is required; found {torch.__version__}")
    print(f"Python: {sys.executable} | architecture: {platform.machine()}")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    if require_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment")


def validate_cuda_model(data, design, change, neighbor_cache_dir, model_options, base_mesh=None):
    """CUDA forward/backward of a small DesignChangeHistoryNet with the configured options.

    The first run is read with its edited design's change nodes, so the change
    readouts must receive a gradient (and, with change locality, the locality
    rates). With ``base_mesh`` (training on design 6) that hood is also read as
    a design with no change nodes. Mesh, impact and time are standardized with
    this one example's statistics; nothing reaches training.
    """
    mesh = np.asarray(data["mesh_geometries"][0], dtype=np.float32)
    mean = mesh.mean(axis=0)
    scale = np.maximum(mesh.std(axis=0), 1e-6)
    mesh = torch.as_tensor((mesh - mean) / scale, device="cuda")
    impact = torch.as_tensor(
        (np.asarray(data["indentor_positions"][0]) - mean[:2]) / scale[:2],
        device="cuda", dtype=torch.float32,
    ).unsqueeze(0)
    time = np.asarray(data["time_arrays"][0], dtype=np.float32)
    time_mean, time_scale = float(time.mean()), max(float(time.std()), 1e-6)
    time = torch.as_tensor((time - time_mean) / time_scale, device="cuda")
    model = DesignChangeHistoryNet(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=1,
                                   dropout=0.0, **model_options).to("cuda")
    # Wiring probe only: training registers each design's own meshes instead.
    changes = {design: change}
    if base_mesh is not None:
        changes[BASE_DESIGN_ID] = {"static": np.zeros((0, 6), np.float32)}
    model.design_changes = DesignChangeTable.from_changes(changes)
    model.design_changes.register(mesh, design)
    if model.design_change_features == "edit":
        model.set_change_displacement_scale(change_displacement_rms({design: change}, [design]))
    model.set_coordinate_scalers(mean, scale, mean[:2], scale[:2])
    model.set_time_scaler(time_mean, time_scale)
    if model.neighborhood_blocks:
        model.neighbor_cache.directory = Path(neighbor_cache_dir)
        model.neighbor_cache.register_geometry(data["mesh_geometries"][0], mesh * model.mesh_scale,
                                               model.neighborhood_k)
    prediction = model(mesh=mesh, mesh_batch=torch.zeros(len(mesh), device="cuda", dtype=torch.long),
                       indentor=impact, time=time,
                       time_batch=torch.zeros(len(time), device="cuda", dtype=torch.long), batch_size=1)
    if prediction.shape != time.shape or not torch.isfinite(prediction).all():
        raise ValueError("CUDA model probe produced an invalid acceleration history")
    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if (model.local_latents and model.local_log_precision.grad is None) or not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise ValueError("CUDA model probe produced missing spatial-bias or nonfinite gradients")
    if any(parameter.grad is None for parameter in model.neighborhood_blocks.parameters()):
        raise ValueError("CUDA model probe produced missing neighborhood gradients")
    if any(readout.output.weight.grad is None or not readout.output.weight.grad.abs().sum() > 0
           for readout in model.change_readouts):
        raise ValueError("CUDA model probe produced no gradient for a design-change readout")
    if model.change_locality and model.change_log_rate.grad is None:
        raise ValueError("CUDA model probe: the change-locality rates are not in the graph")
    if base_mesh is not None:
        base = torch.as_tensor((np.asarray(base_mesh, dtype=np.float32) - mean) / scale, device="cuda")
        model.design_changes.register(base, BASE_DESIGN_ID)
        if model.neighborhood_blocks:
            model.neighbor_cache.register_geometry(base_mesh, base * model.mesh_scale, model.neighborhood_k)
        with torch.no_grad():
            base_prediction = model(mesh=base, mesh_batch=torch.zeros(len(base), device="cuda", dtype=torch.long),
                                    indentor=impact, time=time,
                                    time_batch=torch.zeros(len(time), device="cuda", dtype=torch.long),
                                    batch_size=1)
        if not torch.isfinite(base_prediction).all():
            raise ValueError("CUDA model probe: design 6 (no change nodes) gave an invalid history")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)} | {type(model).__name__} "
          f"(change tokens: {model.design_change_features}, change locality: {model.change_locality}) | "
          f"neighborhood_layers={len(model.neighborhood_blocks)} | forward/backward: passed"
          + (" | design 6 read with no change nodes: passed" if base_mesh is not None else ""))
    if model.neighborhood_blocks:
        print(f"Neighbor graph cache: built={model.neighbor_cache.builds} | loaded from disk={model.neighbor_cache.disk_hits}")


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
    base = (f"design 6 trained on too ({LOCATIONS} runs, no change nodes)" if args.train_on_base_design
            else "design 6 not trained on")
    print(f"Split: training {len(training)} edited designs, {len(training) * EDITED_LOCATIONS} runs | {base}")
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
        inp_dir = args.inp_dir or str(DEFAULT_ROOT / "inp_files")
        manifest_path = args.impact_coords_path or str(DEFAULT_ROOT / "manifest_local_edits.csv")
        acceleration_dir = args.acceleration_dir or str(DEFAULT_ROOT / "output_history_acc")
        cuda = (args.device or "").startswith("cuda")
        validate_runtime(cuda)

        geometry, manifest, base_deck = LocalEditGeometry.from_manifest(manifest_path, inp_dir)
        manifest = dataset_rows(manifest, include_base=True)
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
            acceleration_dir=acceleration_dir, time_subsample_stride=stride, max_train_time=args.max_train_time))
        design = sorted(args.test_designs)[0]
        row = manifest[manifest["design_id"] == design].iloc[0]
        base_row = manifest[(manifest["design"] == BASE_DESIGN) & (manifest["location"] == row["location"])].iloc[0]
        times, accel = processor.load_acceleration_history(int(row["run"]))
        base_times, base_accel = processor.load_acceleration_history(int(base_row["run"]))
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
            cache = args.neighbor_cache_dir or default_neighbor_cache_directory(inp_dir)
            # The probe's own small width, heads, latents and depth; every other option as configured.
            options = {key: getattr(args, key) for key in MODEL_DEFAULTS
                       if key not in ("width", "num_heads", "num_latents", "latent_layers", "temporal_layers",
                                      "dropout")}
            validate_cuda_model(data, design, geometry.changes(design), cache, options,
                                base_mesh=geometry.mesh(BASE_DESIGN_ID) if args.train_on_base_design else None)
        else:
            print("CUDA probe: skipped (--device is not cuda)")
        print("Preflight passed.")
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
