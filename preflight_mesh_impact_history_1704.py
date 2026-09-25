"""Check inputs and the execution environment for a 1,704-run attention job.

This reads the same mesh, impact XY and acceleration files as training. It
checks every required filename and preprocesses the first run, without writing
artifacts or fitting training scalers. Training validates every loaded record.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# Import the actual training entry point so missing training dependencies fail
# here, before the scheduler starts the full dataset load.
from train_mesh_impact_history import (
    Config,
    DataPreprocessor,
    DECODER_BLOCKS,
    DEFAULT_TEST_DESIGNS,
    DEFAULT_VAL_DESIGNS,
    MODEL_DEFAULTS,
    MeshImpactHistoryNet,
    parse_args as parse_training_args,
    resolve_splits,
    validate_data,
)


NUM_RUNS = 1704
SAMPLES_PER_DESIGN = 142
NUM_DESIGNS = NUM_RUNS // SAMPLES_PER_DESIGN
DEFAULT_DATA_ROOT = "Data/HoodImpact_1704_EuroNCAP"

# The 12 design IDs are only 4 distinct hoods. Verified by comparing the hood
# surfaces on a common XY grid (visualize_design_sensitivity_1704.py): inside a
# cluster the surfaces agree to 0.06 mm, between clusters they differ by at
# least 2.13 mm. Holding out one design while its clones stay in training makes
# the test score measure duplicate retrieval instead of learned physics.
from mesh_design_sensitivity import GEOMETRY_CLUSTERS

# Model knobs this experiment adds on top of the baseline architecture.
EXPERIMENT_MODEL_KEYS = tuple(
    key for key in MODEL_DEFAULTS
    if key.startswith("neighborhood_") or key == "impactor_nodes"
)


def validate_cluster_holdout(test_designs, val_designs):
    """Require the *test* designs to take their whole geometry cluster with them.

    A clone of a test design left in training is fatal: the score then measures
    duplicate retrieval. A clone of a validation design is not -- it only makes
    checkpoint selection slightly optimistic, and holding out a validation
    design's whole cluster would cost a second cluster of training data. That
    case is reported as a warning and the run proceeds.
    """
    held_out = set(test_designs) | set(val_designs)
    test_ids, val_ids = set(test_designs), set(val_designs)
    test_leaks, val_leaks = [], []
    for name, members in GEOMETRY_CLUSTERS.items():
        members = set(members)
        remaining = members - held_out  # still in training
        if not remaining:
            continue
        if test_ids & members:
            test_leaks.append(
                f"cluster {name}: test design(s) {sorted(test_ids & members)} held out "
                f"while near-identical {sorted(remaining)} stay in training"
            )
        elif val_ids & members:
            val_leaks.append(
                f"cluster {name}: validation design(s) {sorted(val_ids & members)} held "
                f"out while near-identical {sorted(remaining)} stay in training"
            )
    if test_leaks:
        raise ValueError(
            "Test designs leave near-clones in training, so the test score would "
            "measure retrieval rather than geometry: " + "; ".join(test_leaks)
            + ". Hold out whole clusters "
            + ", ".join(f"{n}={list(m)}" for n, m in GEOMETRY_CLUSTERS.items())
            + ", or pass --allow-clone-leak to accept this deliberately."
        )
    held_clusters = sorted(name for name, members in GEOMETRY_CLUSTERS.items()
                           if set(members) <= held_out)
    return held_clusters, val_leaks


def validate_splits(test_designs, val_designs):
    """Use the training split logic with the complete, fixed 1704-run mapping."""
    for name, identifiers in (("test", test_designs), ("validation", val_designs)):
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            or not 0 <= value < NUM_DESIGNS
            for value in identifiers
        ):
            raise ValueError(f"{name} design IDs must be integers in 0..{NUM_DESIGNS - 1}")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} design IDs must not contain duplicates")
    return resolve_splits(
        {"run_numbers": list(range(1, NUM_RUNS + 1))},
        SimpleNamespace(samples_per_design=SAMPLES_PER_DESIGN),
        test_designs,
        val_designs,
    )


def validate_dataset(data_root):
    """Require every consumed file, not merely a matching directory count."""
    root = Path(data_root).expanduser().resolve()
    required = [root / "ImpactCoords_1704.csv"]
    for run in range(1, NUM_RUNS + 1):
        required.extend((
            root / "inp_files" / f"HoodImpact_{run}.inp",
            root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv",
        ))
    problems = []
    for path in required:
        if not path.is_file():
            problems.append(f"Missing file: {path}")
        elif path.stat().st_size == 0:
            problems.append(f"Empty file: {path}")
    if problems:
        details = "\n".join(problems[:12])
        extra = f"\n... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise ValueError(f"Dataset is incomplete ({len(problems)} problems):\n{details}{extra}")

    coords = pd.read_csv(root / "ImpactCoords_1704.csv")
    coords.columns = [column.strip() for column in coords.columns]
    if len(coords) != NUM_RUNS:
        raise ValueError(f"ImpactCoords_1704.csv must have {NUM_RUNS} rows; found {len(coords)}")
    missing_columns = {"X1", "X2"} - set(coords.columns)
    if missing_columns:
        raise ValueError(f"ImpactCoords_1704.csv is missing columns: {sorted(missing_columns)}")
    try:
        xy = coords[["X1", "X2"]].to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("ImpactCoords_1704.csv X1/X2 must be numeric and finite") from error
    if not np.isfinite(xy).all():
        raise ValueError("ImpactCoords_1704.csv X1/X2 must be numeric and finite")
    return root, xy


def validate_impactor_boundary(root, impactor_nodes):
    """Confirm the leading nodes are the moving headform and the rest are not.

    Two impacts on one design share every structural node and differ only in
    where the headform sits. Verifying that against the files here means the
    neighborhood graph may be cached per design, and that the node count is
    measured rather than trusted. Runs 1 and 2 both belong to design 0.
    """
    config = SimpleNamespace(
        data_format="euroncap1704", inp_dir=str(root / "inp_files"),
        acceleration_dir=str(root / "output_history_acc"),
        time_subsample_stride=Config.time_subsample_stride,
        max_train_time=Config.max_train_time,
    )
    processor = DataPreprocessor(config)
    first, second = processor.load_mesh_geometry(1), processor.load_mesh_geometry(2)
    if first.shape != second.shape:
        raise ValueError(
            f"Runs 1 and 2 share design 0 but parsed to {first.shape} and {second.shape} nodes"
        )
    moved = np.flatnonzero((first != second).any(axis=1))
    if not np.array_equal(moved, np.arange(impactor_nodes)):
        span = f"{moved.min()}..{moved.max()}" if len(moved) else "none"
        raise ValueError(
            f"--impactor-nodes {impactor_nodes} does not match the data: "
            f"{len(moved)} nodes move between runs 1 and 2, at indices {span}. "
            f"The held-out block must be exactly the leading moving nodes."
        )
    print(f"Impactor holdout: nodes 0..{impactor_nodes - 1} move between impacts; "
          f"the remaining {len(first) - impactor_nodes} structural nodes are identical")


def validate_first_run(root, impact_xy):
    """Check mesh parsing and the configured cutoff/stride against source data."""
    stride = Config.time_subsample_stride
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"Config.time_subsample_stride must be a positive integer; got {stride!r}")
    cutoff = Config.max_train_time
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError("Config.max_train_time must be None or finite and positive")
    # Config() creates its output directory. Only the preprocessor fields are
    # needed for this read-only check, so use the shared class defaults directly.
    config = SimpleNamespace(
        data_format="euroncap1704",
        inp_dir=str(root / "inp_files"),
        acceleration_dir=str(root / "output_history_acc"),
        time_subsample_stride=stride,
        max_train_time=cutoff,
    )
    processor = DataPreprocessor(config)
    mesh = processor.load_mesh_geometry(1)
    frame = pd.read_csv(root / "output_history_acc" / "HoodImpact_1_SAE1000_interp1000.csv")
    source_time = frame["Time"].to_numpy(dtype=np.float32)
    source_accel = frame["A(in g)"].to_numpy(dtype=np.float32)
    data = {
        "run_numbers": [1],
        "mesh_geometries": [mesh],
        "indentor_positions": [impact_xy[0]],
        "time_arrays": [source_time],
        "accelerations": [source_accel],
    }
    validate_data(data)
    truncated_time, truncated_accel = processor.truncate_by_time(source_time, source_accel)
    times, acceleration = processor.load_acceleration_history(1)
    if not np.array_equal(times, truncated_time[::stride]) or not np.array_equal(
        acceleration, truncated_accel[::stride]
    ):
        raise ValueError("First-run preprocessing does not preserve the configured time stride")
    data["time_arrays"] = [times]
    data["accelerations"] = [acceleration]
    validate_data(data)
    return data, len(source_time), len(truncated_time)


def validate_runtime(require_cuda):
    version = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not version or tuple(map(int, version.groups())) < (2, 8):
        raise ValueError(f"PyTorch >= 2.8 is required; found {torch.__version__}")
    print(f"Python: {sys.executable} | architecture: {platform.machine()}")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    if require_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment")


def validate_cuda_model(data, decoder=MODEL_DEFAULTS["decoder"], **neighborhood_options):
    """Exercise the selected decoder and learned spatial-bias gradients."""
    # Standardize this one example only for numerical stability in the probe.
    # These temporary statistics never reach training or a saved checkpoint.
    mesh = np.asarray(data["mesh_geometries"][0], dtype=np.float32)
    mean = mesh.mean(axis=0)
    scale = np.maximum(mesh.std(axis=0), 1e-6)
    mesh = torch.as_tensor((mesh - mean) / scale, device="cuda")
    impact = torch.as_tensor(
        (np.asarray(data["indentor_positions"][0]) - mean[:2]) / scale[:2],
        device="cuda", dtype=torch.float32,
    ).unsqueeze(0)
    time = np.asarray(data["time_arrays"][0], dtype=np.float32)
    time = torch.as_tensor((time - time.mean()) / max(float(time.std()), 1e-6), device="cuda")
    model = MeshImpactHistoryNet(
        width=16, num_heads=2, num_latents=4, latent_layers=1,
        temporal_layers=1, dropout=0.0, decoder=decoder, **neighborhood_options,
    ).to("cuda")
    model.set_coordinate_scalers(mean, scale, mean[:2], scale[:2])
    prediction = model(
        mesh=mesh,
        mesh_batch=torch.zeros(len(mesh), device="cuda", dtype=torch.long),
        indentor=impact,
        time=time,
        time_batch=torch.zeros(len(time), device="cuda", dtype=torch.long),
        batch_size=1,
    )
    if prediction.shape != time.shape or not torch.isfinite(prediction).all():
        raise ValueError("CUDA model probe produced an invalid acceleration history")
    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if model.local_log_precision.grad is None or not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise ValueError("CUDA model probe produced missing spatial-bias or nonfinite gradients")
    if any(parameter.grad is None for parameter in model.neighborhood_blocks.parameters()):
        raise ValueError("CUDA model probe produced missing neighborhood gradients")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)} | decoder={decoder} | "
          f"neighborhood_layers={len(model.neighborhood_blocks)} forward/backward: passed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--resume-from", help="Read experiment settings from the checkpoint's run.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--test-designs", type=int, nargs="+", default=list(DEFAULT_TEST_DESIGNS))
    parser.add_argument("--val-designs", type=int, nargs="+", default=list(DEFAULT_VAL_DESIGNS))
    parser.add_argument("--decoder", choices=sorted(DECODER_BLOCKS), default=MODEL_DEFAULTS["decoder"])
    for key in EXPERIMENT_MODEL_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(MODEL_DEFAULTS[key]),
                            default=MODEL_DEFAULTS[key])
    parser.add_argument("--design-difference-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--allow-clone-leak", action="store_true",
                        help="Permit a holdout that leaves near-clones of a held-out "
                             "design in training. Scores from such a run measure "
                             "retrieval, not geometry.")
    args = parser.parse_args(argv)
    if args.resume_from:
        saved = parse_training_args(["--resume-from", args.resume_from])
        if saved.data_format != "euroncap1704" or saved.num_samples != NUM_RUNS or saved.samples_per_design != SAMPLES_PER_DESIGN:
            parser.error("The 1704 resume launcher requires a full euroncap1704 run")
        for key in (*EXPERIMENT_MODEL_KEYS, "decoder", "test_designs", "val_designs",
                    "batch_size", "design_difference_weight"):
            setattr(args, key, getattr(saved, key))
        if args.data_root is None:
            args.data_root = str(Path(saved.inp_dir).parent)
    args.data_root = args.data_root or DEFAULT_DATA_ROOT
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_runtime(args.require_cuda)
        if not np.isfinite(args.design_difference_weight) or args.design_difference_weight < 0:
            raise ValueError("design_difference_weight must be finite and nonnegative")
        if args.design_difference_weight and (args.batch_size < 2 or args.batch_size % 2):
            raise ValueError("Design sensitivity requires an even batch_size >= 2")
        splits = validate_splits(args.test_designs, args.val_designs)
        if args.allow_clone_leak:
            print("Cluster check: SKIPPED (--allow-clone-leak); scores may measure retrieval")
        else:
            held_clusters, val_leaks = validate_cluster_holdout(
                args.test_designs, args.val_designs
            )
            for leak in val_leaks:
                print(f"Cluster check: WARNING - {leak}. The test split is unaffected; "
                      f"checkpoint selection is slightly optimistic.")
            print(f"Cluster check: passed | whole clusters held out: "
                  f"{held_clusters or 'none (no held-out design shares a cluster)'}")
        root, impact_xy = validate_dataset(args.data_root)
        if args.impactor_nodes:
            validate_impactor_boundary(root, args.impactor_nodes)
        elif args.neighborhood_layers:
            print("WARNING: --impactor-nodes 0 keeps the moving headform in the local graph, "
                  "so the neighbor graph is rebuilt for every run instead of once per design")
        data, source_count, cutoff_count = validate_first_run(root, impact_xy)
        if args.require_cuda:
            validate_cuda_model(data, decoder=args.decoder, **{
                key: getattr(args, key) for key in EXPERIMENT_MODEL_KEYS
            })
        print(f"Dataset: {root} | {NUM_RUNS} mesh/history pairs and impact XY rows")
        for name, split in splits.items():
            print(f"{name}: designs={split['design_ids']} | runs={len(split['run_numbers'])}")
        print(
            f"First run: {len(data['mesh_geometries'][0])} mesh nodes | "
            f"history {source_count} source -> {cutoff_count} after cutoff -> "
            f"{len(data['time_arrays'][0])} sampled | Config.time_subsample_stride="
            f"{Config.time_subsample_stride}"
        )
        print("Preflight passed.")
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
