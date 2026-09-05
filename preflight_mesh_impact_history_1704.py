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
    MeshImpactHistoryNet,
    resolve_splits,
    validate_data,
)


NUM_RUNS = 1704
SAMPLES_PER_DESIGN = 142
NUM_DESIGNS = NUM_RUNS // SAMPLES_PER_DESIGN
DEFAULT_DATA_ROOT = "Data/HoodImpact_1704_EuroNCAP"


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


def validate_cuda_model(data):
    """Exercise the new model's attention and learned spatial-bias gradients."""
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
        temporal_layers=1, dropout=0.0,
    ).to("cuda")
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
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)} | new-model forward/backward: passed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--test-designs", type=int, nargs="+", default=[2])
    parser.add_argument("--val-designs", type=int, nargs="+", default=[10])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_runtime(args.require_cuda)
        splits = validate_splits(args.test_designs, args.val_designs)
        root, impact_xy = validate_dataset(args.data_root)
        data, source_count, cutoff_count = validate_first_run(root, impact_xy)
        if args.require_cuda:
            validate_cuda_model(data)
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
