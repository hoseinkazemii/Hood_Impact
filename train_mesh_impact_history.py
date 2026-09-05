"""Train a new irregular-mesh attention model and predict acceleration histories.

Only data processing and optimization utilities are shared with previous work.
The network is defined independently in :mod:`mesh_impact_history`.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wandb

from mesh_impact_history import MeshImpactHistoryNet
from utils.utils import (
    Config,
    DataPreprocessor,
    Evaluator,
    Trainer,
    create_data_loaders,
    run_to_design_id,
    set_seed,
)


MODEL_DEFAULTS = {
    "width": 128,
    "num_heads": 4,
    "num_latents": 32,
    "latent_layers": 3,
    "temporal_layers": 2,
    "dropout": 0.1,
}
CHECKPOINT_NAME = "hood_impact_best_model.pt"  # Filename used by shared Trainer.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-format", choices=["legacy", "industrylike", "euroncap1704"])
    for name in ("inp-dir", "impact-coords-path", "mesh-geometry-dir", "doe-path", "acceleration-dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--samples-per-design", type=int)
    parser.add_argument("--test-designs", type=int, nargs="+", default=[2], help="Zero-based design IDs.")
    parser.add_argument("--val-designs", type=int, nargs="+", default=[10], help="Zero-based design IDs.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-time", type=float, help="Optional cutoff before the configured time stride is applied.")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu, cuda, or cuda:0.")
    for key, default in MODEL_DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--output-dir", help="New run directory; defaults to runs/mesh_impact_history/<timestamp>.")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="hood-impact-mesh-attention")
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def build_config(args):
    """Inherit current preprocessing/training defaults, without model presets."""
    overrides = {"prediction_target": "acceleration"}
    mapping = {"epochs": "num_epochs", "lr": "learning_rate"}
    for key in (
        "data_format", "inp_dir", "impact_coords_path", "mesh_geometry_dir", "doe_path",
        "acceleration_dir", "num_samples", "samples_per_design", "epochs", "batch_size",
        "lr", "weight_decay", "seed", "max_train_time",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[mapping.get(key, key)] = value
    if args.device is not None:
        overrides["device"] = torch.device(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    overrides["output_dir"] = str(Path(args.output_dir or Path("runs") / "mesh_impact_history" / stamp).resolve())
    config = Config(**overrides)
    for key in ("num_samples", "samples_per_design", "num_epochs", "batch_size", "time_subsample_stride"):
        value = getattr(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.max_train_time is not None and (not np.isfinite(config.max_train_time) or config.max_train_time <= 0):
        raise ValueError("max_train_time must be finite and positive")
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return config


def validate_data(data):
    """Catch malformed records before fitting scalers or launching training."""
    run_numbers = data["run_numbers"]
    if not run_numbers:
        raise ValueError("No valid acceleration runs loaded. Check dataset paths and preprocessing warnings.")
    if len(set(run_numbers)) != len(run_numbers):
        raise ValueError("Run numbers must be unique")
    for key in ("mesh_geometries", "indentor_positions", "time_arrays", "accelerations"):
        if len(data[key]) != len(run_numbers):
            raise ValueError(f"{key} does not match the number of loaded runs")
    for i, run in enumerate(run_numbers):
        mesh = np.asarray(data["mesh_geometries"][i])
        impact = np.asarray(data["indentor_positions"][i])
        times = np.asarray(data["time_arrays"][i])
        accel = np.asarray(data["accelerations"][i])
        if mesh.ndim != 2 or mesh.shape[1] != 3 or not len(mesh):
            raise ValueError(f"Run {run}: mesh must have shape (N, 3) with at least one node")
        if impact.shape != (2,):
            raise ValueError(f"Run {run}: impact location must have shape (2,)")
        if times.ndim != 1 or len(times) < 1 or accel.shape != times.shape:
            raise ValueError(f"Run {run}: sampled times and accelerations must be matching nonempty vectors")
        if not all(np.isfinite(array).all() for array in (mesh, impact, times, accel)):
            raise ValueError(f"Run {run}: nonfinite mesh, impact, time, or acceleration values")
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Run {run}: sampled times must be strictly increasing")


def resolve_splits(data, config, test_designs, val_designs):
    """Validate design separation using runs that actually loaded successfully."""
    test_ids, val_ids = set(test_designs), set(val_designs)
    if not test_ids or not val_ids:
        raise ValueError("Validation and test design lists must both be nonempty")
    if test_ids & val_ids:
        raise ValueError(f"Validation and test design IDs overlap: {sorted(test_ids & val_ids)}")
    design_ids = {run_to_design_id(run, config.samples_per_design) for run in data["run_numbers"]}
    missing = (test_ids | val_ids) - design_ids
    if missing:
        raise ValueError(f"Requested held-out designs have no loaded runs: {sorted(missing)}; loaded designs: {sorted(design_ids)}")
    split = {
        "train": {"design_ids": sorted(design_ids - test_ids - val_ids), "run_numbers": []},
        "validation": {"design_ids": sorted(val_ids), "run_numbers": []},
        "test": {"design_ids": sorted(test_ids), "run_numbers": []},
    }
    for run in data["run_numbers"]:
        design = run_to_design_id(run, config.samples_per_design)
        name = "test" if design in test_ids else "validation" if design in val_ids else "train"
        split[name]["run_numbers"].append(int(run))
    for name, group in split.items():
        if not group["run_numbers"]:
            raise ValueError(f"The {name} split is empty; choose held-out designs leaving training data")
    return split


def resolved_config(config, args, model_kwargs, prediction_grid):
    data_keys = ["data_format", "num_samples", "samples_per_design"]
    path_keys = (
        ["mesh_geometry_dir", "doe_path", "acceleration_dir"] if config.data_format == "legacy"
        else ["inp_dir", "impact_coords_path", "acceleration_dir"]
    )
    data_config = {key: getattr(config, key) for key in data_keys}
    data_config.update({key: str(Path(getattr(config, key)).resolve()) for key in path_keys})
    return {
        "format_version": 1,
        "architecture": {"name": "MeshImpactHistoryNet", "kwargs": model_kwargs},
        "data": data_config,
        "preprocessing": {
            "prediction_target": "acceleration",
            "time_subsample_stride": config.time_subsample_stride,
            "max_train_time": config.max_train_time,
            "mesh_columns": ["X1", "X2", "X3"],
            "impact_columns": ["X1", "X2"],
            "acceleration_units": "g",
            "scalers_fit_on": "train",
        },
        "training": {
            **{key: getattr(config, key) for key in ("num_epochs", "batch_size", "learning_rate", "weight_decay", "seed")},
            "device": str(config.device),
            "initialization": "from_scratch",
            "optimizer": "AdamW",
            "loss": "normalized_acceleration_mse",
            "scheduler": "CosineAnnealingLR",
            "gradient_clip_norm": 1.0,
        },
        "prediction_grid": prediction_grid,
        "wandb": {"mode": args.wandb_mode, "project": args.wandb_project, "run_name": args.run_name},
        "output_dir": config.output_dir,
    }


def write_json(path, payload):
    # Strict JSON: undefined R2 for constant targets is represented by null.
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    Path(path).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_predictions(path, dataset, predictions, targets):
    """Flatten test histories in the same sequential order as the test loader."""
    counts = [len(times) for times in dataset.time_arrays]
    if sum(counts) != len(predictions) or len(targets) != len(predictions):
        raise ValueError("Test predictions do not match the sampled histories")
    pd.DataFrame({
        "run_number": np.repeat(dataset.run_numbers, counts),
        "time": np.concatenate(dataset.time_arrays),
        "acceleration_true_g": targets,
        "acceleration_pred_g": predictions,
    }).to_csv(path, index=False)


class HistoryPredictor:
    """Inference from raw geometry and impact coordinates, with saved output times.

    ``predict(mesh_xyz, impact_xy)`` returns acceleration in g at
    ``predictor.time_points``. Optional time points must already be subsampled.
    """

    def __init__(self, model, preprocessor, time_points, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        self.time_points = np.asarray(time_points, dtype=np.float32).copy()

    @classmethod
    def from_run(cls, output_dir, device="cpu"):
        run_dir = Path(output_dir)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved.get("format_version") != 1 or saved.get("architecture", {}).get("name") != "MeshImpactHistoryNet":
            raise ValueError("This directory is not a supported MeshImpactHistoryNet run")
        config = SimpleNamespace(**saved["preprocessing"], device=torch.device(device))
        preprocessor = DataPreprocessor(config)
        preprocessor.load_scalers(str(run_dir / "scalers.joblib"))
        model = MeshImpactHistoryNet(**saved["architecture"]["kwargs"])
        checkpoint = torch.load(run_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        times = np.load(run_dir / "prediction_times.npy", allow_pickle=False)
        return cls(model, preprocessor, times, device=device)

    @torch.no_grad()
    def predict(self, mesh_xyz, impact_xy, sampled_time_points=None):
        mesh = np.asarray(mesh_xyz, dtype=np.float32)
        impact = np.asarray(impact_xy, dtype=np.float32)
        times = self.time_points if sampled_time_points is None else np.asarray(sampled_time_points, dtype=np.float32)
        if mesh.ndim != 2 or mesh.shape[1] != 3 or not len(mesh):
            raise ValueError("mesh_xyz must have shape (N, 3), with at least one node")
        if impact.shape != (2,):
            raise ValueError("impact_xy must have shape (2,), in the same physical coordinate frame as the mesh")
        if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError("sampled_time_points must be a nonempty, strictly increasing vector")
        if not all(np.isfinite(array).all() for array in (mesh, impact, times)):
            raise ValueError("Mesh, impact coordinates, and sampled times must all be finite")

        def tensor(array):
            return torch.as_tensor(array, dtype=torch.float32, device=self.device)

        # Time slicing occurs only while loading training data, never here.
        normalized_mesh = tensor(self.preprocessor.transform_mesh(mesh))
        normalized_impact = tensor(self.preprocessor.transform_indentor(impact)).unsqueeze(0)
        normalized_times = tensor(self.preprocessor.transform_time(times))
        prediction = self.model(
            normalized_mesh,
            torch.zeros(len(mesh), dtype=torch.long, device=self.device),
            normalized_impact,
            normalized_times,
            torch.zeros(len(times), dtype=torch.long, device=self.device),
            batch_size=1,
        )
        return self.preprocessor.inverse_transform_acceleration(prediction.cpu().numpy())


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    output_dir = Path(config.output_dir)
    # A new run cannot accidentally evaluate a leftover checkpoint from old work.
    if any((output_dir / name).exists() for name in ("config.json", CHECKPOINT_NAME, "scalers.joblib")):
        raise FileExistsError(f"Run artifacts already exist in {output_dir}; choose a new --output-dir")
    set_seed(config.seed)
    logger = logging.getLogger("mesh_impact_history.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(output_dir / "training.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    run = None
    try:
        model_kwargs = {key: getattr(args, key) for key in MODEL_DEFAULTS}
        model = MeshImpactHistoryNet(**model_kwargs)
        logger.info("New MeshImpactHistoryNet from scratch: %s", model_kwargs)
        logger.info("Output: %s | device: %s | configured time stride: %s", output_dir, config.device, config.time_subsample_stride)
        preprocessor = DataPreprocessor(config)
        data = preprocessor.load_all_data()
        validate_data(data)
        if len(data["run_numbers"]) != config.num_samples:
            logger.warning("Loaded %s of %s runs; see preprocessing warnings for missing files.", len(data["run_numbers"]), config.num_samples)
        splits = resolve_splits(data, config, args.test_designs, args.val_designs)
        loaders = create_data_loaders(
            data, preprocessor, config,
            test_design_ids=splits["test"]["design_ids"],
            val_design_id=splits["validation"]["design_ids"],
        )
        train_loader, val_loader, test_loader, test_dataset = loaders
        train_dataset = train_loader.dataset
        preprocessor.fit_scalers(
            mesh_geometries=train_dataset.mesh_geometries,
            indentor_positions=train_dataset.indentor_positions,
            time_arrays=train_dataset.time_arrays,
            accelerations=train_dataset.accelerations,
        )
        model.set_coordinate_scalers(
            preprocessor.mesh_scaler.mean_, preprocessor.mesh_scaler.scale_,
            preprocessor.indentor_scaler.mean_, preprocessor.indentor_scaler.scale_,
        )
        first_times = np.asarray(train_dataset.time_arrays[0], dtype=np.float32)
        matching_grid = all(np.array_equal(first_times, times) for times in data["time_arrays"])
        prediction_grid = {
            "file": "prediction_times.npy",
            "reference_run_number": int(train_dataset.run_numbers[0]),
            "num_time_points": len(first_times),
            "same_grid_for_all_loaded_runs": matching_grid,
            "already_subsampled": True,
            "time_units": "source Time column (seconds for the supplied datasets)",
        }
        if not matching_grid:
            logger.warning("Loaded runs have different sampled time grids. Default inference uses training run %s; pass sampled_time_points for another already-subsampled grid.", train_dataset.run_numbers[0])
        saved_config = resolved_config(config, args, model_kwargs, prediction_grid)
        write_json(output_dir / "config.json", saved_config)
        write_json(output_dir / "splits.json", splits)
        preprocessor.save_scalers(str(output_dir / "scalers.joblib"))
        np.save(output_dir / "prediction_times.npy", first_times, allow_pickle=False)
        logger.info("Saved train-only scalers and sampled prediction grid (%s points); parameters: %s", len(first_times), sum(parameter.numel() for parameter in model.parameters()))
        run = wandb.init(
            project=args.wandb_project, name=args.run_name or output_dir.name,
            config=saved_config, dir=str(output_dir), mode=args.wandb_mode,
        )
        trainer = Trainer(model, train_loader, val_loader, config, preprocessor)
        history = trainer.train()
        write_json(output_dir / "training_history.json", history)
        pd.DataFrame({
            "epoch": np.arange(1, len(history["train_losses"]) + 1),
            "train_mse_normalized": history["train_losses"],
            "validation_mse_normalized": history["val_losses"],
        }).to_csv(output_dir / "training_history.csv", index=False)
        checkpoint_path = output_dir / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise RuntimeError("Training produced no finite best validation checkpoint; inspect losses and input data")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        evaluator = Evaluator(model, config, preprocessor)
        metrics, predictions, targets = evaluator.evaluate(test_loader)
        write_json(output_dir / "metrics.json", metrics)
        export_predictions(output_dir / "test_acceleration_histories.csv", test_dataset, predictions, targets)
        wandb.log({f"test/{key}": value for key, value in metrics.items() if np.isfinite(value)})
        logger.info("Best-checkpoint test metrics (acceleration in g): %s", metrics)
        logger.info("Finished. Run artifacts: %s", output_dir)
        return metrics
    finally:
        if run is not None:
            run.finish()
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
