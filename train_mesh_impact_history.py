"""Train the mesh-change attention model (DesignChangeHistoryNet) on the local-edit dataset.

Inputs are the hood geometry (the design-6 inner panel with each design's
edited nodes) and the impact location; the target is the acceleration history.
No simulated response is ever a network input.

Design 6, the unedited base hood, is by default not trained on: its true curves
are read only after training, as the reference the edit effects are measured
from (local_edit_metrics.json). With ``--train-on-base-design true`` its 142 runs
are also ordinary training samples (its geometry in, its curve as the target,
no change nodes); it is never held out, and the edit-effect metrics and their
baselines still score the edited designs only.

    python train_mesh_impact_history.py --test-designs 54 --val-designs 55 --device cuda

run_mesh_change_local_edits_delta.sbatch is the launcher on NCSA Delta.
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

from local_edit_metrics import local_edit_log_values, local_edit_report, time_window_metrics, write_local_edit_report
from local_edits_data import (
    BASE_DESIGN, BASE_DESIGN_ID, base_reference_curves, design_name, load_design_changes, load_manifest,
    sample_design, sample_location,
)
from mesh_design_change import (
    CHANGE_FEATURES, TABLE_FILE, DesignChangeHistoryNet, DesignChangeTable, change_displacement_rms,
    register_design_changes, saved_design_change_kwargs,
)
from mesh_neighborhood import default_neighbor_cache_directory, prepare_model_neighbor_graphs
from utils.utils import (
    CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, Config, DataPreprocessor, Evaluator, Trainer, create_data_loaders,
    set_seed,
)

ARCHITECTURE = "DesignChangeHistoryNet"
# config.json format. 2 (since 2026-11-27) lists exactly the current model options;
# runs of version 1 still resume and reload (mesh_impact_history.saved_model_kwargs).
CONFIG_FORMAT_VERSION = 2
# The configuration of run_mesh_change_local_edits_delta.sbatch.
MODEL_DEFAULTS = {
    "width": 128,
    "num_heads": 4,
    "num_latents": 256,
    # Pooling queries with a radial bias around the impact point. 0 (since
    # 2026-11-22): every query reads the whole hood. Earlier runs used half.
    "local_latents": 0,
    "latent_layers": 3,
    "temporal_layers": 2,
    "dropout": 0.1,
    "fourier_num_frequencies": 6,
    "fourier_min_frequency_hz": 20.0,
    "fourier_max_frequency_hz": 640.0,
    "neighborhood_layers": 2,
    "neighborhood_k": 256,
    # 0 = kNN offsets in units of the mesh's median distance to the k-th
    # neighbour (since 2026-11-22; earlier runs used a fixed 20 mm).
    "neighborhood_scale_mm": 0.0,
    "neighborhood_chunk_size": 1024,
    # Change tokens: reference position + displacement ("edit", since
    # 2026-11-24) or the position after the edit ("xyz").
    "design_change_features": "edit",
    # Learned per-head fall-off of the change readouts' attention with a change
    # node's distance from the impact point (mesh_design_change.py, 2026-11-28).
    "change_locality": False,
}
# Test N54 and validation N55: the strongest, most repeatable edit effects.
DEFAULT_TEST_DESIGNS = (54,)
DEFAULT_VAL_DESIGNS = (55,)
# time_windows.json splits test errors into this many windows of the time grid.
TIME_WINDOWS = 5
# Job 3317059 died when api.wandb.ai answered slower than W&B's 90 s default.
# Online starts get a longer timeout and retries, then fail before data loading.
WANDB_INIT_ATTEMPTS = 3
WANDB_INIT_TIMEOUT_SECONDS = 300
WANDB_RETRY_DELAY_SECONDS = 60


def parse_bool(text):
    """'true'/'false' (also 1/0, yes/no) for switches given as values, as the launcher passes them."""
    value = str(text).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false; got {text!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp-dir", help="Folder with a design-6 deck (default Data/HoodImpact_LocalEdits/inp_files).")
    parser.add_argument("--impact-coords-path", help="The run manifest (default .../manifest_local_edits.csv).")
    parser.add_argument("--acceleration-dir", help="The curves (default .../output_history_acc).")
    parser.add_argument("--test-designs", type=int, nargs="+", default=list(DEFAULT_TEST_DESIGNS),
                        help="Edited design numbers: 54 is N54.")
    parser.add_argument("--val-designs", type=int, nargs="+", default=list(DEFAULT_VAL_DESIGNS),
                        help="Edited design numbers: 55 is N55.")
    parser.add_argument("--epochs", type=int, help="Total epoch target, including completed epochs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-time", type=float, help="Optional cutoff before the configured time stride is applied.")
    parser.add_argument("--time-subsample-stride", type=int,
                        help="Keep every Nth source time/target sample (16 by default: 63 of 1000 points).")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu, cuda, or cuda:0.")
    parser.add_argument("--train-on-base-design", type=parse_bool, default=False, metavar="{true,false}",
                        help="Also train on design 6, the unedited base hood (its 142 runs, no change nodes). "
                             "It is never held out or scored as an edit.")
    for key, default in MODEL_DEFAULTS.items():
        option = f"--{key.replace('_', '-')}"
        if key == "design_change_features":
            parser.add_argument(option, choices=CHANGE_FEATURES, default=default)
        elif isinstance(default, bool):
            parser.add_argument(option, type=parse_bool, default=default, metavar="{true,false}")
        else:
            parser.add_argument(option, type=type(default), default=default)
    parser.add_argument("--output-dir", help="New run directory; defaults to runs/mesh_impact_history/<timestamp>.")
    parser.add_argument("--neighbor-cache-dir", help="Shared persistent kNN directory; defaults to neighbor_graphs beside inp_files.")
    parser.add_argument("--resume-from", help="Prior run directory (latest, else best checkpoint) or checkpoint file; writes a new run.")
    parser.add_argument(
        "--wandb-mode", choices=["disabled", "offline", "online"],
        default=os.environ.get("WANDB_MODE") or "online",
        help="Tracking mode; defaults to WANDB_MODE, or online when unset.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-mesh-attention")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None,
                        help="W&B team/account; defaults to WANDB_ENTITY or your W&B default team.")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    # argparse does not check a default (here from WANDB_MODE) against choices.
    if args.wandb_mode not in ("disabled", "offline", "online"):
        parser.error("WANDB_MODE must be disabled, offline, or online; override it with --wandb-mode")
    if args.resume_from:
        restore_resume_arguments(args, parser, sys.argv[1:] if argv is None else argv)
    return args


def restore_resume_arguments(args, parser, argv):
    """Restore saved experimental settings; allow data paths to be relocated."""
    source = Path(args.resume_from).expanduser().resolve()
    if source.is_dir():
        latest, best = source / LAST_CHECKPOINT_NAME, source / CHECKPOINT_NAME
        source = latest if latest.is_file() else best
        if latest.is_file() and best.is_file():
            # A timeout may fall between the atomic best and latest writes.
            # If best is newer, that completed epoch is the recovery point.
            latest_count = len(torch.load(latest, map_location="cpu", weights_only=True)["train_losses"])
            best_count = len(torch.load(best, map_location="cpu", weights_only=True)["train_losses"])
            if best_count > latest_count:
                source = best
    source_dir = source.parent
    for path in (source, *(source_dir / name for name in
                          ("config.json", "splits.json", "scalers.joblib", "prediction_times.npy", TABLE_FILE))):
        if not path.is_file():
            parser.error(f"Missing resume artifact: {path}")
    saved = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    if (saved.get("format_version") not in (1, 2) or saved.get("architecture", {}).get("name") != ARCHITECTURE
            or saved["data"].get("data_format") != "localedits"):
        parser.error(f"Resume requires a local-edit {ARCHITECTURE} run")
    if saved["training"].get("loss", "normalized_acceleration_mse") != "normalized_acceleration_mse":
        parser.error("This checkpoint used a different training objective; start a fresh acceleration-MSE run")
    splits = json.loads((source_dir / "splits.json").read_text(encoding="utf-8"))
    try:
        model_kwargs = saved_design_change_kwargs(saved["architecture"]["kwargs"],
                                                  format_version=saved["format_version"])
    except ValueError as error:
        parser.error(str(error))
    settings = {**MODEL_DEFAULTS, **model_kwargs,
                **{key: saved["training"][key] for key in ("batch_size", "weight_decay", "seed")},
                "epochs": saved["training"]["num_epochs"],
                "lr": saved["training"]["learning_rate"],
                "max_train_time": saved["preprocessing"]["max_train_time"],
                "time_subsample_stride": saved["preprocessing"]["time_subsample_stride"],
                # Runs saved before 2026-11-28 never trained on design 6.
                "train_on_base_design": bool(saved["data"].get("local_edits", {}).get("train_on_base_design", False)),
                "test_designs": splits["test"]["design_ids"],
                "val_designs": splits["validation"]["design_ids"]}
    explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for key, value in settings.items():
        option = "--" + key.replace("_", "-")
        if option in explicit and getattr(args, key) != value:
            parser.error(f"Resume must retain saved {option}={value!r}; omit the conflicting override")
        setattr(args, key, value)
    for key in ("inp_dir", "impact_coords_path", "acceleration_dir"):
        if getattr(args, key) is None:
            setattr(args, key, saved["data"][key])
    args.resume_from = str(source)
    args.resume_config, args.resume_splits = saved, splits


def build_config(args):
    """Config from the command line, over utils.utils.Config defaults."""
    overrides = {}
    mapping = {"epochs": "num_epochs", "lr": "learning_rate"}
    for key in ("inp_dir", "impact_coords_path", "acceleration_dir", "epochs", "batch_size", "lr",
                "weight_decay", "seed", "max_train_time", "time_subsample_stride"):
        value = getattr(args, key)
        if value is not None:
            overrides[mapping.get(key, key)] = value
    if args.device is not None:
        overrides["device"] = torch.device(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    overrides["output_dir"] = str(Path(args.output_dir or Path("runs") / "mesh_impact_history" / stamp).resolve())
    config = Config(**overrides)
    for key in ("num_epochs", "batch_size", "time_subsample_stride"):
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
        raise ValueError("No valid acceleration runs loaded. Check dataset paths.")
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


def resolve_splits(data, test_designs, val_designs):
    """Design-wise train/validation/test split of the runs that actually loaded."""
    test_ids, val_ids = set(test_designs), set(val_designs)
    if not test_ids or not val_ids:
        raise ValueError("Validation and test design lists must both be nonempty")
    if test_ids & val_ids:
        raise ValueError(f"Validation and test design IDs overlap: {sorted(test_ids & val_ids)}")
    if BASE_DESIGN_ID in test_ids | val_ids:
        raise ValueError("Design 6 (design ID 0) is the base hood the edits are measured from; it cannot be held out")
    design_ids = {sample_design(run) for run in data["run_numbers"]}
    missing = (test_ids | val_ids) - design_ids
    if missing:
        raise ValueError(f"Requested held-out designs have no loaded runs: {sorted(missing)}; loaded designs: {sorted(design_ids)}")
    split = {
        "train": {"design_ids": sorted(design_ids - test_ids - val_ids), "run_numbers": []},
        "validation": {"design_ids": sorted(val_ids), "run_numbers": []},
        "test": {"design_ids": sorted(test_ids), "run_numbers": []},
    }
    for run in data["run_numbers"]:
        design = sample_design(run)
        name = "test" if design in test_ids else "validation" if design in val_ids else "train"
        split[name]["run_numbers"].append(int(run))
    for name, group in split.items():
        if not group["run_numbers"]:
            raise ValueError(f"The {name} split is empty; choose held-out designs leaving training data")
    return split


def resolved_config(config, args, model_kwargs, prediction_grid):
    data_config = {"data_format": "localedits",
                   **{key: str(Path(getattr(config, key)).resolve())
                      for key in ("inp_dir", "impact_coords_path", "acceleration_dir")}}
    base_use = (
        "D6 is also a training design: its runs (no change nodes) are ordinary training samples and enter "
        "the scalers; it is never held out, and local_edit_metrics.json scores only the edited designs, "
        "against D6's true curves (never a network input)" if args.train_on_base_design else
        "D6, excluded from training, validation, test and scalers; its geometry is the mesh every design "
        "is built on, and its true curves are read only after training, as the reference for "
        "local_edit_metrics.json (never a network input)")
    data_config["local_edits"] = {
        "excluded_impact_roles": ["far"],
        "sample_number": "142 * n + location for design Nn (design ID n)",
        "train_on_base_design": bool(args.train_on_base_design),
        "base_design": base_use,
        "mesh": "inner panel of one design-6 deck plus dz from design_changes/<design>_changed_nodes.csv",
        "samples_file": "local_edit_samples.csv",
    }
    return {
        "format_version": CONFIG_FORMAT_VERSION,
        "architecture": {"name": ARCHITECTURE, "kwargs": model_kwargs},
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
            "initialization": "checkpoint" if args.resume_from else "from_scratch",
            "resume_from": args.resume_from,
            "optimizer": "AdamW",
            "loss": "normalized_acceleration_mse",
            "batch_sampling": "shuffle",
            "scheduler": "CosineAnnealingLR",
            "gradient_clip_norm": 1.0,
        },
        "prediction_grid": prediction_grid,
        "wandb": {
            "mode": args.wandb_mode, "project": args.wandb_project,
            "entity": args.wandb_entity, "run_name": args.run_name or Path(config.output_dir).name,
        },
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


def initialize_wandb(args, initial_config, output_dir, logger):
    """Start tracking before dataset loading and save the resolved destination."""
    run = None
    try:
        logger.info("Initializing W&B: mode=%s | project=%s | entity=%s",
                    args.wandb_mode, args.wandb_project, args.wandb_entity or "default team")
        attempts = WANDB_INIT_ATTEMPTS if args.wandb_mode == "online" else 1
        for attempt in range(1, attempts + 1):
            try:
                run = wandb.init(
                    project=args.wandb_project, entity=args.wandb_entity,
                    name=args.run_name or output_dir.name, config=initial_config,
                    dir=str(output_dir), mode=args.wandb_mode,
                    settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT_SECONDS),
                )
                break
            except Exception as exc:
                if attempt == attempts:
                    raise
                logger.warning("W&B initialization attempt %s/%s failed (%s); retrying in %s s",
                               attempt, attempts, exc, WANDB_RETRY_DELAY_SECONDS)
                time.sleep(WANDB_RETRY_DELAY_SECONDS)
        run.define_metric("epoch")
        for metric in ("train/loss", "val/loss", "val/mae_g", "lr"):
            run.define_metric(metric, step_metric="epoch")
        actual_mode = run.settings.mode
        metadata = {
            "id": run.id, "name": run.name, "project": run.project,
            "entity": run.entity, "mode": actual_mode,
            "url": run.url if actual_mode == "online" else None,
        }
        write_json(output_dir / "wandb_run.json", metadata)
        logger.info("W&B run: mode=%s | project=%s | entity=%s | id=%s | URL=%s",
                    actual_mode, run.project, run.entity, run.id, metadata["url"] or "none (local tracking)")
        if actual_mode != args.wandb_mode:
            raise RuntimeError(f"Requested W&B mode {args.wandb_mode!r}, but initialized {actual_mode!r}")
        if actual_mode == "offline":
            logger.info("W&B is offline: history is saved locally and will appear online only after wandb sync.")
        elif actual_mode == "disabled":
            logger.info("W&B is disabled: this run will not appear on the W&B website.")
        return run
    except Exception as exc:
        message = (
            f"W&B initialization failed (mode={args.wandb_mode}, project={args.wandb_project}, "
            f"entity={args.wandb_entity or 'default team'}). "
            "For online logging, run `python -m wandb login` in the training environment "
            "(on the cluster login node before submission), and check the selected team/project "
            "and network connection. For intentional local tracking use --wandb-mode offline; "
            "to turn tracking off use --wandb-mode disabled."
        )
        logger.error("%s Cause: %s", message, exc)
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError(message) from exc


def export_predictions(path, dataset, predictions, targets):
    """Flatten histories in the dataset's sequential (unshuffled loader) order."""
    counts = [len(times) for times in dataset.time_arrays]
    if sum(counts) != len(predictions) or len(targets) != len(predictions):
        raise ValueError("Predictions do not match the sampled histories")
    pd.DataFrame({
        "run_number": np.repeat(dataset.run_numbers, counts),
        "time": np.concatenate(dataset.time_arrays),
        "acceleration_true_g": targets,
        "acceleration_pred_g": predictions,
    }).to_csv(path, index=False)


def export_training_history_plot(history, output_dir):
    """Save training_history.png: train/validation normalized MSE, marking the best epoch."""
    train = np.asarray(history["train_losses"], dtype=np.float64)
    validation = np.asarray(history["val_losses"], dtype=np.float64)
    if train.ndim != 1 or not train.size or train.shape != validation.shape:
        raise ValueError("Train and validation losses must be nonempty vectors of equal length")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Train and validation losses must be finite")
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(validation))
    path = Path(output_dir) / "training_history.png"
    fig, ax = plt.subplots(figsize=(9, 5.3))
    try:
        ax.plot(epochs, train, color="#1965b0", linewidth=1.8, label="Training")
        ax.plot(epochs, validation, color="#dc7014", linewidth=1.8, label="Validation")
        ax.axvline(best + 1, color="#686868", linestyle="--", linewidth=1,
                   label=f"Best validation: epoch {best + 1} ({validation[best]:.5f})")
        ax.scatter([best + 1], [validation[best]], color="#dc7014", s=28, zorder=3)
        ax.set(xlabel="Epoch", ylabel="Normalized acceleration MSE", title=f"{ARCHITECTURE} — training history")
        ax.set_xlim(1 if len(train) > 1 else 0.5, len(train) if len(train) > 1 else 1.5)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


class HistoryPredictor:
    """Inference from raw geometry and impact coordinates, with saved output times.

    ``predict(mesh_xyz, impact_xy)`` returns acceleration in g at
    ``predictor.time_points``. Optional time points must already be subsampled.
    A hood that was not part of the training run must first be registered with
    its change nodes: ``predictor.model.design_changes`` (DesignChangeTable).
    """

    def __init__(self, model, preprocessor, time_points, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        self.time_points = np.asarray(time_points, dtype=np.float32).copy()

    @classmethod
    def from_run(cls, output_dir, device="cpu", neighbor_cache_dir=None):
        run_dir = Path(output_dir)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved.get("format_version") not in (1, 2) or saved.get("architecture", {}).get("name") != ARCHITECTURE:
            raise ValueError(f"This directory is not a saved {ARCHITECTURE} run")
        preprocessor = DataPreprocessor(SimpleNamespace(**saved["preprocessing"]))
        preprocessor.load_scalers(str(run_dir / "scalers.joblib"))
        model = DesignChangeHistoryNet(**saved_design_change_kwargs(saved["architecture"]["kwargs"],
                                                                    format_version=saved["format_version"]))
        checkpoint = torch.load(run_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Only hoods saved with the run are known; DesignChangeTable.register adds others.
        model.design_changes = DesignChangeTable.load(run_dir / TABLE_FILE)
        if model.neighborhood_blocks:
            stored = (saved.get("neighbor_graph_cache") or {}).get("directory")
            directory = neighbor_cache_dir or (stored if stored and Path(stored).is_dir() else None)
            model.neighbor_cache.directory = (Path(directory).expanduser() if directory
                                              else default_neighbor_cache_directory())
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
        if self.model.neighborhood_blocks:
            self.model.neighbor_cache.register_geometry(
                mesh, normalized_mesh * self.model.mesh_scale, self.model.neighborhood_k,
            )
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
    if any((output_dir / name).exists() for name in ("config.json", CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, "scalers.joblib")):
        raise FileExistsError(f"Run artifacts already exist in {output_dir}; choose a new --output-dir")
    source_dir = Path(args.resume_from).parent if args.resume_from else None
    if source_dir is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        completed = len(checkpoint["train_losses"])
        if not 0 < completed < config.num_epochs:
            raise ValueError(f"Checkpoint has {completed} completed epochs; target is {config.num_epochs}, so no training remains")
        if len(checkpoint["val_losses"]) != completed or checkpoint.get("completed_epochs", completed) != completed:
            raise ValueError("Checkpoint epoch count and loss histories disagree")
        if checkpoint["scheduler_state_dict"]["T_max"] != config.num_epochs:
            raise ValueError("Checkpoint scheduler does not match the saved total epoch target")
        # Seed best-model evaluation even if no later epoch improves on it.
        best_source = source_dir / CHECKPOINT_NAME
        best = torch.load(best_source, map_location="cpu", weights_only=True)
        best_count = len(best["val_losses"])
        if (not 0 < best_count <= completed or best["val_losses"] != checkpoint["val_losses"][:best_count]
                or best["best_val_loss"] != checkpoint["best_val_loss"]):
            raise ValueError("Source best checkpoint is inconsistent with the resume checkpoint; use artifacts from the same completed job")
        shutil.copy2(best_source, output_dir / CHECKPOINT_NAME)
        del checkpoint, best
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
        model = DesignChangeHistoryNet(**model_kwargs)
        logger.info("%s: %s", ARCHITECTURE, model_kwargs)
        if source_dir is not None:
            logger.info("Resume: %s | completed epochs: %s | next epoch: %s/%s",
                        args.resume_from, completed, completed + 1, config.num_epochs)
        logger.info("Output: %s | device: %s | configured time stride: %s", output_dir, config.device, config.time_subsample_stride)
        run = initialize_wandb(
            args, resolved_config(config, args, model_kwargs, prediction_grid=None), output_dir, logger,
        )
        # Every edited design's moved nodes, listed exactly by the generator.
        manifest = load_manifest(config.impact_coords_path)
        edited = sorted(manifest.loc[manifest["design"] != BASE_DESIGN, "design_id"].unique())
        # Design 6 itself, when trained on, is a design with no change nodes.
        design_changes, design_change_report = load_design_changes(
            config.impact_coords_path, config.inp_dir,
            [BASE_DESIGN_ID, *edited] if args.train_on_base_design else edited,
        )
        counts = [len(design_changes[design]["rows"]) for design in edited]
        logger.info("Design-change nodes: %s edited designs | %s-%s moved nodes each | base %s",
                    len(counts), min(counts), max(counts), design_change_report["base_deck"])
        preprocessor = DataPreprocessor(config)
        data = preprocessor.load_all_data(include_base=args.train_on_base_design)
        validate_data(data)
        logger.info("Mesh: inner panel, %s nodes | %s runs loaded", len(data["mesh_geometries"][0]),
                    len(data["run_numbers"]))
        splits = resolve_splits(data, args.test_designs, args.val_designs)
        if source_dir is not None and splits != args.resume_splits:
            raise ValueError("Loaded train/validation/test runs differ from the saved resume split")
        split_of = {run: name for name, group in splits.items() for run in group["run_numbers"]}
        pd.DataFrame({
            "sample": data["run_numbers"], "source_run": data["source_runs"],
            "design": [design_name(sample_design(s)) for s in data["run_numbers"]],
            "location": [sample_location(s) for s in data["run_numbers"]],
            "split": [split_of[s] for s in data["run_numbers"]],
        }).to_csv(output_dir / "local_edit_samples.csv", index=False)
        base_runs = sum(sample_design(run) == BASE_DESIGN_ID for run in splits["train"]["run_numbers"])
        logger.info("Local edits: test %s (%s runs) | validation %s (%s runs) | training %s designs (%s runs) "
                    "| %s", [design_name(d) for d in splits["test"]["design_ids"]],
                    len(splits["test"]["run_numbers"]),
                    [design_name(d) for d in splits["validation"]["design_ids"]],
                    len(splits["validation"]["run_numbers"]), len(splits["train"]["design_ids"]),
                    len(splits["train"]["run_numbers"]),
                    f"design 6 trained on ({base_runs} of the training runs)" if base_runs
                    else "design 6 not trained on")
        if args.train_on_base_design and not base_runs:
            raise ValueError("--train-on-base-design is set, but no design-6 run was loaded")
        train_loader, val_loader, test_loader, test_dataset = create_data_loaders(data, preprocessor, config, splits)
        train_dataset = train_loader.dataset
        if source_dir is not None:
            preprocessor.load_scalers(str(source_dir / "scalers.joblib"))
        else:
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
        model.set_time_scaler(preprocessor.time_scaler.mean_[0], preprocessor.time_scaler.scale_[0])
        design_change_report["registration"] = register_design_changes(
            model, design_changes, data["mesh_geometries"],
            [sample_design(run) for run in data["run_numbers"]], preprocessor.transform_mesh,
        )
        if source_dir is not None and not model.design_changes.same_as(DesignChangeTable.load(source_dir / TABLE_FILE)):
            raise ValueError("The design-change table differs from the resume run's; the decks or scalers changed")
        if model.design_change_features == "edit":
            scale = change_displacement_rms(design_changes, splits["train"]["design_ids"])
            model.set_change_displacement_scale(scale)
            design_change_report["registration"]["edit_displacement_scale_mm"] = scale
            logger.info("Edit tokens: reference position + displacement / %.3f mm (RMS displacement length "
                        "of the training designs' moved nodes)", scale)
        model.design_changes.save(output_dir / TABLE_FILE)
        write_json(output_dir / "design_change_nodes.json", design_change_report)
        logger.info("Design-change table: %s", design_change_report["registration"])
        first_times = np.asarray(train_dataset.time_arrays[0], dtype=np.float32)
        if source_dir is not None and not np.array_equal(
                first_times, np.load(source_dir / "prediction_times.npy", allow_pickle=False)):
            raise ValueError("Loaded prediction time grid differs from the resume run")
        matching_grid = all(np.array_equal(first_times, times) for times in data["time_arrays"])
        nyquist = min((.5 / float(np.diff(times).max()) for times in data["time_arrays"] if len(times) > 1),
                      default=None)
        frequencies = model.time_frequencies_hz.tolist()
        prediction_grid = {
            "file": "prediction_times.npy",
            "reference_run_number": int(train_dataset.run_numbers[0]),
            "num_time_points": len(first_times),
            "same_grid_for_all_loaded_runs": matching_grid,
            "already_subsampled": True,
            "time_units": "source Time column (seconds)",
            "time_feature_count": model.time_feature_count,
            "fourier": {
                "frequencies_hz": frequencies,
                "feature_order": "sin(f0), cos(f0), sin(f1), cos(f1), ...",
                "phase_definition": "2*pi*frequency_hz*(normalized_time*time_scale_seconds+time_mean_seconds)",
                "time_mean_seconds": float(model.time_mean),
                "time_scale_seconds": float(model.time_scale),
                "minimum_sampled_nyquist_hz": nyquist,
            },
        }
        logger.info("Fourier time encoding: %s Hz | %s features | minimum sampled Nyquist: %s Hz",
                    frequencies, model.time_feature_count, nyquist)
        if nyquist is not None and frequencies[-1] >= nyquist:
            logger.warning("Fourier maximum frequency %s Hz reaches/exceeds sampled Nyquist %s Hz; "
                           "use a lower --fourier-max-frequency-hz for this grid", frequencies[-1], nyquist)
        if not matching_grid:
            logger.warning("Loaded runs have different sampled time grids. Default inference uses training run %s; pass sampled_time_points for another already-subsampled grid.", train_dataset.run_numbers[0])
        saved_config = resolved_config(config, args, model_kwargs, prediction_grid)
        saved_config["design_change"] = {key: design_change_report[key]
                                         for key in ("rule", "static_channels", "registration")}
        saved_config["design_change"]["file"] = "design_change_nodes.json"
        if model.design_change_features == "edit":
            saved_config["design_change"]["edit_displacement_scale_mm"] = float(model.change_displacement_scale)
        if source_dir is not None:
            saved_config["training"]["resumed_after_epoch"] = completed
        write_json(output_dir / "config.json", saved_config)
        write_json(output_dir / "splits.json", splits)
        preprocessor.save_scalers(str(output_dir / "scalers.joblib"))
        np.save(output_dir / "prediction_times.npy", first_times, allow_pickle=False)
        logger.info("Saved train-only scalers and sampled prediction grid (%s points); parameters: %s", len(first_times), sum(parameter.numel() for parameter in model.parameters()))
        run.config.update(saved_config, allow_val_change=True)
        trainer = Trainer(model, train_loader, val_loader, config, preprocessor)
        if source_dir is not None:
            restored = trainer.load_checkpoint(args.resume_from)
            if "torch_rng_state" not in restored:
                logger.info("Legacy checkpoint has no random-generator state; stochastic draws restart from seed %s", config.seed)
            run.summary["resumed_after_epoch"] = completed
            run.summary["best_val_loss"] = trainer.best_val_loss
            del restored
        if model.neighborhood_blocks:
            directory = args.neighbor_cache_dir or default_neighbor_cache_directory(config.inp_dir)
            saved_config["neighbor_graph_cache"] = prepare_model_neighbor_graphs(
                model, preprocessor, data["mesh_geometries"], directory,
                progress=lambda count, cache: logger.info(
                    "Neighbor graphs prepared: %s unique | built=%s | loaded from disk=%s",
                    count, cache.builds, cache.disk_hits,
                ),
            )
            write_json(output_dir / "config.json", saved_config)
            run.config.update({"neighbor_graph_cache": saved_config["neighbor_graph_cache"]}, allow_val_change=True)
        history = trainer.train()
        write_json(output_dir / "training_history.json", history)
        pd.DataFrame({
            "epoch": np.arange(1, len(history["train_losses"]) + 1),
            "train_mse_normalized": history["train_losses"],
            "validation_mse_normalized": history["val_losses"],
        }).to_csv(output_dir / "training_history.csv", index=False)
        history_plot = export_training_history_plot(history, output_dir)
        logger.info("Saved training history plot: %s", history_plot)
        run.summary["best_epoch"] = int(np.argmin(history["val_losses"])) + 1
        if args.wandb_mode != "disabled":
            run.log({"training/history": wandb.Image(str(history_plot))})
        checkpoint_path = output_dir / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise RuntimeError("Training produced no finite best validation checkpoint; inspect losses and input data")
        evaluator = Evaluator(model, config, preprocessor)
        # The training loader shuffles; export train histories in dataset order.
        train_eval_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                       collate_fn=train_loader.collate_fn)
        # Design 6's true curves: the reference the edit effects are measured from
        # (with --train-on-base-design they were also training targets; never an input).
        base_curves, base_times = base_reference_curves(preprocessor)

        def evaluate_checkpoint(path, directory, prefix=""):
            """Test metrics, train/validation/test histories, edit-effect report and time windows of one checkpoint."""
            checkpoint = torch.load(path, map_location=config.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            metrics, predictions, targets = evaluator.evaluate(test_loader)
            write_json(directory / "metrics.json", metrics)
            export_predictions(directory / "test_acceleration_histories.csv", test_dataset, predictions, targets)
            _, train_predictions, train_targets = evaluator.evaluate(train_eval_loader)
            export_predictions(directory / "train_acceleration_histories.csv", train_dataset,
                               train_predictions, train_targets)
            val_dataset = val_loader.dataset
            _, val_predictions, val_targets = evaluator.evaluate(val_loader)
            export_predictions(directory / "validation_acceleration_histories.csv", val_dataset,
                               val_predictions, val_targets)
            difference, local_runs, local_curves = local_edit_report({
                "train": (train_dataset, train_predictions, train_targets),
                "validation": (val_dataset, val_predictions, val_targets),
                "test": (test_dataset, predictions, targets),
            }, base_curves, base_times, config.impact_coords_path)
            figures = write_local_edit_report(directory, difference, local_runs, local_curves, base_times)
            values = local_edit_log_values(difference)
            if args.wandb_mode != "disabled":
                run.log({f"{prefix}local_edits/{Path(figure).stem}": wandb.Image(str(figure)) for figure in figures})
            for split in ("train", "validation", "test"):
                entry = difference["splits"][split]
                sibling = entry["baselines"]["sibling_copy"]
                logger.info("%s%s edit-effect difference R2 %s (windows 0-5/5-12/12+ ms %s) | sibling copy %s vs "
                            "model %s on its %s runs | location mean %s | HIC change MAE %s pct points "
                            "(copy design 6 %s)", prefix, split, entry["difference_r2"],
                            entry["windows_difference_r2"], sibling["difference_r2"],
                            sibling["model_r2_same_runs"], sibling["runs"],
                            entry["baselines"]["location_mean"]["difference_r2"],
                            entry["hic15_sampled"]["change_mae_pct_points"]["model"],
                            entry["hic15_sampled"]["change_mae_pct_points"]["copy_d6"])
                if "base_design_fit" in entry:
                    logger.info("%s%s design 6 (a training design): %s", prefix, split, entry["base_design_fit"])
            if matching_grid:
                windows = time_window_metrics((train_dataset, train_predictions, train_targets),
                                              (test_dataset, predictions, targets), TIME_WINDOWS)
                write_json(directory / "time_windows.json", windows)
                names = ("model_rmse_g", "location_mean_rmse_g")
                for name in names:
                    for index, value in enumerate(windows[name] or ()):
                        values[f"time_windows/{name}/window_{index + 1}"] = value
                logger.info("%sTest RMSE (g) per time window %s ms: model %s | location mean of the training "
                            "designs %s", prefix, windows["window_ms"],
                            *(np.round(windows[name], 2).tolist() if windows[name] else None for name in names))
            values = {prefix + key: value for key, value in values.items()}
            run.summary.update(values)
            run.log(values)
            return metrics, difference

        metrics, _ = evaluate_checkpoint(checkpoint_path, output_dir)
        wandb.log({f"test/{key}": value for key, value in metrics.items() if np.isfinite(value)})
        logger.info("Best-checkpoint test metrics (acceleration in g): %s", metrics)
        # A long run's best validation epoch can come early, so the final epoch's
        # network is scored too, in its own folder (same file names).
        last_dir = output_dir / "last_checkpoint"
        last_dir.mkdir(exist_ok=True)
        last_metrics, last_difference = evaluate_checkpoint(output_dir / LAST_CHECKPOINT_NAME, last_dir,
                                                            prefix="last_checkpoint/")
        last_values = {f"last_checkpoint/test/{key}": value for key, value in last_metrics.items()
                       if np.isfinite(value)}
        logger.info("Last-checkpoint (epoch %s) test metrics (acceleration in g): %s | edit-effect difference "
                    "R2: train=%s, validation=%s, test=%s", len(history["train_losses"]), last_metrics,
                    *(last_difference["splits"][split]["difference_r2"] for split in ("train", "validation", "test")))
        run.summary.update(last_values)
        run.log(last_values)
        logger.info("Finished. Run artifacts: %s", output_dir)
        return metrics
    finally:
        if run is not None:
            run.finish(exit_code=0 if sys.exc_info()[0] is None else 1)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
