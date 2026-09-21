"""Train January's best acceleration Temporal DeepONet on the local-change set."""

import argparse
from copy import deepcopy
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wandb

from temporal_deeponet import ACCELERATION_BEST_ARCH, LegacyHoodImpactNeuralOperator
from utils.utils import Config, DataPreprocessor, Evaluator, Trainer, create_data_loaders, set_seed
from train_mesh_impact_history import (
    CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, HistoryPredictor, export_predictions,
    initialize_wandb, resolve_splits, validate_data, write_json,
)
from mesh_impact_history_reporting import export_training_history_plot
from preflight_mesh_impact_history_1704 import (
    DEFAULT_DATA_ROOT, NUM_RUNS, SAMPLES_PER_DESIGN, validate_cluster_holdout,
    validate_dataset, validate_runtime, validate_splits,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--test-designs", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--val-designs", type=int, nargs="+", default=[11])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-subsample-stride", type=int, default=16)
    parser.add_argument("--max-train-time", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate all filenames, splits, and a real model forward/backward; write no run.")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE") or "online")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-mesh-attention")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if args.wandb_mode not in ("disabled", "offline", "online"):
        parser.error("WANDB_MODE must be disabled, offline, or online")
    for key in ("epochs", "batch_size", "time_subsample_stride"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight_decay must be finite and nonnegative")
    if args.max_train_time is not None and (not np.isfinite(args.max_train_time) or args.max_train_time <= 0):
        parser.error("max_train_time must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in 0..2**32-1")
    return args


def config_values(args):
    root = Path(args.data_root).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return {
        **deepcopy(ACCELERATION_BEST_ARCH),
        "prediction_target": "acceleration", "data_format": "euroncap1704",
        "num_samples": NUM_RUNS, "samples_per_design": SAMPLES_PER_DESIGN,
        "inp_dir": str(root / "inp_files"),
        "impact_coords_path": str(root / "ImpactCoords_1704.csv"),
        "acceleration_dir": str(root / "output_history_acc"),
        "num_epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "time_subsample_stride": args.time_subsample_stride, "max_train_time": args.max_train_time,
        "device": torch.device(args.device), "save_last_checkpoint": True,
        "output_dir": str(Path(args.output_dir or Path("runs") / "temporal_deeponet_1704" / stamp).resolve()),
    }


def preflight(args, config):
    validate_runtime(config.device.type == "cuda")
    splits = validate_splits(args.test_designs, args.val_designs)
    clusters, warnings = validate_cluster_holdout(args.test_designs, args.val_designs)
    for warning in warnings:
        print(f"Cluster check: WARNING - {warning}. Test geometry remains held out.")
    root, xy = validate_dataset(args.data_root)
    processor = DataPreprocessor(config)
    mesh = processor.load_mesh_geometry(1)
    times, acceleration = processor.load_acceleration_history(1)
    validate_data(dict(run_numbers=[1], mesh_geometries=[mesh], indentor_positions=[xy[0]],
                       time_arrays=[times], accelerations=[acceleration]))
    # Disposable single-example normalization exercises the actual architecture.
    # The training run fits fresh scalers exclusively on its training designs.
    processor.fit_scalers([mesh], xy[:1], [times], [acceleration])
    model = LegacyHoodImpactNeuralOperator(config).to(config.device)
    mesh_tensor = torch.as_tensor(processor.transform_mesh(mesh), device=config.device)
    time_tensor = torch.as_tensor(processor.transform_time(times), device=config.device)
    prediction = model(
        mesh_tensor, torch.zeros(len(mesh), dtype=torch.long, device=config.device),
        torch.as_tensor(processor.transform_indentor(xy[0]), device=config.device).unsqueeze(0),
        time_tensor, torch.zeros(len(times), dtype=torch.long, device=config.device), batch_size=1,
    )
    if prediction.shape != time_tensor.shape or not torch.isfinite(prediction).all():
        raise ValueError("Temporal DeepONet preflight produced an invalid acceleration history")
    prediction.square().mean().backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Temporal DeepONet preflight produced missing or nonfinite gradients")
    print(f"Temporal DeepONet: {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"{config.device} forward/backward passed | {len(mesh)} nodes | {len(times)} sampled times")
    print(f"Dataset: {root} | {NUM_RUNS} runs | held-out clusters: {clusters}")
    for name, group in splits.items():
        print(f"{name}: designs={group['design_ids']} | runs={len(group['run_numbers'])}")
    print("Preflight passed.")
    return splits


def resolved_config(config, args, grid=None):
    return {
        "format_version": 1,
        "architecture": {"name": "LegacyHoodImpactNeuralOperator", "module": "temporal_deeponet",
                         "preset": "acceleration_best_20260116", "kwargs": deepcopy(ACCELERATION_BEST_ARCH),
                         "film_layers": 1, "dropout": 0.1},
        "reference_run": "runs/acceleration_history_target_value/20260116_160228_best",
        "data": {key: getattr(config, key) for key in (
            "data_format", "num_samples", "samples_per_design", "inp_dir", "impact_coords_path", "acceleration_dir")},
        "preprocessing": {"prediction_target": "acceleration", "acceleration_units": "g",
                          "time_subsample_stride": config.time_subsample_stride,
                          "max_train_time": config.max_train_time, "scalers_fit_on": "train",
                          "mesh_nodes": "complete INP node block, including headform",
                          "impact_columns": ["X1", "X2"]},
        "training": {
            **{key: getattr(config, key) for key in (
                "num_epochs", "batch_size", "learning_rate", "weight_decay", "seed")},
            "device": str(config.device), "initialization": "from_scratch",
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "gradient_clip_norm": 1.0,
            "loss": "normalized_acceleration_mse", "design_difference_weight": 0.0,
            "batch_sampling": "shuffle",
        },
        "prediction_grid": grid,
        "wandb": {"mode": args.wandb_mode, "project": args.wandb_project,
                  "entity": args.wandb_entity, "run_name": args.run_name or Path(config.output_dir).name},
        "output_dir": config.output_dir,
    }


class TemporalDeepONetPredictor(HistoryPredictor):
    """Reload this baseline using train-fitted scalers and already-sampled times."""

    @classmethod
    def from_run(cls, output_dir, device="cpu"):
        import json
        root = Path(output_dir)
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if saved.get("format_version") != 1 or saved["architecture"]["name"] != "LegacyHoodImpactNeuralOperator":
            raise ValueError("Expected a Temporal DeepONet acceleration baseline run")
        config = SimpleNamespace(**saved["architecture"]["kwargs"], prediction_target="acceleration")
        model = LegacyHoodImpactNeuralOperator(config)
        checkpoint = torch.load(root / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        processor = DataPreprocessor(config)
        processor.load_scalers(str(root / "scalers.joblib"))
        return cls(model, processor, np.load(root / "prediction_times.npy", allow_pickle=False), device)


def main(argv=None):
    args = parse_args(argv)
    values = config_values(args)
    output_dir = Path(values["output_dir"])
    if not args.preflight_only and any((output_dir / name).exists() for name in (
            "config.json", CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, "scalers.joblib")):
        raise FileExistsError(f"Run artifacts already exist in {output_dir}; choose a new --output-dir")
    expected_splits = preflight(args, SimpleNamespace(**values))
    if args.preflight_only:
        return
    config = Config(**values)
    set_seed(config.seed)  # Reset after the disposable preflight model.
    logger = logging.getLogger("temporal_deeponet_1704.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(output_dir / "training.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    run = None
    try:
        model = LegacyHoodImpactNeuralOperator(config)
        logger.info("Training January's acceleration architecture from scratch: %s", ACCELERATION_BEST_ARCH)
        run = initialize_wandb(args, resolved_config(config, args), output_dir, logger)
        processor = DataPreprocessor(config)
        data = processor.load_all_data()
        validate_data(data)
        if data["run_numbers"] != list(range(1, NUM_RUNS + 1)):
            raise ValueError("Expected all 1704 runs to load successfully; inspect preprocessing warnings")
        splits = resolve_splits(data, config, args.test_designs, args.val_designs)
        if splits != expected_splits:
            raise ValueError("Loaded splits differ from the complete preflight mapping")
        train_loader, val_loader, test_loader, test_dataset = create_data_loaders(
            data, processor, config, test_design_ids=args.test_designs, val_design_id=args.val_designs,
        )
        train_dataset = train_loader.dataset
        processor.fit_scalers(train_dataset.mesh_geometries, train_dataset.indentor_positions,
                              train_dataset.time_arrays, train_dataset.accelerations)
        times = np.asarray(train_dataset.time_arrays[0], dtype=np.float32)
        grid = {"file": "prediction_times.npy", "reference_run_number": int(train_dataset.run_numbers[0]),
                "num_time_points": len(times), "already_subsampled": True, "time_units": "seconds",
                "same_grid_for_all_loaded_runs": all(np.array_equal(times, t) for t in data["time_arrays"])}
        saved = resolved_config(config, args, grid)
        write_json(output_dir / "config.json", saved)
        write_json(output_dir / "splits.json", splits)
        processor.save_scalers(str(output_dir / "scalers.joblib"))
        np.save(output_dir / "prediction_times.npy", times, allow_pickle=False)
        run.config.update(saved, allow_val_change=True)
        trainer = Trainer(model, train_loader, val_loader, config, processor)
        history = trainer.train()
        write_json(output_dir / "training_history.json", history)
        pd.DataFrame({"epoch": np.arange(1, len(history["train_losses"]) + 1),
                      "train_mse_normalized": history["train_losses"],
                      "validation_mse_normalized": history["val_losses"]}).to_csv(
                          output_dir / "training_history.csv", index=False)
        history_plot = export_training_history_plot(history, output_dir)
        run.summary["best_epoch"] = int(np.argmin(history["val_losses"])) + 1
        if args.wandb_mode != "disabled":
            run.log({"training/history": wandb.Image(str(history_plot))})
        if not (output_dir / CHECKPOINT_NAME).is_file():
            raise RuntimeError("Training produced no finite best validation checkpoint")
        checkpoint = torch.load(output_dir / CHECKPOINT_NAME, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics, predictions, targets = Evaluator(model, config, processor).evaluate(test_loader)
        write_json(output_dir / "metrics.json", metrics)
        export_predictions(output_dir / "test_acceleration_histories.csv", test_dataset, predictions, targets)
        wandb.log({f"test/{key}": value for key, value in metrics.items() if np.isfinite(value)})
        logger.info("Best-checkpoint test metrics (acceleration in g): %s", metrics)
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
