"""Train change-region attention without opening held-out test meshes or targets.

Training and evaluation are deliberately separate invocations. The objective is
plain normalized acceleration MSE; the geometry atlas and the normalization
scalers are the only quantities fitted, and both use training designs alone.
"""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mesh_change_attention import MeshChangeAttentionNet
from mesh_change_atlas import fit_change_atlas, group_designs_by_geometry
from mesh_impact_history_reporting import export_training_history_plot
from train_mesh_impact_history import (
    CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, HistoryPredictor, export_predictions,
    initialize_wandb, validate_data, write_json,
)
from utils.utils import (
    Config, DataPreprocessor, Evaluator, HoodImpactDataset, Trainer, collate_fn,
    run_to_design_id, set_seed,
)


MODEL_DEFAULTS = {
    "width": 128, "num_heads": 4, "num_latents": 32, "latent_layers": 3,
    "temporal_layers": 2, "dropout": 0.1, "decoder": "temporal",
    "impactor_nodes": 286, "change_anchors": 32, "change_k": 256,
    "change_layers": 2, "change_scale_mm": 20.0, "change_chunk_size": 256,
}
DATA_PATHS = ("inp_dir", "impact_coords_path", "acceleration_dir")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-run", help="Explicitly evaluate a frozen run on its saved test split; never refits.")
    parser.add_argument("--data-format", choices=["euroncap1704"], default="euroncap1704")
    for name in DATA_PATHS:
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--num-samples", type=int, default=1704)
    parser.add_argument("--samples-per-design", type=int, default=142)
    parser.add_argument("--test-designs", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--val-designs", type=int, nargs="+", default=[11])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-time", type=float)
    parser.add_argument("--time-subsample-stride", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--candidate-spacing-mm", type=float, default=10.0)
    parser.add_argument("--anchor-separation-mm", type=float, default=30.0)
    parser.add_argument("--change-scope", choices=["within_family", "all_designs"], default="within_family",
                        help="Rank anchors by within-near-clone-family variation, or by pooled across-design variation.")
    parser.add_argument("--min-change-mm", type=float, default=0.5)
    parser.add_argument("--family-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--family-max-fraction", type=float, default=0.05)
    for name, default in MODEL_DEFAULTS.items():
        if name != "decoder":
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--output-dir", help="New training run directory (evaluation writes into --evaluate-run).")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE") or "online")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-mesh-attention")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.decoder = "temporal"
    if args.resume_from:
        parser.error("Resume is not supported by this new experiment; choose a fresh --output-dir.")
    if args.wandb_mode not in ("disabled", "offline", "online"):
        parser.error("WANDB_MODE must be disabled, offline, or online")
    if args.evaluate_run:
        allowed = {"--evaluate-run", "--device", "--batch-size", *("--" + key.replace("_", "-") for key in DATA_PATHS)}
        explicit = {value.split("=", 1)[0] for value in (sys.argv[1:] if argv is None else argv) if value.startswith("--")}
        if explicit - allowed:
            parser.error("Evaluation uses the frozen configuration; allowed overrides are device, batch size, and dataset paths.")
    return args


def declared_splits(num_samples, samples_per_design, test_designs, val_designs):
    """Resolve design membership before inspecting any data file."""
    if num_samples < 1 or samples_per_design < 1 or num_samples % samples_per_design:
        raise ValueError("num_samples must contain a positive, complete number of designs")
    designs = set(range(num_samples // samples_per_design))
    test, validation = set(test_designs), set(val_designs)
    if not test or not validation or test & validation:
        raise ValueError("Validation and test design lists must be nonempty and disjoint")
    if not (test | validation) <= designs:
        raise ValueError("Held-out design IDs exceed the declared dataset")
    train = designs - test - validation
    if len(train) < 2:
        raise ValueError("At least two training designs are required to fit a change atlas")
    return {
        name: {"design_ids": sorted(ids), "run_numbers": [
            run for run in range(1, num_samples + 1)
            if run_to_design_id(run, samples_per_design) in ids
        ]}
        for name, ids in (("train", train), ("validation", validation), ("test", test))
    }


def build_config(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    options = {"data_format": "euroncap1704", "prediction_target": "acceleration",
               "num_samples": args.num_samples, "samples_per_design": args.samples_per_design,
               "num_epochs": args.epochs, "time_subsample_stride": args.time_subsample_stride,
               "save_last_checkpoint": True,
               "output_dir": str(Path(args.output_dir or Path("runs") / "mesh_change_attention" / stamp).resolve())}
    mapping = {"lr": "learning_rate"}
    for name in (*DATA_PATHS, "batch_size", "lr", "weight_decay", "seed", "max_train_time"):
        if getattr(args, name) is not None:
            options[mapping.get(name, name)] = getattr(args, name)
    if args.device:
        options["device"] = torch.device(args.device)
    config = Config(**options)
    for name in ("num_samples", "samples_per_design", "num_epochs", "batch_size", "time_subsample_stride"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("learning_rate", config.learning_rate), ("candidate_spacing_mm", args.candidate_spacing_mm),
                        ("anchor_separation_mm", args.anchor_separation_mm),
                        ("family_tolerance_mm", args.family_tolerance_mm)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("weight_decay", config.weight_decay), ("min_change_mm", args.min_change_mm)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < args.family_max_fraction < 1:
        raise ValueError("family_max_fraction must lie strictly between zero and one")
    if config.max_train_time is not None and (not np.isfinite(config.max_train_time) or config.max_train_time <= 0):
        raise ValueError("max_train_time must be finite and positive")
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return config


def load_selected_data(preprocessor, run_numbers):
    """Open only requested meshes/histories, parsing only requested coordinate rows.

    The shared coordinate file has one physical CSV line per run. Skipped rows
    are counted as text only, so malformed test-coordinate values are harmless
    during training. Missing selected runs are errors, never silently skipped.
    """
    runs = [int(run) for run in run_numbers]
    wanted = set(runs)
    if not runs or len(wanted) != len(runs) or min(runs) < 1:
        raise ValueError("Selected run numbers must be unique, positive and nonempty")
    positions = {}
    with open(preprocessor.config.impact_coords_path, newline="", encoding="utf-8-sig") as source:
        header = [field.strip() for field in next(csv.reader([next(source)]))]
        if "X1" not in header or "X2" not in header:
            raise ValueError("Impact coordinate CSV must have X1 and X2 columns")
        columns = (header.index("X1"), header.index("X2"))
        for run, line in enumerate(source, start=1):
            if run in wanted:
                row = next(csv.reader([line]))
                positions[run] = np.asarray([float(row[index]) for index in columns], dtype=np.float32)
            if run >= max(wanted):
                break
    if wanted - set(positions):
        raise ValueError(f"Missing selected impact coordinate rows: {sorted(wanted - set(positions))}")
    data = {key: [] for key in ("mesh_geometries", "indentor_positions", "time_arrays", "accelerations", "run_numbers")}
    for run in runs:
        try:
            mesh = preprocessor.load_mesh_geometry(run)
            times, acceleration = preprocessor.load_acceleration_history(run)
        except Exception as error:
            raise RuntimeError(f"Cannot load required selected run {run}: {error}") from error
        for key, value in (("mesh_geometries", mesh), ("indentor_positions", positions[run]),
                           ("time_arrays", times), ("accelerations", acceleration), ("run_numbers", run)):
            data[key].append(value)
    data["indentor_positions"] = np.asarray(data["indentor_positions"], dtype=np.float32)
    validate_data(data)
    return data


def make_dataset(data, preprocessor):
    return HoodImpactDataset(preprocessor=preprocessor, **data)


def sequential_loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      pin_memory=config.device.type == "cuda")


def training_design_meshes(dataset, samples_per_design, impactor_nodes):
    """One full mesh per training design; validate the hood is location-independent."""
    meshes = {}
    for run, mesh in zip(dataset.run_numbers, dataset.mesh_geometries):
        design = run_to_design_id(run, samples_per_design)
        if len(mesh) <= impactor_nodes:
            raise ValueError(f"Run {run} has no hood nodes after the configured impactor prefix")
        if design in meshes:
            first_hood, hood = meshes[design][impactor_nodes:], mesh[impactor_nodes:]
            if first_hood.shape != hood.shape or not np.array_equal(first_hood, hood):
                raise ValueError(f"Design {design} hood geometry changes across impact locations; verify impactor_nodes and run mapping")
        else:
            meshes[design] = mesh
    return meshes


def resolved_config(config, args, model_kwargs, prediction_grid):
    return {
        "format_version": 1,
        "architecture": {"name": "MeshChangeAttentionNet", "kwargs": model_kwargs},
        "data": {"data_format": config.data_format, "num_samples": config.num_samples,
                 "samples_per_design": config.samples_per_design,
                 **{key: str(Path(getattr(config, key)).resolve()) for key in DATA_PATHS}},
        "preprocessing": {"prediction_target": "acceleration", "time_subsample_stride": config.time_subsample_stride,
                          "max_train_time": config.max_train_time, "scalers_fit_on": "train",
                          "mesh_columns": ["X1", "X2", "X3"], "impact_columns": ["X1", "X2"],
                          "acceleration_units": "g"},
        "training": {**{key: getattr(config, key) for key in
                        ("num_epochs", "batch_size", "learning_rate", "weight_decay", "seed")},
                     "device": str(config.device), "initialization": "from_scratch",
                     "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "gradient_clip_norm": 1.0,
                     "loss": "normalized_acceleration_mse", "design_difference_weight": 0.0,
                     "batch_sampling": "shuffle",
                     "checkpoint_selection": "validation_acceleration_mse", "test_loaded_during_training": False},
        "change_atlas": {"fit_on": "train_only", "file": "change_atlas.npz", "metadata_file": "change_atlas.json",
                         "candidate_spacing_mm": args.candidate_spacing_mm,
                         "anchor_separation_mm": args.anchor_separation_mm,
                         "change_scope": args.change_scope, "min_change_mm": args.min_change_mm,
                         "family_tolerance_mm": args.family_tolerance_mm,
                         "family_max_fraction": args.family_max_fraction,
                         "families_from": "train_geometry" if args.change_scope == "within_family" else None,
                         "response_values_used": False, "impact_locations_used": False},
        "prediction_grid": prediction_grid,
        "wandb": {"mode": args.wandb_mode, "project": args.wandb_project, "entity": args.wandb_entity,
                  "run_name": args.run_name or Path(config.output_dir).name},
        "output_dir": config.output_dir,
    }


def save_atlas(atlas, output_dir):
    arrays = {key: value for key, value in atlas.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output_dir / "change_atlas.npz", **arrays)
    metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in atlas.items()}
    write_json(output_dir / "change_atlas.json", metadata)


def evaluate_run(args):
    """Only this explicit action can read test meshes/targets; all state is frozen."""
    from mesh_change_metrics import design_sensitivity_metrics

    run_dir = Path(args.evaluate_run).expanduser().resolve()
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if saved.get("format_version") != 1 or saved.get("architecture", {}).get("name") != "MeshChangeAttentionNet":
        raise ValueError("Evaluation requires a supported MeshChangeAttentionNet run")
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    expected_splits = declared_splits(saved["data"]["num_samples"], saved["data"]["samples_per_design"],
                                      splits["test"]["design_ids"], splits["validation"]["design_ids"])
    if splits != expected_splits:
        raise ValueError("Saved run numbers do not agree with the declared design split")
    predictor = HistoryPredictor.from_run(run_dir, device=args.device or "cpu")
    config_values = {**saved["data"], **saved["preprocessing"],
                     "device": predictor.device, "batch_size": args.batch_size or saved["training"]["batch_size"]}
    for key in DATA_PATHS:
        if getattr(args, key) is not None:
            config_values[key] = getattr(args, key)
    config = SimpleNamespace(**config_values)
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    preprocessor = predictor.preprocessor
    preprocessor.config = config
    data = load_selected_data(preprocessor, splits["test"]["run_numbers"])
    dataset = make_dataset(data, preprocessor)
    metrics, predictions, targets = Evaluator(predictor.model, config, preprocessor).evaluate(sequential_loader(dataset, config))
    sensitivity = design_sensitivity_metrics(dataset, predictions, targets, config.samples_per_design)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "test_design_sensitivity.json", sensitivity)
    export_predictions(run_dir / "test_acceleration_histories.csv", dataset, predictions, targets)
    # The shared postprocessor integrates the exact sampled grid, in seconds.
    from hic15 import batched_hic
    rows, offset = [], 0
    for run, times in zip(dataset.run_numbers, dataset.time_arrays):
        count = len(times)
        truth, prediction = targets[offset:offset + count], predictions[offset:offset + count]
        gt_hic = float(batched_hic(times, truth))
        predicted_hic = float(batched_hic(times, prediction))
        rows.append({"run_number": run, "design_id": run_to_design_id(run, config.samples_per_design),
                     "location_id": (run - 1) % config.samples_per_design + 1,
                     "hic15_true": gt_hic, "hic15_pred": predicted_hic,
                     "hic15_error": predicted_hic - gt_hic,
                     "acceleration_rmse_g": float(np.sqrt(np.mean((truth - prediction) ** 2)))})
        offset += count
    pd.DataFrame(rows).to_csv(run_dir / "test_hic15.csv", index=False)
    write_json(run_dir / "test_evaluation.json", {
        "evaluated_at": datetime.now().isoformat(), "checkpoint": CHECKPOINT_NAME,
        "test_design_ids": splits["test"]["design_ids"], "test_run_numbers": dataset.run_numbers,
        "fit_or_optimization_performed": False,
        "hic15_grid": "saved sampled histories; no additional resampling; time units seconds",
    })
    print(f"Frozen test evaluation saved in {run_dir}")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if args.evaluate_run:
        return evaluate_run(args)
    # No existence checks, globbing, geometry inspection or target reads of test runs.
    splits = declared_splits(args.num_samples, args.samples_per_design, args.test_designs, args.val_designs)
    config = build_config(args)
    output_dir = Path(config.output_dir)
    protected = ("config.json", CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, "scalers.joblib", "change_atlas.npz")
    if any((output_dir / name).exists() for name in protected):
        raise FileExistsError(f"Run artifacts already exist in {output_dir}; choose a fresh --output-dir")
    set_seed(config.seed)
    logger = logging.getLogger("mesh_change_attention.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(output_dir / "training.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    run = None
    try:
        kwargs = {key: getattr(args, key) for key in MODEL_DEFAULTS}
        model = MeshChangeAttentionNet(**kwargs)
        logger.info("MeshChangeAttentionNet: %s", kwargs)
        logger.info("Train designs: %s | validation: %s | unopened test: %s", splits["train"]["design_ids"],
                    splits["validation"]["design_ids"], splits["test"]["design_ids"])
        run = initialize_wandb(args, resolved_config(config, args, kwargs, None), output_dir, logger)
        preprocessor = DataPreprocessor(config)
        train_data = load_selected_data(preprocessor, splits["train"]["run_numbers"])
        train_dataset = make_dataset(train_data, preprocessor)
        preprocessor.fit_scalers(mesh_geometries=train_dataset.mesh_geometries,
                                 indentor_positions=train_dataset.indentor_positions,
                                 time_arrays=train_dataset.time_arrays, accelerations=train_dataset.accelerations)
        meshes = training_design_meshes(train_dataset, config.samples_per_design, args.impactor_nodes)
        grouping = None
        if args.change_scope == "within_family":
            # Families come from the training meshes themselves, so no external
            # cluster table (which would encode held-out membership) is needed.
            grouping = group_designs_by_geometry(meshes, impactor_nodes=args.impactor_nodes,
                                                 tolerance_mm=args.family_tolerance_mm,
                                                 max_family_fraction=args.family_max_fraction)
            logger.info("Training design families: %s | within-family moved <= %.4f, cross-family >= %s, margin %s",
                        grouping["families"], grouping["max_within_family_moved_fraction"],
                        grouping["min_cross_family_moved_fraction"], grouping["separation_margin"])
            if grouping["num_families"] < 2:
                logger.warning("All training designs formed one family; anchors rank by variation inside it.")
            elif grouping["separation_margin"] is not None and grouping["separation_margin"] < 3:
                logger.warning("Family separation margin %.2f is small; inspect change_atlas.json before trusting the split.",
                               grouping["separation_margin"])
        atlas = fit_change_atlas(meshes, num_anchors=args.change_anchors,
                                 candidate_spacing_mm=args.candidate_spacing_mm,
                                 anchor_separation_mm=args.anchor_separation_mm,
                                 impactor_nodes=args.impactor_nodes, neighborhood_k=args.change_k,
                                 groups_by_design=grouping["design_family"] if grouping else None,
                                 min_change_mm=args.min_change_mm)
        if grouping is not None:
            # Nested values must stay JSON serializable for the saved metadata.
            atlas = {**atlas, "design_family_grouping": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in grouping.items() if key != "design_family"}}
        model.set_change_atlas(atlas)
        model.set_coordinate_scalers(preprocessor.mesh_scaler.mean_, preprocessor.mesh_scaler.scale_,
                                     preprocessor.indentor_scaler.mean_, preprocessor.indentor_scaler.scale_)
        val_data = load_selected_data(preprocessor, splits["validation"]["run_numbers"])
        val_dataset = make_dataset(val_data, preprocessor)
        train_loader = sequential_loader(train_dataset, config, shuffle=True)
        val_loader = sequential_loader(val_dataset, config)
        times = np.asarray(train_dataset.time_arrays[0], dtype=np.float32)
        matching_grid = all(np.array_equal(times, other) for other in (*train_dataset.time_arrays, *val_dataset.time_arrays))
        grid = {"file": "prediction_times.npy", "reference_run_number": train_dataset.run_numbers[0],
                "num_time_points": len(times), "same_grid_for_all_loaded_runs": matching_grid,
                "already_subsampled": True, "time_units": "seconds"}
        saved = resolved_config(config, args, kwargs, grid)
        saved["change_atlas"]["train_design_ids"] = sorted(meshes)
        saved["change_atlas"]["scoring_families"] = atlas["scoring_families"]
        saved["change_atlas"]["num_anchors"] = atlas["num_anchors"]
        write_json(output_dir / "config.json", saved)
        write_json(output_dir / "splits.json", splits)
        write_json(output_dir / "data_access_audit.json", {
            "loaded_training_runs": train_data["run_numbers"], "loaded_validation_runs": val_data["run_numbers"],
            "test_meshes_or_targets_loaded": False, "atlas_fit_design_ids": sorted(meshes),
            "normalization_fit_run_numbers": train_data["run_numbers"],
            "atlas_scoring_families": atlas["scoring_families"],
            "families_derived_from": "train_geometry" if grouping else None,
        })
        save_atlas(atlas, output_dir)
        preprocessor.save_scalers(str(output_dir / "scalers.joblib"))
        np.save(output_dir / "prediction_times.npy", times, allow_pickle=False)
        run.config.update(saved, allow_val_change=True)
        logger.info("Atlas: %s anchors (%s scope, %s scoring families) from %s training designs; "
                    "%s changed candidates of %s; %s parameters",
                    len(atlas["anchors_mm"]), atlas["change_scope"], atlas["num_scoring_families"], len(meshes),
                    atlas["num_changed_candidates"], atlas["num_candidates"],
                    sum(parameter.numel() for parameter in model.parameters()))
        trainer = Trainer(model, train_loader, val_loader, config, preprocessor)
        history = trainer.train()
        write_json(output_dir / "training_history.json", history)
        pd.DataFrame({"epoch": np.arange(1, len(history["train_losses"]) + 1),
                      "train_mse_normalized": history["train_losses"],
                      "validation_mse_normalized": history["val_losses"]}).to_csv(
            output_dir / "training_history.csv", index=False)
        export_training_history_plot(history, output_dir)
        if not (output_dir / CHECKPOINT_NAME).is_file():
            raise RuntimeError("Training did not produce a finite best validation checkpoint")
        run.summary["best_epoch"] = int(np.argmin(history["val_losses"])) + 1
        logger.info("Training finished. Test remains unopened. Explicit frozen evaluation: python train_mesh_change_attention.py --evaluate-run %s", output_dir)
        return history
    finally:
        if run is not None:
            run.finish(exit_code=0 if sys.exc_info()[0] is None else 1)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
