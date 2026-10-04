"""Train a direct geometry-coefficient correction of a frozen saved baseline.

The baseline's cohort, preprocessing and splits are retained. Geometry basis
fitting sees training designs only; test data is opened after model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wandb

from geometry_correction_basis import GeometryCorrectionBasis
from mesh_geometry_correction import CoefficientHistoryCorrection, load_correction_geometry
from mesh_geometry_correction_reporting import export_correction_evaluation
from mesh_impact_history_reporting import export_training_history_plot
from train_mesh_impact_history import HistoryPredictor, initialize_wandb, write_json
from train_mesh_change_attention import load_selected_data, training_design_meshes
from utils.utils import DataPreprocessor, set_seed


BASELINE_FILES = ("config.json", "splits.json", "scalers.joblib", "prediction_times.npy",
                  "hood_impact_best_model.pt")
DEFAULT_BASELINE = "runs/mesh_impact_history/20260930_204155_3283115_1704"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE)
    parser.add_argument("--data-root", help="Relocate inp_files, ImpactCoords_1704.csv, and output_history_acc.")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-spacing-mm", type=float, default=10.)
    parser.add_argument("--profile-k", type=int, default=8)
    parser.add_argument("--geometry-noise-floor-mm", type=float, default=.1)
    parser.add_argument("--correction-width", type=int, default=128)
    parser.add_argument("--correction-layers", type=int, default=3)
    parser.add_argument("--fourier-num-frequencies", type=int, default=6)
    parser.add_argument("--fourier-min-frequency-hz", type=float, default=20.)
    parser.add_argument("--fourier-max-frequency-hz", type=float, default=640.)
    parser.add_argument("--baseline-neighborhood-chunk-size", type=int, default=128,
                        help="Inference workspace size only; does not change baseline weights or neighbors.")
    parser.add_argument("--neighbor-cache-dir")
    parser.add_argument("--resume-from", help="Resume an interrupted correction run into a new output directory.")
    parser.add_argument("--evaluate-run", help="Re-export frozen test results from a completed correction run.")
    parser.add_argument("--no-plots", action="store_true", help="Skip expensive figures, retaining all numerical results.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "hood-impact-geometry-correction"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    explicit = {token.split("=", 1)[0] for token in (argv if argv is not None else __import__("sys").argv[1:])
                if token.startswith("--")}
    if args.resume_from and args.evaluate_run:
        parser.error("Choose resume or evaluation, not both")
    if args.evaluate_run:
        allowed = {"--evaluate-run", "--data-root", "--device", "--batch-size", "--no-plots",
                   "--neighbor-cache-dir", "--baseline-neighborhood-chunk-size"}
        if explicit - allowed:
            parser.error("Evaluation retains the saved settings; only data paths, device, batch size and plots can change")
    if args.resume_from:
        source = Path(args.resume_from).expanduser().resolve()
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if config.get("architecture", {}).get("name") != "CoefficientHistoryCorrection":
            parser.error("Resume requires a coefficient correction run")
        mapping = {"epochs": "num_epochs", "batch_size": "batch_size", "lr": "learning_rate",
                   "weight_decay": "weight_decay", "seed": "seed"}
        for name, saved_name in mapping.items():
            value = config["training"][saved_name]
            if "--" + name.replace("_", "-") in explicit and getattr(args, name) != value:
                parser.error(f"Resume must retain saved --{name.replace('_', '-')}={value}")
            setattr(args, name, value)
        allowed = {"--resume-from", "--output-dir", "--data-root", "--device", "--no-plots", "--run-name",
                   "--wandb-mode", "--wandb-project", "--wandb-entity", "--neighbor-cache-dir",
                   "--baseline-neighborhood-chunk-size", *("--" + key.replace("_", "-") for key in mapping)}
        if explicit - allowed:
            parser.error("Resume uses the saved geometry basis and correction architecture")
        args.resume_from = str(source)
    for name in ("epochs", "batch_size", "profile_k", "correction_width", "correction_layers",
                 "fourier_num_frequencies", "baseline_neighborhood_chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    for name in ("lr", "anchor_spacing_mm", "geometry_noise_floor_mm"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight_decay must be finite and nonnegative")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def validate_baseline(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    for name in BASELINE_FILES:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Missing baseline artifact: {run_dir / name}")
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if saved.get("format_version") != 1 or saved.get("architecture", {}).get("name") != "MeshImpactHistoryNet":
        raise ValueError("Baseline must be a saved MeshImpactHistoryNet run")
    if saved["data"]["data_format"] != "euroncap1704":
        raise ValueError("This experiment requires the EuroNCAP INP input format")
    if saved["preprocessing"]["prediction_target"] != "acceleration":
        raise ValueError("Baseline must predict acceleration")
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    count = saved["data"]["samples_per_design"]
    seen, locations = set(), None
    for split in ("train", "validation", "test"):
        ids, runs = splits[split]["design_ids"], splits[split]["run_numbers"]
        if not ids or not runs or len(set(ids)) != len(ids) or len(set(runs)) != len(runs):
            raise ValueError(f"Malformed {split} split")
        if seen & set(ids):
            raise ValueError("Baseline design splits overlap")
        seen.update(ids)
        if any(not 1 <= run <= saved["data"]["num_samples"] or (run - 1) // count not in ids for run in runs):
            raise ValueError("Saved runs do not match declared designs")
        cohorts = [{(run - 1) % count + 1 for run in runs if (run - 1) // count == design} for design in ids]
        if not all(cohort == cohorts[0] for cohort in cohorts):
            raise ValueError("Baseline designs do not share the same location cohort")
        if locations is not None and cohorts[0] != locations:
            raise ValueError("Baseline splits use different location cohorts")
        locations = cohorts[0]
    if len(splits["train"]["design_ids"]) < 2:
        raise ValueError("At least two training designs are required")
    times = np.load(run_dir / "prediction_times.npy", allow_pickle=False)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid saved prediction time grid")
    if not saved.get("prediction_grid", {}).get("same_grid_for_all_loaded_runs", False):
        raise ValueError("Coefficient experiment requires a common saved output time grid")
    return run_dir, saved, splits


def data_preprocessor(saved, data_root=None):
    values = {**saved["data"], **saved["preprocessing"]}
    if data_root:
        root = Path(data_root).expanduser().resolve()
        values.update(inp_dir=str(root / "inp_files"), impact_coords_path=str(root / "ImpactCoords_1704.csv"),
                      acceleration_dir=str(root / "output_history_acc"))
    return DataPreprocessor(SimpleNamespace(**values))


def _input_signature(preprocessor, runs):
    # Check content-independent file identity before reusing a local resume cache.
    # A changed path, size or modification time invalidates cached predictions.
    entries = []
    for run in runs:
        for directory, filename in ((preprocessor.config.inp_dir, f"HoodImpact_{run}.inp"),
                                    (preprocessor.config.acceleration_dir, f"HoodImpact_{run}_SAE1000_interp1000.csv")):
            path = Path(directory) / filename
            stat = path.stat()
            entries.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    coordinate_path = Path(preprocessor.config.impact_coords_path)
    coordinate_stat = coordinate_path.stat()
    entries.append((str(coordinate_path.resolve()), coordinate_stat.st_size, coordinate_stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _geometry_state(data, baseline, preprocessor):
    count, start = preprocessor.config.samples_per_design, baseline.model.impactor_nodes
    meshes = training_design_meshes(SimpleNamespace(**data), count, start)
    labels = {}
    for design, mesh in meshes.items():
        run = next(run for run in data["run_numbers"] if (run - 1) // count == design)
        parsed, part = load_correction_geometry(Path(preprocessor.config.inp_dir) / f"HoodImpact_{run}.inp", start)
        if parsed.shape != mesh.shape or not np.array_equal(parsed, mesh):
            raise ValueError("Part-label parser and baseline geometry parser disagree")
        labels[design] = part
    return meshes, labels


def prepare_split(split, runs, preprocessor, baseline, basis, output_dir, logger, data=None):
    """Compute frozen baseline histories once; no baseline call occurs in epochs."""
    output_dir = Path(output_dir)
    path = output_dir / f"{split}_cache.npz"
    binding = dict(baseline_artifact_sha256={name: file_sha256(output_dir / "baseline" / name)
                                            for name in BASELINE_FILES},
                   basis_sha256=file_sha256(output_dir / "geometry_basis.npz"),
                   inputs=_input_signature(preprocessor, runs), run_numbers=list(runs))
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored["binding_json"])) == binding:
                cache = {key: stored[key].copy() for key in ("run_numbers", "impact_xy", "normalized_impact",
                         "coefficients", "targets", "baseline", "times", "geometry_diagnostics_json")}
                if (not np.array_equal(cache["run_numbers"], np.asarray(runs))
                        or cache["impact_xy"].shape != (len(runs), 2)
                        or cache["normalized_impact"].shape != (len(runs), 2)
                        or cache["baseline"].shape != cache["targets"].shape
                        or cache["baseline"].shape != (len(runs), len(baseline.time_points))
                        or cache["coefficients"].shape != (len(runs), basis.coefficient_width)
                        or not np.array_equal(cache["times"], baseline.time_points)
                        or not all(np.isfinite(cache[key]).all() for key in
                                   ("impact_xy", "normalized_impact", "coefficients", "targets", "baseline"))):
                    raise ValueError(f"Malformed {split} baseline cache")
                logger.info("Loaded frozen %s predictions from cache (%s curves)", split, len(runs))
                return cache
    data = load_selected_data(preprocessor, runs) if data is None else data
    if data["run_numbers"] != list(runs):
        raise ValueError("Loaded run order differs from saved split")
    if not all(np.array_equal(time, baseline.time_points) for time in data["time_arrays"]):
        raise ValueError("Loaded sampled time grid differs from the baseline; no resampling is allowed")
    meshes, labels = _geometry_state(data, baseline, preprocessor)
    coefficients, diagnostics = {}, {}
    for design, mesh in meshes.items():
        coefficients[design], diagnostics[str(design)] = basis.transform(
            mesh[baseline.model.impactor_nodes:], labels[design])
    frozen = []
    logger.info("Computing frozen %s baseline histories: %s curves", split, len(runs))
    for index, (mesh, impact) in enumerate(zip(data["mesh_geometries"], data["indentor_positions"])):
        frozen.append(baseline.predict(mesh, impact))
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(runs):
            logger.info("Frozen %s baseline: %s/%s", split, index + 1, len(runs))
    cache = dict(run_numbers=np.asarray(runs, dtype=np.int64),
                 impact_xy=np.asarray(data["indentor_positions"], dtype=np.float32),
                 normalized_impact=np.stack([baseline.preprocessor.transform_indentor(p)
                                             for p in data["indentor_positions"]]).astype(np.float32),
                 coefficients=np.stack([coefficients[(run - 1) // preprocessor.config.samples_per_design]
                                        for run in runs]).astype(np.float32),
                 targets=np.stack(data["accelerations"]).astype(np.float32),
                 baseline=np.stack(frozen).astype(np.float32), times=baseline.time_points.copy(),
                 geometry_diagnostics_json=np.asarray(json.dumps(diagnostics)))
    if not all(np.isfinite(cache[key]).all() for key in ("coefficients", "targets", "baseline")):
        raise ValueError("Nonfinite coefficients, targets or baseline predictions")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **cache, binding_json=np.asarray(json.dumps(binding)))
    os.replace(temporary, path)
    write_json(output_dir / f"{split}_geometry_diagnostics.json", diagnostics)
    return cache


def _tensor_cache(cache, baseline, device):
    scale, mean = float(baseline.preprocessor.accel_scaler.scale_[0]), float(baseline.preprocessor.accel_scaler.mean_[0])
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in
            dict(coefficients=cache["coefficients"], impact=cache["normalized_impact"], time=cache["times"],
                 baseline=(cache["baseline"] - mean) / scale, target=(cache["targets"] - mean) / scale).items()}


@torch.no_grad()
def correction_predictions(model, cache, batch_size):
    model.eval()
    return torch.cat([cache["baseline"][start:start + batch_size] + model(
        cache["coefficients"][start:start + batch_size], cache["impact"][start:start + batch_size], cache["time"])
        for start in range(0, len(cache["coefficients"]), batch_size)])


def _difference_skill(prediction, target, run_numbers, samples_per_design):
    prediction, target = prediction.detach().cpu().double().numpy(), target.detach().cpu().double().numpy()
    energy = error = 0.
    locations = (np.asarray(run_numbers) - 1) % samples_per_design
    for location in np.unique(locations):
        rows = np.flatnonzero(locations == location)
        for i, first in enumerate(rows):
            for second in rows[i + 1:]:
                truth = target[second] - target[first]
                delta = prediction[second] - prediction[first]
                energy += float(truth @ truth)
                error += float((delta - truth) @ (delta - truth))
    return None if energy == 0 else 1 - error / energy


def _save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_loss, best_epoch):
    config = json.loads((Path(path).parent / "config.json").read_text(encoding="utf-8"))
    state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                 scheduler_state_dict=scheduler.state_dict(), completed_epochs=epoch,
                 history=history, best_val_loss=best_loss, best_epoch=best_epoch,
                 torch_rng_state=torch.get_rng_state(),
                 baseline_artifact_sha256=config["baseline"]["artifact_sha256"],
                 basis_sha256=config["geometry_basis"]["sha256"],
                 cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_correction(model, baseline, train, validation, args, output_dir, logger, run=None, resume=None):
    device = torch.device(args.device)
    train_tensor, val_tensor = _tensor_cache(train, baseline, device), _tensor_cache(validation, baseline, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = dict(train_losses=[], val_losses=[], train_design_difference_skill=[], learning_rates=[])
    start, best_loss, best_epoch = 0, float("inf"), 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if (state["baseline_artifact_sha256"] != config["baseline"]["artifact_sha256"]
                or state["basis_sha256"] != config["geometry_basis"]["sha256"]):
            raise ValueError("Resume checkpoint does not match baseline/basis artifacts")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start, history, best_loss, best_epoch = state["completed_epochs"], state["history"], state["best_val_loss"], state["best_epoch"]
        if not 0 < start < args.epochs or scheduler.T_max != args.epochs:
            raise ValueError("Resume checkpoint must have unfinished epochs under the saved schedule")
        if any(len(values) != start for values in history.values()):
            raise ValueError("Checkpoint history lengths disagree")
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("cuda_rng_states") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
    for epoch in range(start, args.epochs):
        model.train()
        permutation = torch.randperm(len(train["run_numbers"]))
        total, count = 0., 0
        for begin in range(0, len(permutation), args.batch_size):
            rows = permutation[begin:begin + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = train_tensor["baseline"][rows] + model(
                train_tensor["coefficients"][rows], train_tensor["impact"][rows], train_tensor["time"])
            loss = (predicted - train_tensor["target"][rows]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            count += len(rows)
        val_prediction = correction_predictions(model, val_tensor, args.batch_size)
        val_loss = float((val_prediction - val_tensor["target"]).square().mean())
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        train_prediction = correction_predictions(model, train_tensor, args.batch_size)
        skill = _difference_skill(train_prediction, train_tensor["target"], train["run_numbers"],
                                  baseline.preprocessor.config.samples_per_design)
        history["train_losses"].append(total / count)
        history["val_losses"].append(val_loss)
        history["train_design_difference_skill"].append(skill)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch + 1
            _save_checkpoint(output_dir / "correction_best.pt", model, optimizer, scheduler,
                             epoch + 1, history, best_loss, best_epoch)
        _save_checkpoint(output_dir / "correction_last.pt", model, optimizer, scheduler,
                         epoch + 1, history, best_loss, best_epoch)
        logger.info("Epoch %s/%s | train MSE %.6f | val MSE %.6f | train difference skill %s%s",
                    epoch + 1, args.epochs, total / count, val_loss,
                    "undefined" if skill is None else f"{skill:.5f}", " | best" if improved else "")
        if run:
            payload = dict(epoch=epoch + 1, **{"train/loss": total / count, "val/loss": val_loss,
                                             "lr": history["learning_rates"][-1]})
            if skill is not None:
                payload["train/design_difference_skill"] = skill
            run.log(payload)
    write_json(output_dir / "training_history.json", history)
    pd.DataFrame(dict(epoch=np.arange(1, len(history["train_losses"]) + 1),
                      train_mse_normalized=history["train_losses"], validation_mse_normalized=history["val_losses"],
                      train_design_difference_skill=history["train_design_difference_skill"],
                      learning_rate=history["learning_rates"])).to_csv(output_dir / "training_history.csv", index=False)
    if not args.no_plots:
        export_training_history_plot(history, output_dir)
    model.load_state_dict(torch.load(output_dir / "correction_best.pt", map_location=device,
                                    weights_only=True)["model_state_dict"], strict=True)
    if run:
        run.summary.update(best_epoch=best_epoch, best_val_loss=best_loss)
    return history


def evaluate_cache(model, baseline, cache, args, output_dir, split, samples_per_design):
    tensors = _tensor_cache(cache, baseline, args.device)
    model.eval()
    with torch.no_grad():
        delta = torch.cat([model(tensors["coefficients"][start:start + args.batch_size],
                                 tensors["impact"][start:start + args.batch_size], tensors["time"])
                           for start in range(0, len(cache["run_numbers"]), args.batch_size)]).cpu().numpy()
    corrected = cache["baseline"] + delta * float(baseline.preprocessor.accel_scaler.scale_[0])
    records = [dict(run_number=int(run), design_id=int((run - 1) // samples_per_design),
                    location_id=int((run - 1) % samples_per_design + 1), impact_xy=cache["impact_xy"][index],
                    ground_truth=cache["targets"][index], baseline=cache["baseline"][index], corrected=corrected[index])
               for index, run in enumerate(cache["run_numbers"])]
    return export_correction_evaluation(output_dir, records, cache["times"], prefix=split,
                                        make_plots=not args.no_plots, samples_per_design=samples_per_design)


def _configure_baseline(baseline, args, preprocessor):
    baseline.model.requires_grad_(False)
    baseline.model.eval()
    # Predictor's scaler namespace originally includes preprocessing only.
    baseline.preprocessor.config.samples_per_design = preprocessor.config.samples_per_design
    for block in baseline.model.neighborhood_blocks:
        block.chunk_size = args.baseline_neighborhood_chunk_size


def evaluate_run(args):
    from mesh_geometry_correction import GeometryCorrectionPredictor
    output_dir = Path(args.evaluate_run).expanduser().resolve()
    saved = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    predictor = GeometryCorrectionPredictor.from_run(output_dir, device=args.device,
                                                    neighbor_cache_dir=args.neighbor_cache_dir)
    _, base_config, splits = validate_baseline(output_dir / "baseline")
    preprocessor = data_preprocessor(base_config, args.data_root or saved.get("data_root"))
    _configure_baseline(predictor.baseline, args, preprocessor)
    logger = logging.getLogger("geometry_correction")
    cache = prepare_split("test", splits["test"]["run_numbers"], preprocessor, predictor.baseline,
                          predictor.basis, output_dir, logger)
    return evaluate_cache(predictor.correction, predictor.baseline, cache, args, output_dir, "test",
                          base_config["data"]["samples_per_design"])


def main(argv=None):
    args = parse_args(argv)
    if args.evaluate_run:
        return evaluate_run(args)
    source = Path(args.resume_from) if args.resume_from else None
    baseline_source = source / "baseline" if source else Path(args.baseline_run)
    baseline_source, base_config, splits = validate_baseline(baseline_source)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir or Path("runs") / "mesh_geometry_correction" / stamp).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new, empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("geometry_correction")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "training.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    run, succeeded = None, False
    try:
        snapshot = output_dir / "baseline"
        snapshot.mkdir()
        for name in BASELINE_FILES:
            shutil.copy2(baseline_source / name, snapshot / name)
        hashes = {name: file_sha256(snapshot / name) for name in BASELINE_FILES}
        for name in ("splits.json", "scalers.joblib", "prediction_times.npy"):
            shutil.copy2(snapshot / name, output_dir / name)
        saved_resume = json.loads((source / "config.json").read_text(encoding="utf-8")) if source else None
        if source and not args.data_root:
            args.data_root = saved_resume.get("data_root")
        preprocessor = data_preprocessor(base_config, args.data_root)
        cache_directory = args.neighbor_cache_dir or str(Path(preprocessor.config.inp_dir).parent / "neighbor_graphs")
        baseline = HistoryPredictor.from_run(snapshot, device=args.device, neighbor_cache_dir=cache_directory)
        _configure_baseline(baseline, args, preprocessor)
        set_seed(args.seed)
        train_data = None
        if source:
            for name in ("geometry_basis.npz", "geometry_basis.json", "correction_best.pt",
                         "train_cache.npz", "validation_cache.npz", "train_geometry_diagnostics.json",
                         "validation_geometry_diagnostics.json"):
                shutil.copy2(source / name, output_dir / name)
            basis = GeometryCorrectionBasis.load(output_dir / "geometry_basis.npz")
            kwargs = saved_resume["architecture"]["kwargs"]
            for name, expected in saved_resume["baseline"]["artifact_sha256"].items():
                if hashes.get(name) != expected:
                    raise ValueError("Resume baseline snapshot differs from the source run")
        else:
            logger.info("Loading training designs only to fit the geometry basis")
            train_data = load_selected_data(preprocessor, splits["train"]["run_numbers"])
            meshes, labels = _geometry_state(train_data, baseline, preprocessor)
            designs = sorted(meshes)
            start = baseline.model.impactor_nodes
            basis = GeometryCorrectionBasis.fit([meshes[d][start:] for d in designs], designs,
                                                 part_labels=[labels[d] for d in designs],
                                                 anchor_spacing_mm=args.anchor_spacing_mm, profile_k=args.profile_k,
                                                 noise_floor_mm=args.geometry_noise_floor_mm)
            basis.save(output_dir / "geometry_basis.npz")
            write_json(output_dir / "geometry_basis.json", dict(config=basis.config,
                       training_design_ids=list(basis.training_design_ids), coefficient_width=basis.coefficient_width,
                       num_anchors=len(basis.anchors_mm), fit_on="training_geometry_only"))
            kwargs = dict(coefficient_width=basis.coefficient_width, width=args.correction_width,
                          layers=args.correction_layers, fourier_num_frequencies=args.fourier_num_frequencies,
                          fourier_min_frequency_hz=args.fourier_min_frequency_hz,
                          fourier_max_frequency_hz=args.fourier_max_frequency_hz)
        if list(basis.training_design_ids) != sorted(splits["train"]["design_ids"]):
            raise ValueError("Basis fitted design IDs do not match the saved training split")
        model = CoefficientHistoryCorrection(**kwargs).to(args.device)
        nyquist = .5 / float(np.diff(baseline.time_points).max())
        if float(model.frequencies_hz.max()) >= nyquist:
            raise ValueError(f"Correction Fourier frequencies must stay below the sampled Nyquist ({nyquist:g} Hz)")
        saved = dict(format_version=1, architecture=dict(name="CoefficientHistoryCorrection", kwargs=kwargs),
                     baseline=dict(source_run=str(baseline_source), snapshot="baseline", frozen=True,
                                   artifact_sha256=hashes), geometry_basis=dict(file="geometry_basis.npz",
                     sha256=file_sha256(output_dir / "geometry_basis.npz"), fit_on="train_only",
                     training_design_ids=list(basis.training_design_ids), config=basis.config),
                     data=base_config["data"], preprocessing=base_config["preprocessing"],
                     prediction_grid=base_config["prediction_grid"], location_filter=base_config.get("location_filter"),
                     data_root=args.data_root, training=dict(num_epochs=args.epochs, batch_size=args.batch_size,
                     learning_rate=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                     optimizer="AdamW", scheduler="CosineAnnealingLR", loss="normalized_acceleration_mse",
                     checkpoint_selection="validation_acceleration_mse", baseline_updated=False,
                     test_loaded_during_training=False, resume_from=args.resume_from),
                     correction_fourier=dict(frequencies_hz=model.frequencies_hz.cpu().tolist(), time_units="seconds"),
                     output_dir=str(output_dir))
        write_json(output_dir / "config.json", saved)
        logger.info("Geometry basis: %s training designs | %s anchors | %s coefficients",
                    len(basis.training_design_ids), len(basis.anchors_mm), basis.coefficient_width)
        logger.info("Baseline frozen: %s | correction trainable parameters: %s",
                    baseline_source, sum(p.numel() for p in model.parameters()))
        run = initialize_wandb(args, saved, output_dir, logger)
        run.define_metric("train/design_difference_skill", step_metric="epoch")
        train = prepare_split("train", splits["train"]["run_numbers"], preprocessor, baseline, basis,
                              output_dir, logger, data=train_data)
        del train_data
        validation = prepare_split("validation", splits["validation"]["run_numbers"], preprocessor, baseline,
                                   basis, output_dir, logger)
        resume_checkpoint = source / "correction_last.pt" if source else None
        train_correction(model, baseline, train, validation, args, output_dir, logger, run, resume_checkpoint)
        # These calls occur only after validation selected and loaded the best checkpoint.
        test = prepare_split("test", splits["test"]["run_numbers"], preprocessor, baseline, basis,
                             output_dir, logger)
        metrics = evaluate_cache(model, baseline, test, args, output_dir, "test", preprocessor.config.samples_per_design)
        for split, cache in (("train", train), ("validation", validation)):
            evaluate_cache(model, baseline, cache, args, output_dir, split, preprocessor.config.samples_per_design)
        logs = {}
        for arm in ("baseline", "corrected"):
            for kind in ("acceleration", "hic15"):
                for key, value in metrics[arm][kind].items():
                    if isinstance(value, (int, float)):
                        logs[f"test/{arm}/{kind}/{key}"] = value
            for key in ("design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine", "difference_gain"):
                value = metrics[arm]["design_sensitivity"].get(key)
                if value is not None:
                    logs[f"test/{arm}/design_sensitivity/{key}"] = value
                hic_value = metrics[arm]["design_sensitivity"]["hic15"].get(key)
                if hic_value is not None:
                    logs[f"test/{arm}/design_sensitivity/hic15/{key}"] = hic_value
        run.summary.update(logs)
        run.log(logs)
        final_hashes = {name: file_sha256(snapshot / name) for name in BASELINE_FILES}
        if final_hashes != hashes or any(p.requires_grad or p.grad is not None for p in baseline.model.parameters()):
            raise RuntimeError("Frozen baseline integrity check failed")
        write_json(output_dir / "baseline_integrity.json", dict(unchanged=True, artifact_sha256=hashes,
                   trainable_parameters=0, optimizer_contains_baseline=False))
        write_json(output_dir / "data_access_audit.json", dict(basis_fit_design_ids=list(basis.training_design_ids),
                   training_run_numbers=splits["train"]["run_numbers"],
                   validation_run_numbers=splits["validation"]["run_numbers"],
                   test_run_numbers=splits["test"]["run_numbers"], test_access="after_best_checkpoint_selection"))
        logger.info("Finished: %s", output_dir)
        succeeded = True
        return metrics
    finally:
        if run is not None:
            run.finish(exit_code=0 if succeeded else 1)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
