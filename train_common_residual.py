"""Train the common-mode + design-residual surrogate (common_residual.py) on the 1704 set.

Split as every recent run: test = cluster B (designs 4, 5), validation = design
11, training = the other nine designs, all 142 impact locations. A batch holds
``--locations-per-batch`` impact locations with every training design at each
(or a random ``--designs-per-location`` of them), so the pair term always has
same-location design pairs. The checkpoint is selected on design 11 with the
absolute + residual terms.

Outputs (in --output-dir):
    config.json, splits.json, training_history.csv, best_model.pt, last_model.pt
    metrics.json            absolute / residual / HIC metrics per split, pair-group
                            difference R2, the common-only baseline
    design_difference_r2.json   the approved near-clone difference R2 (train/test),
                            written exactly as train_mesh_impact_history.py does
    predictions.npz         total, common and residual curves (g), 12 x 142 x 63
    {train,test}_acceleration_histories.csv   in the main trainer's format

Usage:
    python train_common_residual.py                                  # DeltaAI defaults
    python train_common_residual.py --lambda-pair 0                  # ablation
    python train_common_residual.py --resume runs/common_residual/<run>
    python train_common_residual.py --max-locations 4 --epochs 1 --neighborhood-layers 0 \
        --num-latents 16 --width 32 --wandb-mode disabled            # smoke test
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

import design_difference_data as dd
from common_residual import (
    DEVIATION_CHANNELS, LOSS_TERMS, CommonResidualNet, deviation_features, family_references, hic_pair_weights,
    reformulation_losses,
)
from hic15 import batched_hic

DEFAULT_ROOT = dd.CODE_ROOT / "runs" / "common_residual"
RESIDUAL_DEFAULTS = {
    # The no-location-shortcut configuration of run 20261016_151923 (k256 FiLM
    # neighborhoods, Fourier time). The rows hold no headform, so impactor_nodes
    # is 0 and nothing location-specific enters pooling.
    "width": 128, "num_heads": 4, "num_latents": 256, "latent_layers": 3, "temporal_layers": 2,
    "dropout": 0.1, "neighborhood_layers": 2, "neighborhood_k": 256, "neighborhood_scale_mm": 20.0,
    "neighborhood_chunk_size": 1024, "impactor_nodes": 0, "impact_conditioning": "film",
    "time_encoding": "fourier", "pooling_query_residual": False, "pool_impactor_nodes": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="run directory to continue from its last checkpoint")
    parser.add_argument("--test-designs", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--val-designs", type=int, nargs="+", default=[11])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--locations-per-batch", type=int, default=1)
    parser.add_argument("--designs-per-location", type=int, default=0, help="0 = every training design")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    for term, default in (("abs", 1.0), ("common", 1.0), ("delta", 1.0), ("pair", 1.0), ("hic", 0.0)):
        parser.add_argument(f"--lambda-{term}", type=float, default=default)
    parser.add_argument("--pair-weighting", choices=["hic_variance", "none"], default="hic_variance")
    parser.add_argument("--pair-weight-max", type=float, default=5.0)
    for key, default in RESIDUAL_DEFAULTS.items():
        if key in ("impactor_nodes", "impact_conditioning", "time_encoding", "pooling_query_residual",
                   "pool_impactor_nodes"):
            continue
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--common-width", type=int, default=128)
    parser.add_argument("--max-locations", type=int, help="use only the first N locations (smoke tests)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE") or "online")
    # Same project as MeshImpactHistoryNet runs, so the curves sit side by side.
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-mesh-attention")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if args.resume is not None:
        saved = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))["arguments"]
        for key, value in saved.items():
            if key not in ("resume", "output_dir", "device", "wandb_mode", "epochs"):
                setattr(args, key, Path(value) if key == "data_dir" else value)
        args.output_dir = args.resume
    return args


class Data:
    """Everything the model reads, on one device, plus the numpy truth."""

    def __init__(self, args, device):
        times, acceleration = dd.load_histories(args.data_dir)
        self.times, self.acceleration = dd.model_grid(times, acceleration)
        locations = np.arange(1, dd.SAMPLES_PER_DESIGN + 1)
        if args.max_locations:
            locations = locations[:args.max_locations]
        self.locations = locations
        designs = set(range(dd.NUM_DESIGNS))
        self.test, self.validation = sorted(args.test_designs), sorted(args.val_designs)
        if set(self.test) & set(self.validation) or not set(self.test) | set(self.validation) < designs:
            raise ValueError("test and validation designs must be disjoint and leave training designs")
        self.train = sorted(designs - set(self.test) - set(self.validation))

        train_curves = self.acceleration[self.train]
        self.mean_curve = train_curves.mean(axis=0)                     # a_bar_p, (142, T)
        self.mu, self.sigma = float(train_curves.mean()), float(train_curves.std())
        self.residual_scale = float(np.sqrt(np.mean((train_curves - self.mean_curve) ** 2)))
        self.hic = batched_hic(self.times, self.acceleration)          # (12, 142) on the model grid
        self.pair_weight = (hic_pair_weights(self.hic, self.train, args.pair_weight_max)
                            if args.pair_weighting == "hic_variance" else np.ones(dd.SAMPLES_PER_DESIGN))

        geometries = {d: dd.load_design_geometry(d, args.data_dir) for d in range(dd.NUM_DESIGNS)}
        self.family_of, self.references = family_references(geometries, self.train)
        reference_xyz = np.concatenate([r["xyz"] for r in self.references])
        self.mesh_mean, self.mesh_scale = reference_xyz.mean(axis=0), reference_xyz.std(axis=0)
        impact = dd.load_locations(args.data_dir)[["X1", "X2"]].to_numpy(float)
        self.impact_mean, self.impact_scale = impact.mean(axis=0), impact.std(axis=0)
        self.time_mean, self.time_scale = float(self.times.mean()), float(self.times.std())

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)
        self.reference_inputs = [tensor((r["xyz"] - self.mesh_mean) / self.mesh_scale) for r in self.references]
        self.design_inputs = [torch.cat((self.reference_inputs[self.family_of[d]],
                                         tensor(deviation_features(geometries[d], self.references[self.family_of[d]]))),
                                        dim=1) for d in range(dd.NUM_DESIGNS)]
        self.deviation_rms_mm = {d: float(np.sqrt(np.mean(np.sum(
            (geometries[d]["xyz"] - self.references[self.family_of[d]]["xyz"]) ** 2, axis=1)))) for d in range(12)}
        self.impact = tensor((impact - self.impact_mean) / self.impact_scale)
        self.time_seconds = tensor(self.times)
        self.time_normalized = tensor((self.times - self.time_mean) / self.time_scale)
        self.target = tensor(self.acceleration)
        self.mean_target = tensor(self.mean_curve)
        self.weight = tensor(self.pair_weight)
        self.scales = {"mu": self.mu, "sigma": self.sigma, "residual": self.residual_scale}

    def inputs(self, designs, locations):
        """Model inputs for parallel lists of designs and 1-based locations."""
        meshes = [self.design_inputs[d] for d in designs]
        mesh = torch.cat(meshes)
        mesh_batch = torch.repeat_interleave(torch.arange(len(meshes), device=mesh.device),
                                             torch.tensor([len(m) for m in meshes], device=mesh.device))
        location_index = torch.as_tensor(np.asarray(locations) - 1, device=mesh.device)
        design_index = torch.as_tensor(np.asarray(designs), device=mesh.device)
        return (mesh, mesh_batch, self.impact[location_index], self.time_normalized, self.time_seconds), \
            self.target[design_index, location_index], self.mean_target[location_index], location_index


def build_model(args, data, device):
    residual_kwargs = {**RESIDUAL_DEFAULTS, **{key: getattr(args, key) for key in RESIDUAL_DEFAULTS
                                               if hasattr(args, key)}}
    model = CommonResidualNet(residual_kwargs, common_width=args.common_width).to(device)
    model.residual.set_coordinate_scalers(data.mesh_mean, data.mesh_scale, data.impact_mean, data.impact_scale)
    model.residual.set_time_scaler(data.time_mean, data.time_scale)
    if model.residual.neighborhood_blocks:
        from mesh_neighborhood import NeighborGraphCache
        model.residual.neighbor_cache = NeighborGraphCache(directory=Path(args.data_dir) / "neighbor_graphs")
        for reference, normalized in zip(data.references, data.reference_inputs):
            model.residual.neighbor_cache.register_geometry(
                reference["xyz"], normalized * model.residual.mesh_scale, model.residual.neighborhood_k)
    return model, residual_kwargs


def predict_all(model, data, designs, batch_size=8):
    """(len(designs), len(locations), T) total, common, residual in g."""
    model.eval()
    pairs = [(d, p) for d in designs for p in data.locations]
    total, common_g, residual_g = [], [], []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            inputs, _, _, _ = data.inputs([d for d, _ in chunk], [p for _, p in chunk])
            common, residual = model(*inputs)
            common = data.mu + data.sigma * common
            residual = data.residual_scale * residual
            total.append((common + residual).cpu().numpy())
            common_g.append(common.cpu().numpy())
            residual_g.append(residual.cpu().numpy())
    shape = (len(designs), len(data.locations), len(data.times))
    return tuple(np.concatenate(part).reshape(shape) for part in (total, common_g, residual_g))


def r2(prediction, truth):
    prediction, truth = np.ravel(prediction), np.ravel(truth)
    denominator = np.sum((truth - truth.mean()) ** 2)
    return float(1 - np.sum((prediction - truth) ** 2) / denominator) if denominator > 0 else float("nan")


def evaluate(total, common, residual, data):
    """Metrics on every design: absolute, residual, HIC, pair-group difference R2, baselines."""
    take = data.locations - 1
    truth = data.acceleration[:, take]
    mean = data.mean_curve[take]
    metrics = {}
    for split, designs in (("train", data.train), ("validation", data.validation), ("test", data.test)):
        a, p, c = truth[designs], total[designs], common[designs]
        hic_true, hic_pred = data.hic[designs][:, take], batched_hic(data.times, p)
        metrics[split] = {
            "rmse_g": float(np.sqrt(np.mean((p - a) ** 2))), "r2": r2(p, a),
            "residual_r2": r2(p - mean, a - mean),
            "residual_amplitude_ratio": float(np.sqrt(np.mean((p - mean) ** 2)) / np.sqrt(np.mean((a - mean) ** 2))),
            # The shape-level effect alone: residual averaged over the split's
            # designs. Undefined for training designs, whose mean residual is 0
            # by construction of the common curve.
            "split_mean_residual_r2": None if split == "train" else r2((p - mean).mean(0), (a - mean).mean(0)),
            "common_path_rmse_to_mean_g": float(np.sqrt(np.mean((c - mean) ** 2))),
            "hic_r2": r2(hic_pred, hic_true), "hic_mape_percent": float(100 * np.mean(np.abs(hic_pred - hic_true) / hic_true)),
            "baseline_common_only_rmse_g": float(np.sqrt(np.mean((mean - a) ** 2))),
            "baseline_common_only_hic_r2": r2(np.broadcast_to(batched_hic(data.times, mean), hic_true.shape), hic_true),
        }
    groups = {}
    for group, pairs in dd.PAIR_GROUPS.items():
        groups[group] = dd.difference_r2(np.stack([total[b] - total[a] for a, b in pairs]),
                                         np.stack([truth[b] - truth[a] for a, b in pairs]))
    metrics["near_clone_difference_r2"] = groups
    return metrics


def export_histories(path, data, designs, total):
    take = data.locations - 1
    rows = []
    for i, design in enumerate(designs):
        for j, location in enumerate(data.locations):
            rows.append(pd.DataFrame({"run_number": dd.run_number(design, int(location)), "time": data.times,
                                      "acceleration_true_g": data.acceleration[design, location - 1],
                                      "acceleration_pred_g": total[design, j]}))
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"common_residual_pair{args.lambda_pair:g}_seed{args.seed}"
    output_dir = args.output_dir or DEFAULT_ROOT / f"{stamp}_{os.environ.get('SLURM_JOB_ID', 'local')}_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("common_residual")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.FileHandler(output_dir / "training.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = Data(args, device)
    model, residual_kwargs = build_model(args, data, device)
    weights = {term: getattr(args, f"lambda_{term}") for term in LOSS_TERMS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    arguments = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "format_version": 1, "architecture": {"name": "CommonResidualNet", "residual_kwargs": residual_kwargs,
                                              "common_width": args.common_width,
                                              "deviation_channels": list(DEVIATION_CHANNELS)},
        "arguments": arguments,
        "data": {"data_format": "euroncap1704", "samples_per_design": dd.SAMPLES_PER_DESIGN,
                 "num_samples": dd.NUM_DESIGNS * dd.SAMPLES_PER_DESIGN, "data_dir": str(args.data_dir)},
        "splits": {"train": data.train, "validation": data.validation, "test": data.test},
        "references": [{"designs": r["designs"], "nodes": len(r["ids"])} for r in data.references],
        "deviation_rms_mm": data.deviation_rms_mm,
        "scales": data.scales, "loss_weights": weights,
        "pair_weighting": {"scheme": args.pair_weighting, "max": args.pair_weight_max,
                           "min_weight": float(data.pair_weight.min()), "max_weight": float(data.pair_weight.max())},
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    dd.write_json(output_dir / "config.json", config)
    dd.write_json(output_dir / "splits.json", config["splits"])
    log.info("CommonResidualNet: %s parameters | train %s | val %s | test %s | %s locations | loss weights %s",
             config["parameters"], data.train, data.validation, data.test, len(data.locations), weights)
    log.info("Scales: mu %.2f g, sigma %.2f g, residual %.2f g | deviation RMS from reference (mm): %s",
             data.mu, data.sigma, data.residual_scale, {d: round(v, 2) for d, v in data.deviation_rms_mm.items()})

    start_epoch, best, history = 1, math.inf, []
    if args.resume is not None:
        state = torch.load(output_dir / "last_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch, best, history = state["epoch"] + 1, state["best"], state["history"]
        log.info("Resumed after epoch %s (best validation %.4f)", state["epoch"], best)
    wandb_run = dd.start_wandb(args.wandb_mode, args.wandb_project, args.wandb_entity, name, config, output_dir, log)
    rng = np.random.default_rng(args.seed + start_epoch)
    per_location = args.designs_per_location or len(data.train)
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            order = rng.permutation(data.locations)
            sums, count = {term: 0.0 for term in ("total",) + LOSS_TERMS}, 0
            for start in range(0, len(order), args.locations_per_batch):
                designs, locations = [], []
                for location in order[start:start + args.locations_per_batch]:
                    chosen = rng.choice(data.train, size=min(per_location, len(data.train)), replace=False)
                    designs += chosen.tolist()
                    locations += [int(location)] * len(chosen)
                inputs, target, mean, group = data.inputs(designs, locations)
                common, residual = model(*inputs)
                loss, terms, _ = reformulation_losses(common, residual, target, mean, group, data.weight[group],
                                                      data.scales, weights, data.time_seconds)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sums["total"] += float(loss.detach())
                for term, value in terms.items():
                    sums[term] += value
                count += 1
            scheduler.step()
            validation = validation_loss(model, data, weights)
            record = {"epoch": epoch, **{f"train_{k}": v / count for k, v in sums.items()}, **validation}
            history.append(record)
            if validation["val_selection"] < best:
                best = validation["val_selection"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, **validation},
                           output_dir / "best_model.pt")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best": best,
                        "history": history}, output_dir / "last_model.pt")
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/loss": record["train_total"], "val/loss": validation["val_selection"],
                               "lr": scheduler.get_last_lr()[0],
                               **{f"train/{t}": record[f"train_{t}"] for t in LOSS_TERMS},
                               **{f"val/{k[4:]}": v for k, v in validation.items()}})
            log.info("epoch %3d | train %.4f (abs %.4f com %.4f delta %.4f pair %.4f hic %.4f) | val abs %.4f "
                     "delta %.4f | best %.4f | %.0f s", epoch, record["train_total"], record["train_abs"],
                     record["train_common"], record["train_delta"], record["train_pair"], record["train_hic"],
                     validation["val_abs"], validation["val_delta"], best, time.perf_counter() - started)

        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        total, common, residual = predict_all(model, data, list(range(dd.NUM_DESIGNS)))
        np.savez_compressed(output_dir / "predictions.npz", times=data.times, locations=data.locations,
                            total=total, common=common, residual=residual,
                            truth=data.acceleration[:, data.locations - 1], mean_curve=data.mean_curve[data.locations - 1])
        metrics = evaluate(total, common, residual, data)
        metrics["best_epoch"] = checkpoint["epoch"]
        dd.write_json(output_dir / "metrics.json", metrics)
        write_difference_report(output_dir, data, total)
        for split in ("train", "test"):
            designs = data.train if split == "train" else data.test
            export_histories(output_dir / f"{split}_acceleration_histories.csv", data, designs, total)
        log.info("Best epoch %s. Metrics:\n%s", checkpoint["epoch"], json.dumps(metrics, indent=2))
        if wandb_run is not None:
            flat = {f"{split}/{key}": value for split, values in metrics.items() if isinstance(values, dict)
                    for key, value in values.items()}
            wandb_run.summary.update({k: v for k, v in flat.items() if isinstance(v, float) and np.isfinite(v)})
        log.info("Finished: %s", output_dir)
        return metrics
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def validation_loss(model, data, weights):
    """Absolute and residual terms on the validation designs (no pairs: one design)."""
    total, common, residual = predict_all(model, data, data.validation)
    take = data.locations - 1
    truth = data.acceleration[data.validation][:, take]
    mean = data.mean_curve[take]
    values = {"val_abs": float(np.mean(((total - truth) / data.sigma) ** 2)),
              "val_delta": float(np.mean(((residual - (truth - mean)) / data.residual_scale) ** 2)),
              "val_common": float(np.mean(((common - mean) / data.sigma) ** 2))}
    values["val_selection"] = weights["abs"] * values["val_abs"] + weights["delta"] * values["val_delta"]
    return values


def write_difference_report(output_dir, data, total):
    """design_difference_r2.json exactly as train_mesh_impact_history.py writes it."""
    from evaluate_design_difference_r2 import design_difference_report, write_design_difference_report

    splits = {}
    for split, designs in (("train", data.train), ("test", data.test)):
        runs, times, predictions, targets = [], [], [], []
        for i, design in enumerate(designs):
            for j, location in enumerate(data.locations):
                runs.append(dd.run_number(design, int(location)))
                times.append(data.times)
                predictions.append(total[design, j])
                targets.append(data.acceleration[design, location - 1])
        dataset = SimpleNamespace(run_numbers=runs, time_arrays=times)
        splits[split] = (dataset, np.concatenate(predictions), np.concatenate(targets))
    report = design_difference_report("euroncap1704", dd.SAMPLES_PER_DESIGN, splits)
    write_design_difference_report(output_dir, report)
    return report


if __name__ == "__main__":
    main()
