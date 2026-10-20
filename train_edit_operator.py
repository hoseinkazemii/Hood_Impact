"""Train the difference-first edit operator (edit_operator.py) on near-clone pairs.

Training data: the 12 near-clone pairs of families A and C, both directions,
at all 142 impact locations (3,408 difference curves). Checkpoint selection:
the validation pair (10, 11), an edit never seen in training. Test: (4, 5).

The target is the difference curve itself, normalized by the RMS of the
training differences, so 100% of the loss is spent on what the absolute-target
surrogates ignore. Reported with the approved metric (difference R2, a < b
orientation, 63-point grid), overall and by XY distance to the edit, with and
without the time-step nuisance term.

Usage:
    python train_edit_operator.py                                   # linear + dt nuisance
    python train_edit_operator.py --operator attention
    python train_edit_operator.py --operator none --nuisance dt     # time step only
    python train_edit_operator.py --context baseline                # + baseline response code
    python train_edit_operator.py --max-train-distance-mm 250       # near/mid edits only
"""
import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import design_difference_data as dd
from edit_operator import EDIT_CHANNELS, NODE_FEATURES, NUISANCES, OPERATORS, EditOperator

DEFAULT_ROOT = dd.CODE_ROOT / "runs" / "design_difference" / "edit_operator"
CONTEXTS = ("location", "baseline")
BASELINE_COMPONENTS = 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="defaults to runs/design_difference/edit_operator/<stamp>")
    parser.add_argument("--operator", choices=OPERATORS, default="linear")
    parser.add_argument("--nuisance", choices=NUISANCES, default="dt")
    parser.add_argument("--context", choices=CONTEXTS, default="location",
                        help="baseline adds a PCA code of the baseline design's own curve (breaks exact antisymmetry)")
    parser.add_argument("--max-train-distance-mm", type=float, default=math.inf,
                        help="train only on locations within this XY distance of the edit")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--basis-size", type=int, default=16)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--length-scale-mm", type=float, default=100.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE") or "online")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-design-difference")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


class PairData:
    """Padded per-pair edit arrays and per-sample indices, all on one device."""

    def __init__(self, args, device):
        times, acceleration = dd.load_histories(args.data_dir)
        self.times, acceleration = dd.model_grid(times, acceleration)
        runs = dd.load_run_table(args.data_dir)
        increments = dd.design_increments(runs)
        penetration = dd.per_run_array(runs, "sphere_pen_mm")
        in_contact = dd.per_run_array(runs, "in_contact").astype(float)
        surface_z = dd.per_run_array(runs, "surf_z_mm")
        impact_xy = dd.load_locations(args.data_dir)[["X1", "X2"]].to_numpy(float)
        geometry = {d: dd.load_design_geometry(d, args.data_dir) for d in range(dd.NUM_DESIGNS)}
        self.pairs = [pair for pairs in dd.PAIR_GROUPS.values() for pair in pairs]
        self.group = {pair: group for group, pairs in dd.PAIR_GROUPS.items() for pair in pairs}

        train_difference = dd.pair_differences(acceleration, dd.TRAIN_PAIRS).reshape(-1, len(self.times))
        self.target_scale = float(np.sqrt(np.mean(train_difference ** 2)))
        # Principal directions of the training differences start the temporal basis.
        _, singular, components = np.linalg.svd(train_difference / self.target_scale, full_matrices=False)
        self.basis = components[:args.basis_size] * (singular[:args.basis_size, None] / np.sqrt(len(train_difference)))
        baseline_pca = None
        if args.context == "baseline":
            from sklearn.decomposition import PCA
            baseline_pca = PCA(BASELINE_COMPONENTS, random_state=0).fit(
                acceleration[list(dd.TRAIN_DESIGNS)].reshape(-1, len(self.times)))
            codes = baseline_pca.transform(acceleration.reshape(-1, len(self.times)))
            codes = (codes / codes[:len(dd.TRAIN_DESIGNS)].std(axis=0)).reshape(dd.NUM_DESIGNS, dd.SAMPLES_PER_DESIGN, -1)

        nodes, features, edits, masks, points, rhos, distances, truths = [], [], [], [], [], [], [], []
        for a, b in self.pairs:
            pair = dd.pair_geometry(geometry[a], geometry[b])
            keep = pair["edit"]
            panel = np.eye(len(dd.PANELS))[pair["panel"][keep]]
            area = pair["area_mid"][keep] / max(pair["area_mid"].mean(), 1e-9)
            features.append(np.column_stack((panel, pair["normal_mid"][keep], pair["gap_mid"][keep] / 50.0, area)))
            edits.append(np.column_stack((pair["dx"][keep], pair["dgap"][keep])) / 10.0)
            nodes.append(pair["x_mid"][keep])
            point = np.column_stack((impact_xy, surface_z[a]))
            points.append(point)
            centre = runs[runs.design == a].sort_values("loc")[["X1", "X2", "X3"]].to_numpy()
            distances.append(dd.edit_distance(pair, centre)[0])
            rhos.append(dd.log_dt_ratio(increments, a, b))
            truths.append(acceleration[b] - acceleration[a])
        width = max(len(n) for n in nodes)
        pad = lambda arrays, dim: np.stack([np.pad(x, ((0, width - len(x)), (0, 0))) for x in arrays])  # noqa: E731
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
        self.nodes = tensor(pad(nodes, 3))
        self.features = tensor(pad(features, NODE_FEATURES))
        self.edits = tensor(pad(edits, EDIT_CHANNELS))
        self.masks = tensor(np.stack([np.arange(width) < len(n) for n in nodes]), torch.bool)
        self.points = tensor(np.stack(points))
        self.rho = tensor(rhos)
        self.distance = np.stack(distances)
        self.truth = np.stack(truths)  # (P, 142, T) g, a < b orientation
        self.edit_counts = [len(n) for n in nodes]

        base_context = np.stack([np.column_stack((impact_xy / 1000.0, penetration[a], in_contact[a]))
                                 for a, _ in self.pairs])  # (P, 142, 4)
        self.context = {1: tensor(base_context), -1: tensor(base_context)}
        if baseline_pca is not None:
            for sign in (1, -1):
                base = [a if sign == 1 else b for a, b in self.pairs]
                self.context[sign] = tensor(np.concatenate((base_context, codes[base]), axis=2))
        self.context_dim = self.context[1].shape[-1]
        self.target = tensor(self.truth / self.target_scale)

    def samples(self, pairs, signs=(1, -1), max_distance=math.inf):
        """Long tensor (S, 3) of (pair index, location index, sign)."""
        rows = []
        for pair in pairs:
            p = self.pairs.index(pair)
            for location in np.nonzero(self.distance[p] <= max_distance)[0]:
                rows += [(p, location, sign) for sign in signs]
        return torch.as_tensor(rows, dtype=torch.long, device=self.nodes.device)

    def batch(self, samples):
        p, location, sign = samples[:, 0], samples[:, 1], samples[:, 2]
        relative = self.nodes[p] - self.points[p, location][:, None, :]
        direction = sign.to(self.nodes.dtype)
        context = torch.where((sign == 1)[:, None], self.context[1][p, location], self.context[-1][p, location])
        return (relative, self.features[p], self.edits[p] * direction[:, None, None], self.masks[p], context,
                self.rho[p] * direction), self.target[p, location] * direction[:, None]


def predict(model, data, samples, batch_size=512):
    model.eval()
    geometric, nuisance = [], []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            inputs, _ = data.batch(samples[start:start + batch_size])
            g, n = model(*inputs)
            geometric.append(g)
            nuisance.append(n)
    return torch.cat(geometric).cpu().numpy() * data.target_scale, torch.cat(nuisance).cpu().numpy() * data.target_scale


def evaluate(model, data):
    """Difference R2 per group and band, full prediction and geometric term only."""
    rows, curves = [], {}
    for pair in data.pairs:
        samples = data.samples([pair], signs=(1,))
        geometric, nuisance = predict(model, data, samples)
        curves[pair] = {"geometric": geometric, "full": geometric + nuisance}
    for group, pairs in dd.PAIR_GROUPS.items():
        truth = np.stack([data.truth[data.pairs.index(p)] for p in pairs])
        bands = np.stack([dd.distance_band(data.distance[data.pairs.index(p)]) for p in pairs])
        for term in ("full", "geometric"):
            prediction = np.stack([curves[p][term] for p in pairs])
            for band in ("all", "near", "mid", "far"):
                mask = np.ones(bands.shape, bool) if band == "all" else bands == band
                if mask.any():
                    rows.append({"group": group, "term": term, "band": band, "n": int(mask.sum()),
                                 "difference_r2": dd.difference_r2(prediction[mask], truth[mask]),
                                 "amplitude_ratio": float(np.sqrt(np.mean(prediction[mask] ** 2))
                                                          / np.sqrt(np.mean(truth[mask] ** 2)))})
    return pd.DataFrame(rows), curves


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"edit_{args.operator}_{args.nuisance}_{args.context}_seed{args.seed}"
    output_dir = args.output_dir or DEFAULT_ROOT / f"{stamp}_{os.environ.get('SLURM_JOB_ID', 'local')}_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("edit_operator")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.FileHandler(output_dir / "training.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = PairData(args, device)
    model = EditOperator(len(data.times), data.context_dim, args.operator, args.nuisance, args.width,
                         args.basis_size, attention_layers=args.attention_layers,
                         length_scale_mm=args.length_scale_mm, dropout=args.dropout).to(device)
    model.set_temporal_basis(torch.as_tensor(data.basis, dtype=torch.float32, device=device))
    train = data.samples(dd.TRAIN_PAIRS, max_distance=args.max_train_distance_mm)
    validation = data.samples(dd.VALIDATION_PAIRS)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "output_dir": str(output_dir), "target_scale_g": data.target_scale,
              "train_samples": len(train), "edit_nodes_per_pair": dict(zip(map(str, data.pairs), data.edit_counts)),
              "parameters": sum(p.numel() for p in model.parameters())}
    if not math.isfinite(args.max_train_distance_mm):
        config["max_train_distance_mm"] = None
    dd.write_json(output_dir / "config.json", config)
    log.info("Edit operator %s | nuisance %s | context %s | %s train samples | %s parameters | target scale %.2f g",
             args.operator, args.nuisance, args.context, len(train), config["parameters"], data.target_scale)
    wandb_run = dd.start_wandb(args.wandb_mode, args.wandb_project, args.wandb_entity, name, config, output_dir, log)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best, history, started = math.inf, [], time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = train[torch.randperm(len(train), generator=generator).to(device)]
            total = 0.0
            for start in range(0, len(order), args.batch_size):
                inputs, target = data.batch(order[start:start + args.batch_size])
                geometric, nuisance = model(*inputs)
                loss = torch.mean((geometric + nuisance - target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(target)
            scheduler.step()
            geometric, nuisance = predict(model, data, validation)
            _, target = data.batch(validation)
            val_loss = float(np.mean(((geometric + nuisance) / data.target_scale - target.cpu().numpy()) ** 2))
            history.append({"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                           output_dir / "best_model.pt")
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/loss": total / len(train), "val/loss": val_loss,
                               "lr": scheduler.get_last_lr()[0]})
            if epoch == 1 or epoch % max(1, args.epochs // 20) == 0:
                log.info("epoch %4d | train %.4f | val %.4f | best %.4f | %.0f s", epoch, total / len(train),
                         val_loss, best, time.perf_counter() - started)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        # The last epoch shows capacity (how well the edits seen in training are
        # fitted); the validation-selected checkpoint shows transfer.
        last, _ = evaluate(model, data)
        last.to_csv(output_dir / "difference_r2_last_epoch.csv", index=False)
        log.info("Last epoch, difference R2 (all locations):\n%s", last[last.band == "all"].pivot_table(
            index="term", columns="group", values="difference_r2").round(4).to_string())
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        table, curves = evaluate(model, data)
        table.to_csv(output_dir / "difference_r2.csv", index=False)
        np.savez_compressed(output_dir / "predictions.npz", times=data.times,
                            pairs=np.asarray(data.pairs), truth=data.truth,
                            full=np.stack([curves[p]["full"] for p in data.pairs]),
                            geometric=np.stack([curves[p]["geometric"] for p in data.pairs]))
        pivot = table[table.band == "all"].pivot_table(index="term", columns="group", values="difference_r2")
        log.info("Best epoch %s. Difference R2 (all locations):\n%s", checkpoint["epoch"], pivot.round(4).to_string())
        test = table[(table.group == "test")].pivot_table(index="term", columns="band", values="difference_r2")
        log.info("Test pair (4, 5) by XY distance to the edit:\n%s", test.round(4).to_string())
        metrics = {f"{r.group}/{r.term}/{r.band}/difference_r2": r.difference_r2 for r in table.itertuples()}
        metrics["best_epoch"] = checkpoint["epoch"]
        dd.write_json(output_dir / "metrics.json", metrics)
        if wandb_run is not None:
            wandb_run.summary.update({k: v for k, v in metrics.items() if np.isfinite(v)})
        log.info("Finished: %s", output_dir)
        return table
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
