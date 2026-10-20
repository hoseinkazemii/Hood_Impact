"""Design-code oracle: can the surrogate's decoder separate near-clones at all?

Replaces the mesh encoder of MeshImpactHistoryNet with a lookup code and keeps
its decoder (HistoryDecoderBlock, Fourier time queries, acceleration head):

    none     impact location only            -> the per-location mean curve
    cluster  impact + 4-way shape-family code (A, B, C, D)
    design   impact + 12-way design code      -> knows every design exactly

All 12 designs train on the 71 even-parity checkerboard locations; the 71 odd
locations are never seen (design 11's odd locations select the checkpoint).
Difference R2 (design_difference_data.difference_r2) is then read for the
near-clone pairs at seen (even) and unseen (odd) locations.

How to read it:

* design code, seen locations, LAST epoch (``last_*`` columns): R2 near 1
  means the decoder can represent the near-clone difference once it is told
  which design it is looking at; a low value means the decoder or the
  objective is the bottleneck, not the geometry encoder. The checkpoint
  selected on unseen locations stops early and is not a capacity measure.
* design code, unseen locations: what a model that knows the designs perfectly
  achieves by interpolating across impact locations. Compare with the
  model-free ceiling of oracle_interpolation_ceiling.py.
* cluster vs none: the between-shape signal, which earlier runs show is
  learnable; the near-clone difference R2 of both must be ~0 by construction.

Usage:
    python oracle_design_code.py                       # 3 codes x 3 seeds
    python oracle_design_code.py --codes design --seeds 0 --epochs 50 --wandb-mode disabled
"""
import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import design_difference_data as dd
from mesh_impact_history import HistoryDecoderBlock, LatentMixingBlock

DEFAULT_OUT = dd.CODE_ROOT / "runs" / "design_difference" / "design_code_oracle"
CODES = {"none": 0, "cluster": 4, "design": dd.NUM_DESIGNS}
CLUSTER_OF_DESIGN = [list(dd.FAMILIES).index(dd.family_of(d)) for d in range(dd.NUM_DESIGNS)]


class DesignCodeHistoryNet(nn.Module):
    """Impact token + learned code tokens -> mixing -> the surrogate's decoder."""

    def __init__(self, num_codes, width=128, num_heads=4, code_tokens=8, mixing_layers=2,
                 decoder_layers=2, dropout=0.1, num_frequencies=6, min_frequency_hz=20.0,
                 max_frequency_hz=640.0):
        super().__init__()
        self.num_codes = num_codes
        self.impact_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        # Without a code every sample shares one token set: an impact-only model.
        self.code_tokens = nn.Parameter(0.02 * torch.randn(max(num_codes, 1), code_tokens, width))
        self.mixing = nn.ModuleList([LatentMixingBlock(width, num_heads, dropout) for _ in range(mixing_layers)])
        self.memory_norm = nn.LayerNorm(width)
        ratio = max_frequency_hz / min_frequency_hz
        frequencies = [min_frequency_hz * ratio ** (i / max(num_frequencies - 1, 1)) for i in range(num_frequencies)]
        self.register_buffer("frequencies_hz", torch.tensor(frequencies, dtype=torch.float32))
        self.time_embedding = nn.Sequential(nn.Linear(2 * num_frequencies, width), nn.GELU(),
                                            nn.Linear(width, width))
        self.decoder = nn.ModuleList([HistoryDecoderBlock(width, num_heads, dropout) for _ in range(decoder_layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, impact, code, time_seconds):
        """impact (B, 2) normalized, code (B,) long, time_seconds (T,) -> (B, T) normalized."""
        batch = impact.shape[0]
        index = code if self.num_codes else torch.zeros_like(code)
        tokens = torch.cat((self.impact_embedding(impact)[:, None], self.code_tokens[index]), dim=1)
        for block in self.mixing:
            tokens = block(tokens)
        memory = self.memory_norm(tokens)
        phases = 2 * math.pi * time_seconds[:, None] * self.frequencies_hz
        features = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2)
        queries = self.time_embedding(features)[None].expand(batch, -1, -1)
        padding = torch.zeros(queries.shape[:2], dtype=torch.bool, device=queries.device)
        for block in self.decoder:
            queries = block(queries, memory, padding)
        return self.head(queries).squeeze(-1)


def code_of(code_name, designs):
    designs = np.asarray(designs)
    if code_name == "cluster":
        return np.asarray(CLUSTER_OF_DESIGN)[designs]
    return designs if code_name == "design" else np.zeros_like(designs)


def train_one(code_name, seed, data, args, log, output_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    model = DesignCodeHistoryNet(CODES[code_name], width=args.width, dropout=args.dropout).to(device)
    acceleration, impact, time_seconds = data["acceleration"], data["impact"], data["time"]
    designs, locations = np.meshgrid(np.arange(dd.NUM_DESIGNS), np.arange(1, dd.SAMPLES_PER_DESIGN + 1),
                                     indexing="ij")
    designs, locations = designs.ravel(), locations.ravel()
    even = np.isin(locations, data["even"])
    train = np.nonzero(even)[0]
    validation = np.nonzero(~even & (designs == dd.VALIDATION_PAIRS[0][1]))[0]

    target = acceleration.reshape(-1, acceleration.shape[-1])
    mean, scale = target[train].mean(), target[train].std()
    impact_all = impact[locations - 1]
    impact_mean, impact_scale = impact_all[train].mean(0), impact_all[train].std(0)
    tensor = lambda array, dtype=torch.float32: torch.as_tensor(array, dtype=dtype, device=device)  # noqa: E731
    x = tensor((impact_all - impact_mean) / impact_scale)
    c = tensor(code_of(code_name, designs), torch.long)
    y = tensor((target - mean) / scale)
    t = tensor(time_seconds)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    run_name = f"design_code_{code_name}_seed{seed}"
    wandb_run = dd.start_wandb(args.wandb_mode, args.wandb_project, args.wandb_entity, run_name,
                               {"code": code_name, "seed": seed, **vars(args)}, output_dir, log)
    best, best_state, history = math.inf, None, []
    generator = torch.Generator().manual_seed(seed)
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.as_tensor(train)[torch.randperm(len(train), generator=generator)]
            total = 0.0
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size].to(device)
                loss = torch.mean((model(x[batch], c[batch], t) - y[batch]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(batch)
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_index = torch.as_tensor(validation, device=device)
                val_loss = torch.mean((model(x[val_index], c[val_index], t) - y[val_index]) ** 2).item()
            history.append({"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/loss": total / len(train), "val/loss": val_loss,
                               "lr": scheduler.get_last_lr()[0]})
            if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
                log.info("  %s seed %s epoch %4d | train %.4f | val %.4f", code_name, seed, epoch,
                         total / len(train), val_loss)
        def predict_all():
            model.eval()
            with torch.no_grad():
                output = torch.cat([model(x[i:i + 256], c[i:i + 256], t) for i in range(0, len(x), 256)])
            return (output.cpu().numpy() * scale + mean).reshape(acceleration.shape)

        # Last epoch = capacity on the seen locations; the checkpoint selected on
        # design 11's unseen locations = what transfers across locations.
        last = evaluate(predict_all(), acceleration, data["even"], data["odd"])
        model.load_state_dict(best_state)
        prediction = predict_all()
        metrics = evaluate(prediction, acceleration, data["even"], data["odd"])
        metrics.update({f"last_{key}": value for key, value in last.items()})
        metrics.update({"code": code_name, "seed": seed, "best_val_loss": best,
                        "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1)})
        if wandb_run is not None:
            wandb_run.summary.update({key: value for key, value in metrics.items()
                                      if isinstance(value, (int, float)) and np.isfinite(value)})
        np.save(output_dir / f"prediction_{run_name}.npy", prediction.astype(np.float32))
        pd.DataFrame(history).to_csv(output_dir / f"history_{run_name}.csv", index=False)
        return metrics
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def evaluate(prediction, truth, even, odd):
    metrics = {}
    for group, pairs in dd.PAIR_GROUPS.items():
        for name, locations in (("seen", even), ("unseen", odd)):
            take = locations - 1
            predicted = np.stack([prediction[b][take] - prediction[a][take] for a, b in pairs])
            actual = np.stack([truth[b][take] - truth[a][take] for a, b in pairs])
            metrics[f"{group}_{name}_difference_r2"] = dd.difference_r2(predicted, actual)
            metrics[f"{group}_{name}_amplitude_ratio"] = float(
                np.sqrt(np.mean(predicted ** 2)) / np.sqrt(np.mean(actual ** 2)))
    for name, locations in (("seen", even), ("unseen", odd)):
        take = locations - 1
        for label, designs in (("train_designs", dd.TRAIN_DESIGNS), ("test_designs", (4, 5))):
            error = prediction[list(designs)][:, take] - truth[list(designs)][:, take]
            metrics[f"{label}_{name}_rmse_g"] = float(np.sqrt(np.mean(error ** 2)))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--codes", nargs="+", default=list(CODES), choices=list(CODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE") or "online")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or "hood-impact-design-difference")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("design_code_oracle")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.FileHandler(args.out_dir / "design_code_oracle.log", mode="w"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)

    times, acceleration = dd.load_histories(args.data_dir)
    times, acceleration = dd.model_grid(times, acceleration)
    even, odd = dd.checkerboard_locations()
    data = {"acceleration": acceleration, "time": times,
            "impact": dd.load_locations(args.data_dir)[["X1", "X2"]].to_numpy(float), "even": even, "odd": odd}
    log.info("Device %s | %s codes x %s seeds | %s epochs | train: 12 designs x %s even locations",
             args.device, args.codes, args.seeds, args.epochs, len(even))
    rows = []
    for code_name in args.codes:
        for seed in args.seeds:
            started = time.perf_counter()
            metrics = train_one(code_name, seed, data, args, log, args.out_dir)
            rows.append(metrics)
            log.info("%s seed %s (%.0f s): difference R2 train pairs seen %.3f unseen %.3f | 4-5 seen %.3f "
                     "unseen %.3f | test-design RMSE unseen %.2f g | last epoch: train pairs seen %.3f, "
                     "4-5 seen %.3f unseen %.3f", code_name, seed, time.perf_counter() - started,
                     metrics["train_seen_difference_r2"], metrics["train_unseen_difference_r2"],
                     metrics["test_seen_difference_r2"], metrics["test_unseen_difference_r2"],
                     metrics["test_designs_unseen_rmse_g"], metrics["last_train_seen_difference_r2"],
                     metrics["last_test_seen_difference_r2"], metrics["last_test_unseen_difference_r2"])
            pd.DataFrame(rows).to_csv(args.out_dir / "design_code_oracle.csv", index=False)
    table = pd.DataFrame(rows)
    columns = ["train_seen_difference_r2", "train_unseen_difference_r2", "validation_unseen_difference_r2",
               "test_seen_difference_r2", "test_unseen_difference_r2", "train_designs_unseen_rmse_g",
               "test_designs_unseen_rmse_g", "last_train_seen_difference_r2", "last_train_unseen_difference_r2",
               "last_test_seen_difference_r2", "last_test_unseen_difference_r2", "last_train_designs_seen_rmse_g"]
    summary = table.groupby("code")[columns].agg(["mean", "std"])
    log.info("Mean (std) over seeds:\n%s", summary.round(3).to_string())
    dd.write_json(args.out_dir / "design_code_oracle.json",
                  {code: {column: {"mean": float(group[column].mean()), "std": float(group[column].std())}
                          for column in columns} for code, group in table.groupby("code")})
    return table


if __name__ == "__main__":
    main()
