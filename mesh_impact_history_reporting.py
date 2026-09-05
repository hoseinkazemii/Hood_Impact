"""Export loss plots and recover saved mesh-history metrics into W&B.

By default this command only writes the missing local training-history plot.
Pass --upload-wandb to publish the saved epoch losses and final test metrics as
an explicitly labelled recovered run. It does not retrain or invent telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _loss_arrays(history):
    train = np.asarray(history["train_losses"], dtype=np.float64)
    validation = np.asarray(history["val_losses"], dtype=np.float64)
    if train.ndim != 1 or not train.size or train.shape != validation.shape:
        raise ValueError("Train and validation losses must be nonempty vectors of equal length")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Train and validation losses must be finite")
    return train, validation


def load_history(run_dir):
    """Load the original epoch history; CSV supports older exported runs too."""
    run_dir = Path(run_dir)
    json_path = run_dir / "training_history.json"
    if json_path.is_file():
        history = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        frame = pd.read_csv(run_dir / "training_history.csv")
        if not np.array_equal(frame["epoch"].to_numpy(), np.arange(1, len(frame) + 1)):
            raise ValueError("Saved epochs must be consecutive and start at 1")
        history = {
            "train_losses": frame["train_mse_normalized"].tolist(),
            "val_losses": frame["validation_mse_normalized"].tolist(),
        }
    _loss_arrays(history)
    return history


def export_training_history_plot(history, output_dir):
    """Save a plot of measured train/validation MSE, marking the best epoch."""
    train, validation = _loss_arrays(history)
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(validation))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_history.png"
    fig, ax = plt.subplots(figsize=(9, 5.3))
    try:
        ax.plot(epochs, train, color="#1965b0", linewidth=1.8, label="Training")
        ax.plot(epochs, validation, color="#dc7014", linewidth=1.8, label="Validation")
        ax.axvline(best + 1, color="#686868", linestyle="--", linewidth=1,
                   label=f"Best validation: epoch {best + 1} ({validation[best]:.5f})")
        ax.scatter([best + 1], [validation[best]], color="#dc7014", s=28, zorder=3)
        ax.set(xlabel="Epoch", ylabel="Normalized acceleration MSE",
               title="MeshImpactHistoryNet — training history")
        ax.set_xlim(1 if len(train) > 1 else 0.5, len(train) if len(train) > 1 else 1.5)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def upload_saved_run(run_dir, project=None, entity=None, run_name=None):
    """Publish only metrics actually present in the saved results directory.

    The original config and checkpoints are kept as evidence of the original
    run. A separate wandb_backfill.json records this upload and its URL.
    """
    import wandb

    run_dir = Path(run_dir).resolve()
    history = load_history(run_dir)
    train, validation = _loss_arrays(history)
    saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    plot_path = export_training_history_plot(history, run_dir)
    original = saved_config.get("wandb", {})
    project = project or os.environ.get("WANDB_PROJECT") or original.get("project") or "hood-impact-mesh-attention"
    entity = entity or os.environ.get("WANDB_ENTITY") or original.get("entity")
    run_name = run_name or f"{original.get('run_name') or run_dir.name}_recovered"
    receipt_path = run_dir / "wandb_backfill.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    if receipt.get("status") == "complete":
        raise ValueError(f"This run has already been recovered to W&B: {receipt.get('url')}")
    if receipt and (receipt.get("project") != project or receipt.get("requested_entity") != entity):
        raise ValueError("An incomplete recovery exists for another project/entity; use its original destination")
    run_id = receipt.get("id") or wandb.util.generate_id()
    run = wandb.init(
        project=project, entity=entity, name=run_name, id=run_id, resume="allow",
        mode="online", dir=str(run_dir), job_type="historical_backfill",
        config={
            **saved_config,
            "recovery": {
                "source": run_dir.name,
                "original_wandb_mode": original.get("mode"),
                "available_epoch_metrics": ["train/loss", "val/loss"],
                "missing_epoch_metrics": ["val/mae_g", "lr", "original_wall_time"],
            },
        },
        notes="Recovered from saved epoch losses and test metrics after training; original runtime telemetry is unavailable.",
    )
    upload_complete = False
    try:
        receipt = {
            "id": run.id, "url": run.url, "project": run.project,
            "entity": run.entity, "requested_entity": entity,
            "source": run_dir.name, "status": "uploading",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        for epoch, (train_loss, val_loss) in enumerate(zip(train, validation), start=1):
            run.log({"epoch": epoch, "train/loss": float(train_loss), "val/loss": float(val_loss)}, step=epoch - 1)
        final_metrics = {
            f"test/{key}": float(value) for key, value in metrics.items()
            if value is not None and np.isfinite(value)
        }
        run.summary.update({
            **final_metrics, "best_val_loss": float(validation.min()),
            "best_epoch": int(validation.argmin()) + 1, "recovered_from_saved_results": True,
        })
        run.log({**final_metrics, "training/history": wandb.Image(str(plot_path))}, step=len(train))
        upload_complete = True
    finally:
        run.finish(exit_code=0 if upload_complete else 1)
    receipt["status"] = "complete"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--upload-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    plot = export_training_history_plot(load_history(args.run_dir), args.run_dir)
    print(f"Saved training history plot: {plot}")
    if args.upload_wandb:
        receipt = upload_saved_run(args.run_dir, args.wandb_project, args.wandb_entity, args.run_name)
        print(f"Recovered W&B run: {receipt['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
