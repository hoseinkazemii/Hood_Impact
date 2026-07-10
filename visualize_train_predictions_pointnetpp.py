import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pointnetpp import HoodImpactNeuralOperator
from utils.utils import (
    Config,
    DataPreprocessor,
    Evaluator,
    create_data_loaders,
    run_to_design_id,
    set_seed,
)


DEFAULT_RUN_DIR = Path(
    r"./runs/acceleration_history_target_value/"
    r"20260406_031209_10train_10test_pointnetpp"
)
DEFAULT_TEST_DESIGNS = [1, 2, 6, 7, 8, 9, 11, 15, 17, 19]
DEFAULT_VAL_DESIGN = 12


def parse_design_ids(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def train_split_run_numbers(
    run_numbers: List[int],
    test_design_ids: List[int],
    val_design_id: int,
    samples_per_design: int,
) -> List[int]:
    train_runs = []
    for run_number in run_numbers:
        design_id = run_to_design_id(run_number, samples_per_design)
        if design_id in test_design_ids or design_id == val_design_id:
            continue
        train_runs.append(run_number)
    return train_runs


def load_pointnetpp_model(config: Config, checkpoint_path: Path) -> HoodImpactNeuralOperator:
    model = HoodImpactNeuralOperator(config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    # The model has one-time debug print blocks used during training.
    if hasattr(model, "trunk") and hasattr(model.trunk, "fourier"):
        model.trunk.fourier.debug = False
    model._debug_local_printed = 5

    return model


def evaluate_train_samples(
    evaluator: Evaluator,
    train_dataset,
    train_run_numbers: List[int],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    sample_count = len(train_dataset) if max_samples is None else min(max_samples, len(train_dataset))
    results = []

    for idx in range(sample_count):
        mesh = train_dataset.mesh_geometries[idx]
        indentor = train_dataset.indentor_positions[idx]
        time = train_dataset.time_arrays[idx]
        target = train_dataset.accelerations[idx]

        pred = evaluator.predict_single(mesh, indentor[0], indentor[1], time)

        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        run_number = train_run_numbers[idx]

        results.append(
            {
                "sample_index": idx,
                "run_number": run_number,
                "design_id": run_to_design_id(run_number),
                "time": time,
                "target": target,
                "prediction": pred,
                "mse": mse,
                "mae": mae,
            }
        )

    return results


def export_train_predictions(results: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for result in results:
        idx = result["sample_index"]
        stem = f"train_sample_{idx:02d}"

        csv_name = f"{stem}.csv"
        png_name = f"{stem}.png"
        csv_path = output_dir / csv_name
        png_path = output_dir / png_name

        time = result["time"]
        target = result["target"]
        pred = result["prediction"]

        pd.DataFrame(
            {
                "time": time,
                "ground_truth": target,
                "prediction": pred,
                "error": pred - target,
            }
        ).to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(time, target, label="Ground Truth", linewidth=2)
        plt.plot(time, pred, "--", label="Prediction", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Acceleration (g)")
        plt.title(f"Train Sample {idx:02d} | MAE = {result['mae']:.4f} g")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

        summary_rows.append(
            {
                "sample_index": idx,
                "run_number": result["run_number"],
                "design_id": result["design_id"],
                "mse": result["mse"],
                "mae": result["mae"],
                "csv_file": csv_name,
                "plot_file": png_name,
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "train_predictions_summary.csv", index=False)


def summarize_metrics(results: List[Dict]) -> Dict[str, float]:
    predictions = np.concatenate([result["prediction"] for result in results])
    targets = np.concatenate([result["target"] for result in results])

    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predictions - targets)))
    r2 = float(
        1.0
        - np.sum((targets - predictions) ** 2)
        / np.sum((targets - np.mean(targets)) ** 2)
    )

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained PointNet++ hood-impact checkpoint and export "
            "acceleration-history prediction plots/CSVs for the training split."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scalers", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--test-design-ids",
        type=parse_design_ids,
        default=DEFAULT_TEST_DESIGNS,
        help="Comma-separated design IDs used as the test split during training.",
    )
    parser.add_argument("--val-design-id", type=int, default=DEFAULT_VAL_DESIGN)
    parser.add_argument("--samples-per-design", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick checks. Omit to export all training samples.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_dir = args.run_dir
    checkpoint_path = args.checkpoint or run_dir / "hood_impact_best_model.pt"
    scalers_path = args.scalers or run_dir / "scalers.joblib"
    output_dir = args.output_dir or run_dir / "train_predictions"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers file not found: {scalers_path}")

    config = Config(
        output_dir=str(run_dir),
        device=resolve_device(args.device),
        prediction_target="acceleration",
    )
    set_seed(config.seed)

    print(f"[INFO] Loading data using device: {config.device}")
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.load_all_data()
    preprocessor.load_scalers(str(scalers_path))

    train_loader, _, _, _ = create_data_loaders(
        data_dict,
        preprocessor,
        config,
        test_design_ids=args.test_design_ids,
        val_design_id=args.val_design_id,
        samples_per_design=args.samples_per_design,
    )
    train_dataset = train_loader.dataset
    train_run_numbers = train_split_run_numbers(
        data_dict["run_numbers"],
        args.test_design_ids,
        args.val_design_id,
        args.samples_per_design,
    )

    if len(train_dataset) != len(train_run_numbers):
        raise RuntimeError(
            "Training dataset size does not match computed training run numbers: "
            f"{len(train_dataset)} != {len(train_run_numbers)}"
        )

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = load_pointnetpp_model(config, checkpoint_path)
    evaluator = Evaluator(model, config, preprocessor)

    print(f"[INFO] Evaluating {len(train_dataset) if args.max_samples is None else min(args.max_samples, len(train_dataset))} training samples")
    results = evaluate_train_samples(
        evaluator,
        train_dataset,
        train_run_numbers,
        max_samples=args.max_samples,
    )

    print(f"[INFO] Exporting training predictions to: {output_dir}")
    export_train_predictions(results, output_dir)

    metrics = summarize_metrics(results)
    print("[INFO] Train metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"[INFO] Exported {len(results)} training-sample plots + CSVs.")


if __name__ == "__main__":
    main()
