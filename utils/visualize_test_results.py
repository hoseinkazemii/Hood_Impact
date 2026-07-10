import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_validation_mse(save_dir: str = "./plots"):
    df = pd.read_csv("training_log.csv")
    epochs = df["epoch"]
    plt.figure()
    plt.plot(epochs, df["train_mse_norm"], label="Train MSE")
    plt.plot(epochs, df["val_mse_norm"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training vs Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_validation_mse.png"), dpi=200)
    plt.show()


def plot_single_run(csv_path: str, save_dir: str, show: bool = False):
    """
    Plot ground truth vs DeepONet prediction for a single test run.
    """
    df = pd.read_csv(csv_path)

    time = df["Time"].values
    y_true = df["A_true_g"].values
    y_pred = df["A_pred_g"].values
    error = y_pred - y_true

    rmse = np.sqrt(np.mean(error ** 2))
    run_id = os.path.basename(csv_path).split("_")[2]

    fig, axs = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Acceleration ---
    axs[0].plot(time, y_true, label="Ground Truth", linewidth=2)
    axs[0].plot(time, y_pred, label="DeepONet Prediction", linestyle="--")
    axs[0].set_ylabel("Acceleration (g)")
    axs[0].set_title(f"Test Run {run_id} | RMSE = {rmse:.2f} g")
    axs[0].legend()
    axs[0].grid(True)

    # --- Error ---
    axs[1].plot(time, error, color="red")
    axs[1].axhline(0.0, linestyle="--", linewidth=1)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error (g)")
    axs[1].grid(True)

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"run_{run_id}_deeponet_result.png")
    plt.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return rmse


def plot_multiple_runs(csv_files, save_dir, max_plots=10):
    """
    Plot up to max_plots runs and report RMSE statistics.
    """
    rmses = []

    for i, csv_path in enumerate(csv_files[:max_plots]):
        rmse = plot_single_run(csv_path, save_dir, show=False)
        rmses.append(rmse)

    print(f"Plotted {len(rmses)} runs")
    print(f"RMSE (g): mean={np.mean(rmses):.2f}, std={np.std(rmses):.2f}, max={np.max(rmses):.2f}")


def plot_rmse_histogram(csv_files, save_dir):
    """
    Plot histogram of RMSE across all test runs.
    """
    rmses = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        error = df["A_pred_g"].values - df["A_true_g"].values
        rmses.append(np.sqrt(np.mean(error ** 2)))

    plt.figure(figsize=(7, 4))
    plt.hist(rmses, bins=20, edgecolor="black")
    plt.xlabel("RMSE (g)")
    plt.ylabel("Number of Test Runs")
    plt.title("DeepONet Test RMSE Distribution")
    plt.grid(True)

    out_path = os.path.join(save_dir, "deeponet_test_rmse_histogram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved RMSE histogram to {out_path}")


def main():
    os.makedirs("./plots", exist_ok=True)

    csv_files = sorted(
        glob.glob(os.path.join(".", "test_run_*_prediction.csv"))
    )

    if len(csv_files) == 0:
        raise RuntimeError("No test_run_*_prediction.csv files found.")

    print(f"Found {len(csv_files)} test prediction files")

    # 1. Plot individual runs
    plot_multiple_runs(csv_files, "./plots", max_plots=10)

    # 2. Plot RMSE histogram
    plot_rmse_histogram(csv_files, "./plots")

    # 3. Plot training vs validation MSE
    plot_training_validation_mse()

if __name__ == "__main__":
    main()
