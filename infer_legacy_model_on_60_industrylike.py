"""
Run the BEST legacy acceleration model (trained on 950 samples, 19 designs x 50)
as a pure inference model on the NEW industry-like dataset
(HoodImpact_60_IndustryLike, 60 samples) treated entirely as a test set.

For every one of the 60 new samples we:
  1. predict the acceleration history with the frozen legacy model,
  2. compute HIC (HIC15, trapezoidal integration) from BOTH the predicted and
     the ground-truth acceleration using the exact formula in the repo's
     visualize_hic_pred_vs_gt.py,
  3. export per-sample CSVs + plots and aggregate metrics.

IMPORTANT design choices (all faithful to "use the old model as-is"):
  * The legacy SCALERS (fit on the 950-sample training set) are loaded and
    reused. We do NOT refit on the new data -- that is the whole point of an
    out-of-the-box transfer test.
  * The legacy model was trained on sequences sampled at dt ~= 1.17e-5 s over
    [0, 0.025] s (~2140 points). The new files are interp1000 (1000 points,
    dt ~= 2.5e-5 s). We resample each new ground-truth history onto the legacy
    resolution grid so the model receives its training-time sampling density.
    (SAE1000-filtered signals are smooth, so linear up-sampling is lossless.)
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from numba import njit
from sklearn.metrics import mean_squared_error, r2_score

# repo imports
import torch.nn as nn
from utils.utils import Config, DataPreprocessor
from temporal_deeponet import PointNetEncoder, FiLMLayer, TrunkNetwork, OutputNetwork


class LegacyHoodImpactNeuralOperator(nn.Module):
    """Replica of the architecture used for the 20260116_160228_best checkpoint.

    Differs from the current temporal_deeponet.HoodImpactNeuralOperator only in
    that it uses a SINGLE FiLM layer (the checkpoint has `film_layer.*`, not
    `film_layer1/2.*`). Everything else is identical.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.mesh_encoder = PointNetEncoder(
            pointnet_input_dim=config.pointnet_input_dim,
            pointnet_hidden_dims=config.pointnet_hidden_dims,
            pointnet_output_dim=config.pointnet_output_dim,
        )
        self.film_layer = FiLMLayer(
            film_feature_dim=config.pointnet_output_dim,
            film_condition_dim=config.film_condition_dim,
            film_hidden_dim=config.film_hidden_dim,
        )
        self.branch_transform = nn.Sequential(
            nn.Linear(config.pointnet_output_dim, config.pointnet_output_dim),
            nn.LayerNorm(config.pointnet_output_dim),
            nn.GELU(),
        )
        self.trunk = TrunkNetwork(
            trunk_output_dim=config.trunk_output_dim,
            num_frequencies=config.num_fourier_frequencies,
            trunk_hidden_dims=config.trunk_hidden_dims,
            num_tcn_layers=config.num_tcn_layers,
            dropout=0.1,
        )
        self.output_net = OutputNetwork(
            operator_head_feature_dim=config.trunk_output_dim,
            operator_head_hidden_dims=config.operator_head_hidden_dims,
            dropout=0.1,
        )

    def forward(self, mesh, mesh_batch, indentor, time, time_batch, batch_size=1):
        branch = self.mesh_encoder(mesh, mesh_batch, indentor, batch_size)
        branch = self.film_layer(branch, indentor)
        branch = self.branch_transform(branch)
        trunk = self.trunk(time, time_batch, batch_size)
        branch_expanded = branch[time_batch]
        combined = branch_expanded * trunk
        return self.output_net(combined)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BEST_RUN_DIR = "./runs/acceleration_history_target_value/20260116_160228_best"
CKPT_PATH = os.path.join(BEST_RUN_DIR, "hood_impact_best_model.pt")
SCALER_PATH = os.path.join(BEST_RUN_DIR, "scalers.joblib")

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./runs/legacy_model_on_60_industrylike_{STAMP}"
CSV_DIR = os.path.join(OUTPUT_DIR, "test_predictions", "test_samples_csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "test_predictions", "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

N_SAMPLES = 60
LEGACY_GRID_POINTS = 2140          # matches legacy stride-16 resolution
MAX_WINDOW = 0.015                  # HIC15 (seconds)
SAMPLES_PER_DESIGN = 5             # new dataset: 12 designs x 5 impact locations


# ----------------------------------------------------------------------------
# HIC (identical formula to repo's visualize_hic_pred_vs_gt.py)
# ----------------------------------------------------------------------------
@njit
def compute_hic_numba(time, acc_g, max_window):
    n = len(time)
    cum_int = np.zeros(n)
    for i in range(1, n):
        cum_int[i] = cum_int[i - 1] + 0.5 * (acc_g[i] + acc_g[i - 1]) * (time[i] - time[i - 1])

    hic_max = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dt = time[j] - time[i]
            if dt > max_window:
                break
            if dt <= 0:
                continue
            a_avg = (cum_int[j] - cum_int[i]) / dt
            hic = dt * (a_avg ** 2.5)
            if hic > hic_max:
                hic_max = hic
    return hic_max


def build_config() -> Config:
    """Config with the EXACT architecture of the saved best model and the
    industry-like data paths. Architecture dims recovered from the checkpoint."""
    cfg = Config(
        data_format="industrylike",
        prediction_target="acceleration",
        # ---- architecture (must match the checkpoint) ----
        pointnet_input_dim=3,
        pointnet_hidden_dims=[64, 128, 128, 128, 256],
        pointnet_output_dim=128,
        trunk_hidden_dims=[64, 128, 256],   # <- legacy used 256 (not 128)
        trunk_output_dim=128,
        num_fourier_frequencies=8,
        num_tcn_layers=5,
        film_condition_dim=2,
        film_hidden_dim=64,
        operator_head_hidden_dims=[256, 256],
        # ---- data handling ----
        time_subsample_stride=1,            # keep native 1000 pts; we resample ourselves
        max_train_time=None,
        output_dir=OUTPUT_DIR,
    )
    return cfg


def main():
    cfg = build_config()
    device = cfg.device
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    # --- preprocessor with LEGACY scalers (do NOT refit) ---
    # NOTE: the legacy scalers.joblib predates the 'hic' scaler key, so we
    # assign the dict entries directly instead of using pre.load_scalers().
    import joblib
    pre = DataPreprocessor(cfg)
    sc = joblib.load(SCALER_PATH)
    pre.mesh_scaler = sc["mesh"]
    pre.indentor_scaler = sc["indentor"]
    pre.time_scaler = sc["time"]
    pre.accel_scaler = sc["accel"]
    pre._fitted = True
    print("Loaded legacy scalers (fit on 950-sample training set):")
    print(f"  mesh     mean={pre.mesh_scaler.mean_}  scale={pre.mesh_scaler.scale_}")
    print(f"  indentor mean={pre.indentor_scaler.mean_}  scale={pre.indentor_scaler.scale_}")
    print(f"  time     mean={pre.time_scaler.mean_}  scale={pre.time_scaler.scale_}")
    print(f"  accel    mean={pre.accel_scaler.mean_}  scale={pre.accel_scaler.scale_}")

    # --- model ---
    model = LegacyHoodImpactNeuralOperator(cfg).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])   # strict=True -> verifies arch
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded best model ({n_params:,} params) from {CKPT_PATH}")

    # --- official HIC values shipped with the new dataset (sanity check) ---
    official_hic = pre.load_hic()  # columns: Job ID, HIC Value
    official_map = dict(zip(official_hic["Job ID"].astype(int), official_hic["HIC Value"].astype(float)))

    impact_df = pre.load_impact_coords()

    rows = []          # per-sample summary
    all_pred, all_gt = [], []   # pooled acceleration points for global metrics

    for run in range(1, N_SAMPLES + 1):
        # ---- inputs ----
        mesh = pre.load_mesh_geometry(run)                 # (N,3) from .inp
        pos = impact_df.iloc[run - 1]
        ind_x, ind_y = float(pos["X1"]), float(pos["X2"])  # codebase convention

        # ---- native ground-truth acceleration (1000 pts) ----
        t_native, a_native = pre.load_acceleration_history(run)  # stride=1 -> native

        # ---- resample onto legacy-resolution grid ----
        grid = np.linspace(float(t_native[0]), float(t_native[-1]), LEGACY_GRID_POINTS).astype(np.float32)
        gt = np.interp(grid, t_native, a_native).astype(np.float32)

        # ---- predict ----
        with torch.no_grad():
            mesh_norm = pre.transform_mesh(mesh)
            ind_norm = pre.transform_indentor(np.array([ind_x, ind_y], dtype=np.float32))
            time_norm = pre.transform_time(grid)

            mesh_t = torch.from_numpy(mesh_norm).to(device)
            mesh_b = torch.zeros(len(mesh_norm), dtype=torch.long, device=device)
            ind_t = torch.from_numpy(ind_norm).unsqueeze(0).to(device)
            time_t = torch.from_numpy(time_norm).to(device)
            time_b = torch.zeros(len(time_norm), dtype=torch.long, device=device)

            pred_norm = model(mesh_t, mesh_b, ind_t, time_t, time_b, batch_size=1)
            pred = pre.inverse_transform_acceleration(pred_norm.cpu().numpy()).astype(np.float32)

        # ---- per-sample metrics ----
        mse = float(np.mean((pred - gt) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred - gt)))
        denom = float(np.sum((gt - gt.mean()) ** 2))
        r2 = float(1.0 - np.sum((gt - pred) ** 2) / denom) if denom > 0 else float("nan")

        hic_gt = float(compute_hic_numba(grid.astype(np.float64), gt.astype(np.float64), MAX_WINDOW))
        hic_pred = float(compute_hic_numba(grid.astype(np.float64), pred.astype(np.float64), MAX_WINDOW))
        hic_official = official_map.get(run, np.nan)

        all_pred.append(pred)
        all_gt.append(gt)

        # ---- per-sample CSV (legacy column layout) ----
        design_id = (run - 1) // SAMPLES_PER_DESIGN
        df = pd.DataFrame({
            "run_number": run,
            "design_id": design_id,
            "time": grid,
            "ground_truth": gt,
            "prediction": pred,
            "error": pred - gt,
        })
        df.to_csv(os.path.join(CSV_DIR, f"test_sample_{run-1:02d}.csv"), index=False)

        # ---- per-sample plot ----
        plt.figure(figsize=(10, 4))
        plt.plot(grid, gt, label="Ground Truth", linewidth=2)
        plt.plot(grid, pred, "--", label="Prediction", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.title(f"Run {run} (design {design_id}) | MAE={mae:.2f} g | "
                  f"HIC gt={hic_gt:.0f} pred={hic_pred:.0f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, f"test_sample_{run-1:02d}.png"), dpi=120, bbox_inches="tight")
        plt.close()

        rows.append({
            "run_number": run,
            "design_id": design_id,
            "acc_mse": mse, "acc_rmse": rmse, "acc_mae": mae, "acc_r2": r2,
            "hic_ground_truth": hic_gt,
            "hic_prediction": hic_pred,
            "hic_official_dataset": hic_official,
            "hic_abs_error": abs(hic_pred - hic_gt),
        })
        print(f"run {run:2d} | MAE={mae:7.2f} g | R2={r2:6.3f} | "
              f"HIC gt={hic_gt:8.1f} pred={hic_pred:8.1f} (official {hic_official:8.1f})")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "per_sample_metrics.csv"), index=False)

    pooled_pred = np.concatenate(all_pred)
    pooled_gt = np.concatenate(all_gt)
    g_mse = float(np.mean((pooled_pred - pooled_gt) ** 2))
    g_rmse = float(np.sqrt(g_mse))
    g_mae = float(np.mean(np.abs(pooled_pred - pooled_gt)))
    g_r2 = float(1.0 - np.sum((pooled_gt - pooled_pred) ** 2) /
                 np.sum((pooled_gt - pooled_gt.mean()) ** 2))

    hic_gt_arr = summary["hic_ground_truth"].values
    hic_pred_arr = summary["hic_prediction"].values
    hic_off_arr = summary["hic_official_dataset"].values
    hic_rmse = float(np.sqrt(mean_squared_error(hic_gt_arr, hic_pred_arr)))
    hic_r2 = float(r2_score(hic_gt_arr, hic_pred_arr))
    hic_mae = float(np.mean(np.abs(hic_pred_arr - hic_gt_arr)))
    # sanity: how well does our integrated GT HIC match the dataset's official HIC?
    mask = ~np.isnan(hic_off_arr)
    gt_vs_official_r2 = float(r2_score(hic_off_arr[mask], hic_gt_arr[mask])) if mask.sum() > 1 else float("nan")

    # HIC pred-vs-gt scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(hic_gt_arr, hic_pred_arr, alpha=0.8)
    lo = min(hic_gt_arr.min(), hic_pred_arr.min())
    hi = max(hic_gt_arr.max(), hic_pred_arr.max())
    plt.plot([lo, hi], [lo, hi], "--", linewidth=2)
    plt.xlabel("Ground Truth HIC")
    plt.ylabel("Predicted HIC")
    plt.title("Legacy model on 60 industry-like samples: HIC pred vs GT")
    plt.text(0.05, 0.95, f"R2 = {hic_r2:.4f}\nRMSE = {hic_rmse:.1f}",
             transform=plt.gca().transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hic_pred_vs_gt.png"), dpi=200)
    plt.close()

    summary_txt = [
        "=" * 70,
        "LEGACY MODEL  ->  60 INDUSTRY-LIKE SAMPLES (all test)",
        "=" * 70,
        f"Checkpoint: {CKPT_PATH}",
        f"Scalers   : {SCALER_PATH}  (fit on the 950-sample training set)",
        f"Samples   : {N_SAMPLES}",
        "",
        "--- Acceleration (pooled over all points & samples) ---",
        f"  MSE  : {g_mse:.4f}",
        f"  RMSE : {g_rmse:.4f} g",
        f"  MAE  : {g_mae:.4f} g",
        f"  R^2  : {g_r2:.4f}",
        "",
        "--- Per-sample acceleration R^2 ---",
        f"  mean : {summary['acc_r2'].mean():.4f}",
        f"  median: {summary['acc_r2'].median():.4f}",
        f"  min / max: {summary['acc_r2'].min():.4f} / {summary['acc_r2'].max():.4f}",
        "",
        "--- HIC (predicted vs ground-truth, both via integration formula) ---",
        f"  R^2  : {hic_r2:.4f}",
        f"  RMSE : {hic_rmse:.1f}",
        f"  MAE  : {hic_mae:.1f}",
        "",
        "--- Sanity: integrated GT HIC vs dataset's official HIC ---",
        f"  R^2  : {gt_vs_official_r2:.4f}",
        "=" * 70,
    ]
    summary_str = "\n".join(summary_txt)
    print("\n" + summary_str)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
