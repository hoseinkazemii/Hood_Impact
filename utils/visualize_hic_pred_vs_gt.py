import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# HIC computation
# ===============================

max_window = 0.015  # seconds (HIC15)

@njit
def compute_hic_numba(time, acc_g, max_window):
    n = len(time)

    # cumulative trapezoidal integral
    cum_int = np.zeros(n)
    for i in range(1, n):
        cum_int[i] = (
            cum_int[i - 1]
            + 0.5 * (acc_g[i] + acc_g[i - 1]) * (time[i] - time[i - 1])
        )

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

# ===============================
# Load test samples and compute HIC
# ===============================

hic_gt = []
hic_pred = []
sample_ids = []

data_dir = Path(__file__).resolve().parent
sample_files = sorted(
    data_dir.glob("test_sample_*.csv"),
    key=lambda path: int(re.search(r"test_sample_(\d+)\.csv$", path.name).group(1)),
)
if not sample_files:
    data_dir = Path.cwd()
    sample_files = sorted(
        data_dir.glob("test_sample_*.csv"),
        key=lambda path: int(re.search(r"test_sample_(\d+)\.csv$", path.name).group(1)),
    )

if not sample_files:
    raise FileNotFoundError(f"No test_sample_*.csv files found in {data_dir}")

for path in sample_files:
    match = re.search(r"test_sample_(\d+)\.csv$", path.name)
    sample_id = int(match.group(1))
    df = pd.read_csv(path)

    time = df["time"].values
    acc_gt = df["ground_truth"].values
    acc_pred = df["prediction"].values

    print(f"Computing HIC for sample {sample_id:02d}...")
    hic_gt_val = compute_hic_numba(time, acc_gt, max_window)
    hic_pred_val = compute_hic_numba(time, acc_pred, max_window)

    hic_gt.append(hic_gt_val)
    hic_pred.append(hic_pred_val)
    sample_ids.append(sample_id)

hic_gt = np.array(hic_gt)
hic_pred = np.array(hic_pred)
df_results = pd.DataFrame({
    "sample_id": sample_ids,
    "hic_ground_truth": hic_gt,
    "hic_prediction": hic_pred
})
df_results.to_csv(data_dir / "hic_results_pred_vs_gt.csv", index=False)

# ===============================
# Metrics
# ===============================
print("Calculating metrics...")
rmse = np.sqrt(mean_squared_error(hic_gt, hic_pred))
r2 = r2_score(hic_gt, hic_pred)

print(f"RMSE (HIC): {rmse:.3f}")
print(f"R² (HIC): {r2:.4f}")

# ===============================
# Scatter plot
# ===============================

plt.figure(figsize=(7, 7))
plt.scatter(hic_gt, hic_pred, alpha=0.8)

min_val = min(hic_gt.min(), hic_pred.min())
max_val = max(hic_gt.max(), hic_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Ground Truth HIC")
plt.ylabel("Predicted HIC")
plt.title("Predicted vs Ground Truth HIC")

plt.text(
    0.05, 0.95,
    f"R² = {r2:.4f}\nRMSE = {rmse:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.grid(True)
plt.tight_layout()
plt.savefig(data_dir / "hic_pred_vs_gt.png", dpi=300)
plt.show()
