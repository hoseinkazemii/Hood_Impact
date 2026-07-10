import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Load direct HIC predictions
# ===============================

csv_path = "hic_test_predictions.csv"
assert os.path.exists(csv_path), f"Missing file: {csv_path}"

df = pd.read_csv(csv_path)

# Required columns
required_cols = {
    "run_number",
    "hic_ground_truth",
    "hic_prediction",
    "error",
    "abs_error",
}
missing = required_cols - set(df.columns)
assert not missing, f"Missing columns in CSV: {missing}"

hic_gt = df["hic_ground_truth"].values
hic_pred = df["hic_prediction"].values
error = df["error"].values
abs_error = df["abs_error"].values
run_numbers = df["run_number"].values

# ===============================
# Metrics
# ===============================

rmse = np.sqrt(mean_squared_error(hic_gt, hic_pred))
r2 = r2_score(hic_gt, hic_pred)

print("Direct HIC regression metrics")
print("-" * 40)
print(f"RMSE (HIC): {rmse:.3f}")
print(f"R²   (HIC): {r2:.4f}")

# ===============================
# Scatter: Predicted vs Ground Truth
# ===============================

plt.figure(figsize=(7, 7))
plt.scatter(hic_gt, hic_pred, alpha=0.8)

min_val = min(hic_gt.min(), hic_pred.min())
max_val = max(hic_gt.max(), hic_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Ground Truth HIC")
plt.ylabel("Predicted HIC")
plt.title("Predicted vs Ground Truth HIC (Direct Regression)")

plt.text(
    0.05, 0.95,
    f"R² = {r2:.4f}\nRMSE = {rmse:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.grid(True)
plt.tight_layout()
plt.savefig("hic_pred_vs_gt_direct.png", dpi=300)
plt.show()

# ===============================
# Error plot: Absolute error per run
# ===============================

plt.figure(figsize=(10, 4))
plt.bar(run_numbers, abs_error, width=0.8)

plt.xlabel("Run Number")
plt.ylabel("Absolute HIC Error")
plt.title("Absolute HIC Prediction Error per Test Sample")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("hic_abs_error_per_sample.png", dpi=300)
plt.show()

# ===============================
# Error distribution (histogram)
# ===============================

plt.figure(figsize=(6, 4))
plt.hist(error, bins=20, edgecolor="black", alpha=0.8)

plt.xlabel("Prediction Error (Pred − GT)")
plt.ylabel("Count")
plt.title("Distribution of HIC Prediction Error")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hic_error_distribution.png", dpi=300)
plt.show()

print("\nSaved plots:")
print("  - hic_pred_vs_gt_direct.png")
print("  - hic_abs_error_per_sample.png")
print("  - hic_error_distribution.png")
