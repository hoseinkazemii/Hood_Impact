import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data_dir = "."

# ===============================
# Relative error analysis: first vs second half
# ===============================

first_half_errors = []
second_half_errors = []

split_time = 0.0125  # seconds

for i in range(50):
    fname = f"test_sample_{i:02d}.csv"
    path = os.path.join(data_dir, fname)

    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    time = df["time"].values
    acc_gt = df["ground_truth"].values
    acc_pred = df["prediction"].values

    # Split indices
    first_mask = time <= split_time
    second_mask = time > split_time

    # Avoid empty segments
    if first_mask.sum() < 5 or second_mask.sum() < 5:
        continue

    # Relative L2 error
    def rel_l2(a, b):
        return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8)

    err_first = rel_l2(acc_pred[first_mask], acc_gt[first_mask])
    err_second = rel_l2(acc_pred[second_mask], acc_gt[second_mask])

    first_half_errors.append(err_first)
    second_half_errors.append(err_second)

first_half_errors = np.array(first_half_errors)
second_half_errors = np.array(second_half_errors)

print("\n=== Relative Error Summary ===")
print(f"First half  mean error: {first_half_errors.mean():.4f}")
print(f"Second half mean error: {second_half_errors.mean():.4f}")
print(f"Error ratio (second / first): {(second_half_errors.mean() / first_half_errors.mean()):.2f}x")


plt.figure(figsize=(6,6))
plt.boxplot(
    [first_half_errors, second_half_errors],
    labels=["First Half (0–0.0125s)", "Second Half (0.0125–0.025s)"],
    showmeans=True
)
plt.ylabel("Relative L2 Error")
plt.title("Model Error in Early vs Late Time Window")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("relative_error_first_vs_second_half_boxplot.png", dpi=300)
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(first_half_errors, second_half_errors, alpha=0.7)

max_val = max(first_half_errors.max(), second_half_errors.max())
plt.plot([0, max_val], [0, max_val], linestyle="--")

plt.xlabel("Relative Error — First Half")
plt.ylabel("Relative Error — Second Half")
plt.title("Per-Sample Error Degradation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("relative_error_first_vs_second_half_scatter.png", dpi=300)
plt.show()
