import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from numba import njit

max_window = 0.015  # seconds for HIC15


@njit
def compute_hic_numba(time, acc_g, max_window):
    n = len(time)

    # cumulative trapezoidal integral
    cum_int = np.zeros(n)
    for i in range(1, n):
        cum_int[i] = (
            cum_int[i-1]
            + 0.5 * (acc_g[i] + acc_g[i-1]) * (time[i] - time[i-1])
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


def compute_hic_fast(time, acc_g, max_window=0.015):
    """
    Fast HIC computation using cumulative integration.
    """
    # cumulative trapezoidal integral
    cum_int = np.zeros_like(acc_g)
    cum_int[1:] = np.cumsum(
        0.5 * (acc_g[1:] + acc_g[:-1]) * np.diff(time)
    )

    hic_max = 0.0
    n = len(time)

    j = 1
    for i in range(n):
        while j < n and (time[j] - time[i]) <= max_window:
            j += 1

        for k in range(i + 1, j):
            dt = time[k] - time[i]
            if dt <= 0:
                continue

            a_avg = (cum_int[k] - cum_int[i]) / dt
            hic = dt * (a_avg ** 2.5)

            if hic > hic_max:
                hic_max = hic

    return hic_max


hic_reference = pd.read_csv("HIC_values_SAE1000.csv")
hic_reference = hic_reference.set_index("Job ID")["HIC Value"]


derived_hic = {}
experiment_ids = []

for exp_id in hic_reference.index:
    fname = f"HoodImpactor_{exp_id}_SAE1000.csv"
    if not os.path.exists(fname):
        print(f"Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    time = df["Time"].values
    acc_g = df["A(in g)"].values

    print(f"Computing HIC for Experiment {exp_id}...")
    hic_val = compute_hic_numba(time, acc_g, max_window=max_window)
    derived_hic[exp_id] = hic_val
    experiment_ids.append(exp_id)

comparison = pd.DataFrame({
    "Experiment": experiment_ids,
    "HIC_Reported": [hic_reference[i] for i in experiment_ids],
    "HIC_Derived": [derived_hic[i] for i in experiment_ids],
})

comparison["Relative Error (%)"] = (
    100 * (comparison["HIC_Derived"] - comparison["HIC_Reported"])
    / comparison["HIC_Reported"]
)


plt.figure()
plt.scatter(
    comparison["HIC_Reported"],
    comparison["HIC_Derived"],
)
plt.plot(
    [comparison["HIC_Reported"].min(), comparison["HIC_Reported"].max()],
    [comparison["HIC_Reported"].min(), comparison["HIC_Reported"].max()],
    linestyle="--",
)
plt.xlabel("Reported HIC")
plt.ylabel("Derived HIC (from acceleration)")
plt.title("Reported vs Derived HIC")
plt.grid(True)
plt.savefig("hic_reported_vs_derived.png")
plt.show()


plt.figure()
plt.plot(
    comparison["Experiment"],
    comparison["Relative Error (%)"],
    marker="o"
)
plt.axhline(0, linestyle="--")
plt.xlabel("Experiment ID")
plt.ylabel("Relative Error (%)")
plt.title("HIC Relative Error by Experiment")
plt.grid(True)
plt.savefig("hic_relative_error.png")
plt.show()


print(
    comparison
    .sort_values("Relative Error (%)", key=np.abs, ascending=False)
    .head(5)
)
