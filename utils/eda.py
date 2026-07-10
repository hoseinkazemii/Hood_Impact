import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "."  # change if needed
FILE_GLOB = os.path.join(DATA_DIR, "HoodImpactor_*_SAE1000.csv")

# Columns expected in each experiment file
COL_TIME = "Time"
COL_A1 = "A1"
COL_A2 = "A2"
COL_A3 = "A3"
COL_AG = "A(in g)"
COL_AMM = "A(mm/s2)"

# Resampling grid for aggregate waveform stats (seconds)
# Your signals appear to run ~0.025s, but we'll infer duration per file and clip.
RESAMPLE_T_START = 0.0
RESAMPLE_T_END = 0.025  # adjust if your records are longer/shorter
RESAMPLE_N = 2000       # points for resampling (controls smoothness/size)

# Waveform overlay sampling for inspection
N_OVERLAY = 40
RANDOM_SEED = 7

# HIC settings
COMPUTE_HIC = True
HIC_MAX_WINDOW = 0.015  # HIC15 (seconds)

# Output folder for figures + tables
OUT_DIR = os.path.join(DATA_DIR, "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Optional: fast HIC via numba
# -----------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

if COMPUTE_HIC and (not NUMBA_AVAILABLE):
    print("Numba not available. Set COMPUTE_HIC=False or install numba: pip install numba")

if COMPUTE_HIC and NUMBA_AVAILABLE:
    @njit
    def compute_hic_numba(time, acc_g, max_window):
        n = len(time)

        # cumulative trapezoidal integral of acc_g over time
        cum_int = np.zeros(n)
        for i in range(1, n):
            cum_int[i] = cum_int[i-1] + 0.5 * (acc_g[i] + acc_g[i-1]) * (time[i] - time[i-1])

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

# -----------------------------
# Helpers
# -----------------------------
def extract_experiment_id(path):
    # Extract integer between "HoodImpactor_" and "_SAE1000"
    m = re.search(r"HoodImpactor_(\d+)_SAE1000\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None

def safe_read_csv(path):
    df = pd.read_csv(path)
    # Minimal validation
    required = [COL_TIME, COL_A1, COL_A2, COL_A3, COL_AG]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df

def basic_features(time, a_g, a1, a2, a3):
    # time stats
    dt = np.diff(time)
    dt_pos = dt[dt > 0]

    duration = float(time[-1] - time[0]) if len(time) > 1 else np.nan
    dt_med = float(np.median(dt_pos)) if len(dt_pos) else np.nan
    dt_mean = float(np.mean(dt_pos)) if len(dt_pos) else np.nan
    dt_std = float(np.std(dt_pos)) if len(dt_pos) else np.nan

    # acceleration stats
    peak_g = float(np.max(a_g))
    peak_idx = int(np.argmax(a_g))
    t_peak = float(time[peak_idx])

    mean_g = float(np.mean(a_g))
    std_g = float(np.std(a_g))
    rms_g = float(np.sqrt(np.mean(a_g**2)))

    # impulse proxy: ∫ a(t) dt over record (g·s)
    impulse_gs = float(np.trapezoid(a_g, time))

    # "energy proxy": ∫ a(t)^2 dt (g^2·s)
    energy_g2s = float(np.trapezoid(a_g**2, time))

    # component summaries (useful for diagnosing coordinate conventions)
    peak_a1 = float(np.max(np.abs(a1)))
    peak_a2 = float(np.max(np.abs(a2)))
    peak_a3 = float(np.max(np.abs(a3)))

    return {
        "duration_s": duration,
        "dt_median_s": dt_med,
        "dt_mean_s": dt_mean,
        "dt_std_s": dt_std,
        "n_samples": int(len(time)),
        "peak_g": peak_g,
        "t_peak_s": t_peak,
        "mean_g": mean_g,
        "std_g": std_g,
        "rms_g": rms_g,
        "impulse_gs": impulse_gs,
        "energy_g2s": energy_g2s,
        "peak_abs_A1": peak_a1,
        "peak_abs_A2": peak_a2,
        "peak_abs_A3": peak_a3,
    }

def resample_to_grid(time, y, t_grid):
    # Linear interpolation; assumes time is increasing
    # If time has duplicates, this can misbehave; we guard by making time strictly increasing.
    time = np.asarray(time)
    y = np.asarray(y)

    # enforce strictly increasing time by small epsilon jitter on duplicates
    # (rare, but can happen with exported signals)
    eps = 1e-15
    for i in range(1, len(time)):
        if time[i] <= time[i-1]:
            time[i] = time[i-1] + eps

    return np.interp(t_grid, time, y, left=np.nan, right=np.nan)

# -----------------------------
# Load file list
# -----------------------------
files = sorted(glob.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched: {FILE_GLOB}")

print(f"Found {len(files)} files.")

# -----------------------------
# Main loop: extract per-file features
# -----------------------------
rows = []
failed = []

# For aggregate waveform statistics
t_grid = np.linspace(RESAMPLE_T_START, RESAMPLE_T_END, RESAMPLE_N)
resampled_ag = []  # list of arrays

for idx, path in enumerate(files, start=1):
    exp_id = extract_experiment_id(path)
    if exp_id is None:
        failed.append((path, "Could not parse experiment id"))
        continue

    try:
        df = safe_read_csv(path)

        time = df[COL_TIME].to_numpy(dtype=float)
        a1 = df[COL_A1].to_numpy(dtype=float)
        a2 = df[COL_A2].to_numpy(dtype=float)
        a3 = df[COL_A3].to_numpy(dtype=float)
        ag = df[COL_AG].to_numpy(dtype=float)

        feat = basic_features(time, ag, a1, a2, a3)

        # Optional: compute HIC15
        if COMPUTE_HIC and NUMBA_AVAILABLE:
            hic15 = float(compute_hic_numba(time, ag, HIC_MAX_WINDOW))
            feat["hic15_derived"] = hic15
        else:
            feat["hic15_derived"] = np.nan

        feat["experiment_id"] = exp_id
        feat["file"] = os.path.basename(path)

        rows.append(feat)

        # Resample for aggregate waveform envelope
        ag_r = resample_to_grid(time, ag, t_grid)
        resampled_ag.append(ag_r)

    except Exception as e:
        failed.append((path, str(e)))

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(files)} files...")

features_df = pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)
features_path = os.path.join(OUT_DIR, "per_experiment_features.csv")
features_df.to_csv(features_path, index=False)
print(f"Saved per-experiment feature table: {features_path}")

if failed:
    fail_df = pd.DataFrame(failed, columns=["file", "error"])
    fail_path = os.path.join(OUT_DIR, "failed_files.csv")
    fail_df.to_csv(fail_path, index=False)
    print(f"Some files failed. Saved log: {fail_path}")

# Convert resampled list to array for envelope stats
resampled_ag = np.vstack(resampled_ag) if len(resampled_ag) else np.empty((0, RESAMPLE_N))

# -----------------------------
# Visualizations (Dataset-level)
# -----------------------------
def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

# 1) Time discretization: dt distribution
plt.figure()
plt.hist(features_df["dt_median_s"].dropna(), bins=50)
plt.xlabel("Median dt per experiment (s)")
plt.ylabel("Count")
plt.title("Distribution of Median Time Step (dt)")
plt.grid(True)
savefig("dt_median_distribution.png")
plt.close()

plt.figure()
plt.hist(features_df["duration_s"].dropna(), bins=50)
plt.xlabel("Duration per experiment (s)")
plt.ylabel("Count")
plt.title("Distribution of Record Duration")
plt.grid(True)
savefig("duration_distribution.png")
plt.close()

# 2) Acc magnitude distributions
plt.figure()
plt.hist(features_df["peak_g"].dropna(), bins=60)
plt.xlabel("Peak acceleration (g)")
plt.ylabel("Count")
plt.title("Distribution of Peak Acceleration")
plt.grid(True)
savefig("peak_g_distribution.png")
plt.close()

plt.figure()
plt.hist(features_df["rms_g"].dropna(), bins=60)
plt.xlabel("RMS acceleration (g)")
plt.ylabel("Count")
plt.title("Distribution of RMS Acceleration")
plt.grid(True)
savefig("rms_g_distribution.png")
plt.close()

# 3) Component peak distributions
plt.figure()
plt.hist(features_df["peak_abs_A1"].dropna(), bins=60, alpha=0.7, label="|A1| peak")
plt.hist(features_df["peak_abs_A2"].dropna(), bins=60, alpha=0.7, label="|A2| peak")
plt.hist(features_df["peak_abs_A3"].dropna(), bins=60, alpha=0.7, label="|A3| peak")
plt.xlabel("Peak absolute component acceleration (units of A1/A2/A3)")
plt.ylabel("Count")
plt.title("Distribution of Peak Component Magnitudes")
plt.legend()
plt.grid(True)
savefig("component_peak_distributions.png")
plt.close()

# 4) Scatter relationships (peak vs impulse, peak vs energy)
plt.figure()
plt.scatter(features_df["peak_g"], features_df["impulse_gs"], s=12)
plt.xlabel("Peak g")
plt.ylabel("Impulse proxy ∫a dt (g·s)")
plt.title("Peak vs Impulse Proxy")
plt.grid(True)
savefig("peak_vs_impulse.png")
plt.close()

plt.figure()
plt.scatter(features_df["peak_g"], features_df["energy_g2s"], s=12)
plt.xlabel("Peak g")
plt.ylabel("Energy proxy ∫a^2 dt (g²·s)")
plt.title("Peak vs Energy Proxy")
plt.grid(True)
savefig("peak_vs_energy.png")
plt.close()

# 5) Time-to-peak distribution
plt.figure()
plt.hist(features_df["t_peak_s"].dropna(), bins=60)
plt.xlabel("Time of peak acceleration (s)")
plt.ylabel("Count")
plt.title("Distribution of Time-to-Peak")
plt.grid(True)
savefig("time_to_peak_distribution.png")
plt.close()

# 6) HIC distribution (if computed)
if "hic15_derived" in features_df.columns and features_df["hic15_derived"].notna().any():
    plt.figure()
    plt.hist(features_df["hic15_derived"].dropna(), bins=60)
    plt.xlabel("HIC15 (derived)")
    plt.ylabel("Count")
    plt.title("Distribution of Derived HIC15")
    plt.grid(True)
    savefig("hic15_distribution.png")
    plt.close()

    plt.figure()
    plt.scatter(features_df["peak_g"], features_df["hic15_derived"], s=12)
    plt.xlabel("Peak g")
    plt.ylabel("HIC15 (derived)")
    plt.title("Peak g vs Derived HIC15")
    plt.grid(True)
    savefig("peak_vs_hic15.png")
    plt.close()

# 7) Identify and save top outliers for manual inspection
outlier_table = features_df.sort_values("peak_g", ascending=False).head(20)
outlier_path = os.path.join(OUT_DIR, "top20_by_peak_g.csv")
outlier_table.to_csv(outlier_path, index=False)
print(f"Saved: {outlier_path}")

if "hic15_derived" in features_df.columns and features_df["hic15_derived"].notna().any():
    outlier_hic = features_df.sort_values("hic15_derived", ascending=False).head(20)
    outlier_hic_path = os.path.join(OUT_DIR, "top20_by_hic15.csv")
    outlier_hic.to_csv(outlier_hic_path, index=False)
    print(f"Saved: {outlier_hic_path}")

# 8) Overlay a random subset of waveforms (A in g)
rng = np.random.default_rng(RANDOM_SEED)
if resampled_ag.shape[0] > 0:
    k = min(N_OVERLAY, resampled_ag.shape[0])
    idxs = rng.choice(resampled_ag.shape[0], size=k, replace=False)

    plt.figure()
    for i in idxs:
        plt.plot(t_grid, resampled_ag[i], alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title(f"Overlay of {k} Random Acceleration Histories (Resampled)")
    plt.grid(True)
    savefig("waveform_overlay_random.png")
    plt.close()

# 9) Envelope statistics (median + percentiles) for resampled waveforms
if resampled_ag.shape[0] > 0:
    median = np.nanmedian(resampled_ag, axis=0)
    p10 = np.nanpercentile(resampled_ag, 10, axis=0)
    p90 = np.nanpercentile(resampled_ag, 90, axis=0)
    p25 = np.nanpercentile(resampled_ag, 25, axis=0)
    p75 = np.nanpercentile(resampled_ag, 75, axis=0)

    plt.figure()
    plt.plot(t_grid, median, label="Median")
    plt.plot(t_grid, p10, label="P10", linestyle="--")
    plt.plot(t_grid, p90, label="P90", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("Acceleration Envelope (P10/P90) and Median")
    plt.legend()
    plt.grid(True)
    savefig("waveform_envelope_p10_p90.png")
    plt.close()

    plt.figure()
    plt.plot(t_grid, median, label="Median")
    plt.plot(t_grid, p25, label="P25", linestyle="--")
    plt.plot(t_grid, p75, label="P75", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("Acceleration Envelope (P25/P75) and Median")
    plt.legend()
    plt.grid(True)
    savefig("waveform_envelope_p25_p75.png")
    plt.close()

# 10) Correlation heatmap of engineered features (no seaborn; use imshow)
corr_cols = [
    "duration_s", "dt_median_s", "n_samples",
    "peak_g", "t_peak_s", "rms_g", "impulse_gs", "energy_g2s",
    "peak_abs_A1", "peak_abs_A2", "peak_abs_A3",
]
if "hic15_derived" in features_df.columns and features_df["hic15_derived"].notna().any():
    corr_cols.append("hic15_derived")

corr_df = features_df[corr_cols].dropna()
if len(corr_df) >= 5:
    C = corr_df.corr().to_numpy()
    labels = corr_df.columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.imshow(C, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title("Correlation Matrix of Per-Experiment Features")
    plt.colorbar()
    savefig("feature_correlation_matrix.png")
    plt.close()

print("EDA complete. Review outputs in:", OUT_DIR)
