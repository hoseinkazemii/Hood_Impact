"""
postprocess_acc.py
==================
Turn the RAW acceleration history extracted from an Abaqus .odb
(Time, A1, A2, A3 -- see extract_acc_odb.py) into the exact
`HoodImpact_<run>_SAE1000_interp1000.csv` format that our partner delivered,
and (optionally) compare against their reference file to verify that our own
Abaqus run reproduces the data set.

Pipeline (matches the delivered files):
    1. resample the raw, slightly non-uniform A1/A2/A3 onto a uniform fine grid
    2. SAE-J211 CFC-1000 filter each component (zero-phase, 4-pole)
    3. sample the filtered signal onto 1000 points: linspace(0, t_final, 1000)
    4. A(mm/s2) = sqrt(A1^2 + A2^2 + A3^2)      (resultant of filtered comps)
       A(in g)  = A(mm/s2) / 9800               (partner's g convention)
    5. HIC15 from A(in g)

Conventions were reverse-engineered from the delivered data:
    * output length          = 1000 points (1000 rows, no resampling to 0.025 s;
                               the grid spans [0, raw final time])
    * g                      = 9800 mm/s^2     (verified: A(mm/s2)/A(in g)==9800)
    * HIC                    = HIC15, dt*(a_avg**2.5), a in g  (see analyze_inp_designs.py)

Plain-Python script (numpy / scipy / pandas / matplotlib) -- run with the same
interpreter you use for the ML code, NOT inside Abaqus.

Examples
--------
    # one run, just produce the SAE1000 csv
    python postprocess_acc.py --raw work/run_1/HoodImpact_1_raw_acc.csv --run 1

    # produce + compare against the delivered reference + save overlay plot
    python postprocess_acc.py --raw work/run_1/HoodImpact_1_raw_acc.csv --run 1 \
        --reference ../Data/HoodImpact_60_IndustryLike/output_history_acc/HoodImpact_1_SAE1000_interp1000.csv \
        --plot
"""
import os
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# constants reverse-engineered from the delivered data set
# ----------------------------------------------------------------------------
G_MM_S2 = 9800.0        # partner's gravitational constant (mm/s^2)
N_OUT = 1000            # delivered files have exactly 1000 rows
CFC = 1000.0           # SAE / ISO channel frequency class -> "SAE1000"
HIC_WINDOW = 0.015     # HIC15 (15 ms), see analyze_inp_designs.py

# Filtering grid. The raw Abaqus/Explicit history is ~sub-microsecond and
# slightly NON-uniform, so we must resample to a uniform grid before the SAE
# filter. That grid MUST be at least as fine as the raw data, otherwise we
# alias the sharp initial contact spikes (the original bug). 5e-7 s matches the
# explicit mass-scaling target time increment and is finer than the raw spacing.
FINE_DT = 5.0e-7
PAD_SECONDS = 0.004    # zero-phase filter edge pad (~several SAE1000 periods)

# Explicit start-up transient: the first 1-2 increments of an explicit step with
# an applied initial velocity into a zero-gap HARD contact are numerically
# meaningless (huge values followed by exact zeros -- see the raw .csv). They are
# above the CFC1000 band, but their net impulse slightly lifts the filtered
# onset. Zeroing this short leading window removes the explicit start-up without
# touching the physical pulse. 0 = keep everything (purest reproduction).
CLIP_STARTUP_S = 1.0e-5

OUT_COLUMNS = ["Time", "A1", "A2", "A3", "A(mm/s2)", "A(in g)"]


# ----------------------------------------------------------------------------
# SAE J211 / ISO 6487 CFC digital filter (2-pole Butterworth, applied
# forward+backward = 4-pole, zero phase).  This is the crash-test standard
# "SAE1000" filter.
# ----------------------------------------------------------------------------
def sae_coeffs(cfc, dt):
    """Return (b, a) scipy-style coefficients for an SAE CFC filter."""
    wd = 2.0 * np.pi * cfc * 2.0775
    wa = np.tan(wd * dt / 2.0)
    den = 1.0 + np.sqrt(2.0) * wa + wa * wa
    a0 = wa * wa / den
    a1 = 2.0 * a0
    a2 = a0
    b1 = -2.0 * (wa * wa - 1.0) / den
    b2 = (-1.0 + np.sqrt(2.0) * wa - wa * wa) / den
    # recurrence: y[i] = a0 x[i] + a1 x[i-1] + a2 x[i-2] + b1 y[i-1] + b2 y[i-2]
    b = np.array([a0, a1, a2])
    a = np.array([1.0, -b1, -b2])
    return b, a


def sae_filter(x, dt, cfc=CFC):
    """Zero-phase SAE CFC filter of a uniformly sampled signal x (spacing dt).

    Uses 'constant' edge padding (not the default odd reflection): the pre-impact
    acceleration is ~0, so padding the leading edge with x[0] (~0) avoids the
    edge ringing that produced the spurious initial spike.
    """
    from scipy.signal import filtfilt
    b, a = sae_coeffs(cfc, dt)
    padlen = min(int(round(PAD_SECONDS / dt)), len(x) - 1)
    return filtfilt(b, a, x, padtype="constant", padlen=padlen)


# ----------------------------------------------------------------------------
# HIC15
# ----------------------------------------------------------------------------
def compute_hic(time, acc_g, window=HIC_WINDOW):
    """HIC over a sliding window <= `window` seconds. time uniform, acc in g."""
    t = np.asarray(time, float)
    a = np.asarray(acc_g, float)
    n = len(t)
    # cumulative integral of a dt -> fast windowed average
    cum = np.concatenate([[0.0], np.cumsum((a[1:] + a[:-1]) * 0.5 * np.diff(t))])
    hic_max = 0.0
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < n and (t[j + 1] - t[i]) <= window:
            j += 1
        if j > i:
            dt = t[j] - t[i]
            if dt > 0:
                a_avg = (cum[j] - cum[i]) / dt
                hic = dt * (a_avg ** 2.5)
                if hic > hic_max:
                    hic_max = hic
    return hic_max


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------
def process_raw(raw_csv, cfc=CFC, n_out=N_OUT, clip_startup_s=CLIP_STARTUP_S):
    """Read raw Time,A1,A2,A3 and return the SAE1000/interp1000 dataframe."""
    raw = pd.read_csv(raw_csv)
    raw.columns = [c.strip() for c in raw.columns]
    t_raw = raw["Time"].to_numpy(float)

    # ensure strictly increasing time (explicit can emit duplicate stamps)
    keep = np.concatenate([[True], np.diff(t_raw) > 0])
    t_raw = t_raw[keep]
    comps_raw = {c: raw[c].to_numpy(float)[keep] for c in ("A1", "A2", "A3")}

    # zero out the explicit start-up transient (see CLIP_STARTUP_S)
    if clip_startup_s and clip_startup_s > 0:
        mask = t_raw < (t_raw[0] + clip_startup_s)
        for c in comps_raw:
            comps_raw[c][mask] = 0.0

    t0, t1 = t_raw[0], t_raw[-1]

    # 1) uniform fine grid for filtering -- finer than the raw data to avoid
    #    aliasing the sharp initial spikes (must satisfy dt_fine <= raw spacing).
    dt_fine = min(FINE_DT, float(np.median(np.diff(t_raw))))
    n_fine = int(round((t1 - t0) / dt_fine)) + 1
    t_fine = np.linspace(t0, t1, n_fine)
    dt_fine = t_fine[1] - t_fine[0]

    # 3) output grid (exactly what the delivered files use)
    t_out = np.linspace(t0, t1, n_out)

    out = {"Time": t_out}
    for c in ("A1", "A2", "A3"):
        a_fine = np.interp(t_fine, t_raw, comps_raw[c])     # resample (upsample)
        a_filt = sae_filter(a_fine, dt_fine, cfc)            # SAE1000, zero-phase
        out[c] = np.interp(t_out, t_fine, a_filt)            # to 1000 pts

    mag = np.sqrt(out["A1"] ** 2 + out["A2"] ** 2 + out["A3"] ** 2)
    out["A(mm/s2)"] = mag
    out["A(in g)"] = mag / G_MM_S2

    return pd.DataFrame(out, columns=OUT_COLUMNS)


def compare(df, reference_csv):
    """Return a dict of agreement metrics vs the delivered reference file."""
    ref = pd.read_csv(reference_csv)
    ref.columns = [c.strip() for c in ref.columns]

    # align on the output grid (defensive; should be identical length)
    t = df["Time"].to_numpy(float)
    g_mine = df["A(in g)"].to_numpy(float)
    g_ref = np.interp(t, ref["Time"].to_numpy(float), ref["A(in g)"].to_numpy(float))

    rmse = float(np.sqrt(np.mean((g_mine - g_ref) ** 2)))
    peak_mine = float(np.max(g_mine))
    peak_ref = float(np.max(g_ref))
    hic_mine = compute_hic(t, g_mine)
    hic_ref = compute_hic(ref["Time"].to_numpy(float), ref["A(in g)"].to_numpy(float))

    denom = max(abs(peak_ref), 1e-9)
    return {
        "rmse_g": rmse,
        "peak_g_mine": peak_mine,
        "peak_g_ref": peak_ref,
        "peak_pct_err": 100.0 * (peak_mine - peak_ref) / denom,
        "hic_mine": hic_mine,
        "hic_ref": hic_ref,
        "hic_pct_err": 100.0 * (hic_mine - hic_ref) / max(hic_ref, 1e-9),
    }


def overlay_plot(df, reference_csv, out_png, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["Time"] * 1e3, df["A(in g)"], label="this Abaqus run", lw=2)
    if reference_csv and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        ref.columns = [c.strip() for c in ref.columns]
        ax.plot(ref["Time"] * 1e3, ref["A(in g)"], "--",
                label="delivered reference", lw=1.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("headform acceleration (g)")
    ax.set_title("HoodImpact_%s  resultant acceleration" % run)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Saved overlay plot -> %s" % out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw Time,A1,A2,A3 csv from extract_acc_odb.py")
    ap.add_argument("--run", required=True, help="run id (used for output file naming)")
    ap.add_argument("--out-dir", default=".", help="where to write the SAE1000 csv")
    ap.add_argument("--reference", default=None, help="delivered reference csv to compare against")
    ap.add_argument("--plot", action="store_true", help="save an overlay plot")
    ap.add_argument("--clip-startup-us", type=float, default=CLIP_STARTUP_S * 1e6,
                    help="zero the explicit start-up transient over this many "
                         "microseconds (default %.1f; 0 = purest reproduction)"
                         % (CLIP_STARTUP_S * 1e6))
    args = ap.parse_args()

    df = process_raw(args.raw, clip_startup_s=args.clip_startup_us * 1e-6)

    out_csv = os.path.join(args.out_dir, "HoodImpact_%s_SAE1000_interp1000.csv" % args.run)
    df.to_csv(out_csv, index=False)
    hic = compute_hic(df["Time"].to_numpy(float), df["A(in g)"].to_numpy(float))
    print("Wrote %s   (peak=%.2f g, HIC15=%.1f)" %
          (out_csv, df["A(in g)"].max(), hic))

    if args.reference:
        m = compare(df, args.reference)
        print("\n--- comparison vs delivered reference ---")
        print("  peak g : mine=%.2f  ref=%.2f  (%.2f%%)" %
              (m["peak_g_mine"], m["peak_g_ref"], m["peak_pct_err"]))
        print("  HIC15  : mine=%.1f  ref=%.1f  (%.2f%%)" %
              (m["hic_mine"], m["hic_ref"], m["hic_pct_err"]))
        print("  RMSE   : %.4f g over the curve" % m["rmse_g"])

    if args.plot:
        out_png = os.path.join(args.out_dir, "HoodImpact_%s_acc_overlay.png" % args.run)
        overlay_plot(df, args.reference, out_png, args.run)


if __name__ == "__main__":
    main()
