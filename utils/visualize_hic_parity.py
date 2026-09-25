"""Parity plot for direct-HIC predictions, with a +/-10% error band.

Reads the ``hic_test_predictions.csv`` written by the HIC branch of
temporal_deeponet_pointnetpp.py and draws:

  (a) predicted vs ground-truth HIC, with the 1:1 line and a +/-10% band,
      in-band and out-of-band points distinguished;
  (b) relative error vs ground-truth HIC, with the +/-10% limits.

Usage:
    python utils/visualize_hic_parity.py runs/<run_dir>
    python utils/visualize_hic_parity.py runs/<run_dir> --tolerance 5
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_BAND = "#2a7f62"
OUT_BAND = "#c8553d"
BAND_FILL = "#4c72b0"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory containing hic_test_predictions.csv")
    p.add_argument("--tolerance", type=float, default=10.0,
                   help="error band half-width in percent (default: 10)")
    p.add_argument("--csv", default="hic_test_predictions.csv")
    p.add_argument("--out", default=None,
                   help="output png (default: <run_dir>/hic_pred_vs_gt_<tol>pct.png)")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.run_dir, args.csv)
    if not os.path.exists(csv_path):
        raise SystemExit(f"not found: {csv_path}")

    df = pd.read_csv(csv_path)
    gt = df["hic_ground_truth"].to_numpy(float)
    pred = df["hic_prediction"].to_numpy(float)

    tol = args.tolerance / 100.0
    rel = (pred - gt) / gt * 100.0
    inside = np.abs(rel) <= args.tolerance

    resid = pred - gt
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    mape = float(np.mean(np.abs(rel)))
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((gt - gt.mean()) ** 2)
    frac = inside.mean() * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # ---------------------------------------------------------------- parity
    lo = min(gt.min(), pred.min())
    hi = max(gt.max(), pred.max())
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    line = np.array([lo, hi])

    ax.fill_between(line, line * (1 - tol), line * (1 + tol),
                    color=BAND_FILL, alpha=0.13, lw=0,
                    label=f"$\\pm${args.tolerance:g}% band", zorder=1)
    ax.plot(line, line * (1 + tol), color=BAND_FILL, lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax.plot(line, line * (1 - tol), color=BAND_FILL, lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax.plot(line, line, color="0.25", lw=1.6, ls="-", label="1:1", zorder=3)

    ax.scatter(gt[inside], pred[inside], s=34, facecolor=IN_BAND, edgecolor="white",
               linewidth=0.5, alpha=0.9, zorder=4,
               label=f"within {args.tolerance:g}%  (n={int(inside.sum())})")
    ax.scatter(gt[~inside], pred[~inside], s=44, facecolor=OUT_BAND, edgecolor="white",
               linewidth=0.5, alpha=0.95, zorder=5, marker="D",
               label=f"outside  (n={int((~inside).sum())})")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth HIC15")
    ax.set_ylabel("Predicted HIC15")
    ax.set_title("Predicted vs ground-truth HIC")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)

    stats = (f"$R^2$ = {r2:.4f}\n"
             f"RMSE = {rmse:.1f}\n"
             f"MAE  = {mae:.1f}\n"
             f"MAPE = {mape:.2f}%\n"
             f"bias = {bias:+.1f}\n"
             f"within {args.tolerance:g}% : {frac:.1f}%")
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.75", alpha=0.94))

    # -------------------------------------------------------- relative error
    ax2.axhspan(-args.tolerance, args.tolerance, color=BAND_FILL, alpha=0.13, lw=0,
                label=f"$\\pm${args.tolerance:g}% band")
    ax2.axhline(args.tolerance, color=BAND_FILL, lw=1.0, ls="--", alpha=0.7)
    ax2.axhline(-args.tolerance, color=BAND_FILL, lw=1.0, ls="--", alpha=0.7)
    ax2.axhline(0.0, color="0.25", lw=1.6)

    ax2.scatter(gt[inside], rel[inside], s=34, facecolor=IN_BAND, edgecolor="white",
                linewidth=0.5, alpha=0.9)
    ax2.scatter(gt[~inside], rel[~inside], s=44, facecolor=OUT_BAND, edgecolor="white",
                linewidth=0.5, alpha=0.95, marker="D")

    ax2.set_xlabel("Ground-truth HIC15")
    ax2.set_ylabel("Relative error  (pred $-$ GT) / GT  [%]")
    ax2.set_title("Relative error vs HIC magnitude")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=9)

    run_label = os.path.basename(os.path.normpath(args.run_dir))
    fig.suptitle(f"Direct-HIC DeepONet + PointNet++  |  {run_label}  |  n = {len(df)}",
                 fontsize=11.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    out = args.out or os.path.join(
        args.run_dir, f"hic_pred_vs_gt_{args.tolerance:g}pct.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"saved: {out}")

    # ------------------------------------------------------------- terminal
    print()
    print(f"n = {len(df)}   R2 = {r2:.4f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}   "
          f"MAPE = {mape:.2f}%   bias = {bias:+.2f}")
    for t in (5, 10, 15, 20):
        print(f"  within {t:2d}% : {np.mean(np.abs(rel) <= t) * 100:5.1f}%")
    worst = df.assign(rel_pct=rel).reindex(
        np.argsort(-np.abs(rel))).head(10)
    print("\nworst 10 by relative error:")
    print(worst[["run_number", "hic_ground_truth", "hic_prediction",
                 "error", "rel_pct"]].to_string(index=False,
                                                float_format=lambda v: f"{v:9.2f}"))


if __name__ == "__main__":
    main()
