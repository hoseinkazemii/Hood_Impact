"""
make_verification_report.py
===========================
Summary visualizations for the 60-run verification batch (batch_verify_60.py):

1. verification_overlay_grid.png    -- small-multiples grid, one panel per run:
   our regenerated SAE1000 acceleration (blue, solid) overlaid on the delivered
   reference (orange, dashed).
2. verification_overlay_pages/*.png -- clearer page figures, five runs per
   figure by default.
3. verification_metrics_summary.png -- per-run agreement metrics: RMSE (g),
   signed peak-g error (%), signed HIC15 error (%).

Reads work/verification_metrics.csv (last row per run wins) and the per-run
SAE1000 csv files. Table-view twin of the charts = verification_metrics.csv.

    python abaqus_scripts\\make_verification_report.py
"""
import os
import math
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "output_history_acc")

# reference palette (dataviz skill, light mode, pre-validated)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis / labels
GRID = "#e1e0d9"         # hairline grid
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"         # series: this Abaqus run  (also diverging + pole)
ORANGE = "#eb6834"       # series: delivered reference
RED = "#e34948"          # diverging - pole

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.linewidth": 0.6, "axes.axisbelow": True,
})


def load_metrics(work_dir):
    df = pd.read_csv(os.path.join(work_dir, "verification_metrics.csv"))
    df = df.drop_duplicates(subset="run", keep="last")
    ok = df[df["status"] == "ok"].sort_values("run").reset_index(drop=True)
    failed = df[df["status"] != "ok"]["run"].tolist()
    return ok, failed


def _load_overlay_curves(row, work_dir):
    run = int(row["run"])
    mine = pd.read_csv(os.path.join(
        work_dir, "run_%d" % run, "HoodImpact_%d_SAE1000_interp1000.csv" % run))
    ref = pd.read_csv(os.path.join(
        REF_DIR, "HoodImpact_%d_SAE1000_interp1000.csv" % run))
    ref.columns = [c.strip() for c in ref.columns]
    return run, mine, ref


def _style_time_axis(ax, y_top=None):
    ax.set_xlim(0, 25)
    ax.set_ylim(0, y_top)
    ax.margins(x=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _plot_overlay_panel(ax, row, work_dir, compact=False):
    run, mine, ref = _load_overlay_curves(row, work_dir)

    mine_t = mine["Time"].to_numpy() * 1e3
    ref_t = ref["Time"].to_numpy() * 1e3
    mine_a = mine["A(in g)"].to_numpy()
    ref_a = ref["A(in g)"].to_numpy()
    y_top = max(float(np.nanmax(mine_a)), float(np.nanmax(ref_a))) * 1.08

    ax.plot(mine_t, mine_a, color=BLUE, lw=1.45 if compact else 1.85,
            solid_capstyle="round", zorder=3)
    ax.plot(ref_t, ref_a, color=ORANGE, lw=1.25 if compact else 1.55,
            ls=(0, (4, 3)), zorder=4)
    _style_time_axis(ax, y_top)

    if compact:
        ax.set_title("run %d" % run, fontsize=8.5, color=INK, pad=2.5)
        ax.text(0.97, 0.94, "RMSE %.2f g" % row["rmse_g"], transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=MUTED)
        ax.tick_params(labelsize=6, length=2)
    else:
        ax.set_title("run %d" % run, loc="left", fontsize=10.5,
                     color=INK, fontweight="semibold", pad=3)
        ax.set_title("RMSE %.2f g | peak %+.2f%% | HIC %+.2f%%"
                     % (row["rmse_g"], row["peak_pct_err"], row["hic_pct_err"]),
                     loc="right", fontsize=8.6, color=INK_2, pad=3)
        ax.set_ylabel("g", fontsize=8.7, color=INK_2)
        ax.tick_params(labelsize=8.3, length=2.5)


def overlay_grid(ok, work_dir, out_png, ncols=6):
    n = len(ok)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.75 * nrows))
    axes = np.atleast_2d(axes)

    for i, row in ok.iterrows():
        ax = axes[i // ncols][i % ncols]
        _plot_overlay_panel(ax, row, work_dir, compact=True)
        # outer labels only
        if i // ncols != nrows - 1 and i + ncols < n:
            ax.set_xticklabels([])

    # hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Our Abaqus runs reproduce the delivered dataset "
                 "(headform resultant acceleration, %d/60 runs)" % n,
                 fontsize=13, color=INK, y=1.0)
    fig.supxlabel("time (ms)", fontsize=9, color=INK_2)
    fig.supylabel("headform acceleration (g)   [y-scale per panel]",
                  fontsize=9, color=INK_2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.8, label="this Abaqus run"),
        Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (4, 3)), label="delivered reference"),
    ], loc="upper right", bbox_to_anchor=(0.995, 1.003), ncols=2,
        fontsize=9, frameon=False)
    fig.tight_layout(rect=(0.015, 0.012, 1, 0.985))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out_png)


def overlay_pages(ok, work_dir, out_dir, runs_per_figure=5, dpi=180):
    if runs_per_figure <= 0:
        raise ValueError("--runs-per-figure must be positive")

    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "verification_overlay_runs_*.png")):
        os.remove(old)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=BLUE, lw=2.0, label="this Abaqus run"),
        Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (4, 3)),
               label="delivered reference"),
    ]

    paths = []
    n = len(ok)
    for page, start in enumerate(range(0, n, runs_per_figure), start=1):
        group = ok.iloc[start:start + runs_per_figure]
        runs = group["run"].astype(int).tolist()
        fig_h = 1.75 * len(group) + 1.25
        fig, axes = plt.subplots(len(group), 1, figsize=(9.4, fig_h), sharex=True)
        axes = np.atleast_1d(axes)

        for ax, (_, row) in zip(axes, group.iterrows()):
            _plot_overlay_panel(ax, row, work_dir, compact=False)

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        fig.suptitle("Headform resultant acceleration: Abaqus vs delivered reference",
                     fontsize=13.2, color=INK, x=0.075, ha="left", y=0.992)
        fig.text(0.075, 0.952, "figure %02d | runs %02d-%02d | %d of 60 verified"
                 % (page, runs[0], runs[-1], n),
                 fontsize=9.2, color=INK_2, ha="left", va="top")
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.995),
                   ncols=2, fontsize=9.2, frameon=False)
        fig.supxlabel("time (ms)", fontsize=9.5, color=INK_2, y=0.012)
        fig.tight_layout(rect=(0.045, 0.035, 0.995, 0.925), h_pad=0.75)

        out_png = os.path.join(out_dir, "verification_overlay_runs_%02d_%02d.png"
                               % (runs[0], runs[-1]))
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_png)
        print("wrote %s" % out_png)

    return paths


def diverging_bars(ax, runs, vals, title, unit):
    colors = [BLUE if v >= 0 else RED for v in vals]
    ax.bar(runs, vals, width=0.72, color=colors, zorder=3)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=4)
    lim = max(abs(v) for v in vals) * 1.25
    ax.set_ylim(-lim, lim)
    # selective direct label: only the extreme run
    i = int(np.argmax(np.abs(vals)))
    ax.annotate("run %d: %+.2f%s" % (runs[i], vals[i], unit),
                (runs[i], vals[i]), textcoords="offset points",
                xytext=(0, 5 if vals[i] >= 0 else -11),
                ha="center", fontsize=7.5, color=INK_2)


def metrics_summary(ok, out_png):
    runs = ok["run"].astype(int).to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True)

    ax = axes[0]
    ax.bar(runs, ok["rmse_g"], width=0.72, color=BLUE, zorder=3)
    ax.set_title("RMSE vs delivered reference (g)  --  curves peak at 98-191 g",
                 fontsize=10.5, color=INK, loc="left", pad=4)
    i = int(np.argmax(ok["rmse_g"].to_numpy()))
    ax.annotate("run %d: %.2f g" % (runs[i], ok["rmse_g"].iloc[i]),
                (runs[i], ok["rmse_g"].iloc[i]), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=7.5, color=INK_2)
    ax.set_ylim(bottom=0)

    diverging_bars(axes[1], runs, ok["peak_pct_err"].to_numpy(),
                   "Peak acceleration error (%)", "%")
    diverging_bars(axes[2], runs, ok["hic_pct_err"].to_numpy(),
                   "HIC15 error (%)", "%")

    axes[2].set_xlabel("run id", fontsize=9)
    axes[2].set_xlim(runs.min() - 1, runs.max() + 1)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, length=2.5)

    fig.suptitle("Verification metrics: regenerated vs delivered, %d runs "
                 "(max RMSE %.2f g, max |peak err| %.2f%%, max |HIC err| %.2f%%)"
                 % (len(ok), ok["rmse_g"].max(),
                    ok["peak_pct_err"].abs().max(), ok["hic_pct_err"].abs().max()),
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--runs-per-figure", type=int, default=5,
                    help="number of run panels in each clear overlay figure")
    ap.add_argument("--overlay-pages-dir", default=None,
                    help="output folder for the split overlay figures")
    args = ap.parse_args()

    ok, failed = load_metrics(args.work_dir)
    print("runs ok: %d, failed/missing: %s" % (len(ok), failed or "none"))

    overlay_grid(ok, args.work_dir,
                 os.path.join(args.work_dir, "verification_overlay_grid.png"))
    overlay_pages(
        ok, args.work_dir,
        args.overlay_pages_dir or os.path.join(args.work_dir, "verification_overlay_pages"),
        runs_per_figure=args.runs_per_figure)
    metrics_summary(ok, os.path.join(args.work_dir, "verification_metrics_summary.png"))

    print("\nsummary over %d runs:" % len(ok))
    print("  RMSE      : mean %.3f g, max %.3f g" % (ok["rmse_g"].mean(), ok["rmse_g"].max()))
    print("  |peak err|: mean %.3f %%, max %.3f %%"
          % (ok["peak_pct_err"].abs().mean(), ok["peak_pct_err"].abs().max()))
    print("  |HIC err| : mean %.3f %%, max %.3f %%"
          % (ok["hic_pct_err"].abs().mean(), ok["hic_pct_err"].abs().max()))


if __name__ == "__main__":
    main()
