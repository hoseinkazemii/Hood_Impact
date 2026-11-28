"""Compare two local-edit runs on the held-out and in-sample edit-effect metrics.

Both runs must already have the outputs of ``evaluate_local_edits_hic.py`` and
``analyze_local_edits_run.py`` (hic_results_pred_vs_gt.csv, diagnostics/). Run A is the
earlier run, run B the new one.

Writes into B's diagnostics folder:
    07_vs_<A>.png    (a) held-out HIC change, A vs B, run by run
                     (b) the headline metrics, A -> B, with the sibling-copy bar and the
                         mirror-twin ceiling where they exist
                     (c) training and validation loss per epoch
    07_vs_<A>.json   the numbers in (b), plus the held-out HIC-change error by location
                     role (near / twin / far), also printed

Usage:
    python compare_local_edit_runs.py runs/.../<run A> runs/.../<run B>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_local_edits_run import pooled_pairs

# Reference palette slots 1-3 (validated all-pairs), a neutral for design 6, text inks.
RUN_A, RUN_B, SIBLING, D6C = "#2a78d6", "#eb6834", "#1baf7a", "#6f6e69"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SIB_REL = "sibling (same site & sign)"


def short_name(run_dir):
    parts = Path(run_dir).name.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else Path(run_dir).name


def slope(p, t):
    return float(np.sum(p * t) / np.sum(t * t))


def r2_zero(p, t):
    return float(1.0 - np.sum((p - t) ** 2) / np.sum(t ** 2))


def load(run_dir):
    run_dir = Path(run_dir)
    diag = run_dir / "diagnostics"
    return {"hic": pd.read_csv(run_dir / "hic_results_pred_vs_gt.csv"),
            "designs": pd.read_csv(diag / "per_design.csv").set_index("design"),
            "pairs": pd.read_csv(diag / "design_pairs.csv"),
            "history": pd.read_csv(run_dir / "training_history.csv")}


def pooled_class_r2(designs, cls):
    g = designs[(designs["split"] == "train") & (designs["class"] == cls)]
    w = g["runs"]
    return float(1.0 - np.sum(w * g["effect_error_rms_g"] ** 2) / np.sum(w * g["effect_rms_true_g"] ** 2))


def headline(run):
    """[(label, value, sibling bar or None, ceiling or None)] on a 0-1 scale."""
    d, h = run["designs"], run["hic"]
    held = d[d["split"] != "train"]
    test = held[held["split"] == "test"].index[0]
    val = held[held["split"] == "validation"].index[0]
    rows = []
    for name, split in ((test, "test"), (val, "validation")):
        r = d.loc[name]
        rows.append((f"{name} ({'val' if split == 'validation' else split}) edit-effect R²", r["difference_r2"], r["sibling_r2"], r["twin_copy_r2"]))
    train = d[d["split"] == "train"]
    w = train["runs"]
    rows.append(("edit-effect R², training designs",
                 float(1.0 - np.sum(w * train["effect_error_rms_g"] ** 2) / np.sum(w * train["effect_rms_true_g"] ** 2)),
                 None, None))
    rows.append(("edit-effect R², deepening (training)", pooled_class_r2(d, "deepening"), None, None))
    for name, split in ((test, "test"), (val, "validation")):
        g = h[h["split"] == split]
        t, p = g["hic_change_true_pct"].to_numpy(), g["hic_change_pred_pct"].to_numpy()
        s = g["hic_change_sibling_pct"].to_numpy()
        ok = np.isfinite(s)
        rows.append((f"{name} HIC-change R²", r2_zero(p, t), r2_zero(s[ok], t[ok]), None))
    for name, split in ((test, "test"), (val, "validation")):
        g = h[h["split"] == split]
        rows.append((f"{name} HIC-change slope",
                     slope(g["hic_change_pred_pct"].to_numpy(), g["hic_change_true_pct"].to_numpy()), None, 1.0))
    ins = pooled_pairs(run["pairs"], ["train-train (in-sample)"]).get(SIB_REL, {})
    out = pooled_pairs(run["pairs"], ["test vs train", "validation vs train"]).get(SIB_REL, {})
    rows.append(("sibling pair R², training", ins.get("pair_r2", np.nan), None, None))
    rows.append(("sibling pair R², held-out", out.get("pair_r2", np.nan), None, None))
    rows.append(("sibling size ratio, held-out", out.get("size_ratio", np.nan), None, 1.0))
    return rows


def role_errors(run):
    h = run["hic"]
    g = h[h["split"] != "train"]
    out = {}
    for role in ("near", "twin", "far"):
        x = g[g["role"] == role]
        t = x["hic_change_true_pct"]
        out[role] = {"runs": int(len(x)), "model": float((x["hic_change_pred_pct"] - t).abs().mean()),
                     "copy_d6": float(t.abs().mean())}
    return out


def plot(a, b, name_a, name_b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig = plt.figure(figsize=(13.0, 9.6))
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.915, top=0.9, bottom=0.115, wspace=0.62, hspace=0.36,
                          width_ratios=(1.0, 1.15))

    def style(ax):
        ax.grid(color=GRID, lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) held-out HIC change, run by run
    ax = fig.add_subplot(gs[0, 0])
    ha = a["hic"][a["hic"]["split"] != "train"].set_index("sample")
    hb = b["hic"][b["hic"]["split"] != "train"].set_index("sample").loc[ha.index]
    t = ha["hic_change_true_pct"].to_numpy()
    pa, pb = ha["hic_change_pred_pct"].to_numpy(), hb["hic_change_pred_pct"].to_numpy()
    lim = (min(t.min(), pa.min(), pb.min()) - 4, max(t.max(), pa.max(), pb.max()) + 4)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    ax.axhline(0, color=D6C, lw=1.0, ls="--")
    ax.text(lim[0] + 1, 1.0, "copy design 6", fontsize=7.5, color=D6C, va="bottom")
    for x, y0, y1 in zip(t, pa, pb):
        ax.plot([x, x], [y0, y1], color=GRID, lw=1.0, zorder=2)
    ax.scatter(t, pa, s=30, facecolor="none", edgecolor=RUN_A, lw=1.3, zorder=3)
    ax.scatter(t, pb, s=30, color=RUN_B, edgecolor="white", lw=0.5, zorder=4)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec=RUN_A, label=f"{name_a}: slope {slope(pa, t):.2f}, R² {r2_zero(pa, t):.2f}"),
        Line2D([], [], marker="o", ls="", color=RUN_B, label=f"{name_b}: slope {slope(pb, t):.2f}, R² {r2_zero(pb, t):.2f}")],
        frameon=False, fontsize=7.8, loc="lower right")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("simulated HIC change vs design 6 (%)")
    ax.set_ylabel("predicted HIC change (%)")
    ax.set_title(f"(a) All {len(t)} held-out runs (test + validation): HIC change\n"
                 "grey line joins the two predictions of the same run", loc="left")
    style(ax)

    # (b) headline metrics, A -> B
    ax = fig.add_subplot(gs[:, 1])
    rows_a, rows_b = headline(a), headline(b)
    y = np.arange(len(rows_a))[::-1]
    for yi, ra, rb in zip(y, rows_a, rows_b):
        va, vb = ra[1], rb[1]
        ax.plot([va, vb], [yi, yi], color=GRID, lw=2.0, zorder=2)
        ax.scatter([va], [yi], s=46, facecolor="white", edgecolor=RUN_A, lw=1.5, zorder=4)
        ax.scatter([vb], [yi], s=46, color=RUN_B, edgecolor="white", lw=0.6, zorder=5)
        if rb[2] is not None and np.isfinite(rb[2]):
            ax.scatter([rb[2]], [yi], s=60, marker="D", facecolor="none", edgecolor=SIBLING, lw=1.3, zorder=3)
        if rb[3] is not None:
            ax.scatter([rb[3]], [yi], s=90, marker="|", color=INK, lw=1.6, zorder=3)
        ax.text(1.02, yi, f"{va:.2f} → {vb:.2f}", va="center", ha="left", fontsize=8, color=INK,
                family="monospace", transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] for r in rows_a], fontsize=8.2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("value (R², slope or size ratio)")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec=RUN_A, label=name_a),
        Line2D([], [], marker="o", ls="", color=RUN_B, label=name_b),
        Line2D([], [], marker="D", ls="", mfc="none", mec=SIBLING, label="sibling copy (same site & sign)"),
        Line2D([], [], marker="|", ls="", color=INK, ms=10, mew=1.6,
               label="mirror-twin repeatability; 1 = right size")],
        frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.4, -0.045), ncol=2)
    ax.set_title("(b) Headline metrics, previous → new", loc="left")
    style(ax)

    # (c) validation loss per epoch
    ax = fig.add_subplot(gs[1, 0])
    for run, color, name in ((a, RUN_A, name_a), (b, RUN_B, name_b)):
        hist = run["history"]
        ax.plot(hist["epoch"], hist["validation_mse_normalized"], color=color, lw=2.0, label=f"{name.split('_')[0]} validation")
        ax.plot(hist["epoch"], hist["train_mse_normalized"], color=color, lw=1.2, ls="--", label=f"{name.split('_')[0]} training")
        best = hist.loc[hist["validation_mse_normalized"].idxmin()]
        ax.scatter([best["epoch"]], [best["validation_mse_normalized"]], s=40, color=color, edgecolor="white", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE, normalized acceleration (log)")
    ax.legend(frameon=False, fontsize=7.6, loc="upper right", bbox_to_anchor=(1.0, 0.47))
    ax.set_title("(c) Loss per epoch; dot = best validation epoch (kept checkpoint)", loc="left")
    style(ax)

    fig.suptitle(f"Local-edit runs: {name_a} (previous) vs {name_b} (new); "
                 "held-out designs N54 (test) and N55 (validation)", x=0.065, ha="left", fontsize=11.5, color=INK)

    ra, rb = role_errors(a), role_errors(b)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return rows_a, rows_b, ra, rb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)
    a, b = load(args.run_a), load(args.run_b)
    name_a, name_b = short_name(args.run_a), short_name(args.run_b)
    out = args.run_b / "diagnostics" / f"07_vs_{name_a}"
    rows_a, rows_b, ra, rb = plot(a, b, name_a, name_b, out.with_suffix(".png"))
    record = {"run_a": str(args.run_a), "run_b": str(args.run_b),
              "headline": [{"metric": x[0], "a": float(x[1]), "b": float(y[1]),
                            "sibling_copy": None if y[2] is None else float(y[2]),
                            "reference": None if y[3] is None else float(y[3])} for x, y in zip(rows_a, rows_b)],
              "held_out_hic_change_mae_by_role": {"a": ra, "b": rb}}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    for x, y in zip(rows_a, rows_b):
        print(f"{x[0]:55s} {x[1]:6.3f} -> {y[1]:6.3f}")
    print("held-out HIC-change MAE by role (pts):")
    for role in ra:
        print(f"  {role:5s} n={ra[role]['runs']:2d}  {ra[role]['model']:5.2f} -> {rb[role]['model']:5.2f}"
              f"   copy D6 {ra[role]['copy_d6']:5.2f}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
