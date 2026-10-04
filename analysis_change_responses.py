"""Post-hoc response analysis of all 12 designs; never trains or changes a model.

This intentionally includes previously held-out simulations for a user-requested
diagnostic. It is descriptive analysis, not an untouched test or causal study.
"""

from __future__ import annotations

import argparse
import ast
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hic15 import batched_hic


ROOT = Path(__file__).resolve().parent
COLORS = ["#2878b5", "#d8782b", "#278368", "#7857a1"]


def cluster_mapping():
    tree = ast.parse((ROOT / "mesh_design_clusters.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GEOMETRY_CLUSTERS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing authoritative GEOMETRY_CLUSTERS")


def load_responses(data_dir, stride=16, max_time=None, run_dir=None):
    """Return aligned physical responses without reading geometry or model weights.

    Keys: curves[12,142,T] (g), times[T] (s), impact_xy[142,2] (X1,X2 mm),
    hic[12,142] (sampled-grid HIC15), hic_full[12,142], cluster_ids[12],
    run_numbers[12,142]. If run_dir is supplied, its saved preprocessing overrides
    stride/max_time and its prediction_times.npy must match exactly. Full HIC
    uses the untruncated original 1000-point history, keeping that distinction
    explicit. No interpolation, normalization fitting, or output-table HIC is used.
    """
    data_dir = Path(data_dir)
    if run_dir is not None:
        run_dir = Path(run_dir)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved["data"]["num_samples"] != 1704 or saved["data"]["samples_per_design"] != 142:
            raise ValueError("The response analysis requires 12 designs x 142 impacts")
        stride = saved["preprocessing"]["time_subsample_stride"]
        max_time = saved["preprocessing"]["max_train_time"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_time is not None and (not np.isfinite(max_time) or max_time <= 0):
        raise ValueError("max_time must be finite and positive")

    mapping = pd.read_csv(data_dir / "run_mapping_1704.csv")
    runs = np.arange(1, 1705)
    if (len(mapping) != 1704 or not np.array_equal(mapping["run"], runs)
            or not np.array_equal(mapping["design"], (runs - 1) // 142)
            or not np.array_equal(mapping["loc"], (runs - 1) % 142 + 1)):
        raise ValueError("The canonical run/design/location mapping is inconsistent")
    positions = pd.read_csv(data_dir / "ImpactCoords_1704.csv")[["X1", "X2"]].to_numpy(dtype=float)
    if positions.shape != (1704, 2) or not np.isfinite(positions).all():
        raise ValueError("Expected 1704 finite impact XY coordinates")
    positions = positions.reshape(12, 142, 2)
    if not np.allclose(positions, positions[:1], rtol=0, atol=.001):
        raise ValueError("Matched impact locations have different XY coordinates")
    locations = pd.read_csv(data_dir / "impact_locations_142.csv")
    if (not np.array_equal(locations["loc"], np.arange(1, 143))
            or not np.allclose(locations[["X1", "X2"]], positions[0], rtol=0, atol=.001)):
        raise ValueError("Impact locations do not agree with the canonical 142-location table")

    full_curves, full_times = [], None
    for run in runs:
        frame = pd.read_csv(data_dir / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv",
                            usecols=["Time", "A(in g)"])
        t, a = frame["Time"].to_numpy(dtype=float), frame["A(in g)"].to_numpy(dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"Run {run}: invalid time grid")
        if a.shape != t.shape or not np.isfinite(a).all():
            raise ValueError(f"Run {run}: invalid acceleration history")
        if full_times is None:
            full_times = t
        if not np.array_equal(t, full_times):
            raise ValueError(f"Run {run}: exact full time grid differs; interpolation is prohibited")
        full_curves.append(a)
    full_curves = np.asarray(full_curves).reshape(12, 142, -1)
    # Reproduce DataPreprocessor's float32, cutoff-then-stride convention.
    times = full_times.astype(np.float32)
    take = np.arange(len(times))
    if max_time is not None:
        take = take[times <= max_time]
        if len(take) < 2:
            raise ValueError("The requested cutoff leaves fewer than two times")
    take = take[::stride]
    times = times[take]
    curves = full_curves[..., take].astype(np.float32)
    if run_dir is not None:
        saved_times = np.load(run_dir / "prediction_times.npy", allow_pickle=False)
        if not np.array_equal(times, saved_times):
            raise ValueError("Reconstructed sampled grid differs from the saved model prediction grid")
    clusters = cluster_mapping()
    cluster_ids = np.asarray([next(name for name, ids in clusters.items() if design in ids) for design in range(12)])
    return {
        "curves": curves, "times": times, "impact_xy": positions[0],
        "hic": batched_hic(times, curves), "hic_full": batched_hic(full_times, full_curves),
        "cluster_ids": cluster_ids, "run_numbers": runs.reshape(12, 142),
        "stride": np.asarray(stride), "full_time_points": np.asarray(len(full_times)),
    }


def variance_decomposition(curves, cluster_ids):
    """Exact design-weighted population decomposition at each location.

    For each time, total variance across designs = mean squared residual from
    its cluster mean + squared cluster-mean deviation from the overall mean,
    weighted by cluster size. Then average each term over time. These components
    describe the given designs, and within-cluster variation is not called noise.
    """
    curves = np.asarray(curves, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if curves.ndim != 3 or curves.shape[0] != len(cluster_ids) or not np.isfinite(curves).all():
        raise ValueError("Expected finite curves[design,location,time] with one cluster ID per design")
    center = curves.mean(axis=0)
    fitted = np.empty_like(curves)
    for cluster in np.unique(cluster_ids):
        select = cluster_ids == cluster
        fitted[select] = curves[select].mean(axis=0)
    total = np.mean((curves - center) ** 2, axis=(0, 2))
    within = np.mean((curves - fitted) ** 2, axis=(0, 2))
    between = np.mean((fitted - center) ** 2, axis=(0, 2))
    np.testing.assert_allclose(total, within + between, rtol=1e-12, atol=1e-12)
    return {"total_variance_g2": total, "within_cluster_variance_g2": within,
            "between_cluster_variance_g2": between}


def summarize_responses(arrays, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = arrays["curves"].astype(float)
    clusters, hic = arrays["cluster_ids"], arrays["hic"]
    variance = variance_decomposition(curves, clusters)
    pairs = np.asarray(list(combinations(range(len(curves)), 2)))
    delta = curves[pairs[:, 0]] - curves[pairs[:, 1]]
    pair_rmse = np.sqrt(np.mean(delta ** 2, axis=-1))
    pair_hic = np.abs(hic[pairs[:, 0]] - hic[pairs[:, 1]])
    same_cluster = clusters[pairs[:, 0]] == clusters[pairs[:, 1]]
    total = variance["total_variance_g2"]
    within = variance["within_cluster_variance_g2"]
    between = variance["between_cluster_variance_g2"]
    frame = pd.DataFrame({
        "location_id": np.arange(1, curves.shape[1] + 1),
        "X1_mm": arrays["impact_xy"][:, 0], "X2_mm": arrays["impact_xy"][:, 1],
        **variance, "total_rms_spread_g": np.sqrt(total),
        "within_cluster_rms_spread_g": np.sqrt(within),
        "between_cluster_rms_spread_g": np.sqrt(between),
        "within_cluster_variance_fraction": np.divide(within, total, out=np.full_like(total, np.nan), where=total > 0),
        "mean_within_cluster_pair_rmse_g": pair_rmse[same_cluster].mean(axis=0),
        "mean_cross_cluster_pair_rmse_g": pair_rmse[~same_cluster].mean(axis=0),
        "mean_all_pair_rmse_g": pair_rmse.mean(axis=0),
        "hic_sampled_min": hic.min(axis=0), "hic_sampled_max": hic.max(axis=0),
        "hic_sampled_std": hic.std(axis=0),
    })
    frame["total_spread_rank_descending"] = frame.total_rms_spread_g.rank(ascending=False, method="min").astype(int)
    frame["within_spread_rank_descending"] = frame.within_cluster_rms_spread_g.rank(ascending=False, method="min").astype(int)
    frame.to_csv(out_dir / "location_response_variation.csv", index=False)
    pair_rows = []
    for i, (a, b) in enumerate(pairs):
        for loc in range(curves.shape[1]):
            pair_rows.append({"design_a": a, "design_b": b, "cluster_a": clusters[a], "cluster_b": clusters[b],
                              "same_cluster": bool(same_cluster[i]), "location_id": loc + 1,
                              "history_difference_rmse_g": pair_rmse[i, loc], "absolute_hic_difference": pair_hic[i, loc]})
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_response_differences.csv", index=False)
    group_rows = []
    for label, select in (("all", np.ones(len(pairs), dtype=bool)), ("within_cluster", same_cluster), ("cross_cluster", ~same_cluster)):
        group_rows.append({"pair_group": label, "num_pairs": int(select.sum()),
                           "pooled_history_difference_rmse_g": float(np.sqrt(np.mean(pair_rmse[select] ** 2))),
                           "mean_pair_location_history_rmse_g": float(pair_rmse[select].mean()),
                           "median_pair_location_history_rmse_g": float(np.median(pair_rmse[select])),
                           "mean_absolute_hic_difference": float(pair_hic[select].mean())})
    pd.DataFrame(group_rows).to_csv(out_dir / "pair_group_summary.csv", index=False)
    cluster_rows = []
    for cluster in np.unique(clusters):
        design_select = clusters == cluster
        pair_select = (clusters[pairs[:, 0]] == cluster) & (clusters[pairs[:, 1]] == cluster)
        cluster_rows.append({"cluster": cluster, "num_designs": int(design_select.sum()),
                             "num_within_pairs": int(pair_select.sum()),
                             "within_cluster_rms_spread_g": float(np.sqrt(np.var(curves[design_select], axis=0).mean())),
                             "pooled_within_pair_history_rmse_g": float(np.sqrt(np.mean(pair_rmse[pair_select] ** 2))),
                             "mean_absolute_within_pair_hic_difference": float(pair_hic[pair_select].mean())})
    pd.DataFrame(cluster_rows).to_csv(out_dir / "cluster_response_summary.csv", index=False)
    phase_rows = []
    for start_ms in (0, 5, 10, 15, 20):
        end_seconds = (start_ms + 5) / 1000
        upper = arrays["times"] <= end_seconds if start_ms == 20 else arrays["times"] < end_seconds
        keep = (arrays["times"] >= start_ms / 1000) & upper
        if not keep.any():
            continue
        components = variance_decomposition(curves[..., keep], clusters)
        for loc in range(curves.shape[1]):
            phase_rows.append({"location_id": loc + 1, "start_ms": start_ms, "end_ms": start_ms + 5,
                               "num_times": int(keep.sum()), **{key: value[loc] for key, value in components.items()}})
    pd.DataFrame(phase_rows).to_csv(out_dir / "time_window_variance.csv", index=False)
    summary = {
        "num_designs": len(curves), "num_locations": curves.shape[1], "num_sampled_times": curves.shape[2],
        "within_cluster_pairs": int(same_cluster.sum()), "cross_cluster_pairs": int((~same_cluster).sum()),
        "total_rms_spread_g": float(np.sqrt(total.mean())),
        "within_cluster_rms_spread_g": float(np.sqrt(within.mean())),
        "between_cluster_rms_spread_g": float(np.sqrt(between.mean())),
        "pooled_within_cluster_variance_fraction": float(within.sum() / total.sum()) if total.sum() else None,
        "pooled_between_cluster_variance_fraction": float(between.sum() / total.sum()) if total.sum() else None,
        "examples": frame[frame.location_id.isin([1, 3, 115, 142])].astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
        "pair_groups": group_rows,
        "cluster_summaries": cluster_rows,
        "interpretation": [
            "All designs including previously held-out designs are included for authorized post-hoc diagnosis.",
            "The 66 design pairs are dependent; the 142 impact locations also have spatial dependence.",
            "Variance decomposition weights each design equally; four clusters have sizes 4,2,4,2.",
            "Within-cluster response variation is retained and is not assumed to be numerical noise.",
            "HIC uses exhaustive windows up to 15 ms, in g and seconds; sampled and full grids are separate.",
        ],
    }
    (out_dir / "response_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    augmented = {**arrays, **variance, "pair_designs": pairs, "pair_same_cluster": same_cluster,
                 "pair_curve_rmse": pair_rmse, "pair_hic_absdiff": pair_hic}
    np.savez_compressed(out_dir / "response_arrays.npz", **augmented)
    return frame, summary


def _save(fig, out_dir, name):
    for extension in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"{name}.{extension}", dpi=190, facecolor="white")
    plt.close(fig)


def plot_responses(arrays, frame, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11})
    curves, times, hic = arrays["curves"], arrays["times"] * 1000, arrays["hic"]
    clusters = cluster_mapping()
    examples = [1, 3, 115, 142]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey="row")
    fig.subplots_adjust(left=.055, right=.985, bottom=.09, top=.90, hspace=.38, wspace=.10)
    styles = ["-", "--", "-.", ":"]
    for row, location in enumerate(examples):
        for column, (cluster, designs) in enumerate(clusters.items()):
            ax = axes[row, column]
            for j, design in enumerate(designs):
                ax.plot(times, curves[design, location - 1], color=COLORS[column], linestyle=styles[j],
                        linewidth=1.5, alpha=.95, label=f"D{design}: HIC {hic[design, location - 1]:.0f}")
            ax.set_title(f"Location {location} | cluster {cluster}")
            ax.legend(fontsize=8, loc="upper right", framealpha=.9)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel("Acceleration (g)")
            if row == len(examples) - 1:
                ax.set_xlabel("Time (ms)")
    fig.suptitle("Actual acceleration histories for all 12 designs\n"
                 "Same four user-specified impact locations; columns separate geometry clusters", fontsize=17)
    fig.text(.055, .018, f"No model predictions. Same vertical scale within each row. HIC15 is computed on the {len(times)} sampled times used for this analysis.\n"
             "Within-cluster differences remain visible rather than being averaged away; examples are descriptive, not evidence of causation.", fontsize=9, color="#444444")
    _save(fig, out_dir, "ground_truth_histories_12_designs")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.subplots_adjust(left=.055, right=.965, top=.83, bottom=.22, wspace=.32)
    xy = arrays["impact_xy"]
    for ax, values, title, cmap, vmax, label in (
        (axes[0], frame.total_rms_spread_g, "All-design response spread", "viridis", None, "Across-design RMS spread (g)"),
        (axes[1], frame.within_cluster_rms_spread_g, "Within-cluster response spread", "magma", None, "Within-cluster RMS spread (g)"),
        (axes[2], 100 * frame.within_cluster_variance_fraction, "Fraction arising within clusters", "cividis", 100, "Within-cluster share of variance (%)"),
    ):
        points = ax.scatter(xy[:, 1], xy[:, 0], c=values, cmap=cmap, vmin=0, vmax=vmax, s=32)
        for loc in examples:
            ax.annotate(str(loc), (xy[loc - 1, 1], xy[loc - 1, 0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.scatter(*xy[loc - 1, ::-1], facecolors="none", edgecolors="black", s=60, linewidths=.8)
        ax.set(title=title, xlabel="X2 / Y (mm)", ylabel="X1 / X (mm)", aspect="equal")
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(points, ax=ax, fraction=.05, pad=.03, label=label)
    fig.suptitle("Where do acceleration histories vary across designs?", fontsize=17)
    fig.text(.055, .045, "Exact design-weighted decomposition: total variance = within-cluster variance + between-cluster variance.\n"
             "Each component is averaged over sampled times before taking its square root. A high percentage can occur where absolute variation is small.\n"
             "All 12 designs are included for post-hoc diagnosis; within-cluster variation is not automatically numerical noise.", fontsize=9, color="#444444")
    _save(fig, out_dir, "response_spread_decomposition")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/change_geometry_analysis/responses")
    args = parser.parse_args(argv)
    arrays = load_responses(args.data_dir, stride=args.stride, run_dir=args.run_dir)
    frame, summary = summarize_responses(arrays, args.out_dir)
    plot_responses(arrays, frame, args.out_dir)
    (args.out_dir / "response_provenance.json").write_text(json.dumps({
        "data_dir": str(args.data_dir.resolve()), "run_dir": str(args.run_dir.resolve()) if args.run_dir else None,
        "stride": int(arrays["stride"]), "canonical_run_mapping_verified": True,
        "canonical_impact_mapping_verified": True, "exact_common_full_time_grid_verified": True,
        "model_or_training_changes": False, "scope": "all 1704 ground-truth responses, post-hoc",
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
