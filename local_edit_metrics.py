"""Edit-effect metrics of a run trained on the local-edit dataset.

Every edited run is scored against design 6's true curve at the same impact
location. Design 6 is the base the edits were made on. When a run also trained
on it (``--train-on-base-design``), its own runs are not edits: they are left
out of every edit-effect score and baseline below, and reported on their own
as ``base_design_fit`` (how well the training fit the unedited hood).

    d(t) = a_design(t) - a_D6(t)        true edit effect
    p(t) = a_predicted(t) - a_D6(t)     predicted edit effect
    difference R2 = 1 - sum((p - d)^2) / sum(d^2)

pooled over the runs and sampled times of a design or split. It equals
1 - MSE(model) / MSE(copy design 6): 0 is no better than predicting design 6's
curve, 1 is perfect. This is the design-difference R2 of the 2026-11-21 data
analysis (Data/HoodImpact_LocalEdits/inspection), so its numbers compare directly:
copying the same-sign siblings scored 0.47 pooled over all designs, the
mirror-twin repeatability 0.63, a perfect model about 0.8.

Baselines on the same runs, built from training designs only:
    sibling_copy   design 6 plus the mean true effect of the training designs
                   with the same site (the same edited beams), the same sign
                   (shallower or deeper) and the same location.
    location_mean  the mean true curve of the training designs at that location
                   (knows the location, ignores which edit it is).
A baseline is scored on the runs it exists for; ``model_r2_same_runs`` is the
model on those runs.

Windows: 0-5, 5-12 and 12 ms to the end (the effect repeated at mirror twins
with R2 0.95, 0.78 and 0.50 there). HIC15 is computed on the sampled grid for
truth, prediction, design 6 and the sibling copy, so it is approximate but the
same for all four.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hic15 import batched_hic
from local_edits_data import BASE_DESIGN, BASE_DESIGN_ID, design_name, load_manifest, sample_design, sample_location

REPORT_FILE = "local_edit_metrics.json"
RUNS_FILE = "local_edit_runs.csv"
WINDOWS_MS = ((0.0, 5.0), (5.0, 12.0), (12.0, None))
DEFINITION = ("d = a_design - a_D6 (true), p = a_predicted - a_D6, both at the same impact location; "
              "difference R2 = 1 - sum((p - d)^2) / sum(d^2) pooled over runs and sampled times. "
              "0 = copying design 6's curve, 1 = perfect.")
# Text and series colours (dataviz reference palette).
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
TRUE_COLOR, PRED_COLOR = "#2a78d6", "#eb6834"


def design_classes(root) -> pd.DataFrame:
    """Per design: class (crease / softening notch / deepening), site and sign, from the DOE tables."""
    meta = pd.read_csv(Path(root) / "local_edit_designs.csv").set_index("design")
    doe = pd.read_csv(Path(root) / "local_edit_doe.csv")
    site = doe.groupby("design")["site"].apply(lambda s: "+".join(sorted(set(s))))
    sign = np.where(meta["dz_max_mm"] > 0.5, 1, -1)
    cls = np.where(meta["variant"].isin(["K", "KH"]), "crease", np.where(sign < 0, "deepening", "softening notch"))
    return pd.DataFrame({"site": site.reindex(meta.index), "sign": sign, "class": cls,
                         "variant": meta["variant"], "kind": meta["kind"]}, index=meta.index)


def _window_masks(times):
    ms = 1e3 * np.asarray(times, dtype=np.float64)
    masks = []
    for start, stop in WINDOWS_MS:
        mask = ms >= start if stop is None else (ms >= start) & (ms < stop)
        masks.append(mask)
    return masks


def _has(curve):
    return isinstance(curve, np.ndarray)


def _r2(error_sq, signal_sq):
    return float(1.0 - error_sq / signal_sq) if signal_sq > 0 else None


class _Pool:
    """Running sums for a pooled difference R2, overall and per window."""

    def __init__(self, masks):
        self.masks = masks
        self.error = np.zeros(len(masks) + 1)
        self.signal = np.zeros(len(masks) + 1)
        self.runs = 0

    def add(self, prediction_effect, true_effect):
        error, signal = np.square(prediction_effect - true_effect), np.square(true_effect)
        self.error[0] += error.sum()
        self.signal[0] += signal.sum()
        for index, mask in enumerate(self.masks, start=1):
            self.error[index] += error[mask].sum()
            self.signal[index] += signal[mask].sum()
        self.runs += 1

    def r2(self):
        return _r2(self.error[0], self.signal[0])

    def windows(self):
        return [_r2(e, s) for e, s in zip(self.error[1:], self.signal[1:])]


def _curves(values, dataset, steps):
    return np.asarray(values, dtype=np.float64).reshape(len(dataset.run_numbers), steps)


def _base_design_fit(truth, predicted, times):
    """Fit of design 6's own runs (a training design when --train-on-base-design): acceleration and HIC15."""
    hic_true, hic_pred = batched_hic(times, truth), batched_hic(times, predicted)
    error = 100.0 * (hic_pred / hic_true - 1.0)
    return {"runs": int(len(truth)),
            "accel_rmse_g": float(np.sqrt(np.mean((predicted - truth) ** 2))),
            "hic15_sampled_error_mean_abs_pct": float(np.abs(error).mean()),
            "hic15_sampled_error_max_abs_pct": float(np.abs(error).max())}


def local_edit_report(splits, base_curves, base_times, manifest_path):
    """Score ``splits`` = {name: (dataset, predictions, targets)}; "train" must be one of them.

    Returns (report dict, per-run DataFrame, curves for plotting).
    """
    root = Path(manifest_path).parent
    manifest = load_manifest(manifest_path).set_index("sample")
    classes = design_classes(root)
    steps = len(base_times)
    for name, (dataset, _, _) in splits.items():
        if any(not np.array_equal(base_times, times) for times in dataset.time_arrays):
            raise ValueError(f"The {name} curves are not on design 6's sampled time grid")
    masks = _window_masks(base_times)

    # Baselines come from the edited training designs only, with or without design 6 in training.
    train_set, _, train_targets = splits["train"]
    train_design = np.array([sample_design(s) for s in train_set.run_numbers])
    edited = train_design != BASE_DESIGN_ID
    train_curves = _curves(train_targets, train_set, steps)[edited]
    train_design = train_design[edited]
    train_location = np.array([sample_location(s) for s in train_set.run_numbers])[edited]
    train_names = [design_name(d) for d in train_design]
    train_site = np.array([classes.loc[n, "site"] for n in train_names])
    train_sign = np.array([classes.loc[n, "sign"] for n in train_names])

    records, plot_curves, base_fits = [], {}, {}
    for split, (dataset, predictions, targets) in splits.items():
        truth, predicted = _curves(targets, dataset, steps), _curves(predictions, dataset, steps)
        base_rows = [i for i, s in enumerate(dataset.run_numbers) if sample_design(s) == BASE_DESIGN_ID]
        if base_rows:
            base_fits[split] = _base_design_fit(truth[base_rows], predicted[base_rows], base_times)
        for index, sample in enumerate(dataset.run_numbers):
            design, location = sample_design(sample), sample_location(sample)
            if design == BASE_DESIGN_ID:
                continue
            name, row = design_name(design), manifest.loc[int(sample)]
            if location not in base_curves:
                raise ValueError(f"No design-6 curve at location {location}")
            base = base_curves[location]
            d, p = truth[index] - base, predicted[index] - base
            info = classes.loc[name]
            others = (train_location == location) & (train_design != design)
            siblings = others & (train_site == info["site"]) & (train_sign == info["sign"])
            sibling = train_curves[siblings].mean(axis=0) - base if siblings.any() else None
            location_mean = train_curves[others].mean(axis=0) - base if others.any() else None
            records.append({
                "split": split, "sample": int(sample), "source_run": int(row["run"]), "design": name,
                "location": location, "role": row["role"], "dist_to_edit_mm": row.get("dist_to_edit_mm"),
                "class": info["class"], "site": info["site"],
                "effect_rms_true_g": float(np.sqrt(np.mean(d ** 2))),
                "effect_rms_pred_g": float(np.sqrt(np.mean(p ** 2))),
                "error_rms_g": float(np.sqrt(np.mean((p - d) ** 2))),
                "difference_r2": _r2(np.square(p - d).sum(), np.square(d).sum()),
                "siblings": int(siblings.sum()),
                "sibling_r2": None if sibling is None else _r2(np.square(sibling - d).sum(), np.square(d).sum()),
                "curve_d": d, "curve_p": p, "curve_sibling": sibling, "curve_location_mean": location_mean,
                "curve_truth": truth[index], "curve_pred": predicted[index], "curve_base": base,
            })
    frame = pd.DataFrame(records)

    # HIC15 on the sampled grid, batched per quantity.
    def hic(column):
        return batched_hic(base_times, np.stack(frame[column].to_list()))

    frame["hic_true"], frame["hic_pred"], frame["hic_d6"] = hic("curve_truth"), hic("curve_pred"), hic("curve_base")
    with_sibling = frame["curve_sibling"].map(_has)
    frame["hic_sibling"] = np.nan
    if with_sibling.any():
        frame.loc[with_sibling, "hic_sibling"] = batched_hic(
            base_times, np.stack([b + s for b, s in zip(frame.loc[with_sibling, "curve_base"],
                                                         frame.loc[with_sibling, "curve_sibling"])]))
    for key in ("true", "pred", "sibling"):
        frame[f"hic_change_{key}_pct"] = 100.0 * (frame[f"hic_{key}"] / frame["hic_d6"] - 1.0)

    def summarize(rows):
        model, sibling, sibling_model = _Pool(masks), _Pool(masks), _Pool(masks)
        location, location_model = _Pool(masks), _Pool(masks)
        for r in rows.itertuples():
            model.add(r.curve_p, r.curve_d)
            if _has(r.curve_sibling):
                sibling.add(r.curve_sibling, r.curve_d)
                sibling_model.add(r.curve_p, r.curve_d)
            if _has(r.curve_location_mean):
                location.add(r.curve_location_mean, r.curve_d)
                location_model.add(r.curve_p, r.curve_d)
        true_change, pred_change = rows["hic_change_true_pct"], rows["hic_change_pred_pct"]
        hic_true, hic_pred = rows["hic_true"], rows["hic_pred"]
        sib_rows = rows[rows["hic_sibling"].notna()]
        return {
            "runs": int(len(rows)), "designs": sorted(rows["design"].unique(), key=lambda n: int(n[1:])),
            "difference_r2": model.r2(), "windows_difference_r2": model.windows(),
            "effect_rms_true_g": float(np.sqrt(np.mean(np.stack(rows["curve_d"].to_list()) ** 2))),
            "error_rms_g": float(np.sqrt(np.mean((np.stack(rows["curve_p"].to_list())
                                                  - np.stack(rows["curve_d"].to_list())) ** 2))),
            "baselines": {
                "copy_d6": {"difference_r2": 0.0},
                "sibling_copy": {"runs": sibling.runs, "difference_r2": sibling.r2(),
                                 "windows_difference_r2": sibling.windows(),
                                 "model_r2_same_runs": sibling_model.r2(),
                                 "model_windows_same_runs": sibling_model.windows()},
                "location_mean": {"runs": location.runs, "difference_r2": location.r2(),
                                  "windows_difference_r2": location.windows(),
                                  "model_r2_same_runs": location_model.r2()},
            },
            "hic15_sampled": {
                "r2": _r2(np.square(hic_pred - hic_true).sum(), np.square(hic_true - hic_true.mean()).sum()),
                "change_r2": _r2(np.square(pred_change - true_change).sum(), np.square(true_change).sum()),
                "change_mae_pct_points": {
                    "model": float(np.abs(pred_change - true_change).mean()),
                    "copy_d6": float(np.abs(true_change).mean()),
                    "sibling_copy_same_runs": float(np.abs(sib_rows["hic_change_sibling_pct"]
                                                           - sib_rows["hic_change_true_pct"]).mean())
                    if len(sib_rows) else None,
                    "model_same_runs": float(np.abs(sib_rows["hic_change_pred_pct"]
                                                    - sib_rows["hic_change_true_pct"]).mean())
                    if len(sib_rows) else None,
                },
            },
        }

    report = {"metric": "edit-effect (design - design 6) difference R2", "definition": DEFINITION,
              "base_design": BASE_DESIGN,
              "base_use": ("reference; also a training design, left out of the edit scores and baselines"
                           if base_fits else "reference only; never trained on"),
              "windows_ms": [[a, b] for a, b in WINDOWS_MS],
              "time_points": steps, "splits": {}}
    for split in splits:
        rows = frame[frame["split"] == split]
        entry = summarize(rows)
        if split in base_fits:
            entry["base_design_fit"] = base_fits[split]
        entry["by_class"] = {cls: summarize(group) for cls, group in rows.groupby("class")}
        if split != "train":
            entry["by_design"] = {name: summarize(group) for name, group in rows.groupby("design")}
        report["splits"][split] = entry
    for split in splits:
        if split != "train":
            for name in report["splits"][split]["designs"]:
                plot_curves[name] = (split, frame[frame["design"] == name])
    runs = frame.drop(columns=[c for c in frame.columns if c.startswith("curve_")])
    return report, runs, plot_curves


def local_edit_log_values(report, prefix="local_edits"):
    values = {}
    for split, entry in report["splits"].items():
        key = f"{prefix}/{split}"
        values[f"{key}/difference_r2"] = entry["difference_r2"]
        for index, value in enumerate(entry["windows_difference_r2"], start=1):
            values[f"{key}/difference_r2_window_{index}"] = value
        sibling = entry["baselines"]["sibling_copy"]
        values[f"{key}/sibling_copy_r2"] = sibling["difference_r2"]
        values[f"{key}/model_r2_on_sibling_runs"] = sibling["model_r2_same_runs"]
        values[f"{key}/location_mean_r2"] = entry["baselines"]["location_mean"]["difference_r2"]
        mae = entry["hic15_sampled"]["change_mae_pct_points"]
        values[f"{key}/hic_change_mae_pct"] = mae["model"]
        values[f"{key}/copy_d6_hic_change_mae_pct"] = mae["copy_d6"]
        values[f"{key}/sibling_hic_change_mae_pct"] = mae["sibling_copy_same_runs"]
        values[f"{key}/hic_change_r2"] = entry["hic15_sampled"]["change_r2"]
        if "base_design_fit" in entry:
            values[f"{key}/base_design_accel_rmse_g"] = entry["base_design_fit"]["accel_rmse_g"]
            values[f"{key}/base_design_hic_error_mean_abs_pct"] = entry["base_design_fit"][
                "hic15_sampled_error_mean_abs_pct"]
    return {k: v for k, v in values.items() if v is not None and np.isfinite(v)}


def plot_design_effects(name, split, rows, times, path):
    """True vs predicted edit effect at each of a design's locations, sibling copy dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {"near": 0, "twin": 1, "far": 2}
    rows = rows.assign(_order=rows["role"].map(order).fillna(3)).sort_values(["_order", "dist_to_edit_mm"])
    count = len(rows)
    columns = 5
    lines = -(-count // columns)
    ms = 1e3 * np.asarray(times)
    fig, axes = plt.subplots(lines, columns, figsize=(3.0 * columns, 2.2 * lines + 0.9),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, rows.itertuples()):
        ax.set_facecolor(SURFACE)
        ax.axhline(0.0, color="#c3c2b7", linewidth=0.8)
        if _has(r.curve_sibling):
            ax.plot(ms, r.curve_sibling, color=MUTED, linewidth=1.2, linestyle="--")
        ax.plot(ms, r.curve_d, color=TRUE_COLOR, linewidth=1.8)
        ax.plot(ms, r.curve_p, color=PRED_COLOR, linewidth=1.8)
        distance = "" if pd.isna(r.dist_to_edit_mm) else f" {r.dist_to_edit_mm:.0f} mm"
        score = "n/a" if r.difference_r2 is None else f"{r.difference_r2:.2f}"
        ax.set_title(f"loc {r.location} ({r.role}{distance})  R² {score}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7.5, colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)", fontsize=8, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("effect (g)", fontsize=8, color="#52514e")
    model = _r2(sum(np.square(p - d).sum() for p, d in zip(rows["curve_p"], rows["curve_d"])),
                sum(np.square(d).sum() for d in rows["curve_d"]))
    with_sibling = rows[rows["curve_sibling"].map(_has)]
    sibling = _r2(sum(np.square(s - d).sum() for s, d in zip(with_sibling["curve_sibling"], with_sibling["curve_d"])),
                  sum(np.square(d).sum() for d in with_sibling["curve_d"])) if len(with_sibling) else None
    fig.suptitle(f"{name} ({split}): edit effect = design - design 6.  Difference R² model "
                 f"{model:.2f}" + ("" if sibling is None else f", sibling copy {sibling:.2f}"),
                 fontsize=11, color=INK, y=0.995)
    handles = [plt.Line2D([], [], color=TRUE_COLOR, linewidth=1.8),
               plt.Line2D([], [], color=PRED_COLOR, linewidth=1.8),
               plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle="--")]
    fig.legend(handles, ["simulation", "model", "sibling copy (training designs)"], loc="upper center",
               ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_local_edit_report(directory, report, runs, plot_curves, times):
    directory = Path(directory)

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    (directory / REPORT_FILE).write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    runs.to_csv(directory / RUNS_FILE, index=False)
    figures = []
    for name, (split, rows) in plot_curves.items():
        figures.append(plot_design_effects(name, split, rows, times, directory / f"local_edit_effects_{name}.png"))
    return figures


def time_window_metrics(train, test, num_windows):
    """Test RMSE (g) in each of ``num_windows`` contiguous windows of the time grid.

    ``train`` and ``test`` are (dataset, predictions, targets), predictions and
    targets flat in dataset order. Windows hold ceil(T / num_windows) times
    each. Besides the model, each window scores the location mean: the mean
    true training curve at the test curve's impact location, which ignores the
    geometry. Location-mean RMSE uses only test curves whose location is in
    training.
    """
    (train_set, _, train_targets), (test_set, predictions, targets) = train, test
    grids = [*train_set.time_arrays, *test_set.time_arrays]
    if any(not np.array_equal(grids[0], grid) for grid in grids):
        raise ValueError("Time windows need one time grid shared by all training and test runs")
    steps = len(grids[0])
    size = -(-steps // num_windows)
    edges = list(range(0, steps, size)) + [steps]

    def curves(values, dataset):
        return np.asarray(values, dtype=np.float64).reshape(len(dataset.run_numbers), steps)

    truth = curves(targets, test_set)
    # The location mean is a baseline of the edited training designs, as in local_edit_report.
    edited = np.array([sample_design(run) != BASE_DESIGN_ID for run in train_set.run_numbers])
    training = curves(train_targets, train_set)[edited]
    train_locations = np.array([sample_location(run) for run in train_set.run_numbers])[edited]
    test_locations = np.array([sample_location(run) for run in test_set.run_numbers])
    means = {place: training[train_locations == place].mean(axis=0) for place in set(train_locations.tolist())}
    covered = np.array([place in means for place in test_locations.tolist()])
    location_mean = np.array([means.get(place, np.zeros(steps)) for place in test_locations.tolist()])

    def rmse(prediction, rows=None):
        if rows is not None and not rows.any():
            return None
        error = curves(prediction, test_set) - truth
        error = error if rows is None else error[rows]
        return [float(np.sqrt(np.square(error[:, a:b]).mean())) for a, b in zip(edges[:-1], edges[1:])]

    times_ms = 1e3 * np.asarray(grids[0], dtype=np.float64)
    return {
        "num_windows": len(edges) - 1,
        "window_ms": [[float(times_ms[a]), float(times_ms[b - 1])] for a, b in zip(edges[:-1], edges[1:])],
        "window_points": [b - a for a, b in zip(edges[:-1], edges[1:])],
        "model_rmse_g": rmse(predictions),
        "location_mean_rmse_g": rmse(location_mean, covered),
        "test_signal_std_g": [float(truth[:, a:b].std()) for a, b in zip(edges[:-1], edges[1:])],
        "num_test_curves": len(truth),
        "num_test_curves_with_location_mean": int(covered.sum()),
    }
