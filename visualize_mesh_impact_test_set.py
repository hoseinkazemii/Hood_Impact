"""Plot every held-out acceleration history and the corresponding HIC errors.

Reads the saved prediction CSV and ``hic_results_pred_vs_gt.csv`` produced by
``evaluate_mesh_impact_hic.py``. No model or dataset loading is required.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


_HISTORY_COLUMNS = (
    "run_number", "time", "acceleration_true_g", "acceleration_pred_g"
)
_METRIC_COLUMNS = (
    "run_number", "design_id", "location_id", "hic_ground_truth",
    "hic_prediction", "hic_error", "hic_relative_error_pct",
    "acceleration_rmse_g", "acceleration_mae_g", "acceleration_r2",
)
_COLORS = {"truth": "#1765a1", "prediction": "#df7126", "residual": "#7861a2"}


def _as_run_ids(values, label):
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} run numbers must be finite integers")
    return numeric.astype(np.int64)


def _load_validated_histories(run_dir, hic_results, histories=None):
    if histories is None:
        histories = pd.read_csv(run_dir / "test_acceleration_histories.csv")
    else:
        histories = histories.copy()
    for frame, columns, label in (
        (histories, _HISTORY_COLUMNS, "Acceleration history"),
        (hic_results, _METRIC_COLUMNS, "HIC results"),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{label} CSV is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{label} must contain at least one test impact")
    metrics = hic_results.copy()
    histories["run_number"] = _as_run_ids(histories["run_number"], "Acceleration history")
    metrics["run_number"] = _as_run_ids(metrics["run_number"], "HIC results")
    if metrics["run_number"].duplicated().any():
        raise ValueError("HIC results must have exactly one row per run")
    history_runs = set(histories["run_number"])
    metric_runs = set(metrics["run_number"])
    if history_runs != metric_runs:
        raise ValueError(
            "History and HIC run sets differ: "
            f"missing HIC={sorted(history_runs - metric_runs)}, "
            f"missing histories={sorted(metric_runs - history_runs)}"
        )
    groups = {}
    for run_number, group in histories.groupby("run_number", sort=False):
        # Validate the stored order; do not reorder or interpolate any history.
        values = group[list(_HISTORY_COLUMNS[1:])].to_numpy(dtype=np.float64)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Run {run_number}: expected at least two finite time/acceleration rows")
        if not np.all(np.diff(values[:, 0]) > 0):
            raise ValueError(f"Run {run_number}: saved time must be strictly increasing")
        groups[int(run_number)] = values
    metrics = metrics.sort_values(["design_id", "run_number"], kind="stable")
    return groups, metrics


def _number(value, decimals=2, signed=False):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "undefined"
    return format(float(value), f"{'+' if signed else ''}.{decimals}f")


def _identifier(value):
    if pd.isna(value):
        return "unknown"
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass
    return str(value)


def _impact_label(row):
    x, y = row.get("impact_x_mm"), row.get("impact_y_mm")
    if x is not None and y is not None and pd.notna(x) and pd.notna(y):
        return f"Impact XY = ({_number(x, 1)}, {_number(y, 1)}) mm"
    return ""


def _title(row):
    return (
        f"Run {_identifier(row['run_number'])} | Design {_identifier(row['design_id'])}"
        f" | Location {_identifier(row['location_id'])}"
    )


def _draw_history(ax, values, row, compact=False):
    time_ms, truth, prediction = values[:, 0] * 1000.0, values[:, 1], values[:, 2]
    linewidth = 1.5 if compact else 2.0
    ax.plot(time_ms, truth, color=_COLORS["truth"], linewidth=linewidth,
            label="Simulation (ground truth)")
    ax.plot(time_ms, prediction, color=_COLORS["prediction"], linewidth=linewidth,
            linestyle="--", label="Model prediction")
    ax.set_xlim(time_ms[0], time_ms[-1])
    ax.set_ylabel("Acceleration (g)")
    ax.grid(alpha=0.22)
    if compact:
        detail = (
            f"HIC: {_number(row['hic_ground_truth'], 1)} / {_number(row['hic_prediction'], 1)}"
            f" (GT / pred), error {_number(row['hic_relative_error_pct'], 1, True)}%\n"
            f"RMSE {_number(row['acceleration_rmse_g'], 2)} g; "
            f"R² {_number(row['acceleration_r2'], 3)}"
        )
        ax.set_title(_title(row) + "\n" + detail, fontsize=9, loc="left", pad=7)
        ax.set_xlabel("Time (ms)")
        ax.tick_params(labelsize=8)


def _export_individual_plot(path, values, row, stride_label):
    fig, (history_ax, residual_ax) = plt.subplots(
        2, 1, figsize=(10.4, 6.7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.1},
    )
    try:
        fig.suptitle(_title(row), fontsize=15, fontweight="bold", y=0.98)
        subtitle = _impact_label(row)
        if subtitle:
            fig.text(0.5, 0.936, subtitle, ha="center", fontsize=10, color="#555555")
        _draw_history(history_ax, values, row)
        history_ax.legend(loc="upper right", frameon=True, fontsize=9)
        history_ax.tick_params(axis="x", labelbottom=False)
        time_ms = values[:, 0] * 1000.0
        residual = values[:, 2] - values[:, 1]
        residual_ax.plot(time_ms, residual, color=_COLORS["residual"], linewidth=1.4)
        residual_ax.fill_between(time_ms, 0, residual, color=_COLORS["residual"], alpha=0.15)
        residual_ax.axhline(0, color="#555555", linewidth=0.8)
        residual_ax.set(xlabel="Time (ms)", ylabel="Pred − GT (g)")
        residual_ax.grid(alpha=0.2)
        metrics_line = (
            f"Acceleration RMSE: {_number(row['acceleration_rmse_g'])} g    "
            f"MAE: {_number(row['acceleration_mae_g'])} g    "
            f"R²: {_number(row['acceleration_r2'], 4)}\n"
            f"HIC15: GT {_number(row['hic_ground_truth'])}  |  "
            f"Prediction {_number(row['hic_prediction'])}  |  "
            f"Error {_number(row['hic_error'], 2, True)} "
            f"({_number(row['hic_relative_error_pct'], 2, True)}%)"
        )
        fig.text(0.5, 0.058, metrics_line, ha="center", va="center", fontsize=10,
                 linespacing=1.7)
        fig.text(0.5, 0.009,
                 f"HIC15 uses the saved sampled histories ({stride_label}); "
                 "HIC error = prediction − ground truth. No additional resampling.",
                 ha="center", fontsize=8, color="#666666")
        fig.subplots_adjust(left=0.085, right=0.975, top=0.89, bottom=0.17)
        fig.savefig(path, dpi=145, facecolor="white")
    finally:
        plt.close(fig)


def _export_pdf(path, groups, rows, stride_label, heading, noun):
    count = len(rows)
    page_count = math.ceil(count / 6)
    with PdfPages(path) as pdf:
        pdf.infodict()["Title"] = f"{heading} — MeshImpactHistoryNet"
        pdf.infodict()["Subject"] = (
            f"All {count} {noun}; HIC15 from saved sampled histories ({stride_label})"
        )
        for page, start in enumerate(range(0, count, 6), start=1):
            fig, axes = plt.subplots(3, 2, figsize=(11.7, 13.0))
            try:
                fig.suptitle(
                    f"{heading}  •  {count} impacts",
                    fontsize=15, y=0.986,
                )
                fig.text(0.5, 0.963,
                         f"Blue: simulation  |  Orange dashed: prediction  |  "
                         f"HIC15 from sampled histories ({stride_label})",
                         ha="center", fontsize=9)
                for ax, row in zip(axes.flat, rows[start:start + 6]):
                    _draw_history(ax, groups[int(row["run_number"])], row, compact=True)
                for ax in list(axes.flat)[len(rows[start:start + 6]):]:
                    ax.set_visible(False)
                fig.text(0.5, 0.014,
                         f"Page {page}/{page_count}  |  {noun.capitalize()} {start + 1}–{min(start + 6, count)} "
                         f"of {count}  |  HIC error = prediction − ground truth; design IDs are zero based.",
                         ha="center", fontsize=8, color="#666666")
                fig.subplots_adjust(left=0.075, right=0.975, top=0.91, bottom=0.06,
                                    hspace=0.49, wspace=0.23)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
    return page_count


def _sort_metric(value):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "-Infinity"
    return str(abs(float(value)))


def _export_gallery(path, rows, image_paths, stride_label, heading, noun, pdf_name):
    escape = html.escape
    cards = []
    for row, image_path in zip(rows, image_paths):
        run = int(row["run_number"])
        card_heading = _title(row)
        location = _impact_label(row)
        search = f"{card_heading} {location}".lower()
        image_url = image_path.relative_to(path.parent).as_posix()
        error = _number(row["hic_relative_error_pct"], 2, True)
        cards.append(
            f'<article class="card" data-run="{run}" '
            f'data-search="{escape(search, quote=True)}" '
            f'data-hic="{_sort_metric(row["hic_relative_error_pct"])}" '
            f'data-rmse="{_sort_metric(row["acceleration_rmse_g"])}">\n'
            f'<a class="plot-link" href="{escape(image_url, quote=True)}" target="_blank" rel="noopener">'
            f'<img src="{escape(image_url, quote=True)}" alt="{escape(card_heading, quote=True)}: '
            'simulation and predicted acceleration with residuals" loading="lazy" width="1508" height="971"></a>\n'
            f'<div class="card-body"><h2>{escape(card_heading)}</h2>'
            f'<p class="location">{escape(location)}</p>'
            f'<p>HIC15: <strong>{_number(row["hic_ground_truth"], 1)}</strong> GT → '
            f'<strong>{_number(row["hic_prediction"], 1)}</strong> predicted '
            f'<span class="badge">{error}%</span></p>'
            f'<p>RMSE {_number(row["acceleration_rmse_g"])} g · '
            f'MAE {_number(row["acceleration_mae_g"])} g · '
            f'R² {_number(row["acceleration_r2"], 4)}</p></div></article>'
        )
    # Raw: the only backslash in this template is the JavaScript \s+ pattern.
    document = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__HEADING__ | MeshImpactHistoryNet</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; background: #f2f5f8; color: #17283c; font: 15px/1.5 system-ui, sans-serif; }
header, main { max-width: 1560px; margin: auto; padding: 25px; }
header { padding-bottom: 10px; }
h1 { font-size: clamp(24px, 3vw, 34px); margin: 0 0 8px; }
p { margin: 6px 0; }
.explanation { max-width: 1000px; color: #48586a; }
.controls { display: flex; gap: 14px; align-items: end; flex-wrap: wrap; margin: 18px 0 10px; }
label { display: flex; flex-direction: column; gap: 5px; font-weight: 600; }
input, select, button { padding: 10px 12px; border: 1px solid #bac5d1; border-radius: 7px;
                       font: inherit; background: white; color: inherit; }
input { width: min(440px, 85vw); }
button { cursor: pointer; }
a { color: #1765a1; }
.count { font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(430px, 100%), 1fr)); gap: 22px; }
.card { border: 1px solid #dce2e9; border-radius: 10px; background: white; overflow: hidden; }
.card[hidden] { display: none; }
.plot-link { display: block; }
img { width: 100%; height: auto; display: block; }
.card-body { border-top: 1px solid #eef0f3; padding: 12px 16px 15px; }
h2 { font-size: 17px; margin: 0 0 5px; }
.location { color: #5b6979; font-size: 13px; }
.badge { background: #edf1f6; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; }
.truth { color: #1765a1; font-weight: 650; }
.prediction { color: #b8540f; font-weight: 650; }
.residual { color: #7861a2; font-weight: 650; }
.empty { padding: 30px; background: white; border-radius: 10px; }
</style>
</head>
<body>
<header>
<h1>__HEADING__</h1>
<p class="explanation">MeshImpactHistoryNet · <strong>__COUNT__ __NOUN__</strong> ·
Click any plot to inspect the full image. The lines show every saved time sample without additional resampling.
Every one of these impacts is included below.</p>
<p class="explanation">HIC15 is calculated from both saved acceleration histories (__STRIDE__).
The ground-truth HIC shown here uses the sampled simulation history, not its full-resolution source.
Signed HIC error = prediction − ground truth. Relative error divides by ground truth.
Design IDs are zero based.</p>
<div class="legend"><span class="truth">━ Simulation (ground truth)</span>
<span class="prediction">┄ Model prediction</span><span class="residual">━ Residual: prediction − ground truth</span></div>
<div class="controls">
<label for="search">Find impact<input id="search" type="search" placeholder="Run, location, design, or XY coordinate" autocomplete="off"></label>
<label for="sort">Order<select id="sort"><option value="run">Run / location order</option>
<option value="hic">Largest absolute HIC relative error first</option>
<option value="rmse">Largest acceleration RMSE first</option></select></label>
<button id="reset" type="button">Reset</button>
<a href="__PDF__">Download all histories as PDF</a>
</div>
<p class="count" id="count" aria-live="polite">Showing __COUNT__ of __COUNT__ __NOUN__</p>
<noscript><p>All plots are visible; search and sorting require JavaScript.</p></noscript>
</header>
<main><div class="grid" id="grid">__CARDS__</div>
<p id="empty" class="empty" hidden>No impacts match this search. Clear the search to see every impact.</p></main>
<script>
const grid = document.getElementById('grid');
const cards = Array.from(grid.querySelectorAll('.card'));
const search = document.getElementById('search');
const sort = document.getElementById('sort');
function update() {
  const words = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const key = sort.value;
  cards.sort((a, b) => {
    if (key === 'run') return Number(a.dataset.run) - Number(b.dataset.run);
    const difference = Number(b.dataset[key]) - Number(a.dataset[key]);
    return (Number.isNaN(difference) ? 0 : difference) || Number(a.dataset.run) - Number(b.dataset.run);
  });
  let visible = 0;
  for (const card of cards) {
    card.hidden = !words.every(word => card.dataset.search.includes(word));
    if (!card.hidden) visible += 1;
    grid.appendChild(card);
  }
  document.getElementById('count').textContent = `Showing ${visible} of ${cards.length} __NOUN__`;
  document.getElementById('empty').hidden = visible !== 0;
}
search.addEventListener('input', update);
sort.addEventListener('change', update);
document.getElementById('reset').addEventListener('click', () => {
  search.value = ''; sort.value = 'run'; update();
});
</script>
</body>
</html>
"""
    document = document.replace("__COUNT__", str(len(rows))).replace(
        "__STRIDE__", escape(stride_label)
    ).replace("__HEADING__", escape(heading)).replace(
        "__NOUN__", escape(noun)
    ).replace("__PDF__", escape(pdf_name, quote=True)).replace("__CARDS__", "\n".join(cards))
    path.write_text(document, encoding="utf-8")


def export_all_test_plots(
    run_dir,
    hic_results: pd.DataFrame,
    output_dir=None,
    histories: pd.DataFrame | None = None,
    slug: str = "test",
    heading: str = "All test acceleration predictions",
    noun: str = "test impacts",
) -> dict:
    """Export one PNG per run, a multipage PDF, and an offline HTML gallery.

    The result contains the exact run IDs, image count, PDF page count, and paths.
    ``output_dir`` is the report base directory and defaults to ``run_dir``.
    Histories and HIC results must contain exactly the same set of run numbers.

    ``histories`` supplies the acceleration histories directly; when omitted they
    are read from ``<run_dir>/test_acceleration_histories.csv``.  ``slug``,
    ``heading`` and ``noun`` name the outputs, so the same report can be produced
    for a set of runs that is not the test split -- training designs, say -- and
    still say so on every page.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else run_dir
    groups, metrics = _load_validated_histories(run_dir, hic_results, histories)
    rows = metrics.to_dict(orient="records")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    stride = config.get("preprocessing", {}).get("time_subsample_stride")
    stride_label = f"stride {stride}" if stride is not None else "saved time grid"
    image_dir = output_dir / f"{slug}_acceleration_plots"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index, row in enumerate(rows, start=1):
        run_number = int(row["run_number"])
        image_path = image_dir / f"run_{run_number:04d}.png"
        _export_individual_plot(image_path, groups[run_number], row, stride_label)
        images.append(image_path)
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(f"Exported acceleration plots ({slug}): {index}/{len(rows)}", flush=True)
    pdf_name = f"all_{slug}_acceleration_predictions.pdf"
    pdf_path = output_dir / pdf_name
    page_count = _export_pdf(pdf_path, groups, rows, stride_label, heading, noun)
    gallery_path = output_dir / f"{slug}_acceleration_gallery.html"
    _export_gallery(gallery_path, rows, images, stride_label, heading, noun, pdf_name)
    if len(images) != len(groups) or any(not path.is_file() for path in images):
        raise RuntimeError("Not every impact received an acceleration plot")
    return {
        "test_impact_count": len(rows),
        "png_count": len(images),
        "run_numbers": [int(row["run_number"]) for row in rows],
        "png_directory": str(image_dir),
        "pdf_path": str(pdf_path),
        "pdf_page_count": page_count,
        "gallery_path": str(gallery_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    results = pd.read_csv(args.run_dir / "hic_results_pred_vs_gt.csv")
    exports = export_all_test_plots(args.run_dir, results, args.output_dir)
    print(json.dumps(exports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
