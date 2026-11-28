"""Diagnose a run trained on the local-edit dataset: acceleration accuracy, how well it
tells designs apart, and every design's geometry next to its siblings.

Inputs: the run's train/validation/test_acceleration_histories.csv (best checkpoint) and
local_edit_runs.csv, plus the dataset (manifest, design_changes/, design-6 histories and
the design-6 deck for the inner-panel outline).

Terms
  effect          a_design(t) - a_D6(t) at the same impact location (D6 = base hood, never trained on)
  difference R2   1 - sum((p - d)^2) / sum(d^2) over runs and times; 0 = copy design 6
  sibling copy    D6 + mean effect of the training designs with the same site and sign there
  twin copy       the effect at the mirror-image location of the same design (solver repeatability)
  pair R2         difference R2 between two edited designs A and B at their shared locations,
                  d = a_A - a_B and p = pred_A - pred_B: does the model tell A from B?
  Training designs are scored in-sample.

Outputs in <run>/diagnostics/:
  01_design_atlas.png             all designs: edit geometry + difference R2
  02_acceleration_accuracy.png    error over time, per-run R2, example curves (train and test)
  03_edit_effect_skill.png        how well design effects are predicted, and why not
  04_design_pair_distinction.png  design-vs-design differences at shared locations
  05_<design>_vs_related.png      held-out design vs its relatives, location by location
  06_<design>_all_locations.png   held-out design, full curves at every location
  designs/<design>.png            one card per design: geometry, relatives, HIC, curves
  per_design.csv, per_run.csv, design_pairs.csv, diagnostics_summary.json

Usage:
    python analyze_local_edits_run.py runs/mesh_impact_history/local_edits/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_local_edits_hic import load_full_resolution, load_histories, resolve_data_dir
from local_edit_metrics import design_classes
from local_edits_data import BASE_DESIGN, base_deck_path, base_inner_panel, load_manifest

SIM, MODEL, SIBLING, D6C = "#2a78d6", "#eb6834", "#1baf7a", "#6f6e69"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CLASSES = ("crease", "softening notch", "deepening")
CLASS_COLORS = dict(zip(CLASSES, ("#2a78d6", "#eb6834", "#1baf7a")))
CLASS_MARKERS = dict(zip(CLASSES, ("o", "s", "^")))
SPLIT_STYLE = {"train": dict(color="#9b9a95", ls="-"), "validation": dict(color=INK, ls="--"),
               "test": dict(color=INK, ls="-")}
DESIGN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
RELATIONS = ("sibling (same site & sign)", "same site, opposite sign", "shares a beam edit", "different site")
RELATION_COLORS = dict(zip(RELATIONS, ("#2a78d6", "#eb6834", "#1baf7a", "#9b9a95")))
MIRROR_Y = 155.3432
ROLE_MARKERS = {"near": "o", "twin": "D", "far": "s"}


def rms(values, axis=None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def r2_zero(prediction, truth):
    """1 - SSE / sum(truth^2): 0 for predicting zero (copy design 6)."""
    total = float(np.sum(np.square(truth)))
    return float(1.0 - np.sum(np.square(prediction - truth)) / total) if total else float("nan")


def site_label(site):
    return (site.replace("xm", "x-").replace("ym", "y-").replace("horiz_", "centre ")
            .replace("diag_", "diag ").replace("vert_", "vert ").replace("+", " + "))


# --------------------------------------------------------------------------- data


class Context:
    """Every array the figures need, row-aligned with ``runs``."""

    def __init__(self, run_dir, data_dir):
        self.run_dir, self.data_dir = Path(run_dir), Path(data_dir)
        config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        stride = int(config["preprocessing"]["time_subsample_stride"])
        runs = pd.read_csv(self.run_dir / "local_edit_runs.csv")
        curves = load_histories(self.run_dir)
        self.time = curves[int(runs["sample"].iloc[0])][0]
        self.ms = 1e3 * self.time
        steps = len(self.time)
        self.T = np.stack([curves[s][1] for s in runs["sample"]])
        self.P = np.stack([curves[s][2] for s in runs["sample"]])

        manifest = load_manifest(self.data_dir / "manifest_local_edits.csv")
        base = manifest[manifest["design"] == BASE_DESIGN]
        self.loc_xy = {int(r.location): (float(r.x_mm), float(r.y_mm)) for r in base.itertuples()}
        d6 = {int(r.location): load_full_resolution(self.data_dir, int(r.run))[1][::stride][:steps]
              for r in base.itertuples()}
        self.d6 = d6
        self.B = np.stack([d6[int(l)] for l in runs["location"]])
        self.D, self.Pe = self.T - self.B, self.P - self.B

        classes = design_classes(self.data_dir)
        meta = pd.read_csv(self.data_dir / "local_edit_designs.csv").set_index("design")
        meta = meta.join(classes[["site", "sign", "class"]])
        meta["depth_mm"] = np.where(meta["sign"] > 0, meta["dz_max_mm"], meta["dz_min_mm"])
        split_of = runs.groupby("design")["split"].first()
        meta["split"] = split_of.reindex(meta.index)
        self.meta = meta
        runs["sign"] = runs["design"].map(meta["sign"])
        runs["variant"] = runs["design"].map(meta["variant"])
        self.runs = runs
        self.locations = runs.groupby("design")["location"].apply(lambda s: set(int(v) for v in s))

        self.S, sibling_count = self._sibling_effects()
        if not np.array_equal(sibling_count, runs["siblings"].to_numpy()):
            raise ValueError("Sibling counts differ from local_edit_runs.csv")
        self.twin_source = self._twin_sources()

        runs["effect_rms_true_g"] = rms(self.D, axis=1)
        runs["effect_rms_pred_g"] = rms(self.Pe, axis=1)
        runs["effect_error_rms_g"] = rms(self.Pe - self.D, axis=1)
        runs["accel_rmse_g"] = rms(self.P - self.T, axis=1)
        runs["accel_r2"] = 1.0 - np.sum((self.P - self.T) ** 2, axis=1) / np.sum(
            (self.T - self.T.mean(axis=1, keepdims=True)) ** 2, axis=1)
        runs["copy_d6_accel_r2"] = 1.0 - np.sum(self.D ** 2, axis=1) / np.sum(
            (self.T - self.T.mean(axis=1, keepdims=True)) ** 2, axis=1)

        deck = base_deck_path(self.data_dir / "inp_files", manifest)
        _, xyz = base_inner_panel(deck)
        self.panel_xy = np.asarray(xyz[:, :2], dtype=np.float64)
        self.changes = {d: pd.read_csv(self.data_dir / "design_changes" / f"{d}_changed_nodes.csv")
                        for d in meta.index}

    def _sibling_effects(self):
        runs, D = self.runs, self.D
        train = runs["split"].eq("train").to_numpy()
        key = (runs["location"].astype(str) + "|" + runs["site"] + "|" + runs["sign"].astype(str)).to_numpy()
        sums, counts = {}, {}
        for k in np.unique(key[train]):
            rows = np.where(train & (key == k))[0]
            sums[k], counts[k] = D[rows].sum(axis=0), len(rows)
        S = np.full_like(D, np.nan)
        count = np.zeros(len(runs), dtype=int)
        for i, k in enumerate(key):
            if k not in sums:
                continue
            total, n = sums[k], counts[k]
            if train[i]:
                total, n = total - D[i], n - 1
            if n > 0:
                S[i], count[i] = total / n, n
        return S, count

    def _twin_sources(self):
        """Row of the mirror-image near run for every twin run, else -1."""
        runs = self.runs
        row_of = {(d, int(l)): i for i, (d, l) in enumerate(zip(runs["design"], runs["location"]))}
        source = np.full(len(runs), -1)
        for i, r in enumerate(runs.itertuples()):
            if r.role != "twin":
                continue
            x, y = self.loc_xy[int(r.location)]
            for loc in self.locations[r.design]:
                lx, ly = self.loc_xy[loc]
                if loc != r.location and abs(lx - x) < 1.0 and abs(ly - (2 * MIRROR_Y - y)) < 1.0:
                    source[i] = row_of[(r.design, loc)]
        return source

    def rows(self, **filters):
        mask = np.ones(len(self.runs), dtype=bool)
        for column, value in filters.items():
            values = value if isinstance(value, (list, tuple, set)) else [value]
            mask &= self.runs[column].isin(values).to_numpy()
        return np.where(mask)[0]

    def label(self, design):
        m = self.meta.loc[design]
        return f"{design} {m['variant_name']} {m['depth_mm']:+.0f} mm", site_label(m["site"])


# ------------------------------------------------------------------------ metrics


def twin_r2(ctx, rows):
    rows = [i for i in rows if ctx.twin_source[i] >= 0]
    if not rows:
        return float("nan")
    return r2_zero(ctx.D[[ctx.twin_source[i] for i in rows]], ctx.D[rows])


def per_design_table(ctx):
    records = []
    for design, m in ctx.meta.iterrows():
        rows = ctx.rows(design=design)
        sib = [i for i in rows if np.isfinite(ctx.S[i]).all()]
        r = ctx.runs.iloc[rows]
        sib_runs = ctx.runs.iloc[sib]
        records.append({
            "design": design, "split": m["split"], "class": m["class"], "variant": m["variant"],
            "variant_name": m["variant_name"], "site": m["site"], "depth_mm": m["depth_mm"],
            "nodes_moved": int(m["n_nodes_moved"]), "runs": len(rows),
            "difference_r2": r2_zero(ctx.Pe[rows], ctx.D[rows]),
            "sibling_runs": len(sib),
            "sibling_r2": r2_zero(ctx.S[sib], ctx.D[sib]) if sib else np.nan,
            "model_r2_sibling_runs": r2_zero(ctx.Pe[sib], ctx.D[sib]) if sib else np.nan,
            "twin_copy_r2": twin_r2(ctx, rows),
            "effect_rms_true_g": float(rms(ctx.D[rows])),
            "effect_rms_pred_g": float(rms(ctx.Pe[rows])),
            "effect_error_rms_g": float(rms(ctx.Pe[rows] - ctx.D[rows])),
            "accel_rmse_g": float(rms(ctx.P[rows] - ctx.T[rows])),
            "hic_change_true_mean_abs_pct": float(r["hic_change_true_pct"].abs().mean()),
            "hic_change_mae_model_pts": float((r["hic_change_pred_pct"] - r["hic_change_true_pct"]).abs().mean()),
            "hic_change_mae_sibling_pts": float((sib_runs["hic_change_sibling_pct"]
                                                 - sib_runs["hic_change_true_pct"]).abs().mean()) if sib else np.nan,
        })
    return pd.DataFrame(records).set_index("design")


def relation(ctx, a, b):
    ma, mb = ctx.meta.loc[a], ctx.meta.loc[b]
    if ma["site"] == mb["site"]:
        return RELATIONS[0] if ma["sign"] == mb["sign"] else RELATIONS[1]
    if set(ma["site"].split("+")) & set(mb["site"].split("+")):
        return RELATIONS[2]
    return RELATIONS[3]


def design_pairs(ctx):
    """Pooled difference R2 between every two edited designs at their shared locations."""
    runs = ctx.runs
    acc = {}
    for _, group in runs.groupby("location"):
        rows = group.index.to_numpy()
        if len(rows) < 2:
            continue
        a, b = np.triu_indices(len(rows), k=1)
        ra, rb = rows[a], rows[b]
        d = ctx.T[ra] - ctx.T[rb]
        p = ctx.P[ra] - ctx.P[rb]
        sse, sst, spp = ((p - d) ** 2).sum(1), (d ** 2).sum(1), (p ** 2).sum(1)
        for i, j, e, t, q in zip(ra, rb, sse, sst, spp):
            da, db = runs.at[i, "design"], runs.at[j, "design"]
            key = (da, db) if int(da[1:]) < int(db[1:]) else (db, da)
            entry = acc.setdefault(key, [0, 0.0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += e
            entry[2] += t
            entry[3] += q
    steps = len(ctx.time)
    records = []
    for (da, db), (n, sse, sst, spp) in acc.items():
        sa, sb = ctx.meta.at[da, "split"], ctx.meta.at[db, "split"]
        held = [s for s in (sa, sb) if s != "train"]
        group = "train-train (in-sample)" if not held else ("held-out vs held-out" if len(held) == 2
                                                           else f"{held[0]} vs train")
        records.append({"design_a": da, "design_b": db, "shared_locations": n, "group": group,
                        "relation": relation(ctx, da, db), "pair_r2": 1.0 - sse / sst if sst else np.nan,
                        "true_diff_rms_g": float(np.sqrt(sst / (n * steps))),
                        "pred_diff_rms_g": float(np.sqrt(spp / (n * steps)))})
    return pd.DataFrame(records)


def pair_lookup(pairs):
    lookup = {}
    for r in pairs.itertuples():
        lookup[(r.design_a, r.design_b)] = lookup[(r.design_b, r.design_a)] = r
    return lookup


def physics_baselines(ctx, design):
    """Depth-scaled half crease and summed single creases, for a crease cut line."""
    m = ctx.meta.loc[design]
    out = {}
    row_of = {(d, int(l)): i for i, (d, l) in enumerate(zip(ctx.runs["design"], ctx.runs["location"]))}
    mine = ctx.rows(design=design)
    if m["variant"] != "K":
        return out
    half = ctx.meta[(ctx.meta["site"] == m["site"]) & (ctx.meta["variant"] == "KH")].index
    if len(half):
        h = half[0]
        rows = [i for i in mine if (h, int(ctx.runs.at[i, "location"])) in row_of]
        src = [row_of[(h, int(ctx.runs.at[i, "location"]))] for i in rows]
        if rows:
            out[f"2 x {h} (half crease)"] = {
                "runs": len(rows), "baseline_r2": r2_zero(2.0 * ctx.D[src], ctx.D[rows]),
                "copy_half_r2": r2_zero(ctx.D[src], ctx.D[rows]),
                "model_r2_same_runs": r2_zero(ctx.Pe[rows], ctx.D[rows])}
    parts = m["site"].split("+")
    if len(parts) > 1:
        singles = []
        for part in parts:
            match = ctx.meta[(ctx.meta["site"] == part) & (ctx.meta["variant"] == "K")].index
            if len(match):
                singles.append(match[0])
        if len(singles) == len(parts):
            rows = [i for i in mine if all((s, int(ctx.runs.at[i, "location"])) in row_of for s in singles)]
            if rows:
                total = sum(ctx.D[[row_of[(s, int(ctx.runs.at[i, "location"]))] for i in rows]] for s in singles)
                out[" + ".join(singles) + " (single creases summed)"] = {
                    "runs": len(rows), "baseline_r2": r2_zero(total, ctx.D[rows]),
                    "model_r2_same_runs": r2_zero(ctx.Pe[rows], ctx.D[rows])}
    return out


def related_designs(ctx, design, k=4):
    m = ctx.meta.loc[design]
    parts = set(m["site"].split("+"))
    mine = ctx.locations[design]

    def rank(d):
        md = ctx.meta.loc[d]
        if md["site"] == m["site"]:
            r = 0 if md["sign"] == m["sign"] else 1
        elif parts & set(md["site"].split("+")):
            r = 2
        else:
            r = 3
        return (r, md["variant"] != m["variant"], -len(mine & ctx.locations[d]))

    candidates = [d for d in ctx.meta.index if d != design and mine & ctx.locations[d]]
    return sorted(candidates, key=rank)[:k]


# ----------------------------------------------------------------------- drawing


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8986", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5})
    return plt


def _style(ax):
    ax.grid(alpha=0.22, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _dz_style():
    from matplotlib.colors import TwoSlopeNorm
    return "RdBu_r", TwoSlopeNorm(vmin=-20.0, vcenter=0.0, vmax=20.0)


def draw_panel(ax, ctx, box=None, size=0.06):
    xy = ctx.panel_xy
    if box is not None:
        (x0, x1), (y0, y1) = box
        keep = (xy[:, 0] > x0 - 30) & (xy[:, 0] < x1 + 30) & (xy[:, 1] > y0 - 30) & (xy[:, 1] < y1 + 30)
        xy = xy[keep]
    ax.scatter(xy[:, 1], xy[:, 0], s=size, c=GRID, lw=0, rasterized=True, zorder=1)
    ax.set_aspect("equal", adjustable="box")


def draw_edit(ax, ctx, design, size=4.0, color=None, alpha=1.0, zorder=3):
    cmap, norm = _dz_style()
    ch = ctx.changes[design]
    ch = ch.iloc[np.argsort(np.abs(ch["dz"].to_numpy()))]
    if color is None:
        return ax.scatter(ch["y"], ch["x"], c=ch["dz"], cmap=cmap, norm=norm, s=size, lw=0,
                          alpha=alpha, rasterized=True, zorder=zorder)
    return ax.scatter(ch["y"], ch["x"], c=color, s=size, lw=0, alpha=alpha, rasterized=True, zorder=zorder)


def edit_box(ctx, designs, locations=(), margin=35.0, half=False):
    """Bounding box of edits (and locations); ``half`` keeps the y >= mirror side only."""
    frames = [ctx.changes[d] for d in designs]
    if half:
        frames = [f[f["y"] >= MIRROR_Y - 20.0] for f in frames]
    xs = np.concatenate([f["x"].to_numpy() for f in frames]
                        + [np.array([ctx.loc_xy[l][0] for l in locations])])
    ys = np.concatenate([f["y"].to_numpy() for f in frames]
                        + [np.array([ctx.loc_xy[l][1] for l in locations])])
    return (xs.min() - margin, xs.max() + margin), (ys.min() - margin, ys.max() + margin)


def set_box(ax, box):
    (x0, x1), (y0, y1) = box
    ax.set_xlim(y0, y1)
    ax.set_ylim(x0, x1)


def r2_colors():
    import matplotlib
    from matplotlib.colors import Normalize
    return matplotlib.colormaps["Purples"], Normalize(0.0, 1.0)


def draw_locations(ax, ctx, design, values, label=True, size=60):
    cmap, norm = r2_colors()
    for i in ctx.rows(design=design):
        r = ctx.runs.iloc[i]
        x, y = ctx.loc_xy[int(r["location"])]
        v = values[i]
        ax.scatter(y, x, s=size, marker=ROLE_MARKERS[r["role"]], color=cmap(norm(np.clip(v, 0, 1))),
                   edgecolor="#e34948" if v < 0 else INK, linewidth=1.4 if v < 0 else 0.7, zorder=5)
        if label:
            ax.annotate(str(int(r["location"])), (y, x), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=INK, zorder=6)


# ----------------------------------------------------------------------- figures


def fig_atlas(ctx, table, path):
    plt = _plt()
    order = table.assign(_s=table["site"], _v=table["variant"]).sort_values(["_s", "_v"]).index
    cols = 9
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.15, rows * 2.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.935, bottom=0.055, wspace=0.06, hspace=0.62)
    for ax, design in zip(axes.flat, order):
        draw_panel(ax, ctx, size=0.09)
        mappable = draw_edit(ax, ctx, design, size=4.0)
        for i in ctx.rows(design=design):
            x, y = ctx.loc_xy[int(ctx.runs.at[i, "location"])]
            ax.scatter(y, x, s=6, marker=ROLE_MARKERS[ctx.runs.at[i, "role"]], color=INK, lw=0, zorder=4)
        t = table.loc[design]
        name, site = ctx.label(design)
        sib = "" if np.isnan(t["sibling_r2"]) else f"  sib {t['sibling_r2']:.2f}"
        ax.set_title(f"{name}\n{site}\nR² {t['difference_r2']:.2f}{sib}", fontsize=7.2, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        edge = {"test": "#e34948", "validation": "#6250d6"}.get(t["split"])
        for spine in ax.spines.values():
            spine.set_visible(edge is not None)
            if edge:
                spine.set_color(edge)
                spine.set_linewidth(2.4)
        if edge:
            ax.text(0.03, 0.03, t["split"].upper(), transform=ax.transAxes, fontsize=7.5, color=edge,
                    fontweight="bold")
    for ax in list(axes.flat)[len(order):]:
        ax.set_axis_off()
    cbar = fig.colorbar(mappable, cax=fig.add_axes([0.35, 0.022, 0.3, 0.008]), orientation="horizontal")
    cbar.set_label("dz of the inner panel (mm):  + shallower beam,  − deeper beam")
    fig.suptitle("All 72 local-edit designs on design 6 (top view, y across, x along the hood). Black marks = the "
                 "design's 20 impact locations (circle near, diamond mirror twin, square far).\nR² = edit-effect "
                 "difference R² of the model (training designs in-sample), sib = sibling copy. Grouped by edit site.",
                 fontsize=10.5, y=0.985)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _example(ax, ctx, i, title_prefix):
    r = ctx.runs.iloc[i]
    ax.plot(ctx.ms, ctx.B[i], color=D6C, lw=1.0, label="design 6")
    ax.plot(ctx.ms, ctx.T[i], color=SIM, lw=1.8, label="simulation")
    ax.plot(ctx.ms, ctx.P[i], color=MODEL, lw=1.6, label="model")
    ax.set_title(f"{title_prefix}: {r['design']} loc {int(r['location'])} ({r['role']} {r['dist_to_edit_mm']:.0f} mm)\n"
                 f"accel R² {r['accel_r2']:.3f}  RMSE {r['accel_rmse_g']:.1f} g  |  effect R² {r['difference_r2']:.2f}",
                 fontsize=8.5)
    ax.set_xlabel("time (ms)")
    _style(ax)


def fig_acceleration(ctx, path):
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(17, 13))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1, 1], hspace=0.5, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0:2])
    for split, style in SPLIT_STYLE.items():
        rows = ctx.rows(split=split)
        ax.plot(ctx.ms, rms(ctx.P[rows] - ctx.T[rows], axis=0), lw=2, label=f"model, {split}", **style)
    held = ctx.rows(split=["validation", "test"])
    ax.plot(ctx.ms, rms(ctx.D[held], axis=0), color=D6C, lw=1.6, ls=":", label="copy design 6, held-out")
    train = ctx.rows(split="train")
    ax.plot(ctx.ms, rms(ctx.D[train], axis=0), color=D6C, lw=1.0, ls="-.", label="copy design 6, train")
    ax.set(xlabel="time (ms)", ylabel="RMSE over runs (g)",
           title="Acceleration error over time (copy design 6 = the size of the edit effect)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    _style(ax)

    ax = fig.add_subplot(gs[0, 2])
    data, labels = [], []
    for split in SPLIT_STYLE:
        rows = ctx.rows(split=split)
        data += [ctx.runs["accel_r2"].iloc[rows], ctx.runs["copy_d6_accel_r2"].iloc[rows]]
        labels += [f"model\n{split}", f"copy D6\n{split}"]
    box = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for k, patch in enumerate(box["boxes"]):
        patch.set_facecolor(MODEL if k % 2 == 0 else D6C)
        patch.set_alpha(0.55)
    for k, values in enumerate(data):
        jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(values))
        ax.scatter(np.full(len(values), k + 1) + jitter, values, s=4 if len(values) > 100 else 14,
                   color=INK, alpha=0.25 if len(values) > 100 else 0.8, lw=0, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7.5)
    ax.set(ylabel="per-run acceleration R²", title="Per-run acceleration R²")
    ax.set_ylim(max(0.4, min(np.min(v) for v in data) - 0.02), 1.0)
    _style(ax)

    for row, split_set, name in ((1, ["train"], "train"), (2, ["validation", "test"], "held-out")):
        rows = ctx.rows(split=split_set)
        order = rows[np.argsort(ctx.runs["accel_r2"].to_numpy()[rows])]
        picks = [(order[-1], "best"), (order[len(order) // 2], "median"), (order[0], "worst")]
        for col, (i, tag) in enumerate(picks):
            ax = fig.add_subplot(gs[row, col])
            _example(ax, ctx, i, f"{name} {tag}")
            if col == 0:
                ax.set_ylabel("acceleration (g)")
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Acceleration history accuracy  |  {ctx.run_dir.name}  |  best checkpoint, 63-point grid",
                 fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_effect_skill(ctx, table, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    held = table[table["split"] != "train"]

    def class_scatter(ax, frame, x, y, size=26):
        frame = frame[frame["split"] == "train"]
        for cls in CLASSES:
            f = frame[frame["class"] == cls]
            ax.scatter(f[x], f[y], s=size, color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], alpha=0.8,
                       edgecolor="white", linewidth=0.4, label=f"{cls} (train, in-sample)")
        for d, r in held.iterrows():
            ax.scatter(r[x], r[y], s=180, marker="*", color=CLASS_COLORS[r["class"]], edgecolor=INK,
                       linewidth=1.0, zorder=6)
            ax.annotate(f"{d} ({r['split']})", (r[x], r[y]), xytext=(6, -10), textcoords="offset points",
                        fontsize=8.5, fontweight="bold")

    ax = axes[0, 0]
    class_scatter(ax, table.dropna(subset=["sibling_r2"]), "sibling_r2", "model_r2_sibling_runs")
    lo = min(-0.5, np.nanmin(table[["sibling_r2", "model_r2_sibling_runs"]].to_numpy()))
    ax.plot([lo, 1], [lo, 1], color="0.3", lw=1)
    ax.set(xlim=(lo, 1), ylim=(lo, 1), xlabel="sibling copy difference R²", ylabel="model difference R² (same runs)",
           title="Per design: model vs sibling copy (above the line = model better)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)

    ax = axes[0, 1]
    class_scatter(ax, table, "effect_rms_true_g", "difference_r2")
    ax.set_xscale("log")
    ax.axhline(0, color="0.5", lw=0.8)
    for cls in CLASSES:
        rows = ctx.rows(split="train", **{"class": cls})
        ax.axhline(twin_r2(ctx, rows), color=CLASS_COLORS[cls], lw=1.2, ls="--", alpha=0.8)
    ax.text(0.02, 0.97, "dashed: mirror-twin repeatability of each class\n(how well the solver repeats its own effect)",
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    ax.set(xlabel="true edit-effect size, RMS over the design's runs (g)", ylabel="model difference R²",
           title="Skill grows with effect size", ylim=(max(-2.0, table["difference_r2"].min() - 0.1), 1))
    _style(ax)

    ax = axes[0, 2]
    for cls in CLASSES:
        rows = ctx.rows(split="train", **{"class": cls})
        ax.scatter(ctx.runs["effect_rms_true_g"].iloc[rows], ctx.runs["effect_error_rms_g"].iloc[rows], s=6,
                   color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], alpha=0.5, lw=0, label=cls)
    rows = ctx.rows(split=["validation", "test"])
    ax.scatter(ctx.runs["effect_rms_true_g"].iloc[rows], ctx.runs["effect_error_rms_g"].iloc[rows], s=60,
               marker="*", color=INK, label="held-out runs (N54, N55)", zorder=5)
    line = np.array([0.5, 60])
    ax.plot(line, line, color="0.3", lw=1, label="error = effect (copy design 6)")
    small = ctx.rows(split="train")
    small = small[ctx.runs["effect_rms_true_g"].to_numpy()[small] < 2.0]
    floor = float(np.median(ctx.runs["effect_error_rms_g"].to_numpy()[small]))
    ax.axhline(floor, color="#e34948", lw=1.2, ls="--")
    ax.text(0.6, floor * 1.08, f"model error floor ≈ {floor:.1f} g (median, runs with effect < 2 g)",
            fontsize=8, color="#e34948")
    ax.set(xscale="log", yscale="log", xlim=line, ylim=(0.5, 60), xlabel="true edit effect, RMS (g)",
           ylabel="model effect error, RMS (g)", title="Per run: below the line the model beats copying design 6")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)

    ax = axes[1, 0]
    for cls in CLASSES:
        rows = ctx.rows(split="train", **{"class": cls})
        ax.scatter(ctx.runs["effect_rms_true_g"].iloc[rows], ctx.runs["effect_rms_pred_g"].iloc[rows], s=6,
                   color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], alpha=0.5, lw=0, label=cls)
    for split, marker in (("validation", "P"), ("test", "*")):
        rows = ctx.rows(split=split)
        x, y = ctx.runs["effect_rms_true_g"].iloc[rows], ctx.runs["effect_rms_pred_g"].iloc[rows]
        ax.scatter(x, y, s=70, marker=marker, color=INK, label=f"{split} runs", zorder=5)
    ax.plot(line, line, color="0.3", lw=1, label="1:1")
    for name, split in (("train", ["train"]), ("held-out", ["validation", "test"])):
        rows = ctx.rows(split=split)
        big = rows[ctx.runs["effect_rms_true_g"].to_numpy()[rows] > 8]
        ratio = np.median(ctx.runs["effect_rms_pred_g"].to_numpy()[big] / ctx.runs["effect_rms_true_g"].to_numpy()[big])
        ax.text(0.97, 0.05 + (0.07 if name == "train" else 0), f"{name}, effects > 8 g: predicted / true = {ratio:.2f}",
                transform=ax.transAxes, ha="right", fontsize=8.5)
    ax.set(xscale="log", yscale="log", xlim=line, ylim=line, xlabel="true edit effect, RMS (g)",
           ylabel="predicted edit effect, RMS (g)", title="Effect amplitude: is it shrunk?")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)

    ax = axes[1, 1]
    edges = np.arange(0, 26.1, 2.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for split, style in SPLIT_STYLE.items():
        rows = ctx.rows(split=split)
        values = []
        for a, b in zip(edges[:-1], edges[1:]):
            t = (ctx.ms >= a) & (ctx.ms < b)
            values.append(r2_zero(ctx.Pe[rows][:, t], ctx.D[rows][:, t]))
        ax.plot(centres, values, marker="o", ms=4, lw=2, label=f"model, {split}", **style)
    rows = ctx.rows(split=["validation", "test"])
    rows = rows[np.isfinite(ctx.S[rows]).all(axis=1)]
    values = [r2_zero(ctx.S[rows][:, (ctx.ms >= a) & (ctx.ms < b)], ctx.D[rows][:, (ctx.ms >= a) & (ctx.ms < b)])
              for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(centres, values, color=SIBLING, lw=2, ls="--", marker="s", ms=4, label="sibling copy, held-out")
    ax2 = ax.twinx()
    ax2.fill_between(ctx.ms, rms(ctx.D[rows], axis=0), color=SIM, alpha=0.12, lw=0)
    ax2.set_ylabel("held-out true effect RMS (g), shaded", color=MUTED)
    ax2.spines["top"].set_visible(False)
    ax.set(xlabel="time (ms), 2 ms bins", ylabel="difference R² in the bin", ylim=(-1, 1),
           title="When in the impact are design effects predicted?")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax)

    ax = axes[1, 2]
    groups = []
    for cls in CLASSES:
        rows = ctx.rows(split="train", **{"class": cls})
        sib = rows[np.isfinite(ctx.S[rows]).all(axis=1)]
        groups.append((f"train\n{cls}", r2_zero(ctx.Pe[rows], ctx.D[rows]),
                       r2_zero(ctx.S[sib], ctx.D[sib]) if len(sib) else np.nan, twin_r2(ctx, rows)))
    for split in ("validation", "test"):
        rows = ctx.rows(split=split)
        sib = rows[np.isfinite(ctx.S[rows]).all(axis=1)]
        design = ctx.runs["design"].iloc[rows[0]]
        groups.append((f"{split}\n{design}", r2_zero(ctx.Pe[rows], ctx.D[rows]), r2_zero(ctx.S[sib], ctx.D[sib]),
                       twin_r2(ctx, rows)))
    x = np.arange(len(groups))
    width = 0.27
    for k, (name, color) in enumerate((("model", MODEL), ("sibling copy", SIBLING), ("twin copy (repeatability)", D6C))):
        values = [g[k + 1] for g in groups]
        ax.bar(x + (k - 1) * width, np.clip(values, -1, None), width, color=color, label=name)
        for xi, v in zip(x, values):
            ax.text(xi + (k - 1) * width, max(v, -1) + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(ylabel="difference R² (clipped at −1)", ylim=(-1, 1.05),
           title="By class: model, sibling copy, solver repeatability")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax)
    fig.suptitle("Edit-effect skill: effect = design − design 6 at the same location; difference R² = 0 for copying "
                 "design 6.  Training designs are scored in-sample.", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pairs(ctx, pairs, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    groups = ["train-train (in-sample)", "validation vs train", "test vs train"]
    width = 0.2
    for k, rel in enumerate(RELATIONS):
        values, labels = [], []
        for g in groups:
            f = pairs[(pairs["group"] == g) & (pairs["relation"] == rel)]
            w = f["true_diff_rms_g"] ** 2 * f["shared_locations"]
            pooled = 1.0 - np.sum(w * (1 - f["pair_r2"])) / np.sum(w) if len(f) else np.nan
            values.append(pooled)
            labels.append(len(f))
        xs = np.arange(len(groups)) + (k - 1.5) * width
        ax.bar(xs, np.clip(values, -1, None), width, color=RELATION_COLORS[rel], label=rel)
        for xi, v, n in zip(xs, values, labels):
            if np.isfinite(v):
                ax.text(xi, max(v, -1) + 0.02, f"{v:.2f}\n{n}p", ha="center", fontsize=7)
    ax.set_xticks(range(len(groups)), groups)
    ax.axhline(0, color="0.4", lw=0.8)
    lines = ["Predicted / true size of the A − B difference", "(pooled RMS; 1 = right size)", ""]
    for g in groups:
        parts = []
        for rel in RELATIONS:
            f = pairs[(pairs["group"] == g) & (pairs["relation"] == rel)]
            if len(f):
                ratio = np.sqrt(np.sum(f["shared_locations"] * f["pred_diff_rms_g"] ** 2)
                                / np.sum(f["shared_locations"] * f["true_diff_rms_g"] ** 2))
                parts.append(f"{rel.split(' (')[0].split(',')[0]} {ratio:.2f}")
        lines.append(f"{g}: " + ", ".join(parts))
    ax.text(0.98, 0.04, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))
    ax.set(ylim=(-1, 1.35), ylabel="pair difference R² (pooled; clipped at −1)",
           title="Can it tell design A from design B at the same location?\n(np = number of design pairs)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    _style(ax)

    for ax, design in zip(axes[1:], ("N54", "N55")):
        if design not in ctx.meta.index:
            ax.set_axis_off()
            continue
        f = pairs[((pairs["design_a"] == design) | (pairs["design_b"] == design)) & (pairs["shared_locations"] >= 8)].copy()
        f["partner"] = np.where(f["design_a"] == design, f["design_b"], f["design_a"])
        f["rank"] = f["relation"].map({r: i for i, r in enumerate(RELATIONS)})
        f = f.sort_values(["rank", "pair_r2"], ascending=[True, False]).head(22)
        y = np.arange(len(f))[::-1]
        ax.barh(y, np.clip(f["pair_r2"], -1, None), color=[RELATION_COLORS[r] for r in f["relation"]])
        names = [f"{p} {ctx.meta.at[p, 'variant_name']} {ctx.meta.at[p, 'depth_mm']:+.0f} · {site_label(ctx.meta.at[p, 'site'])}"
                 for p in f["partner"]]
        ax.set_yticks(y, names, fontsize=7.5)
        for yi, (_, r) in zip(y, f.iterrows()):
            ax.text(max(r["pair_r2"], -1) + 0.02, yi, f"{r['pair_r2']:.2f}  |Δ| {r['true_diff_rms_g']:.1f}→"
                    f"{r['pred_diff_rms_g']:.1f} g  ({r['shared_locations']} locs)", va="center", fontsize=7)
        ax.axvline(0, color="0.4", lw=0.8)
        split = ctx.meta.at[design, "split"]
        ax.set(xlim=(-1, 1.6), xlabel="pair difference R²",
               title=f"{design} ({split}) vs each training design sharing ≥ 8 locations\n"
                     "|Δ| = RMS of the true → predicted difference between the two designs")
        _style(ax)
    fig.suptitle("Design-vs-design distinction: d = a_A − a_B (simulation), p = same from the model, at every shared "
                 "location; pair R² = 1 − Σ(p − d)² / Σd²", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_vs_related(ctx, design, pairs_by, path):
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    related = related_designs(ctx, design, k=5)
    group = [design] + related
    colors = {d: (INK if d == design else DESIGN_COLORS[(k - 1) % len(DESIGN_COLORS)]) for k, d in enumerate(group)}
    rows = ctx.rows(design=design)
    near = [i for i in rows if ctx.runs.at[i, "role"] != "far"]
    near = sorted(near, key=lambda i: -ctx.runs.at[i, "effect_rms_true_g"])
    chosen = []
    for i in near:
        loc = int(ctx.runs.at[i, "location"])
        if sum(loc in ctx.locations[d] for d in related) >= 2:
            chosen.append(i)
        if len(chosen) == 4:
            break
    row_of = {(d, int(l)): i for i, (d, l) in enumerate(zip(ctx.runs["design"], ctx.runs["location"]))}

    fig = plt.figure(figsize=(20, 2.2 + 4.0 * len(chosen)))
    gs = GridSpec(1 + len(chosen), 6, figure=fig, height_ratios=[0.5] + [1] * len(chosen), hspace=0.5, wspace=0.3,
                  top=1.0 - 1.2 / (2.2 + 4.0 * len(chosen)))
    box = edit_box(ctx, group, margin=20, half=True)
    for k, d in enumerate(group):
        ax = fig.add_subplot(gs[0, k])
        draw_panel(ax, ctx, box, size=0.5)
        if d != design:
            draw_edit(ax, ctx, design, size=3, color="#b9b8b2", zorder=2)
        draw_edit(ax, ctx, d, size=6)
        set_box(ax, box)
        ax.set_xticks([])
        ax.set_yticks([])
        name, site = ctx.label(d)
        if d == design:
            info = f"HELD-OUT ({ctx.meta.at[d, 'split']})"
        else:
            p = pairs_by.get((design, d))
            info = (f"{relation(ctx, design, d).split(' (')[0]}\npair R² {p.pair_r2:.2f} over {p.shared_locations} locs"
                    if p is not None else relation(ctx, design, d))
        ax.set_title(f"{name}\n{site}\n{info}", fontsize=8.5, color=colors[d], fontweight="bold" if d == design else None)
        for spine in ax.spines.values():
            spine.set_color(colors[d])
            spine.set_linewidth(2.2)

    for row, i in enumerate(chosen, start=1):
        loc = int(ctx.runs.at[i, "location"])
        ax_s = fig.add_subplot(gs[row, 0:2])
        ax_m = fig.add_subplot(gs[row, 2:4], sharey=ax_s)
        ax_h = fig.add_subplot(gs[row, 4:6])
        present = [d for d in group if (d, loc) in row_of]
        hic = []
        for d in present:
            j = row_of[(d, loc)]
            lw = 2.4 if d == design else 1.5
            ax_s.plot(ctx.ms, ctx.D[j], color=colors[d], lw=lw, label=d)
            ax_m.plot(ctx.ms, ctx.Pe[j], color=colors[d], lw=lw, ls="-" if d == design else "--", label=d)
            hic.append((d, ctx.runs.at[j, "hic_change_true_pct"], ctx.runs.at[j, "hic_change_pred_pct"],
                        ctx.runs.at[j, "split"]))
        r = ctx.runs.iloc[i]
        ax_s.set_title(f"loc {loc} ({r['role']}, {r['dist_to_edit_mm']:.0f} mm from {design}'s edit): SIMULATION effect",
                       fontsize=9)
        ax_m.set_title(f"loc {loc}: MODEL effect ({design} held-out, others in-sample)", fontsize=9)
        for ax in (ax_s, ax_m):
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xlabel("time (ms)")
            _style(ax)
        ax_s.set_ylabel("design − design 6 (g)")
        ax_s.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower left")
        y = np.arange(len(hic))[::-1]
        for yi, (d, t, p, split) in zip(y, hic):
            ax_h.plot([t, p], [yi, yi], color="#b9b8b2", lw=1.5, zorder=1)
            ax_h.scatter(t, yi, s=60, color=colors[d], zorder=3, label="simulation" if yi == y[0] else None)
            ax_h.scatter(p, yi, s=60, facecolor="white", edgecolor=colors[d], linewidth=1.8, marker="D", zorder=4,
                         label="model" if yi == y[0] else None)
        ax_h.set_yticks(y, [f"{d} ({s})" for d, _, _, s in hic], fontsize=8)
        ax_h.axvline(0, color="0.5", lw=0.8)
        ax_h.set(xlabel="HIC change vs design 6 (%)", title=f"loc {loc}: HIC change, filled = simulation, "
                                                            "hollow = model")
        _style(ax_h)
    base = physics_baselines(ctx, design)
    held = ctx.rows(design=design)
    sib = held[np.isfinite(ctx.S[held]).all(axis=1)]
    lines = [f"{design} all 20 runs: model difference R² {r2_zero(ctx.Pe[held], ctx.D[held]):.2f}",
             f"sibling copy {r2_zero(ctx.S[sib], ctx.D[sib]):.2f} (model {r2_zero(ctx.Pe[sib], ctx.D[sib]):.2f} "
             f"on its {len(sib)} runs)"]
    for name, b in base.items():
        lines.append(f"{name}: {b['baseline_r2']:.2f} (model {b['model_r2_same_runs']:.2f} on the same {b['runs']} runs)")
    fig.suptitle(f"{design} next to its relatives: top = edit geometry, one mirror half (gray = {design}'s own edit); "
                 f"rows = its 4 largest-effect locations shared with ≥ 2 relatives\n" + "   |   ".join(lines),
                 fontsize=11, y=1.0)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base


def fig_all_locations(ctx, design, path):
    plt = _plt()
    rows = ctx.rows(design=design)
    rows = sorted(rows, key=lambda i: ({"near": 0, "twin": 1, "far": 2}[ctx.runs.at[i, "role"]],
                                       ctx.runs.at[i, "dist_to_edit_mm"]))
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(rows) / cols)), cols, figsize=(21, 15), sharex=True)
    for ax, i in zip(axes.flat, rows):
        r = ctx.runs.iloc[i]
        ax.plot(ctx.ms, ctx.B[i], color=D6C, lw=1.0, label="design 6")
        if np.isfinite(ctx.S[i]).all():
            ax.plot(ctx.ms, ctx.B[i] + ctx.S[i], color=SIBLING, lw=1.2, ls="--", label="sibling copy")
        ax.plot(ctx.ms, ctx.T[i], color=SIM, lw=1.9, label="simulation")
        ax.plot(ctx.ms, ctx.P[i], color=MODEL, lw=1.6, label="model")
        ax.set_title(f"loc {int(r['location'])} ({r['role']} {r['dist_to_edit_mm']:.0f} mm)  accel R² {r['accel_r2']:.3f}\n"
                     f"effect R² {r['difference_r2']:.2f}  HIC {r['hic_true']:.0f} → pred {r['hic_pred']:.0f} "
                     f"(D6 {r['hic_d6']:.0f})", fontsize=8.5)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("acceleration (g)")
    axes.flat[0].legend(fontsize=8, frameon=False)
    split = ctx.meta.at[design, "split"]
    name, site = ctx.label(design)
    fig.suptitle(f"{name}, {site} ({split}): full acceleration at all 20 locations", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_card(ctx, design, table, pairs_by, path):
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    t = table.loc[design]
    rows = ctx.rows(design=design)
    fig = plt.figure(figsize=(22, 12.8))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1, 1], hspace=0.42, wspace=0.28, top=0.92)

    # Geometry and per-location difference R2.
    ax = fig.add_subplot(gs[0, 0:2])
    box = edit_box(ctx, [design], [int(l) for l in ctx.runs["location"].iloc[rows]], margin=45)
    draw_panel(ax, ctx, box, size=0.6)
    mappable = draw_edit(ax, ctx, design, size=9)
    draw_locations(ax, ctx, design, ctx.runs["difference_r2"].to_numpy())
    set_box(ax, box)
    ax.axhline(MIRROR_Y, color="#b9b8b2", lw=0.8, ls=":")
    ax.set(xlabel="y (mm)", ylabel="x (mm)",
           title="Edit (dz colour) and impact locations coloured by effect R²\n(circle near, diamond mirror twin, square far; "
                 "red edge = R² < 0)")
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("dz (mm): + shallower, − deeper", fontsize=8)
    cmap, norm = r2_colors()
    import matplotlib.cm as cm
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.06)
    cbar2.set_label("effect R² at the location", fontsize=8)

    # Relatives.
    related = related_designs(ctx, design, k=4)
    sub = gs[0, 2:4].subgridspec(2, 2, hspace=0.45, wspace=0.12)
    for k, d in enumerate(related):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        b = edit_box(ctx, [design, d], margin=20, half=True)
        draw_panel(ax, ctx, b, size=0.5)
        draw_edit(ax, ctx, design, size=3, color="#b9b8b2", zorder=2)
        draw_edit(ax, ctx, d, size=5)
        set_box(ax, b)
        ax.set_xticks([])
        ax.set_yticks([])
        name, site = ctx.label(d)
        p = pairs_by.get((design, d))
        rd = table.loc[d]
        pair = f"pair R² {p.pair_r2:.2f}, |Δ| {p.true_diff_rms_g:.1f}→{p.pred_diff_rms_g:.1f} g, {p.shared_locations} locs" \
            if p is not None else "no shared locations"
        ax.set_title(f"{name} · {site}\n{relation(ctx, design, d).split(' (')[0]} · {rd['split']} · own R² "
                     f"{rd['difference_r2']:.2f}\n{pair}", fontsize=7.5)

    # HIC change by location.
    ax = fig.add_subplot(gs[0, 4:6])
    order = sorted(rows, key=lambda i: ({"near": 0, "twin": 1, "far": 2}[ctx.runs.at[i, "role"]],
                                        ctx.runs.at[i, "dist_to_edit_mm"]))
    r = ctx.runs.iloc[order]
    x = np.arange(len(order))
    w = 0.27
    ax.axhspan(-1.4, 1.4, color="0.88", lw=0, label="D6 mirror floor ±1.4%")
    ax.bar(x - w, r["hic_change_true_pct"], w, color=SIM, label="simulation")
    ax.bar(x, r["hic_change_pred_pct"], w, color=MODEL, label="model")
    ax.bar(x + w, r["hic_change_sibling_pct"].fillna(0), w, color=SIBLING, label="sibling copy")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, [f"{int(l)}\n{ro[0]}{d:.0f}" for l, ro, d in zip(r["location"], r["role"], r["dist_to_edit_mm"])],
                  fontsize=6.5)
    ax.set(ylabel="HIC change vs design 6 (%)", title="HIC change by location (n = near, t = twin, f = far; mm to edit)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    _style(ax)

    # Curves at six locations.
    near = [i for i in rows if ctx.runs.at[i, "role"] == "near"]
    pick = sorted(near, key=lambda i: -ctx.runs.at[i, "effect_rms_true_g"])[:4]
    for role in ("twin", "far"):
        cand = [i for i in rows if ctx.runs.at[i, "role"] == role]
        if cand:
            pick.append(max(cand, key=lambda i: ctx.runs.at[i, "effect_rms_true_g"]))
    for k, i in enumerate(pick[:6]):
        rr = ctx.runs.iloc[i]
        top = fig.add_subplot(gs[1, k])
        bottom = fig.add_subplot(gs[2, k], sharex=top)
        top.plot(ctx.ms, ctx.B[i], color=D6C, lw=1.0, label="design 6")
        bottom.axhline(0, color=D6C, lw=1.0)
        if np.isfinite(ctx.S[i]).all():
            top.plot(ctx.ms, ctx.B[i] + ctx.S[i], color=SIBLING, lw=1.2, ls="--", label="sibling copy")
            bottom.plot(ctx.ms, ctx.S[i], color=SIBLING, lw=1.2, ls="--")
        top.plot(ctx.ms, ctx.T[i], color=SIM, lw=1.8, label="simulation")
        top.plot(ctx.ms, ctx.P[i], color=MODEL, lw=1.5, label="model")
        bottom.plot(ctx.ms, ctx.D[i], color=SIM, lw=1.8)
        bottom.plot(ctx.ms, ctx.Pe[i], color=MODEL, lw=1.5)
        top.set_title(f"loc {int(rr['location'])} ({rr['role']} {rr['dist_to_edit_mm']:.0f} mm)\naccel R² "
                      f"{rr['accel_r2']:.3f}, effect R² {rr['difference_r2']:.2f}", fontsize=8.5)
        bottom.set_xlabel("time (ms)")
        for a in (top, bottom):
            _style(a)
        if k == 0:
            top.set_ylabel("acceleration (g)")
            bottom.set_ylabel("effect: design − design 6 (g)")
            top.legend(fontsize=7.5, frameon=False)
    name, site = ctx.label(design)
    status = {"train": "TRAINING design (scores in-sample)", "validation": "VALIDATION design (held out)",
              "test": "TEST design (held out)"}[t["split"]]
    sib = "n/a" if np.isnan(t["sibling_r2"]) else f"{t['sibling_r2']:.2f}"
    fig.suptitle(f"{name} · {site} · {int(t['nodes_moved'])} nodes moved · {t['class']} · {status}\n"
                 f"effect R²: model {t['difference_r2']:.2f}, sibling copy {sib}, mirror-twin repeatability "
                 f"{t['twin_copy_r2']:.2f}  |  effect size {t['effect_rms_true_g']:.1f} g RMS, model error "
                 f"{t['effect_error_rms_g']:.1f} g  |  HIC change MAE: model {t['hic_change_mae_model_pts']:.1f}, "
                 f"sibling {t['hic_change_mae_sibling_pts']:.1f}, copy D6 {t['hic_change_true_mean_abs_pct']:.1f} pts",
                 fontsize=11.5, y=0.995)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- command


def pooled_pairs(pairs, groups):
    """Pooled pair R² and predicted/true size of the A - B difference per relation."""
    out = {}
    f = pairs[pairs["group"].isin(groups)]
    for rel, g in f.groupby("relation"):
        n = g["shared_locations"]
        w = g["true_diff_rms_g"] ** 2 * n
        out[rel] = {"pairs": int(len(g)),
                    "pair_r2": float(1.0 - np.sum(w * (1 - g["pair_r2"])) / np.sum(w)),
                    "size_ratio": float(np.sqrt(np.sum(n * g["pred_diff_rms_g"] ** 2) / np.sum(w)))}
    return out


def fig_summary(ctx, pairs, path, design="N54", location=141):
    """One page: the edit, one held-out run, HIC change on all held-out runs, pair distinction."""
    plt = _plt()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11.6, 8.8))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.985, top=0.885, bottom=0.085, wspace=0.26, hspace=0.46)
    i0 = int(ctx.rows(design=design, location=location)[0])
    r0 = ctx.runs.iloc[i0]
    name, site = ctx.label(design)

    # (a) the geometry change and where the headform hits
    ax = fig.add_subplot(gs[0, 0])
    draw_panel(ax, ctx, size=0.15)
    mappable = draw_edit(ax, ctx, design, size=7.0)
    for i in ctx.rows(design=design):
        loc = int(ctx.runs.at[i, "location"])
        x, y = ctx.loc_xy[loc]
        if loc == location:
            ax.scatter(y, x, s=170, marker="*", color=MODEL, edgecolor=INK, lw=0.8, zorder=6)
            ax.annotate(f"location {loc}", (y, x), xytext=(-8, 14), textcoords="offset points", ha="right",
                        fontsize=8.5, color=INK, zorder=7)
        else:
            ax.scatter(y, x, s=16, marker=ROLE_MARKERS[ctx.runs.at[i, "role"]], color=INK, lw=0, zorder=5)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    cax = ax.inset_axes([0.04, 0.05, 0.32, 0.035])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label("node shift dz (mm)", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"(a) Held-out test design {design}: {name.split(' ', 1)[1]}\n"
                 f"{len(ctx.changes[design]):,} of {len(ctx.panel_xy):,} inner-panel nodes moved; "
                 "dots = its 20 impacts", loc="left")

    # (b) one run: acceleration history
    ax = fig.add_subplot(gs[0, 1])
    pct = lambda v: 100.0 * (v / r0["hic_d6"] - 1.0)
    ax.plot(ctx.ms, ctx.B[i0], color=D6C, lw=1.4, label="design 6, no edit (the reference)")
    ax.plot(ctx.ms, ctx.T[i0], color=SIM, lw=2.0, label=f"simulation {design}   HIC {pct(r0['hic_true']):+.0f}%")
    ax.plot(ctx.ms, ctx.P[i0], color=MODEL, lw=2.0, label=f"model           HIC {pct(r0['hic_pred']):+.0f}%")
    if r0["siblings"] > 0:
        ax.plot(ctx.ms, ctx.B[i0] + ctx.S[i0], color=SIBLING, lw=1.4, ls="--",
                label=f"sibling copy    HIC {pct(r0['hic_sibling']):+.0f}%")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("headform acceleration (g)")
    ax.set_title(f"(b) One held-out run: {design} at location {location} "
                 f"({r0['dist_to_edit_mm']:.0f} mm from the edit)\n"
                 f"edit-effect R² {r0['difference_r2']:.2f}; HIC change relative to design 6", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", prop={"family": "monospace", "size": 7.6})
    ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
    _style(ax)

    # (c) HIC change on every held-out run
    ax = fig.add_subplot(gs[1, 0])
    held = ctx.runs[ctx.runs["split"] != "train"]
    t, p = held["hic_change_true_pct"].to_numpy(), held["hic_change_pred_pct"].to_numpy()
    has_sib = held["siblings"].to_numpy() > 0
    s = held["hic_change_sibling_pct"].to_numpy()
    lim = (min(t.min(), p.min(), np.nanmin(s)) - 4, max(t.max(), p.max(), np.nanmax(s)) + 4)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    ax.axhline(0, color=D6C, lw=1.0, ls="--")
    ax.text(lim[0] + 1, 1.2, "copy design 6 (predicts no change)", ha="left", va="bottom", fontsize=7.5, color=D6C)
    markers = {"test": "o", "validation": "s"}
    for split, mk in markers.items():
        m = (held["split"] == split).to_numpy()
        ax.scatter(t[m & has_sib], s[m & has_sib], s=34, marker=mk, facecolor="none", edgecolor=SIBLING, lw=1.3, zorder=3)
        ax.scatter(t[m], p[m], s=30, marker=mk, color=MODEL, edgecolor="white", lw=0.5, zorder=4)
    ax.scatter([r0["hic_change_true_pct"]], [r0["hic_change_pred_pct"]], s=150, marker="o", facecolor="none",
               edgecolor=INK, lw=1.0, zorder=5)
    r2m, r2s = r2_zero(p, t), r2_zero(s[has_sib], t[has_sib])
    slope = float(np.sum(p * t) / np.sum(t * t))
    big = np.abs(t) >= 3.0
    sign_pct = 100.0 * float(np.mean(np.sign(p[big]) == np.sign(t[big]))) if big.any() else float("nan")
    handles = [Line2D([], [], color=MODEL, marker="o", ls="", label=f"model   R² {r2m:.2f}, slope {slope:.2f}"),
               Line2D([], [], color=SIBLING, marker="o", mfc="none", ls="", label=f"sibling copy   R² {r2s:.2f}"),
               Line2D([], [], color=MUTED, marker="o", ls="", label="circle N54 (test), square N55 (validation)")]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, loc="lower right")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("simulated HIC change vs design 6 (%)")
    ax.set_ylabel("predicted HIC change (%)")
    ax.set_title(f"(c) All {len(held)} held-out runs: HIC change caused by the edit\n"
                 f"right sign on {sign_pct:.0f}% of runs with |change| ≥ 3%; "
                 f"predicted/true size (slope) {slope:.2f}", loc="left")
    _style(ax)

    # (d) telling two designs apart at the same impact location
    ax = fig.add_subplot(gs[1, 1])
    ins = pooled_pairs(pairs, ["train-train (in-sample)"])
    out = pooled_pairs(pairs, ["test vs train", "validation vs train"])
    names = {"sibling (same site & sign)": "same site,\nsame direction\n(siblings)",
             "same site, opposite sign": "same site,\nopposite\ndirection",
             "shares a beam edit": "share one\nbeam edit",
             "different site": "different\nsites"}
    x = np.arange(len(RELATIONS))
    w = 0.38
    for k, (stats, label, color) in enumerate(((ins, "training designs vs each other", "#f2b49b"),
                                                (out, "held-out N54/N55 vs training designs", MODEL))):
        vals = [stats.get(rel, {}).get("pair_r2", np.nan) for rel in RELATIONS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=color, label=label, zorder=3)
        for xi, rel, b in zip(x + (k - 0.5) * w, RELATIONS, bars):
            st = stats.get(rel)
            if st is None:
                continue
            ax.text(xi, max(st["pair_r2"], 0) + 0.02, f"×{st['size_ratio']:.2f}\nn={st['pairs']}",
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [names[r] for r in RELATIONS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R² of the difference curve a_A(t) - a_B(t)")
    ax.set_title("(d) Telling two designs apart at the same impact location\n"
                 "label: predicted / true size of the difference, number of design pairs", loc="left")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    _style(ax)

    train = ctx.runs["split"] == "train"
    parts = ctx.run_dir.name.split("_")
    run_name = "_".join(parts[2:]) if len(parts) > 2 else ctx.run_dir.name
    held_r2 = ", ".join(
        f"{d} {r2_zero(ctx.Pe[(ctx.runs['design'] == d).to_numpy()], ctx.D[(ctx.runs['design'] == d).to_numpy()]):.2f}"
        for d in sorted(held["design"].unique()))
    sib_in = ins.get("sibling (same site & sign)", {}).get("pair_r2", np.nan)
    sib_out = out.get("sibling (same site & sign)", {}).get("pair_r2", np.nan)
    fig.suptitle(f"Local-edit run {run_name} ({ctx.runs.loc[train, 'design'].nunique()} training designs, "
                 f"{int(train.sum()):,} runs)\nheld-out edit-effect R² {held_r2}  |  HIC-change slope {slope:.2f}  |  "
                 f"sibling pair R² {sib_in:.2f} in-sample, {sib_out:.2f} held-out",
                 x=0.06, ha="left", fontsize=11.5, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cards", nargs="*", default=None,
                        help="designs to draw a card for (default: all designs)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    data_dir = resolve_data_dir(run_dir, args.data_dir)
    out = run_dir / "diagnostics"
    (out / "designs").mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir, data_dir)
    table = per_design_table(ctx)
    pairs = design_pairs(ctx)
    pairs_by = pair_lookup(pairs)
    table.to_csv(out / "per_design.csv")
    pairs.to_csv(out / "design_pairs.csv", index=False)
    ctx.runs.to_csv(out / "per_run.csv", index=False)

    fig_summary(ctx, pairs, out / "00_summary.png")
    fig_atlas(ctx, table, out / "01_design_atlas.png")
    fig_acceleration(ctx, out / "02_acceleration_accuracy.png")
    fig_effect_skill(ctx, table, out / "03_edit_effect_skill.png")
    fig_pairs(ctx, pairs, out / "04_design_pair_distinction.png")
    held = table[table["split"] != "train"].index.tolist()
    baselines = {}
    for design in held:
        baselines[design] = fig_vs_related(ctx, design, pairs_by, out / f"05_{design}_vs_related.png")
        fig_all_locations(ctx, design, out / f"06_{design}_all_locations.png")
    cards = args.cards if args.cards is not None else table.index.tolist()
    for k, design in enumerate(cards, 1):
        fig_card(ctx, design, table, pairs_by, out / "designs" / f"{design}.png")
        if k % 12 == 0:
            print(f"  cards {k}/{len(cards)}", flush=True)

    pooled = {}
    for (group, rel), f in pairs.groupby(["group", "relation"]):
        w = f["true_diff_rms_g"] ** 2 * f["shared_locations"]
        pooled.setdefault(group, {})[rel] = {"pairs": int(len(f)),
                                              "pair_r2": float(1.0 - np.sum(w * (1 - f["pair_r2"])) / np.sum(w))}
    summary = {
        "run_dir": str(run_dir),
        "held_out": {d: table.loc[d].to_dict() for d in held},
        "physics_baselines": baselines,
        "pair_r2_pooled": pooled,
        "class_twin_copy_r2_train": {cls: twin_r2(ctx, ctx.rows(split="train", **{"class": cls})) for cls in CLASSES},
    }
    (out / "diagnostics_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"physics_baselines": baselines, "pair_r2_pooled": pooled,
                      "class_twin_copy_r2_train": summary["class_twin_copy_r2_train"]}, indent=1, default=float))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
