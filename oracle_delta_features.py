"""Shallow baseline on explicit geometry-difference features: is the signal in the edit?

Within a mesh family the near-clone edit is an exact displacement field
dx = x_b - x_a on corresponding nodes (design_difference_data.pair_geometry). This
script summarizes that field around each impact point with handcrafted,
direction-aware descriptors and regresses the acceleration difference
a_b(t) - a_a(t) on them with ridge regression and extremely randomized trees.
No mesh encoder, no pooling: if these models predict the difference on edits
they have never seen, the information exists in the geometry and the deep
models are losing it; if they cannot, the deep models had nothing to find.

Features per (directed pair a -> b, impact location p), from nodes within an XY
radius r of p, r in {50, 100, 200, 400} mm, for the outer panel, the inner panel
and all nodes:
    sum |dx|, max |dx|, sum dz, sum dx.n (normal displacement)
plus the panel-gap closing and opening sums and the count of nodes moved
> 0.5 mm, and globally the log XY/3D distance from p to the edit.
Groups that can be switched on:
    location  X1, X2, headform start penetration, in-contact flag
    dt        log(dt_b / dt_a) from the increment counts, and its product with
              the in-contact flag: the suspected solver artifact
    ref       PCA code of the baseline design's own curve a_a(t) (known at
              deployment, but it breaks exact antisymmetry, see below)

Every predictor except those using ``ref`` is antisymmetrized,
    d^(a->b) = (f(phi(a->b)) - f(phi(b->a))) / 2,
so swapping the designs flips the prediction exactly and a zero edit predicts
zero. Training uses both directions of the 12 train pairs.

Protocols (difference R2, a < b orientation, the approved metric):
    in-sample      fit on the 12 train pairs, score them
    unseen-edit    leave one training design out: fit on the pairs without it,
                   score the 3 pairs with it (its edit is never seen)
    test / val     fit on all 12, score (4, 5) and (10, 11)
each overall, by XY distance band to the edit, and (unseen-edit) split into
pairs with matched vs mismatched solver time step.

Usage:
    python oracle_delta_features.py
    python oracle_delta_features.py --models ridge --feature-sets geometry geometry+dt
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import design_difference_data as dd

DEFAULT_OUT = dd.CODE_ROOT / "runs" / "design_difference" / "delta_features"
RADII_MM = (50.0, 100.0, 200.0, 400.0)
PANEL_SETS = {"outer": (0,), "inner": (1,), "all": (0, 1, 2, 3)}
FEATURE_SETS = {
    "location": ("location",),
    "location+dt": ("location", "dt"),
    "location+geometry": ("location", "geometry"),
    "location+geometry+dt": ("location", "geometry", "dt"),
    "location+geometry+dt+ref": ("location", "geometry", "dt", "ref"),
}
MODELS = ("ridge", "trees")
# |log dt ratio| below this counts as a matched time step (pairs 6-9, 7-8, 0-3,
# 1-3 and 10-11, as in the within-family dt-artifact analysis).
MATCHED_DT = 0.11
REF_COMPONENTS = 8


def signed_log(values):
    """sign(x) log(1 + |x|): edits differ in size by orders of magnitude, and raw
    sums make a held-out edit an extreme outlier that a linear model extrapolates."""
    return np.sign(values) * np.log1p(np.abs(values))


def geometry_features(edit, impact_xy, sign):
    """(142, F) direction-aware descriptors of one pair's edit; sign=-1 swaps a and b."""
    dx, dgap = sign * edit["dx"], sign * edit["dgap"]
    magnitude = np.linalg.norm(dx, axis=1)
    normal_disp = np.sum(dx * edit["normal"], axis=1)
    moved = magnitude > dd.MOVED_THRESHOLD_MM
    distance = np.linalg.norm(impact_xy[:, None, :] - edit["x"][None, :, :2], axis=2)
    columns, names = [], []
    for radius in RADII_MM:
        inside = distance < radius
        for panel_name, panels in PANEL_SETS.items():
            mask = inside & np.isin(edit["panel"], panels)[None, :]
            for name, values, reducer in (("abs_sum", magnitude, "sum"), ("max_abs", magnitude, "max"),
                                          ("dz_sum", dx[:, 2], "sum"), ("dn_sum", normal_disp, "sum")):
                masked = np.where(mask, values[None, :], 0.0)
                columns.append(signed_log(masked.max(axis=1) if reducer == "max" else masked.sum(axis=1)))
                names.append(f"{name}_{panel_name}_r{radius:g}")
        columns.append(signed_log(np.where(inside, np.minimum(dgap, 0)[None, :], 0.0).sum(axis=1)))
        names.append(f"gap_close_r{radius:g}")
        columns.append(signed_log(np.where(inside, np.maximum(dgap, 0)[None, :], 0.0).sum(axis=1)))
        names.append(f"gap_open_r{radius:g}")
        columns.append(np.log1p((inside & moved[None, :]).sum(axis=1).astype(float)))
        names.append(f"n_moved_r{radius:g}")
    columns.append(np.log1p(edit["distance_xy"]))
    names.append("log_distance_xy_edit")
    columns.append(np.log1p(edit["distance_3d"]))
    names.append("log_distance_3d_edit")
    return np.stack(columns, axis=1), names


def build_dataset(args):
    """Features and targets for every directed near-clone pair at every location."""
    times, acceleration = dd.load_histories(args.data_dir)
    _, acceleration = dd.model_grid(times, acceleration)
    runs = dd.load_run_table(args.data_dir)
    increments = dd.design_increments(runs)
    penetration = dd.per_run_array(runs, "sphere_pen_mm")
    in_contact = dd.per_run_array(runs, "in_contact").astype(float)
    impact_xy = dd.load_locations(args.data_dir)[["X1", "X2"]].to_numpy(float)
    geometry = {d: dd.load_design_geometry(d, args.data_dir) for d in range(dd.NUM_DESIGNS)}

    from sklearn.decomposition import PCA
    reference_pca = PCA(REF_COMPONENTS, random_state=0).fit(
        acceleration[list(dd.TRAIN_DESIGNS)].reshape(-1, acceleration.shape[-1]))

    rows, blocks = [], {"location": [], "dt": [], "geometry": [], "ref": []}
    targets, names = [], {}
    for group, pairs in dd.PAIR_GROUPS.items():
        for a, b in pairs:
            pair = dd.pair_geometry(geometry[a], geometry[b])
            centre = runs[runs.design == a].sort_values("loc")[["X1", "X2", "X3"]].to_numpy()
            distance_xy, distance_3d = dd.edit_distance(pair, centre)
            keep = pair["edit"]
            edit = {"x": pair["x_mid"][keep], "dx": pair["dx"][keep], "dgap": pair["dgap"][keep],
                    "normal": pair["normal_mid"][keep], "panel": pair["panel"][keep],
                    "distance_xy": distance_xy, "distance_3d": distance_3d}
            rho = dd.log_dt_ratio(increments, a, b)
            for sign, base in ((1, a), (-1, b)):
                features, names["geometry"] = geometry_features(edit, impact_xy, sign)
                blocks["geometry"].append(features)
                blocks["location"].append(np.column_stack((impact_xy / 1000.0, penetration[a], in_contact[a])))
                blocks["dt"].append(np.column_stack((np.full(dd.SAMPLES_PER_DESIGN, sign * rho),
                                                     sign * rho * in_contact[a])))
                blocks["ref"].append(reference_pca.transform(acceleration[base]))
                targets.append(sign * (acceleration[b] - acceleration[a]))
                band = dd.distance_band(distance_xy)
                for location in range(1, dd.SAMPLES_PER_DESIGN + 1):
                    rows.append({"group": group, "a": a, "b": b, "sign": sign, "location": location,
                                 "band": band[location - 1], "abs_log_dt": abs(rho)})
    index = pd.DataFrame(rows)
    blocks = {key: np.concatenate(value) for key, value in blocks.items()}
    names.update({"location": ["x1_m", "x2_m", "sphere_pen_mm", "in_contact"],
                  "dt": ["log_dt_ratio", "log_dt_ratio_x_contact"],
                  "ref": [f"ref_pc{i}" for i in range(REF_COMPONENTS)]})
    return index, blocks, names, np.concatenate(targets)


class Antisymmetric:
    """Fit on both directions; predict (f(x_ab) - f(x_ba)) / 2 when asked to."""

    def __init__(self, kind, alpha=1.0, seed=0):
        self.kind, self.alpha, self.seed = kind, alpha, seed

    def fit(self, features, targets):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self.kind == "ridge":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        else:
            self.model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33,
                                             n_jobs=-1, random_state=self.seed)
        self.model.fit(features, targets)
        return self

    def predict(self, forward, backward=None):
        prediction = self.model.predict(forward)
        return prediction if backward is None else 0.5 * (prediction - self.model.predict(backward))


def select_rows(index, pairs, sign=None):
    mask = index[["a", "b"]].apply(tuple, axis=1).isin(set(pairs)).to_numpy()
    if sign is not None:
        mask &= index["sign"].to_numpy() == sign
    return np.nonzero(mask)[0]


def predict_forward(model, features, index, pairs, antisymmetric):
    """Predictions for the a<b direction, rows ordered like select_rows(index, pairs, +1)."""
    forward = select_rows(index, pairs, 1)
    if not antisymmetric:
        return model.predict(features[forward]), forward
    # Rows of the reverse direction, in the same (pair, location) order.
    backward = select_rows(index, pairs, -1)
    key = lambda rows: list(zip(index.a.iloc[rows], index.b.iloc[rows], index.location.iloc[rows]))  # noqa: E731
    if key(forward) != key(backward):
        raise RuntimeError("forward and backward rows are not aligned")
    return model.predict(features[forward], features[backward]), forward


def score(prediction, truth, index, rows, label, extra=None):
    out = []
    frame = index.iloc[rows].reset_index(drop=True)
    splits = {"all": np.ones(len(rows), bool)}
    for band in ("near", "mid", "far"):
        splits[band] = (frame["band"] == band).to_numpy()
    if extra == "dt":
        splits["matched_dt"] = (frame["abs_log_dt"] <= MATCHED_DT).to_numpy()
        splits["mismatched_dt"] = (frame["abs_log_dt"] > MATCHED_DT).to_numpy()
    for name, mask in splits.items():
        if mask.any():
            out.append({**label, "subset": name, "n": int(mask.sum()),
                        "difference_r2": dd.difference_r2(prediction[mask], truth[mask])})
    return out


def run(args, log):
    index, blocks, names, targets = build_dataset(args)
    log.info("Dataset: %s directed (pair, location) rows, %s geometry features, %s time points",
             len(index), blocks["geometry"].shape[1], targets.shape[1])
    from sklearn.model_selection import GroupKFold  # noqa: F401  (documents the grouping intent)

    rows, importances = [], None
    for model_kind in args.models:
        for set_name in args.feature_sets:
            groups = FEATURE_SETS[set_name]
            features = np.concatenate([blocks[g] for g in groups], axis=1)
            feature_names = [n for g in groups for n in names[g]]
            antisymmetric = "ref" not in groups
            train_rows = select_rows(index, dd.TRAIN_PAIRS)

            alpha = 1.0
            if model_kind == "ridge":
                alpha = choose_alpha(features, targets, index, antisymmetric)
            label = {"model": model_kind, "features": set_name, "antisymmetric": antisymmetric, "alpha": alpha}

            # Unseen edit: hold out one training design (all its pairs).
            fold_predictions, fold_truth, fold_rows = [], [], []
            for held in dd.TRAIN_DESIGNS:
                test_pairs = [p for p in dd.TRAIN_PAIRS if held in p]
                fit_pairs = [p for p in dd.TRAIN_PAIRS if held not in p]
                fit_rows = select_rows(index, fit_pairs)
                model = Antisymmetric(model_kind, alpha, args.seed).fit(features[fit_rows], targets[fit_rows])
                prediction, forward = predict_forward(model, features, index, test_pairs, antisymmetric)
                fold_predictions.append(prediction)
                fold_truth.append(targets[forward])
                fold_rows.append(forward)
            rows += score(np.concatenate(fold_predictions), np.concatenate(fold_truth), index,
                          np.concatenate(fold_rows), {**label, "protocol": "unseen_edit"}, extra="dt")

            model = Antisymmetric(model_kind, alpha, args.seed).fit(features[train_rows], targets[train_rows])
            for protocol, pairs in (("in_sample", dd.TRAIN_PAIRS), ("validation", dd.VALIDATION_PAIRS),
                                    ("test", dd.TEST_PAIRS)):
                prediction, forward = predict_forward(model, features, index, pairs, antisymmetric)
                rows += score(prediction, targets[forward], index, forward, {**label, "protocol": protocol})
            if model_kind == "trees" and set_name == "location+geometry+dt":
                importances = pd.Series(model.model.feature_importances_, index=feature_names)
            summary = {r["protocol"]: r["difference_r2"] for r in rows
                       if r["model"] == model_kind and r["features"] == set_name and r["subset"] == "all"}
            log.info("%-6s %-26s unseen-edit %.3f | in-sample %.3f | val (10,11) %.3f | test (4,5) %.3f",
                     model_kind, set_name, summary["unseen_edit"], summary["in_sample"],
                     summary["validation"], summary["test"])
    return pd.DataFrame(rows), importances


def choose_alpha(features, targets, index, antisymmetric, alphas=np.logspace(-1, 7, 17)):
    """Ridge alpha by leave-one-training-design-out difference R2 (slightly optimistic).

    The grid reaches alphas that shrink every coefficient to ~0, so when no
    alpha transfers the choice falls back to the collapsed predictor (R2 ~ 0).
    """
    best, best_alpha = -np.inf, 1.0
    for alpha in alphas:
        predictions, truth = [], []
        for held in dd.TRAIN_DESIGNS:
            fit_rows = select_rows(index, [p for p in dd.TRAIN_PAIRS if held not in p])
            model = Antisymmetric("ridge", alpha).fit(features[fit_rows], targets[fit_rows])
            prediction, forward = predict_forward(model, features, index,
                                                  [p for p in dd.TRAIN_PAIRS if held in p], antisymmetric)
            predictions.append(prediction)
            truth.append(targets[forward])
        value = dd.difference_r2(np.concatenate(predictions), np.concatenate(truth))
        if value > best:
            best, best_alpha = value, float(alpha)
    return best_alpha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("delta_features")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.FileHandler(args.out_dir / "delta_features.log", mode="w"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)

    table, importances = run(args, log)
    table.to_csv(args.out_dir / "delta_features.csv", index=False)
    overall = table[table.subset == "all"].pivot_table(index=["model", "features"], columns="protocol",
                                                       values="difference_r2")
    log.info("Difference R2 (all locations):\n%s", overall.round(3).to_string())
    unseen = table[table.protocol == "unseen_edit"].pivot_table(index=["model", "features"], columns="subset",
                                                                values="difference_r2")
    log.info("Unseen-edit difference R2 by subset:\n%s", unseen.round(3).to_string())
    test = table[table.protocol == "test"].pivot_table(index=["model", "features"], columns="subset",
                                                       values="difference_r2")
    log.info("Test pair (4, 5) difference R2 by distance band:\n%s", test.round(3).to_string())
    report = {"overall": {f"{m}/{f}": row.to_dict() for (m, f), row in overall.iterrows()},
              "unseen_edit": {f"{m}/{f}": row.to_dict() for (m, f), row in unseen.iterrows()},
              "test_by_band": {f"{m}/{f}": row.to_dict() for (m, f), row in test.iterrows()}}
    if importances is not None:
        top = importances.sort_values(ascending=False).head(15)
        log.info("Top tree importances (location+geometry+dt):\n%s", top.round(4).to_string())
        report["top_importances"] = top.to_dict()
    dd.write_json(args.out_dir / "delta_features.json", report)
    log.info("Wrote %s", args.out_dir)
    return table


if __name__ == "__main__":
    main()
