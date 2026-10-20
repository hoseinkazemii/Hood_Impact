"""Model-free ceiling on predicting a near-clone difference at unseen impact locations.

Take one near-clone pair's TRUE acceleration difference d_p(t) = a_b - a_a at
the training locations and predict it at the held-out ones by spatial
interpolation alone. No geometry and no network: the predictor already knows
the pair perfectly. Any surrogate that must interpolate across impact locations
(even one handed the design identity) is bounded by this, up to the
interpolator's quality. If the ceiling is near zero, the difference is not
spatially coherent at the 100 mm grid spacing and no location-interpolating
model can predict it; only a model that reads the geometry at the new location
could.

Protocols (both keep lateral mirror twins together, since the hood is nearly
mirror-symmetric):

* checkerboard -- fit on the 71 even-parity locations, predict the 71 odd ones
  (location_splits/euroncap1704_checkerboard.json, the same split as the
  unseen-location surrogate run).
* loo -- predict each location from all others except itself and its twin.

Interpolators: zero (collapsed predictor), nearest location, inverse-distance
weighting of the 4 nearest, and a Gaussian process (Matern 3/2 + white noise,
hyperparameters by marginal likelihood) in XY. Also reports a correlogram: the
mean correlation between difference curves of two locations as a function of
their XY distance, and the same ceiling for the difference MAGNITUDE alone
(log RMS of the difference curve per location). The magnitude can be spatially
coherent while the curve shape is not; the two answers mean different things
for a surrogate.

Metric: R2 of the difference curve pooled over time, locations and pairs
(design_difference_data.difference_r2), for train (A and C near-clone pairs),
validation (10, 11) and test (4, 5) pairs, overall and by XY distance from the
impact point to the pair's edit.

Usage:
    python oracle_interpolation_ceiling.py
    python oracle_interpolation_ceiling.py --out-dir runs/design_difference/ceiling --stride 1
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import design_difference_data as dd

DEFAULT_OUT = dd.CODE_ROOT / "runs" / "design_difference" / "interpolation_ceiling"
METHODS = ("zero", "nearest", "idw", "gp")
CORRELOGRAM_BINS_MM = (0, 120, 160, 230, 320, 450, 700, 3000)


def interpolate(method, train_xy, train_values, query_xy, gp_kernel=None):
    """Predict (Q, T) values at query_xy from (N, T) training values."""
    if method == "zero":
        return np.zeros((len(query_xy), train_values.shape[1]))
    distance = np.linalg.norm(query_xy[:, None, :] - train_xy[None, :, :], axis=2)
    if method == "nearest":
        return train_values[np.argmin(distance, axis=1)]
    if method == "idw":
        nearest = np.argsort(distance, axis=1)[:, :4]
        weight = 1.0 / np.maximum(np.take_along_axis(distance, nearest, axis=1), 1e-6) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        return np.einsum("qk,qkt->qt", weight, train_values[nearest])
    if method == "gp":
        model = fit_gp(train_xy, train_values, gp_kernel)
        return model.predict(query_xy)
    raise ValueError(f"unknown method {method}")


def fit_gp(train_xy, train_values, kernel=None):
    """Matern 3/2 GP shared across the time points; kernel=None optimizes it."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    if kernel is None:
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(150.0, (20.0, 3000.0), nu=1.5)
                  + WhiteKernel(0.5, (1e-4, 10.0)))
        optimizer, restarts = "fmin_l_bfgs_b", 2
    else:
        optimizer, restarts = None, 0
    model = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=optimizer,
                                     n_restarts_optimizer=restarts, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_xy, train_values)
    return model


def predict_pair(difference, xy, grid, even, odd, method):
    """{protocol: (142, T) prediction, NaN where the protocol does not predict}."""
    out = {}
    checker = np.full_like(difference, np.nan)
    checker[odd - 1] = interpolate(method, xy[even - 1], difference[even - 1], xy[odd - 1])
    out["checkerboard"] = checker
    kernel = None
    if method == "gp":
        # LOO with the kernel fitted once on all locations: 142 refits of the
        # hyperparameters would be slow and change nothing material.
        kernel = fit_gp(xy, difference).kernel_
    loo = np.empty_like(difference)
    for location in range(1, dd.SAMPLES_PER_DESIGN + 1):
        keep = np.ones(dd.SAMPLES_PER_DESIGN, bool)
        keep[[location - 1, grid.loc[location, "mirror_twin"] - 1]] = False
        loo[location - 1] = interpolate(method, xy[keep], difference[keep], xy[location - 1:location], kernel)[0]
    out["loo"] = loo
    return out


def correlogram(difference, xy):
    """Mean Pearson correlation of two locations' difference curves per XY-distance bin."""
    centred = difference - difference.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1)
    valid = norm > 0
    correlation = (centred @ centred.T) / np.outer(np.where(valid, norm, 1), np.where(valid, norm, 1))
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    upper = np.triu(np.ones_like(distance, bool), 1) & valid[:, None] & valid[None, :]
    rows = []
    for low, high in zip(CORRELOGRAM_BINS_MM[:-1], CORRELOGRAM_BINS_MM[1:]):
        mask = upper & (distance >= low) & (distance < high)
        if mask.any():
            rows.append({"bin_mm": f"{low}-{high}", "pairs_of_locations": int(mask.sum()),
                         "mean_correlation": float(correlation[mask].mean())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=dd.MODEL_TIME_STRIDE,
                        help="time stride; 16 is the surrogates' 63-point grid")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(args.out_dir / "ceiling.log", mode="w"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ceiling")

    times, acceleration = dd.load_histories(args.data_dir)
    _, acceleration = dd.model_grid(times, acceleration, args.stride)
    grid = dd.load_locations(args.data_dir)
    xy = grid[["X1", "X2"]].to_numpy(float)
    even, odd = dd.checkerboard_locations()
    runs = dd.load_run_table(args.data_dir)
    geometry = {d: dd.load_design_geometry(d, args.data_dir) for d in range(dd.NUM_DESIGNS)}
    log.info("Loaded %s histories on a %s-point grid", acceleration.shape[:2], acceleration.shape[2])

    truth, bands, predictions, correlations = {}, {}, {}, []
    for group, pairs in dd.PAIR_GROUPS.items():
        for a, b in pairs:
            truth[(a, b)] = acceleration[b] - acceleration[a]
            centre = runs[runs.design == a].sort_values("loc")[["X1", "X2", "X3"]].to_numpy()
            distance, _ = dd.edit_distance(dd.pair_geometry(geometry[a], geometry[b]), centre)
            bands[(a, b)] = dd.distance_band(distance)
            for row in correlogram(truth[(a, b)], xy):
                correlations.append({"group": group, "pair": f"{a}-{b}", **row})
            for method in args.methods:
                predictions[(method, a, b)] = predict_pair(truth[(a, b)], xy, grid, even, odd, method)
            log.info("pair %s-%s (%s): RMS difference %.2f g", a, b, group, np.sqrt(np.mean(truth[(a, b)] ** 2)))

    rows = []
    for method in args.methods:
        for protocol, locations in (("checkerboard", odd), ("loo", np.arange(1, dd.SAMPLES_PER_DESIGN + 1))):
            for group, pairs in dd.PAIR_GROUPS.items():
                for band in ("all", "near", "mid", "far"):
                    p_parts, d_parts = [], []
                    for pair in pairs:
                        take = locations - 1
                        if band != "all":
                            take = take[bands[pair][take] == band]
                        p_parts.append(predictions[(method, *pair)][protocol][take])
                        d_parts.append(truth[pair][take])
                    count = sum(len(part) for part in d_parts)
                    rows.append({"method": method, "protocol": protocol, "group": group, "band": band,
                                 "locations_x_pairs": count,
                                 "difference_r2": dd.difference_r2(np.concatenate(p_parts), np.concatenate(d_parts))
                                 if count else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "interpolation_ceiling.csv", index=False)

    # Magnitude only: log RMS of the difference curve, checkerboard GP, pooled R2.
    magnitude_rows = []
    for group, pairs in dd.PAIR_GROUPS.items():
        predicted, actual = [], []
        for pair in pairs:
            magnitude = np.log(np.sqrt(np.mean(truth[pair] ** 2, axis=1)) + 1e-3)[:, None]
            predicted.append(np.ravel(interpolate("gp", xy[even - 1], magnitude[even - 1], xy[odd - 1])))
            actual.append(magnitude[odd - 1, 0])
        predicted, actual = np.concatenate(predicted), np.concatenate(actual)
        magnitude_rows.append({"group": group, "log_rms_r2": dd.difference_r2(predicted, actual),
                               "log_rms_correlation": float(np.corrcoef(predicted, actual)[0, 1])})
    magnitude_table = pd.DataFrame(magnitude_rows)
    magnitude_table.to_csv(args.out_dir / "magnitude_ceiling.csv", index=False)
    log.info("Difference MAGNITUDE (log RMS) at odd locations from even ones, GP:\n%s",
             magnitude_table.round(3).to_string(index=False))
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(args.out_dir / "difference_correlogram.csv", index=False)

    overall = table[table.band == "all"].pivot_table(index=["protocol", "method"], columns="group",
                                                      values="difference_r2")[["train", "validation", "test"]]
    log.info("Difference R2 by interpolating the pair's own true differences (all bands):\n%s",
             overall.round(3).to_string())
    by_band = table[(table.method == "gp")].pivot_table(index=["protocol", "group"], columns="band",
                                                       values="difference_r2")
    if not by_band.empty:
        log.info("GP ceiling by XY distance to the edit:\n%s", by_band.round(3).to_string())
    mean_correlation = correlation_table.groupby(["group", "bin_mm"], sort=False)["mean_correlation"].mean().unstack(0)
    log.info("Mean correlation of difference curves vs location distance:\n%s", mean_correlation.round(3).to_string())
    dd.write_json(args.out_dir / "interpolation_ceiling.json", {
        "stride": args.stride, "methods": args.methods,
        "overall": {f"{p}/{m}": row.to_dict() for (p, m), row in overall.iterrows()},
        "gp_by_band": {f"{p}/{g}": row.to_dict() for (p, g), row in by_band.iterrows()},
        "magnitude_gp_checkerboard": magnitude_table.set_index("group").to_dict("index"),
    })
    plot(overall, correlation_table, args.out_dir / "interpolation_ceiling.png")
    log.info("Wrote %s", args.out_dir)
    return table


def plot(overall, correlation_table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    checker = overall.loc["checkerboard"] if "checkerboard" in overall.index.get_level_values(0) else overall
    checker.plot.bar(ax=left, rot=0)
    left.axhline(0, color="black", linewidth=0.8)
    left.set_ylabel("difference R2 at unseen (odd) locations")
    left.set_title("Interpolating the pair's own true differences")
    left.legend(title="pairs", frameon=False)
    for group, frame in correlation_table.groupby("group"):
        mean = frame.groupby("bin_mm", sort=False)["mean_correlation"].mean()
        right.plot(range(len(mean)), mean.to_numpy(), marker="o", label=group)
        right.set_xticks(range(len(mean)))
        right.set_xticklabels(mean.index, rotation=30, fontsize=8)
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xlabel("XY distance between impact locations (mm)")
    right.set_ylabel("mean correlation of difference curves")
    right.set_title("Spatial coherence of the near-clone difference")
    right.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
