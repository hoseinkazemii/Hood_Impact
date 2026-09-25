"""Post-hoc diagnostics of design sensitivity, with no fitted statistics.

These metrics consume predictions and references only after prediction. They
must not be used to fit geometry selection, scalers, or training weights on a
held-out split. No mesh geometry or data outside the supplied split is read.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np


def _summary(pair_error_sum, true_pair_sum, pair_points,
             true_variance_sum, predicted_variance_sum, spread_points):
    if not pair_points:
        return {
            "same_impact_difference_rmse_g": None,
            "collapsed_predictor_difference_rmse_g": None,
            "true_across_design_rms_spread_g": None,
            "predicted_across_design_rms_spread_g": None,
            "predicted_to_true_spread_ratio": None,
        }
    true_spread = float(np.sqrt(true_variance_sum / spread_points))
    predicted_spread = float(np.sqrt(predicted_variance_sum / spread_points))
    return {
        "same_impact_difference_rmse_g": float(np.sqrt(pair_error_sum / pair_points)),
        "collapsed_predictor_difference_rmse_g": float(np.sqrt(true_pair_sum / pair_points)),
        "true_across_design_rms_spread_g": true_spread,
        "predicted_across_design_rms_spread_g": predicted_spread,
        "predicted_to_true_spread_ratio": predicted_spread / true_spread if true_spread > 0 else None,
    }


def design_sensitivity_metrics(dataset, predictions, targets, samples_per_design):
    """Measure response differences for all different designs at matching impacts.

    ``predictions`` and ``targets`` are flattened physical-g histories in the
    order of ``dataset.run_numbers``. Time grids come from ``dataset.time_arrays``;
    impact XY comes from ``dataset.indentor_positions`` in physical millimetres.
    Location IDs in the result are one based. Unequal time grids or impact XY
    differing by more than 0.001 mm are rejected, without interpolation.

    Difference RMSE pools every same-location unordered design pair and time
    point. The collapsed baseline predicts zero design-to-design difference.
    Spread is the square root of mean population variance across designs,
    pooled over matched location/time points. Ratio < 1 indicates suppressed
    spread; the ratio alone cannot establish correct signed differences.
    Single-design locations are excluded and zero reference spread has a null
    ratio. All outputs are JSON serializable and all locations receive equal
    treatment regardless of their observed variation.
    """
    if isinstance(samples_per_design, bool) or not isinstance(samples_per_design, (int, np.integer)) or samples_per_design < 1:
        raise ValueError("samples_per_design must be a positive integer")
    runs = list(dataset.run_numbers)
    if len(dataset.time_arrays) != len(runs) or len(dataset.indentor_positions) != len(runs):
        raise ValueError("Run numbers, time grids and impact XY must have equal lengths")
    if any(isinstance(run, (bool, np.bool_)) or not isinstance(run, (int, np.integer)) or run < 1 for run in runs):
        raise ValueError("Run numbers must be positive integers")
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicate design/location run numbers")

    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Predictions and targets must contain only finite physical-g values")

    times, positions = [], []
    groups = defaultdict(list)
    for index, run in enumerate(runs):
        time = np.asarray(dataset.time_arrays[index])
        xy = np.asarray(dataset.indentor_positions[index])
        if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("Each time grid must be a nonempty, finite, strictly increasing vector")
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Each impact XY must be a finite two-coordinate vector")
        times.append(time)
        positions.append(xy)
        groups[(int(run) - 1) % samples_per_design + 1].append(index)
    offsets = np.concatenate(([0], np.cumsum([len(time) for time in times])))
    if len(prediction) != offsets[-1] or len(target) != offsets[-1]:
        raise ValueError("Flattened prediction/target length does not match the dataset time grids")

    totals = np.zeros(6, dtype=np.float64)
    pair_count = 0
    per_location = []
    for location, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        reference = indices[0]
        for index in indices[1:]:
            if not np.array_equal(times[index], times[reference]):
                raise ValueError(f"Time grids differ at location {location}; no implicit interpolation is allowed")
            if not np.allclose(positions[index], positions[reference], rtol=0, atol=1e-3):
                raise ValueError(f"Impact XY mismatch at location {location}")
        predicted_curves = np.stack([prediction[offsets[i]:offsets[i + 1]] for i in indices])
        true_curves = np.stack([target[offsets[i]:offsets[i + 1]] for i in indices])
        pair_error_sum = true_pair_sum = 0.0
        local_pairs = 0
        for a, b in combinations(range(len(indices)), 2):
            true_delta = true_curves[a] - true_curves[b]
            error_delta = predicted_curves[a] - predicted_curves[b] - true_delta
            pair_error_sum += float(np.square(error_delta).sum())
            true_pair_sum += float(np.square(true_delta).sum())
            local_pairs += 1
        local = np.array([
            pair_error_sum, true_pair_sum, local_pairs * len(times[reference]),
            np.var(true_curves, axis=0, ddof=0).sum(),
            np.var(predicted_curves, axis=0, ddof=0).sum(), len(times[reference]),
        ])
        totals += local
        pair_count += local_pairs
        per_location.append({
            "location_id": location,
            "design_ids": [(int(runs[i]) - 1) // samples_per_design for i in indices],
            "num_designs": len(indices),
            "num_design_pairs": local_pairs,
            **_summary(*local),
        })
    return {
        "num_matched_locations": len(per_location),
        "num_design_pairs": pair_count,
        "num_unmatched_locations": len(groups) - len(per_location),
        **_summary(*totals),
        "per_location": per_location,
    }
