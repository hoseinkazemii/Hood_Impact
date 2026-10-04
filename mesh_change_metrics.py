"""Post-hoc diagnostics of design sensitivity, with no fitted statistics.

These metrics consume predictions and references only after prediction. They
must not be used to fit geometry selection, scalers, or training weights on a
held-out split. No mesh geometry or data outside the supplied split is read.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
from hic15 import batched_hic


def _summary(pair_error_sum, true_pair_sum, pair_points,
             true_variance_sum, predicted_variance_sum, spread_points,
             predicted_pair_sum, pair_dot_sum, unit="_g"):
    if not pair_points:
        return {
            f"same_impact_difference_rmse{unit}": None,
            f"collapsed_predictor_difference_rmse{unit}": None,
            f"true_across_design_rms_spread{unit}": None,
            f"predicted_across_design_rms_spread{unit}": None,
            "predicted_to_true_spread_ratio": None,
            "design_difference_skill": None,
            "difference_alignment_cosine": None,
            "difference_gain": None,
            "difference_amplitude_ratio": None,
        }
    true_spread = float(np.sqrt(true_variance_sum / spread_points))
    predicted_spread = float(np.sqrt(predicted_variance_sum / spread_points))
    return {
        f"same_impact_difference_rmse{unit}": float(np.sqrt(pair_error_sum / pair_points)),
        f"collapsed_predictor_difference_rmse{unit}": float(np.sqrt(true_pair_sum / pair_points)),
        f"true_across_design_rms_spread{unit}": true_spread,
        f"predicted_across_design_rms_spread{unit}": predicted_spread,
        "predicted_to_true_spread_ratio": predicted_spread / true_spread if true_spread > 0 else None,
        "design_difference_skill": 1 - pair_error_sum / true_pair_sum if true_pair_sum > 0 else None,
        "difference_alignment_cosine": pair_dot_sum / np.sqrt(true_pair_sum * predicted_pair_sum)
            if true_pair_sum > 0 and predicted_pair_sum > 0 else None,
        "difference_gain": pair_dot_sum / true_pair_sum if true_pair_sum > 0 else None,
        "difference_amplitude_ratio": np.sqrt(predicted_pair_sum / true_pair_sum) if true_pair_sum > 0 else None,
    }


def _sums(predicted, truth):
    """Sufficient statistics for unordered design pairs; columns are times."""
    true_delta = np.stack([truth[a] - truth[b] for a, b in combinations(range(len(truth)), 2)])
    predicted_delta = np.stack([predicted[a] - predicted[b] for a, b in combinations(range(len(truth)), 2)])
    return np.array([
        np.square(predicted_delta - true_delta).sum(), np.square(true_delta).sum(), true_delta.size,
        np.var(truth, axis=0).sum(), np.var(predicted, axis=0).sum(), truth.shape[1],
        np.square(predicted_delta).sum(), (predicted_delta * true_delta).sum(),
    ], dtype=np.float64)


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
    Skill = 1 - squared difference error / squared true differences. Perfect
    differences score 1; a design-independent curve scores 0; worse differences
    score below 0. Alignment is an uncentered cosine of signed pair differences.
    HIC15 diagnostics use the supplied time grids in seconds, with no resampling.
    Single-design locations are excluded and zero reference spread has null
    normalized scores. Locations with zero true variation still contribute
    spurious predicted differences to the aggregate error. Pairs/time samples
    are equally weighted (locations equally weighted when counts/grids match).
    Optional missing impact XY permits CSV-only backfills; the result explicitly
    records that only the run-number location mapping was checked in that case.
    """
    if isinstance(samples_per_design, bool) or not isinstance(samples_per_design, (int, np.integer)) or samples_per_design < 1:
        raise ValueError("samples_per_design must be a positive integer")
    runs = list(dataset.run_numbers)
    input_positions = getattr(dataset, "indentor_positions", None)
    if len(dataset.time_arrays) != len(runs) or (input_positions is not None and len(input_positions) != len(runs)):
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
        xy = np.asarray(input_positions[index]) if input_positions is not None else None
        if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("Each time grid must be a nonempty, finite, strictly increasing vector")
        if xy is not None and (xy.shape != (2,) or not np.isfinite(xy).all()):
            raise ValueError("Each impact XY must be a finite two-coordinate vector")
        times.append(time)
        positions.append(xy)
        groups[(int(run) - 1) % samples_per_design + 1].append(index)
    offsets = np.concatenate(([0], np.cumsum([len(time) for time in times])))
    if len(prediction) != offsets[-1] or len(target) != offsets[-1]:
        raise ValueError("Flattened prediction/target length does not match the dataset time grids")

    totals = np.zeros(8, dtype=np.float64)
    hic_totals = np.zeros(8, dtype=np.float64)
    pair_count = 0
    per_location = []
    for location, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        reference = indices[0]
        for index in indices[1:]:
            if not np.array_equal(times[index], times[reference]):
                raise ValueError(f"Time grids differ at location {location}; no implicit interpolation is allowed")
            if input_positions is not None and not np.allclose(positions[index], positions[reference], rtol=0, atol=1e-3):
                raise ValueError(f"Impact XY mismatch at location {location}")
        predicted_curves = np.stack([prediction[offsets[i]:offsets[i + 1]] for i in indices])
        true_curves = np.stack([target[offsets[i]:offsets[i + 1]] for i in indices])
        local_pairs = len(indices) * (len(indices) - 1) // 2
        local = _sums(predicted_curves, true_curves)
        local_hic = _sums(batched_hic(times[reference], predicted_curves)[:, None],
                          batched_hic(times[reference], true_curves)[:, None])
        totals += local
        hic_totals += local_hic
        pair_count += local_pairs
        per_location.append({
            "location_id": location,
            "design_ids": [(int(runs[i]) - 1) // samples_per_design for i in indices],
            "num_designs": len(indices),
            "num_design_pairs": local_pairs,
            **_summary(*local),
            "hic15": _summary(*local_hic, unit=""),
        })
    return {
        "schema_version": 2,
        "samples_per_design": int(samples_per_design),
        "design_ids": sorted({(int(run) - 1) // samples_per_design for run in runs}),
        "impact_xy_checked": input_positions is not None,
        "hic15_source": "supplied sampled acceleration histories; seconds; 15 ms maximum window",
        "num_matched_locations": len(per_location),
        "num_design_pairs": pair_count,
        "num_unmatched_locations": len(groups) - len(per_location),
        **_summary(*totals),
        "hic15": _summary(*hic_totals, unit=""),
        "per_location": per_location,
    }
