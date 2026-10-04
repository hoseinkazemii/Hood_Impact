"""Post-hoc design-sensitivity diagnostics for subtle within-family changes.

These read predictions after the fact. Training itself optimizes plain
acceleration MSE, so no pairing sampler or difference loss is involved.
"""

import json
import numpy as np
import pytest
import torch

from mesh_change_metrics import design_sensitivity_metrics


class SyntheticDataset:
    def __init__(self, designs=(0, 1, 6), locations=3):
        self.run_numbers = [design * locations + location + 1 for design in designs for location in range(locations)]
        self.indentor_positions = [np.array([location, 2 * location], dtype=float)
                                   for _ in designs for location in range(locations)]
        self.time_arrays = [np.array([0., .001, .002]) for _ in self.run_numbers]

    def __len__(self):
        return len(self.run_numbers)

    def __getitem__(self, index):
        return dict(mesh=torch.zeros(2, 3), indentor=torch.zeros(2),
                    time=torch.tensor(self.time_arrays[index]), acceleration=torch.zeros(3))


def test_metrics_measure_collapse_signed_differences_and_common_errors():
    data = SyntheticDataset(designs=(4, 5), locations=1)
    target = np.array([0., 2., 4., 2., 4., 6.])
    collapsed = np.array([1., 3., 5., 1., 3., 5.])
    metrics = design_sensitivity_metrics(data, collapsed, target, 1)
    assert metrics["same_impact_difference_rmse_g"] == 2
    assert metrics["collapsed_predictor_difference_rmse_g"] == 2
    assert metrics["true_across_design_rms_spread_g"] == 1
    assert metrics["predicted_across_design_rms_spread_g"] == 0
    assert metrics["predicted_to_true_spread_ratio"] == 0
    assert metrics["design_difference_skill"] == 0
    assert metrics["difference_amplitude_ratio"] == 0
    assert metrics["difference_alignment_cosine"] is None
    assert metrics["num_matched_locations"] == metrics["num_design_pairs"] == 1
    assert metrics["per_location"][0]["design_ids"] == [4, 5]
    correct = design_sensitivity_metrics(data, target + 100, target, 1)
    assert correct["same_impact_difference_rmse_g"] == 0
    assert correct["predicted_to_true_spread_ratio"] == 1
    assert correct["design_difference_skill"] == 1
    assert correct["difference_alignment_cosine"] == 1
    # The correct spread alone is insufficient: reversing design differences
    # preserves spread but doubles delta error relative to collapse.
    reversed_designs = target.reshape(2, 3)[::-1].reshape(-1)
    wrong_sign = design_sensitivity_metrics(data, reversed_designs, target, 1)
    assert wrong_sign["predicted_to_true_spread_ratio"] == 1
    assert wrong_sign["same_impact_difference_rmse_g"] == 4
    assert wrong_sign["design_difference_skill"] == -3
    assert wrong_sign["difference_alignment_cosine"] == -1
    assert wrong_sign["difference_gain"] == -1


def test_partial_correct_design_response_beats_collapse_and_common_curve_bias_cancels():
    data = SyntheticDataset(designs=(4, 5), locations=1)
    truth = np.array([[0., 2., 4.], [2., 4., 6.]])
    predicted = truth.mean(0) + .1 * (truth-truth.mean(0)) + np.array([100., -20., 7.])
    result = design_sensitivity_metrics(data, predicted.ravel(), truth.ravel(), 1)
    assert result["design_difference_skill"] == pytest.approx(.19)
    assert result["difference_amplitude_ratio"] == pytest.approx(.1)
    assert result["difference_alignment_cosine"] == pytest.approx(1)
    assert result["difference_gain"] == pytest.approx(.1)


def test_hic_skill_uses_the_nonlinear_hic_of_each_saved_curve():
    data = SyntheticDataset(designs=(4, 5), locations=1)
    truth = np.array([[10., 10., 10.], [20., 20., 20.]])
    result = design_sensitivity_metrics(data, (.5 * truth).ravel(), truth.ravel(), 1)
    assert result["hic15"]["difference_amplitude_ratio"] == pytest.approx(.5**2.5)
    assert result["hic15"]["design_difference_skill"] == pytest.approx(1-(.5**2.5-1)**2)


def test_spurious_variation_at_identical_truth_locations_is_penalized():
    data = SyntheticDataset(designs=(0, 1), locations=2)
    truth = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2.])
    predicted = truth.copy()
    predicted[6:9] = 2  # Invent a difference at a location with zero true spread.
    result = design_sensitivity_metrics(data, predicted, truth, 2)
    assert result["design_difference_skill"] == 0
    assert result["per_location"][0]["design_difference_skill"] is None
    assert result["per_location"][0]["same_impact_difference_rmse_g"] == 2


def test_metrics_are_invariant_to_design_order_for_more_than_two_designs():
    data = SyntheticDataset(designs=(0, 1, 6), locations=1)
    truth = np.arange(9, dtype=float)
    predicted = np.array([1., 4., 2., 0., 9., 8., 6., 3., 9.])
    first = design_sensitivity_metrics(data, predicted, truth, 1)
    order = [2, 0, 1]
    data.run_numbers = [data.run_numbers[i] for i in order]
    second = design_sensitivity_metrics(data, predicted.reshape(3,3)[order].ravel(), truth.reshape(3,3)[order].ravel(), 1)
    for key in ("design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine"):
        assert first[key] == pytest.approx(second[key])


def test_metrics_keep_low_variation_locations_and_define_null_ratios():
    data = SyntheticDataset(designs=(0, 1), locations=2)
    target = np.array([0., 2., 4., 1., 3., 5., 0., 2., 4., 3., 5., 7.])
    result = design_sensitivity_metrics(data, target, target, 2)
    assert result["num_matched_locations"] == 2
    assert result["per_location"][0]["true_across_design_rms_spread_g"] == 0
    assert result["per_location"][0]["predicted_to_true_spread_ratio"] is None
    assert result["true_across_design_rms_spread_g"] == pytest.approx(np.sqrt(.5))
    json.dumps(result, allow_nan=False)
    single = SyntheticDataset(designs=(0,), locations=1)
    missing = design_sensitivity_metrics(single, np.zeros(3), np.zeros(3), 1)
    assert missing["num_matched_locations"] == 0
    assert missing["num_unmatched_locations"] == 1
    assert missing["same_impact_difference_rmse_g"] is None
    json.dumps(missing, allow_nan=False)


@pytest.mark.parametrize("corruption, message", [
    ("time", "Time grids differ"), ("xy", "XY mismatch"),
    ("length", "length does not match"), ("duplicate", "Duplicate"),
    ("finite", "finite physical-g"),
])
def test_metrics_reject_misaligned_or_invalid_histories(corruption, message):
    data = SyntheticDataset(designs=(0, 1), locations=1)
    prediction = np.zeros(6)
    if corruption == "time":
        data.time_arrays[1][1] += 1e-9
    elif corruption == "xy":
        data.indentor_positions[1][0] += .1
    elif corruption == "length":
        prediction = prediction[:-1]
    elif corruption == "duplicate":
        data.run_numbers[1] = data.run_numbers[0]
    elif corruption == "finite":
        prediction[0] = np.nan
    with pytest.raises(ValueError, match=message):
        design_sensitivity_metrics(data, prediction, np.zeros(6), 1)
