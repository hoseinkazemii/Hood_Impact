"""Direct-output correction reports must distinguish spread from correct effects."""

import copy
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from evaluate_design_sensitivity import evaluate_saved_run
from mesh_geometry_correction_reporting import export_correction_evaluation


TIMES = np.array([0., .005, .010, .015])


def records():
    rows = []
    for design in (4, 5):
        for location in (1, 2):
            truth = np.full(len(TIMES), 10. + (10. * (design - 4) if location == 1 else 0.))
            rows.append({"run_number": design * 2 + location, "design_id": design,
                         "location_id": location, "impact_xy": [location * 20., -location * 10.],
                         "ground_truth": truth, "baseline": np.full(len(TIMES), 15. if location == 1 else 10.),
                         "corrected": truth.copy()})
    return rows


def test_exact_correction_restores_differences_and_absolute_accuracy_on_same_cohort(tmp_path):
    metrics = export_correction_evaluation(tmp_path, records(), TIMES, make_plots=False)
    before, after = metrics["baseline"], metrics["corrected"]
    assert before["acceleration"]["mse"] == pytest.approx(12.5)
    assert before["acceleration"]["rmse"] == pytest.approx(np.sqrt(12.5))
    assert before["acceleration"]["mae"] == pytest.approx(2.5)
    assert after["acceleration"] == {"mse": 0., "rmse": 0., "mae": 0., "r2": 1.}
    assert after["hic15"] == {"mse": 0., "rmse": 0., "mae": 0., "r2": 1.}
    assert before["design_sensitivity"]["design_difference_skill"] == 0
    assert before["design_sensitivity"]["difference_amplitude_ratio"] == 0
    assert before["design_sensitivity"]["difference_alignment_cosine"] is None
    for section in (after["design_sensitivity"], after["design_sensitivity"]["hic15"]):
        assert section["design_difference_skill"] == pytest.approx(1)
        assert section["difference_amplitude_ratio"] == pytest.approx(1)
        assert section["difference_alignment_cosine"] == pytest.approx(1)
    assert metrics["cohort"]["run_numbers"] == [9, 10, 11, 12]
    assert metrics["cohort"]["samples_per_design"] == 2
    assert metrics["comparison"]["design_sensitivity"]["design_difference_skill"] == 1
    assert metrics["comparison"]["design_sensitivity"]["amplitude_ratio_distance_to_one_change"] == -1
    assert metrics["comparison"]["design_sensitivity"]["difference_alignment_cosine"] is None
    assert metrics["comparison"]["acceleration"]["mse"] < 0
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    json.dumps(metrics, allow_nan=False)
    assert not list(tmp_path.glob("*.png"))


def test_reversed_designs_cannot_pass_by_merely_restoring_amplitude(tmp_path):
    rows = records()
    for row in rows:
        if row["location_id"] == 1:
            row["corrected"] = np.full(len(TIMES), 20. if row["design_id"] == 4 else 10.)
    report = export_correction_evaluation(tmp_path, rows, TIMES, make_plots=False)
    sensitivity = report["corrected"]["design_sensitivity"]
    assert sensitivity["difference_amplitude_ratio"] == pytest.approx(1)
    assert sensitivity["design_difference_skill"] == pytest.approx(-3)
    assert sensitivity["difference_alignment_cosine"] == pytest.approx(-1)
    assert report["corrected"]["acceleration"]["rmse"] > report["baseline"]["acceleration"]["rmse"]


def test_hic_uses_seconds_supplied_grid_and_each_curve_before_difference(tmp_path):
    result = export_correction_evaluation(tmp_path, records(), TIMES, make_plots=False)
    frame = pd.read_csv(tmp_path / "test_hic_per_curve.csv")
    np.testing.assert_allclose(frame.hic15_true_sampled_grid, .015 * np.array([10., 10., 20., 10.]) ** 2.5)
    np.testing.assert_allclose(frame.hic15_baseline_sampled_grid, .015 * np.array([15., 10., 15., 10.]) ** 2.5)
    assert "sampled" in result["hic15_source"]
    assert result["cohort"]["prediction_times_seconds"] == TIMES.tolist()


def test_saved_histories_are_existing_evaluator_compatible_and_identical_cohorts(tmp_path):
    report = export_correction_evaluation(tmp_path, list(reversed(records())), TIMES, make_plots=False)
    (tmp_path / "config.json").write_text(json.dumps({"data": {"samples_per_design": 2}}))
    for name, filename in (("baseline", "baseline_test_acceleration_histories.csv"),
                           ("corrected", "test_acceleration_histories.csv")):
        saved = evaluate_saved_run(tmp_path, history_name=filename)
        for key in ("design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine", "same_impact_difference_rmse_g"):
            expected = report[name]["design_sensitivity"][key]
            assert saved[key] is None if expected is None else saved[key] == pytest.approx(expected)
        assert saved["evaluated_run_numbers"] == report["cohort"]["run_numbers"]
        assert saved["source_sha256"] == report[name]["design_sensitivity"]["source_sha256"]
    baseline = pd.read_csv(tmp_path / "baseline_test_acceleration_histories.csv")
    corrected = pd.read_csv(tmp_path / "test_acceleration_histories.csv")
    for key in ("run_number", "time", "acceleration_true_g", "impact_x_mm", "impact_y_mm"):
        np.testing.assert_array_equal(baseline[key], corrected[key])
    assert len(corrected) == 4 * len(TIMES)


def test_zero_variance_and_single_design_have_null_normalized_scores(tmp_path):
    row = records()[1]
    result = export_correction_evaluation(tmp_path, [row], TIMES, make_plots=False, samples_per_design=2)
    assert result["corrected"]["acceleration"]["r2"] is None
    assert result["corrected"]["hic15"]["r2"] is None
    assert result["corrected"]["design_sensitivity"]["num_matched_locations"] == 0
    assert result["corrected"]["design_sensitivity"]["difference_amplitude_ratio"] is None
    rows = [records()[1], records()[3]]
    rows[1]["corrected"] = np.full(len(TIMES), 20.)
    result = export_correction_evaluation(tmp_path, rows, TIMES, make_plots=False, samples_per_design=2)
    assert result["corrected"]["design_sensitivity"]["same_impact_difference_rmse_g"] == 10
    assert result["corrected"]["design_sensitivity"]["design_difference_skill"] is None
    assert result["corrected"]["design_sensitivity"]["difference_alignment_cosine"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("problem", ["empty", "duplicate", "mapping", "xy", "length", "nan", "bool_id", "different_count"])
def test_invalid_or_misaligned_records_fail_before_export(tmp_path, problem):
    rows = records()
    if problem == "empty":
        rows = []
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "mapping":
        rows[0]["run_number"] += 1
    elif problem == "xy":
        rows[2]["impact_xy"][0] += .01
    elif problem == "length":
        rows[0]["corrected"] = rows[0]["corrected"][:-1]
    elif problem == "nan":
        rows[0]["baseline"][0] = np.nan
    elif problem == "bool_id":
        rows[0]["design_id"] = True
    elif problem == "different_count":
        rows[0]["run_number"] = 4 * 3 + 1
    with pytest.raises(ValueError):
        export_correction_evaluation(tmp_path / "result", rows, TIMES, make_plots=False)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("times", [[], [.0], [.0, .001, .001, .003], [.0, .003, .002, .004], [.0, np.nan, .002, .003], [[.0, .001, .002, .003]]])
def test_invalid_time_grids_are_rejected(tmp_path, times):
    with pytest.raises(ValueError, match="prediction_times"):
        export_correction_evaluation(tmp_path, records(), times, make_plots=False)


def test_non_test_prefix_does_not_overwrite_final_test_metrics(tmp_path):
    result = export_correction_evaluation(tmp_path, records(), TIMES, make_plots=False)
    before = (tmp_path / "metrics.json").read_bytes()
    export_correction_evaluation(tmp_path, [records()[0]], TIMES, prefix="validation", make_plots=False)
    assert (tmp_path / "metrics.json").read_bytes() == before
    assert (tmp_path / "validation_metrics.json").is_file()
    assert json.loads(before) == result
    with pytest.raises(ValueError, match="prefix"):
        export_correction_evaluation(tmp_path, records(), TIMES, prefix="../escape")


def test_plots_cover_each_location_and_do_not_leak_figures(tmp_path):
    figures = plt.get_fignums()
    export_correction_evaluation(tmp_path, records(), TIMES, make_plots=True)
    expected = ["test_hic_pred_vs_gt.png", "test_design_sensitivity.png", "baseline_test_design_sensitivity.png",
                "test_acceleration_plots/location_001.png", "test_acceleration_plots/location_002.png"]
    for name in expected:
        pixels = mpimg.imread(tmp_path / name)
        assert min(pixels.shape[:2]) > 100
        assert np.std(pixels) > .01
    assert plt.get_fignums() == figures


def test_zero_spread_matched_location_can_be_plotted(tmp_path):
    rows = [records()[1], records()[3]]
    result = export_correction_evaluation(tmp_path, rows, TIMES, make_plots=True)
    assert result["corrected"]["design_sensitivity"]["design_difference_skill"] is None
    assert (tmp_path / "test_design_sensitivity.png").is_file()
