"""Checks for the HIC15 evaluation of predicted acceleration histories."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import evaluate_mesh_impact_hic as evaluation


def naive_hic(time, acceleration, max_window=evaluation.HIC_WINDOW_S):
    """Direct transcription of the HIC definition, used as the reference."""
    integral = np.zeros(len(acceleration))
    integral[1:] = np.cumsum(0.5 * (acceleration[1:] + acceleration[:-1]) * np.diff(time))
    best = 0.0
    for start in range(len(time)):
        for end in range(start + 1, len(time)):
            window = time[end] - time[start]
            if window > max_window:
                break
            mean = (integral[end] - integral[start]) / window
            if mean <= 0.0:
                continue
            best = max(best, window * mean**2.5)
    return best


class HicFormulaTests(unittest.TestCase):
    def test_constant_acceleration_matches_the_closed_form(self):
        time = np.linspace(0.0, 0.025, 1001)
        for level in (50.0, 100.0, 250.0):
            with self.subTest(level=level):
                hic, start, end = evaluation.hic15(time, np.full_like(time, level))
                self.assertAlmostEqual(hic, 0.015 * level**2.5, delta=1e-6 * level**2.5)
                self.assertAlmostEqual(end - start, 0.015, places=6)

    def test_matches_the_direct_definition_on_pulse_shaped_histories(self):
        generator = np.random.default_rng(7)
        time = np.linspace(0.0, 0.025, 200)
        for trial in range(5):
            pulse = 200.0 * np.exp(-((time - 0.003) ** 2) / (2 * 0.0015**2))
            noisy = pulse + generator.normal(0.0, 6.0, size=len(time))
            with self.subTest(trial=trial):
                self.assertAlmostEqual(
                    evaluation.hic15(time, noisy)[0], naive_hic(time, noisy), places=6
                )

    def test_window_length_limit_is_enforced(self):
        time = np.linspace(0.0, 0.05, 501)
        acceleration = np.full_like(time, 100.0)
        hic, start, end = evaluation.hic15(time, acceleration, max_window=0.005)
        self.assertAlmostEqual(hic, 0.005 * 100.0**2.5, delta=1e-3)
        self.assertAlmostEqual(end - start, 0.005, places=6)

    def test_the_maximising_window_is_reported(self):
        time = np.linspace(0.0, 0.025, 251)
        acceleration = np.where((time >= 0.010) & (time <= 0.014), 200.0, 1.0)
        hic, start, end = evaluation.hic15(time, acceleration)
        self.assertGreater(hic, 0.0)
        self.assertLessEqual(start, 0.010)
        self.assertGreaterEqual(end, 0.014)
        self.assertLessEqual(end - start, evaluation.HIC_WINDOW_S + 1e-12)

    def test_negative_predictions_stay_finite(self):
        time = np.linspace(0.0, 0.025, 101)
        acceleration = np.full_like(time, -25.0)
        acceleration[40:60] = 120.0
        hic, _, _ = evaluation.hic15(time, acceleration)
        self.assertTrue(np.isfinite(hic))
        self.assertGreater(hic, 0.0)
        self.assertAlmostEqual(hic, naive_hic(time, acceleration), places=6)

    def test_all_negative_history_scores_zero_rather_than_failing(self):
        time = np.linspace(0.0, 0.025, 51)
        self.assertEqual(evaluation.hic15(time, np.full_like(time, -10.0))[0], 0.0)

    def test_malformed_histories_are_rejected(self):
        time = np.linspace(0.0, 0.025, 10)
        cases = [
            (time, time[:-1]),
            (time[:1], time[:1]),
            (time[::-1], np.ones(10)),
            (np.zeros(10), np.ones(10)),
            (time, np.full(10, np.nan)),
        ]
        for bad_time, bad_acceleration in cases:
            with self.subTest(time=bad_time.shape), self.assertRaises(ValueError):
                evaluation.hic15(bad_time, bad_acceleration)


class RunEvaluationTests(unittest.TestCase):
    SAMPLES_PER_DESIGN = 4

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hic_evaluation_")
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.time = np.linspace(0.0, 0.025, 40)
        self.runs = (9, 10, 11, 12)
        rows = []
        for offset, run_number in enumerate(self.runs):
            truth = (150.0 + 20.0 * offset) * np.exp(
                -((self.time - 0.003) ** 2) / (2 * 0.0018**2)
            )
            prediction = truth * 1.02
            rows.append(
                pd.DataFrame({
                    "run_number": run_number,
                    "time": self.time,
                    "acceleration_true_g": truth,
                    "acceleration_pred_g": prediction,
                })
            )
        pd.concat(rows).to_csv(self.run_dir / "test_acceleration_histories.csv", index=False)
        (self.run_dir / "config.json").write_text(
            json.dumps({
                "data": {"samples_per_design": self.SAMPLES_PER_DESIGN},
                "preprocessing": {"time_subsample_stride": 16},
            }),
            encoding="utf-8",
        )

    def test_case_identity_and_metrics_are_derived_from_the_saved_histories(self):
        results, report = evaluation.evaluate_run(self.run_dir)
        self.assertEqual(list(results["run_number"]), list(self.runs))
        # Runs 9-12 are the third block of four, so design 2, locations 1-4.
        self.assertEqual(list(results["design_id"]), [2, 2, 2, 2])
        self.assertEqual(list(results["location_id"]), [1, 2, 3, 4])
        self.assertEqual(report["num_test_impacts"], 4)
        self.assertEqual(report["test_design_ids"], [2])
        self.assertEqual(report["time_grid"]["num_points"], len(self.time))
        self.assertIsNone(report["dataset_dir"])
        # A uniform 2% acceleration gain scales HIC by 1.02**2.5.
        expected = results["hic_ground_truth"] * 1.02**2.5
        np.testing.assert_allclose(results["hic_prediction"], expected, rtol=1e-9)
        self.assertAlmostEqual(
            float(results["hic_relative_error_pct"].iloc[0]),
            100.0 * (1.02**2.5 - 1.0),
            places=6,
        )
        self.assertTrue((results["hic_error"] > 0).all())

    def test_report_covers_only_the_comparisons_the_data_supports(self):
        _, report = evaluation.evaluate_run(self.run_dir)
        self.assertEqual(len(report["hic"]), 1)
        self.assertNotIn("recorded_label_check", report)
        summary = next(iter(report["hic"].values()))
        self.assertEqual(summary["n"], 4)
        for limit in evaluation.BAND_LIMITS_PERCENT:
            self.assertIn(f"within_{limit:g}_percent", summary)
        self.assertEqual(len(report["worst_hic_cases"]), 4)
        self.assertGreaterEqual(
            abs(report["worst_hic_cases"][0]["hic_relative_error_pct"]),
            abs(report["worst_hic_cases"][-1]["hic_relative_error_pct"]),
        )

    def test_full_resolution_and_recorded_labels_are_used_when_present(self):
        dataset = self.run_dir / "dataset"
        (dataset / "output_history_acc").mkdir(parents=True)
        dense_time = np.linspace(0.0, 0.025, 400)
        recorded = {}
        for run_number in self.runs:
            offset = self.runs.index(run_number)
            dense = (150.0 + 20.0 * offset) * np.exp(
                -((dense_time - 0.003) ** 2) / (2 * 0.0018**2)
            )
            pd.DataFrame({"Time": dense_time, "A(in g)": dense}).to_csv(
                dataset / "output_history_acc" / f"HoodImpact_{run_number}_SAE1000.csv",
                index=False,
            )
            recorded[run_number] = evaluation.hic15(dense_time, dense)[0]
        pd.DataFrame({
            "run": list(recorded), "hic15": [recorded[run] for run in recorded]
        }).to_csv(dataset / "generation_metrics.csv", index=False)
        pd.DataFrame({
            "loc": [1, 2, 3, 4], "X1": [10.0, 20.0, 30.0, 40.0],
            "X2": [-5.0, -6.0, -7.0, -8.0],
        }).to_csv(dataset / "impact_locations_4.csv", index=False)

        results, report = evaluation.evaluate_run(self.run_dir, dataset)
        self.assertIn("hic_full_resolution", results)
        np.testing.assert_allclose(
            results["hic_full_resolution"], results["hic_solver_reference"], rtol=1e-9
        )
        self.assertEqual(list(results["impact_x_mm"]), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(len(report["hic"]), 4)
        self.assertEqual(report["recorded_label_check"]["labels_matching_correct_hic"], 4)
        self.assertEqual(report["recorded_label_check"]["labels_under_reporting"], 0)
        # The coarse saved grid under-integrates the pulse, so it under-reports HIC.
        self.assertTrue((results["sampling_relative_error_pct"] < 0).all())

    def test_missing_columns_are_reported_clearly(self):
        pd.DataFrame({"run_number": [1], "time": [0.0]}).to_csv(
            self.run_dir / "test_acceleration_histories.csv", index=False
        )
        with self.assertRaises(ValueError) as raised:
            evaluation.evaluate_run(self.run_dir)
        self.assertIn("acceleration_true_g", str(raised.exception))

    def test_command_writes_results_metrics_and_parity_plot(self):
        exit_code = evaluation.main([str(self.run_dir), "--no-history-plots"])
        self.assertEqual(exit_code, 0)
        results = pd.read_csv(self.run_dir / "hic_results_pred_vs_gt.csv")
        self.assertEqual(len(results), 4)
        report = json.loads((self.run_dir / "hic_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["num_test_impacts"], 4)
        for name in ("hic_pred_vs_gt_10pct.png", "hic_error_distribution.png"):
            self.assertGreater((self.run_dir / name).stat().st_size, 10_000)


class SummaryStatisticTests(unittest.TestCase):
    def test_r2_is_one_for_an_exact_prediction_and_zero_for_the_mean(self):
        truth = np.array([700.0, 1100.0, 1500.0, 1900.0])
        self.assertAlmostEqual(evaluation.r2_score(truth, truth), 1.0)
        self.assertAlmostEqual(evaluation.r2_score(truth, np.full(4, truth.mean())), 0.0)

    def test_error_summary_reports_bias_bands_and_sample_count(self):
        truth = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        prediction = np.array([1030.0, 1080.0, 1120.0, 1200.0])
        summary = evaluation.error_summary(truth, prediction, "test")
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["bias"], 107.5)
        self.assertAlmostEqual(summary["mape_percent"], 10.75)
        self.assertAlmostEqual(summary["within_5_percent"], 25.0)
        self.assertAlmostEqual(summary["within_10_percent"], 50.0)
        # The bands are inclusive, so the +20.0% case counts as within 20%.
        self.assertAlmostEqual(summary["within_15_percent"], 75.0)
        self.assertAlmostEqual(summary["within_20_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
