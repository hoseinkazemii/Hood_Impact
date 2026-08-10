import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import hic_direct_surrogate_v5 as surrogate


def brute_force_hic(time, acceleration, max_window):
    integral = np.zeros(len(time), dtype=np.float64)
    integral[1:] = np.cumsum(
        0.5 * (acceleration[1:] + acceleration[:-1]) * np.diff(time)
    )
    best = 0.0
    for start in range(len(time) - 1):
        for end in range(start + 1, len(time)):
            duration = time[end] - time[start]
            if duration > max_window:
                break
            average = (integral[end] - integral[start]) / duration
            best = max(best, duration * max(average, 0.0) ** 2.5)
    return best


class HicCalculationTests(unittest.TestCase):
    def test_constant_acceleration_uses_full_15ms_window(self):
        time = np.linspace(0.0, 0.020, 201)
        acceleration = np.full_like(time, 10.0)
        expected = 0.015 * 10.0**2.5
        actual = surrogate.hic15_numpy(time, acceleration, 0.015)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_all_window_search_matches_brute_force_for_short_pulse(self):
        time = np.linspace(0.0, 0.025, 101)
        acceleration = np.full_like(time, 5.0)
        acceleration[(time >= 0.006) & (time <= 0.010)] = 120.0
        expected = brute_force_hic(time, acceleration, 0.015)
        actual = surrogate.hic15_numpy(time, acceleration, 0.015)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_invalid_time_grid_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            surrogate.hic15_numpy(
                np.asarray([0.0, 0.001, 0.001]),
                np.asarray([1.0, 2.0, 3.0]),
            )

    def test_batch_matches_numpy_on_irregular_grids(self):
        times = [
            np.asarray([0.0, 0.001, 0.0025, 0.004, 0.008, 0.014, 0.019]),
            np.asarray([0.0, 0.0008, 0.0018, 0.003, 0.007, 0.013, 0.020]),
        ]
        accelerations = [
            np.asarray([5.0, 90.0, 120.0, 30.0, 8.0, 5.0, 5.0]),
            np.asarray([3.0, 4.0, 70.0, 150.0, 60.0, 4.0, 3.0]),
        ]
        expected = np.asarray(
            [surrogate.hic15_numpy(t, a, 0.015) for t, a in zip(times, accelerations)]
        )
        np.testing.assert_allclose(
            surrogate.hic15_batch(times, accelerations, 0.015),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_invalid_max_window_fails_for_scalar_and_batch_paths(self):
        time = np.asarray([0.0, 0.001])
        acceleration = np.asarray([1.0, 1.0])
        for invalid in (0.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    surrogate.hic15_numpy(time, acceleration, invalid)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    surrogate.hic15_batch([time], [acceleration], invalid)


class ManifestAndMeshTests(unittest.TestCase):
    def test_manifest_design_location_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            for design in range(2):
                for location in range(1, 3):
                    run = design * 2 + location
                    manifest_rows.append(
                        {
                            "run": run,
                            "design": design,
                            "loc": location,
                            "X1": 10.0 * location,
                            "X2": -5.0 * location,
                            "X3": 100.0,
                            "dz_mm": 0.0,
                            "center_gap_mm": 2.0,
                            "sphere_pen_mm": 0.0,
                            "lift_mm": 0.0,
                            "surf_z_mm": 15.0,
                        }
                    )
            pd.DataFrame(manifest_rows).to_csv(root / "manifest.csv", index=False)
            pd.DataFrame(
                {
                    "loc": [1, 2],
                    "X1": [10.0, 20.0],
                    "X2": [-5.0, -10.0],
                    "wadh_mm": [500.0, 1000.0],
                    "col_k": [1, 2],
                }
            ).to_csv(root / "grid.csv", index=False)

            loaded = surrogate.load_manifest(
                root / "manifest.csv",
                root / "grid.csv",
                expected_designs=2,
                expected_locations=2,
            )
            self.assertEqual(loaded["run"].tolist(), [1, 2, 3, 4])
            self.assertEqual(loaded["design"].tolist(), [0, 0, 1, 1])
            self.assertEqual(loaded["loc"].tolist(), [1, 2, 1, 2])

            shifted_grid = pd.DataFrame(
                {
                    "loc": [2, 3],
                    "X1": [10.0, 20.0],
                    "X2": [-5.0, -10.0],
                    "wadh_mm": [500.0, 1000.0],
                    "col_k": [1, 2],
                }
            )
            shifted_grid.to_csv(root / "grid.csv", index=False)
            with self.assertRaisesRegex(ValueError, "exactly 1..2"):
                surrogate.load_manifest(
                    root / "manifest.csv",
                    root / "grid.csv",
                    expected_designs=2,
                    expected_locations=2,
                )

            shifted_grid["loc"] = [1, 2]
            shifted_grid.loc[0, "wadh_mm"] = np.nan
            shifted_grid.to_csv(root / "grid.csv", index=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                surrogate.load_manifest(
                    root / "manifest.csv",
                    root / "grid.csv",
                    expected_designs=2,
                    expected_locations=2,
                )

    def test_mesh_parser_removes_highest_id_impactor_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.inp"
            path.write_text(
                "\n".join(
                    [
                        "*HEADING",
                        "synthetic",
                        "*NODE",
                        "1, 0.0, 0.0, 0.0",
                        "2, 1.0, 0.0, 0.0",
                        "10, 4.0, 5.0, 20.0",
                        "11, 4.0, 5.0, 30.0",
                        "*ELEMENT, TYPE=S3",
                        "1, 1, 2, 1",
                    ]
                ),
                encoding="utf-8",
            )
            node_ids, xyz = surrogate.parse_hood_mesh(
                path, np.asarray([4.0, 5.0]), n_impactor_nodes=2
            )
            np.testing.assert_array_equal(node_ids, [1, 2])
            np.testing.assert_allclose(xyz[:, 2], [0.0, 0.0])


class PredictionUtilityTests(unittest.TestCase):
    def test_split_ids_reject_numpy_aliases_and_too_few_designs(self):
        surrogate.validate_split_ids(12, 10, 11)
        for validation, test in ((-1, 11), (10, -1), (12, 11), (10, 12), (4, 4)):
            with self.subTest(validation=validation, test=test):
                with self.assertRaises(ValueError):
                    surrogate.validate_split_ids(12, validation, test)
        with self.assertRaisesRegex(ValueError, "at least four"):
            surrogate.validate_split_ids(3, 1, 2)

    def test_neighbor_selection_excludes_query_and_uses_geometry_distance(self):
        pair_features = np.zeros((3, 3, len(surrogate.pair_geometry_feature_names())))
        pair_features[2, 0, 0] = 5.0
        pair_features[2, 1, 0] = 0.5
        nearest = surrogate.nearest_designs(2, [0, 1], pair_features, n_neighbors=1)
        np.testing.assert_array_equal(nearest, [1])
        with self.assertRaisesRegex(ValueError, "must not appear"):
            surrogate.nearest_designs(2, [0, 2], pair_features, n_neighbors=1)

    def test_geometric_blend_endpoints(self):
        retrieval = np.asarray([100.0, 200.0])
        residual = np.asarray([121.0, 162.0])
        np.testing.assert_allclose(
            surrogate.geometric_blend(retrieval, residual, 0.0), retrieval
        )
        np.testing.assert_allclose(
            surrogate.geometric_blend(retrieval, residual, 1.0), residual
        )
        np.testing.assert_allclose(
            surrogate.geometric_blend(retrieval, residual, 0.5),
            np.sqrt(retrieval * residual),
        )

    def test_alpha_selection_can_reject_a_harmful_residual(self):
        target = np.asarray([100.0, 200.0, 300.0])
        retrieval = target.copy()
        residual = target + 100.0
        alpha, table = surrogate.select_blend_alpha(target, retrieval, residual)
        self.assertEqual(alpha, 0.0)
        self.assertEqual(len(table), 21)

    def test_metric_definition(self):
        target = np.asarray([1.0, 2.0, 3.0])
        score = surrogate.metrics(target, target)
        self.assertEqual(score.r2, 1.0)
        self.assertEqual(score.rmse, 0.0)
        self.assertEqual(score.mae, 0.0)
        self.assertTrue(math.isclose(score.bias, 0.0))


class RunGatingTests(unittest.TestCase):
    def test_development_run_never_requests_or_writes_test_targets(self):
        class FakeModel:
            feature_importances_ = np.asarray([1.0])

        manifest = pd.DataFrame(
            [
                {
                    "run": design * 2 + loc,
                    "design": design,
                    "loc": loc,
                    "X1": float(loc),
                    "X2": 0.0,
                    "X3": 1.0,
                }
                for design in range(4)
                for loc in (1, 2)
            ]
        )
        development_targets = manifest.loc[manifest["design"] != 3].copy()
        development_targets["hic15"] = (
            100.0 + 10.0 * development_targets["design"] + development_targets["loc"]
        )
        development_targets["n_time_points"] = 10
        development_targets["max_window_s"] = 0.015
        prediction = {
            "neighbors": np.asarray([0]),
            "distances": np.asarray([1.0]),
            "weights": np.asarray([1.0]),
            "retrieval": np.asarray([101.0, 102.0]),
            "residual": np.asarray([111.0, 112.0]),
        }
        geometry = (
            np.zeros((4, 2, 1)),
            np.zeros((4, 2, 1)),
            np.zeros((4, 4, 1)),
            ["feature"],
            "geometry-fingerprint",
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "development_run"
            args = surrogate.parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--validation-design",
                    "2",
                    "--test-design",
                    "3",
                    "--selection-mode",
                    "validation",
                    "--n-estimators",
                    "1",
                ]
            )
            with (
                mock.patch.object(surrogate, "load_manifest", return_value=manifest),
                mock.patch.object(surrogate, "preflight", return_value={"runs": 8}),
                mock.patch.object(
                    surrogate,
                    "load_or_compute_targets",
                    return_value=(development_targets, "development-fingerprint"),
                ) as target_loader,
                mock.patch.object(
                    surrogate, "load_or_compute_geometry_features", return_value=geometry
                ),
                mock.patch.object(
                    surrogate,
                    "fit_residual_model",
                    return_value=(FakeModel(), 4),
                ),
                mock.patch.object(surrogate, "predict_design", return_value=prediction),
                mock.patch.object(surrogate, "atomic_joblib_dump"),
            ):
                surrogate.run(args)

            self.assertEqual(target_loader.call_count, 1)
            requested_manifest = target_loader.call_args.args[0]
            self.assertEqual(sorted(requested_manifest["design"].unique()), [0, 1, 2])
            self.assertFalse((output / "test_predictions.csv").exists())
            self.assertFalse((output / "test_metrics.json").exists())
            exported = pd.read_csv(output / "canonical_hic15_targets.csv")
            self.assertNotIn(3, exported["design"].unique())


if __name__ == "__main__":
    unittest.main()
