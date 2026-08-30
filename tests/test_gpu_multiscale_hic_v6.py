import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import gpu_multiscale_hic_v6 as v6


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


class HicTests(unittest.TestCase):
    def test_constant_acceleration_uses_15ms_window(self):
        time = np.linspace(0.0, 0.020, 201)
        acceleration = np.full_like(time, 10.0)
        expected = 0.015 * 10.0**2.5
        self.assertAlmostEqual(v6.hic15_numpy(time, acceleration), expected, places=10)

    def test_short_pulse_matches_brute_force(self):
        time = np.linspace(0.0, 0.025, 101)
        acceleration = np.full_like(time, 4.0)
        acceleration[(time >= 0.006) & (time <= 0.010)] = 130.0
        expected = brute_force_hic(time, acceleration, 0.015)
        self.assertAlmostEqual(
            v6.hic15_numpy(time, acceleration, 0.015), expected, places=10
        )

    def test_batch_matches_scalar_and_rejects_bad_windows(self):
        time = np.asarray(
            [
                [0.0, 0.001, 0.0025, 0.006, 0.012, 0.018],
                [0.0, 0.0008, 0.003, 0.007, 0.013, 0.021],
            ]
        )
        acceleration = np.asarray(
            [[3.0, 90.0, 110.0, 20.0, 4.0, 3.0], [2.0, 4.0, 100.0, 60.0, 4.0, 2.0]]
        )
        expected = np.asarray(
            [v6.hic15_numpy(t, a, 0.015) for t, a in zip(time, acceleration)]
        )
        np.testing.assert_allclose(v6.hic15_batch(time, acceleration, 0.015), expected)
        for invalid in (0.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    v6.hic15_numpy(time[0], acceleration[0], invalid)


class GeometryTests(unittest.TestCase):
    def test_raster_is_translation_invariant_and_finite(self):
        x, y = np.meshgrid(np.linspace(-250.0, 250.0, 31), np.linspace(-250.0, 250.0, 31))
        z = 500.0 + 0.05 * x - 0.02 * y
        outer = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        lower = outer.copy()
        lower[:, 2] -= 60.0
        hood = np.vstack([outer, lower])
        connectors = lower[::2].copy()
        impact = np.asarray([20.0, -10.0, 585.0])
        maps, scalars = v6.rasterize_geometry(
            hood, outer, lower, connectors, impact
        )

        shift = np.asarray([700.0, -300.0, 40.0])
        shifted_maps, shifted_scalars = v6.rasterize_geometry(
            hood + shift,
            outer + shift,
            lower + shift,
            connectors + shift,
            impact + shift,
        )
        self.assertEqual(maps.shape, (3, len(v6.MAP_CHANNELS), 48, 48))
        self.assertEqual(scalars.shape, (len(v6.SCALAR_FEATURES),))
        self.assertTrue(np.isfinite(maps).all())
        np.testing.assert_allclose(maps, shifted_maps, atol=1e-5)
        # Absolute impact/surface coordinates shift, while gap and local surface
        # descriptors remain invariant.
        np.testing.assert_allclose(scalars[4:], shifted_scalars[4:], atol=1e-5)

    def test_aggregate_bins_handles_empty_cells(self):
        count, minimum, maximum, mean = v6._aggregate_bins(
            np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([-3.0, 5.0]), 4
        )
        self.assertEqual(count.sum(), 2)
        self.assertEqual(minimum[0, 0], -3.0)
        self.assertEqual(maximum[1, 1], 5.0)
        self.assertEqual(mean[3, 3], 0.0)
        self.assertTrue(np.isfinite(np.stack([count, minimum, maximum, mean])).all())

    def test_connector_parser_reads_only_conn3d2_endpoints(self):
        content = """*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 2, 0, 0
4, 3, 0, 0
*ELEMENT, TYPE=CONN3D2
10, 1, 2
11, 2, 3
*ELEMENT, TYPE=S4
20, 1, 2, 3, 4
"""
        with tempfile.TemporaryDirectory() as directory:
            deck = Path(directory) / "small.inp"
            deck.write_text(content, encoding="utf-8")
            np.testing.assert_array_equal(v6.connector_node_ids(deck), [1, 2, 3])


class MetadataTests(unittest.TestCase):
    @staticmethod
    def write_manifest(root: Path) -> Path:
        folder = root / v6.EURONCAP_DATASET
        folder.mkdir(parents=True)
        rows = []
        for run in range(1, v6.EXPECTED_SAMPLES + 1):
            design = (run - 1) // v6.EURONCAP_LOCATIONS
            location = (run - 1) % v6.EURONCAP_LOCATIONS + 1
            if location <= v6.ORIGINAL_EURONCAP_LOCATIONS:
                origin_dataset = "HoodImpact_600_EuroNCAP"
                origin_location = location
                origin_run = design * v6.ORIGINAL_EURONCAP_LOCATIONS + location
                subset = "selected_50"
            else:
                origin_dataset = "HoodImpact_1104_EuroNCAP_Remaining"
                origin_location = location - v6.ORIGINAL_EURONCAP_LOCATIONS
                origin_run = design * 92 + origin_location
                subset = "remaining_92"
            rows.append(
                {
                    "run": run,
                    "design": design,
                    "loc": location,
                    "X1": float(location),
                    "X2": float(-location),
                    "X3": float(500 + design),
                    "source_dataset": origin_dataset,
                    "source_run": origin_run,
                    "source_loc": origin_location,
                    "source_subset": subset,
                }
            )
        path = folder / v6.EURONCAP_MANIFEST
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_merged_manifest_boundaries_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(root)
            metadata = v6.build_metadata(root)
            self.assertEqual(len(metadata), 1704)
            for run, design, location in (
                (1, 0, 1),
                (142, 0, 142),
                (143, 1, 1),
                (1704, 11, 142),
            ):
                row = metadata.iloc[run - 1]
                self.assertEqual((row.global_run, row.design, row.location),
                                 (run, design, location))
                self.assertEqual(row.source, v6.EURONCAP_SOURCE)
            self.assertEqual(metadata.iloc[49].origin_subset, "selected_50")
            self.assertEqual(metadata.iloc[50].origin_subset, "remaining_92")

    def test_merged_manifest_rejects_wrong_design_location_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_manifest(root)
            frame = pd.read_csv(path)
            frame.loc[142, "loc"] = 2
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "run = 142"):
                v6.build_metadata(root)


class SplitAndNormalizationTests(unittest.TestCase):
    @staticmethod
    def metadata():
        return pd.DataFrame(
            [
                {
                    "design": design,
                    "source": v6.EURONCAP_SOURCE,
                    "location": location,
                }
                for design in range(12)
                for location in range(1, v6.EURONCAP_LOCATIONS + 1)
            ]
        )

    def test_design_split_is_exact_and_rejects_aliases(self):
        train, validation, test = v6.split_indices(self.metadata(), 10, 11)
        self.assertEqual((len(train), len(validation), len(test)), (1420, 142, 142))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        for validation_design, test_design in ((-1, 11), (10, -1), (12, 11), (10, 12), (5, 5)):
            with self.subTest(validation=validation_design, test=test_design):
                with self.assertRaises(ValueError):
                    v6.split_indices(self.metadata(), validation_design, test_design)

    def test_normalizer_uses_only_training_indices(self):
        rng = np.random.default_rng(4)
        metadata = pd.DataFrame(
            [
                {
                    "design": design,
                    "source": v6.EURONCAP_SOURCE,
                    "location": location,
                }
                for design in range(3)
                for location in range(1, v6.EURONCAP_LOCATIONS + 1)
            ]
        )
        samples = len(metadata)
        maps = rng.normal(
            size=(samples, 3, len(v6.MAP_CHANNELS), 4, 4)
        ).astype(np.float32)
        scalars = rng.normal(size=(samples, len(v6.SCALAR_FEATURES))).astype(np.float32)
        times = np.tile(np.linspace(0.0, 0.025, 10), (samples, 1)).astype(np.float32)
        acceleration = np.full((samples, 10), 10.0, dtype=np.float32)
        base = 1000.0 + 2.0 * metadata["location"].to_numpy()
        hic = (base + 10.0 * metadata["design"].to_numpy()).astype(np.float32)
        hic[metadata["design"].to_numpy() == 2] += 1.0e9
        data = v6.PreparedData(metadata, maps, scalars, times, acceleration, hic, "test")
        train_indices = np.flatnonzero(metadata["design"].to_numpy() < 2)
        normalizer = v6.TrainingNormalizer().fit(data, train_indices)
        train_metadata = metadata.iloc[train_indices]
        expected_baseline = base[train_indices] + 5.0
        np.testing.assert_allclose(normalizer.hic_baseline(train_metadata), expected_baseline)
        transformed = normalizer.transform_hic(hic[train_indices], train_metadata)
        self.assertAlmostEqual(float(transformed.mean()), 0.0, places=6)
        np.testing.assert_allclose(np.unique(transformed), [-1.0, 1.0], atol=1e-6)

        shuffled = train_indices[::-1]
        round_trip = normalizer.inverse_hic(
            normalizer.transform_hic(hic[shuffled], metadata.iloc[shuffled]),
            metadata.iloc[shuffled],
        )
        np.testing.assert_allclose(round_trip, hic[shuffled], atol=1e-5)
        self.assertLess(float(normalizer.euroncap_hic_mean.max()), 1300.0)

        with self.assertRaisesRegex(ValueError, "unknown HIC-baseline source"):
            normalizer.hic_baseline(pd.DataFrame([{"source": "bad", "location": 1}]))
        with self.assertRaisesRegex(ValueError, "location 143 is invalid"):
            normalizer.hic_baseline(
                pd.DataFrame([{"source": v6.EURONCAP_SOURCE, "location": 143}])
            )

class ModelTests(unittest.TestCase):
    def test_wandb_metric_flattening_preserves_namespaces(self):
        flattened = v6.flatten_numeric_metrics(
            {
                "combined": {
                    "hic_direct": {"n": 142, "r2": np.float64(0.95)},
                    "label": "ignored",
                }
            },
            "evaluation/validation",
        )
        self.assertEqual(
            flattened,
            {
                "evaluation/validation/combined/hic_direct/n": 142.0,
                "evaluation/validation/combined/hic_direct/r2": 0.95,
            },
        )

    def test_forward_backward_and_checkpoint_parity(self):
        torch.manual_seed(8)
        model = v6.MultiScaleHoodImpactNet()
        model.eval()
        maps = torch.randn(2, 3, len(v6.MAP_CHANNELS), 48, 48)
        scalars = torch.randn(2, len(v6.SCALAR_FEATURES))
        hic, waveform = model(maps, scalars)
        self.assertEqual(tuple(hic.shape), (2,))
        self.assertEqual(tuple(waveform.shape), (2, 1000))
        loss = hic.square().mean() + waveform.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            v6.atomic_torch_save({"model_state_dict": model.state_dict()}, checkpoint)
            restored = v6.MultiScaleHoodImpactNet()
            restored.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
            )
            restored.eval()
            with torch.no_grad():
                restored_hic, restored_waveform = restored(maps, scalars)
            torch.testing.assert_close(restored_hic, hic)
            torch.testing.assert_close(restored_waveform, waveform)

    def test_parameter_budget_and_schedule(self):
        parameters = v6.model_parameter_count(v6.MultiScaleHoodImpactNet())
        self.assertEqual(parameters, 1_900_514)
        self.assertTrue(math.isclose(v6.scheduler_multiplier(0, 10, 200), 0.1))
        self.assertLess(v6.scheduler_multiplier(199, 10, 200), 0.02)


if __name__ == "__main__":
    unittest.main()
