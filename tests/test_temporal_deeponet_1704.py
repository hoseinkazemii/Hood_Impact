"""The January checkpoint, 1704 training contract, and exported predictions."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import torch

import train_temporal_deeponet_1704 as training
from temporal_deeponet import ACCELERATION_BEST_ARCH, LegacyHoodImpactNeuralOperator


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "runs/acceleration_history_target_value/20260116_160228_best/hood_impact_best_model.pt"


class TemporalDeepONetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        return LegacyHoodImpactNeuralOperator(SimpleNamespace(
            **ACCELERATION_BEST_ARCH, prediction_target="acceleration"))

    def test_exact_architecture_has_single_film_and_all_parameters_train(self):
        model = self.make_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1376513)
        self.assertTrue(hasattr(model, "film_layer"))
        self.assertFalse(hasattr(model, "film_layer2"))
        mesh = torch.randn(11, 3)
        output = model(mesh, torch.tensor([0] * 5 + [1] * 6), torch.randn(2, 2),
                       torch.linspace(-1, 1, 12), torch.tensor([0] * 7 + [1] * 5), batch_size=2)
        self.assertEqual(output.shape, (12,))
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    @unittest.skipUnless(REFERENCE.is_file(), "Original January artifact is local and not stored in Git")
    def test_original_best_checkpoint_loads_strictly(self):
        model = self.make_model()
        checkpoint = torch.load(REFERENCE, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(len(model.state_dict()), 85)

    def test_settings_are_explicit_acceleration_comparison_and_do_not_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "not-created"
            args = training.parse_args(["--output-dir", str(destination)])
            values = training.config_values(args)
            self.assertFalse(destination.exists())
            self.assertEqual(values["prediction_target"], "acceleration")
            self.assertEqual(values["num_samples"], 1704)
            self.assertEqual(values["samples_per_design"], 142)
            self.assertEqual(values["time_subsample_stride"], 16)
            self.assertEqual(values["trunk_hidden_dims"], [64, 128, 256])
            self.assertEqual((args.test_designs, args.val_designs), ([4, 5], [11]))

    def test_direct_training_rejects_test_clones_before_loading_dataset(self):
        with mock.patch.object(training, "validate_runtime"), mock.patch.object(training, "validate_dataset") as load:
            with self.assertRaisesRegex(ValueError, "near-clones in training"):
                training.main(["--test-designs", "4", "--val-designs", "11", "--device", "cpu"])
            load.assert_not_called()

    def test_nonfinite_settings_and_invalid_stride_are_rejected(self):
        for option, value in (("--lr", "nan"), ("--weight-decay", "-1"),
                              ("--time-subsample-stride", "0"), ("--epochs", "0")):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                training.parse_args([option, value])

    def test_complete_1704_pipeline_scalers_and_reload_without_hic_file(self):
        # Tiny geometry/history files exercise real parsing, every run boundary,
        # optimizer/checkpoints, evaluation and prediction export in one epoch.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            inp = dataset / "inp_files"
            acc = dataset / "output_history_acc"
            inp.mkdir(parents=True)
            acc.mkdir()
            raw_times = np.linspace(0, 0.025, 33, dtype=np.float32)
            meshes, impacts, accelerations = [], [], []
            for number in range(1, 1705):
                design, location = divmod(number - 1, 142)
                mesh = np.array([[0, 0, 1], [2, 0, 3], [0, 4, 2], [2, 4, 4]], dtype=np.float32)
                mesh[:, 2] += design
                impact = np.array([location / 10, design], dtype=np.float32)
                acceleration = (10 + design * 5 + location / 5 + 20 * np.sin(raw_times * 100)).astype(np.float32)
                text = "*HEADING\n*NODE\n" + "\n".join(
                    f"{index},{x},{y},{z}" for index, (x, y, z) in enumerate(mesh, 1)) + "\n*ELEMENT\n"
                (inp / f"HoodImpact_{number}.inp").write_text(text, encoding="utf-8")
                pd.DataFrame({"Time": raw_times, "A(in g)": acceleration}).to_csv(
                    acc / f"HoodImpact_{number}_SAE1000_interp1000.csv", index=False)
                meshes.append(mesh)
                impacts.append(impact)
                accelerations.append(acceleration)
            pd.DataFrame(impacts, columns=["X1", "X2"]).to_csv(dataset / "ImpactCoords_1704.csv", index=False)
            run_dir = root / "run"
            arguments = ["--data-root", str(dataset), "--output-dir", str(run_dir), "--epochs", "1",
                         "--batch-size", "256", "--device", "cpu", "--wandb-mode", "disabled"]
            training.main(arguments + ["--preflight-only"])
            self.assertFalse(run_dir.exists())
            metrics = training.main(arguments)
            self.assertEqual(set(metrics), {"mse", "rmse", "mae", "r2"})
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            saved = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(saved["architecture"]["name"], "LegacyHoodImpactNeuralOperator")
            self.assertEqual(saved["training"]["initialization"], "from_scratch")
            self.assertEqual(saved["training"]["loss"], "normalized_acceleration_mse")
            splits = json.loads((run_dir / "splits.json").read_text())
            self.assertEqual([len(splits[key]["run_numbers"]) for key in ("train", "validation", "test")],
                             [1278, 142, 284])
            self.assertEqual(splits["test"]["run_numbers"], list(range(569, 853)))
            self.assertEqual(splits["validation"]["run_numbers"], list(range(1563, 1705)))
            train_indices = np.array(splits["train"]["run_numbers"]) - 1
            scalers = joblib.load(run_dir / "scalers.joblib")
            expected_values = {
                "mesh": np.concatenate([meshes[i] for i in train_indices]),
                "indentor": np.array(impacts)[train_indices],
                "time": np.tile(raw_times[::16], len(train_indices)).reshape(-1, 1),
                "accel": np.concatenate([accelerations[i][::16] for i in train_indices]).reshape(-1, 1),
            }
            for key, expected in expected_values.items():
                np.testing.assert_allclose(scalers[key].mean_, expected.astype(np.float64).mean(axis=0), rtol=1e-6)
                np.testing.assert_allclose(scalers[key].scale_, expected.astype(np.float64).std(axis=0), rtol=1e-6)
            for artifact in (training.CHECKPOINT_NAME, training.LAST_CHECKPOINT_NAME, "metrics.json",
                             "training_history.json", "training_history.csv", "training_history.png"):
                self.assertTrue((run_dir / artifact).is_file(), artifact)
            exported = pd.read_csv(run_dir / "test_acceleration_histories.csv")
            self.assertEqual(len(exported), 284 * 3)
            predictor = training.TemporalDeepONetPredictor.from_run(run_dir)
            np.testing.assert_array_equal(predictor.time_points, raw_times[::16])
            for number in (569, 710, 711, 852):
                rows = exported[exported.run_number == number]
                prediction = predictor.predict(meshes[number - 1], impacts[number - 1])
                np.testing.assert_allclose(prediction, rows.acceleration_pred_g, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(rows.acceleration_true_g, accelerations[number - 1][::16], rtol=1e-6)
                np.testing.assert_allclose(rows.time, raw_times[::16], rtol=1e-6)
            with self.assertRaises(FileExistsError):
                training.main(arguments)


if __name__ == "__main__":
    unittest.main()
