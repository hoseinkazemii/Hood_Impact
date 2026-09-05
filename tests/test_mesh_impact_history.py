"""Behavioral checks for the independent mesh-to-acceleration architecture."""

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import mesh_impact_history as architecture
from mesh_impact_history import MeshImpactHistoryNet


class MeshImpactHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(38)
        self.model = MeshImpactHistoryNet(
            width=16,
            num_heads=2,
            num_latents=4,
            latent_layers=1,
            temporal_layers=1,
            dropout=0.0,
        ).eval()
        self.mesh = torch.randn(15, 3)
        self.mesh_batch = torch.tensor([0] * 6 + [1] * 9)
        self.impact = torch.tensor([[0.15, -0.4], [-0.7, 0.3]])
        self.time = torch.tensor([-1.0, -0.6, -0.1, 0.4, 0.9, -0.8, 0.0, 0.8])
        self.time_batch = torch.tensor([0] * 5 + [1] * 3)

    def forward(self, **overrides):
        arguments = dict(
            mesh=self.mesh,
            mesh_batch=self.mesh_batch,
            indentor=self.impact,
            time=self.time,
            time_batch=self.time_batch,
            batch_size=2,
        )
        arguments.update(overrides)
        return self.model(**arguments)

    def test_mesh_node_order_does_not_change_history(self):
        permutation = torch.tensor([9, 2, 12, 0, 6, 14, 4, 7, 11, 1, 10, 5, 13, 3, 8])
        with torch.no_grad():
            expected = self.forward()
            actual = self.forward(
                mesh=self.mesh[permutation], mesh_batch=self.mesh_batch[permutation]
            )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_variable_meshes_and_histories_remain_isolated_and_keep_input_order(self):
        node_order = torch.tensor([7, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14])
        time_order = torch.tensor([6, 3, 0, 5, 2, 7, 4, 1])
        with torch.no_grad():
            expected = []
            for sample in range(2):
                nodes = self.mesh[self.mesh_batch == sample]
                times = self.time[self.time_batch == sample]
                expected.append(
                    self.model(
                        nodes,
                        torch.zeros(len(nodes), dtype=torch.long),
                        self.impact[sample : sample + 1],
                        times,
                        torch.zeros(len(times), dtype=torch.long),
                        batch_size=1,
                    )
                )
            actual = self.forward(
                mesh=self.mesh[node_order],
                mesh_batch=self.mesh_batch[node_order],
                time=self.time[time_order],
                time_batch=self.time_batch[time_order],
            )
            inferred_batch_size = self.forward(batch_size=None)
        expected = torch.cat(expected)
        self.assertEqual(actual.shape, self.time.shape)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected[time_order], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(inferred_batch_size, expected, atol=2e-6, rtol=2e-5)

    def test_geometry_and_impact_both_change_predictions_and_receive_gradients(self):
        with torch.no_grad():
            baseline = self.forward()
            changed_geometry = self.mesh.clone()
            changed_geometry[:6, 2] += torch.linspace(0.25, 1.25, 6)
            changed_impact = self.impact.clone()
            changed_impact[0] += torch.tensor([0.6, -0.5])
            geometry_prediction = self.forward(mesh=changed_geometry)
            impact_prediction = self.forward(indentor=changed_impact)
        self.assertGreater((geometry_prediction[:5] - baseline[:5]).abs().max().item(), 1e-7)
        self.assertGreater((impact_prediction[:5] - baseline[:5]).abs().max().item(), 1e-7)
        torch.testing.assert_close(geometry_prediction[5:], baseline[5:], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(impact_prediction[5:], baseline[5:], atol=2e-6, rtol=2e-5)

        mesh = self.mesh.clone().requires_grad_()
        impact = self.impact.clone().requires_grad_()
        prediction = self.forward(mesh=mesh, indentor=impact)
        loss = (prediction * torch.linspace(0.5, 1.5, len(prediction))).square().mean()
        loss.backward()
        for name, inputs in (("mesh", mesh), ("impact", impact)):
            with self.subTest(input=name):
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        parameter_gradients = [
            parameter.grad for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in parameter_gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in parameter_gradients), 0.0)

    def test_relative_features_put_independently_normalized_impact_in_mesh_units(self):
        mesh_mean = np.array([100.0, -30.0, 500.0], dtype=np.float32)
        mesh_scale = np.array([10.0, 5.0, 20.0], dtype=np.float32)
        impact_mean = np.array([80.0, -20.0], dtype=np.float32)
        impact_scale = np.array([4.0, 2.0], dtype=np.float32)
        self.model.set_coordinate_scalers(mesh_mean, mesh_scale, impact_mean, impact_scale)
        mesh = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 3.0, -0.5]])
        impact = torch.tensor([2.0, -1.0])
        # Physical impact=(88,-22), hence normalized mesh-space impact=(-1.2,1.6).
        relative = mesh[:, :2] - torch.tensor([-1.2, 1.6])
        expected = torch.cat((mesh, relative, relative.square().sum(-1, keepdim=True)), dim=-1)
        actual = self.model.node_features(mesh, impact)
        self.assertEqual(actual.shape, (2, 6))
        torch.testing.assert_close(actual, expected)

    def test_cpu_autocast_forward_and_backward_are_finite(self):
        self.model.train()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = self.forward()
            loss = prediction.float().square().mean()
        self.assertEqual(prediction.shape, self.time.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

    def test_contains_no_convolution_or_legacy_model_import(self):
        self.assertFalse(any(isinstance(module, nn.modules.conv._ConvNd) for module in self.model.modules()))
        tree = ast.parse(inspect.getsource(architecture))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        for module in imports:
            self.assertFalse(
                any(name in module.lower() for name in ("deeponet", "pointnet", "multiscale_hic", "tcn")),
                msg=f"Architecture imports a previous model module: {module}",
            )

    def test_rejects_empty_meshes_and_missing_samples(self):
        cases = [
            dict(mesh=self.mesh[:0], mesh_batch=self.mesh_batch[:0]),
            dict(mesh=self.mesh[:6], mesh_batch=self.mesh_batch[:6]),
            dict(time=self.time[:0], time_batch=self.time_batch[:0]),
            dict(time=self.time[:5], time_batch=self.time_batch[:5]),
        ]
        for overrides in cases:
            with self.subTest(overrides=list(overrides)):
                with self.assertRaisesRegex(ValueError, r"\S"):
                    self.forward(**overrides)

    def test_rejects_wrong_shapes_and_invalid_memberships(self):
        cases = [
            dict(mesh=self.mesh[:, :2]),
            dict(mesh=self.mesh.unsqueeze(0)),
            dict(indentor=torch.zeros(2, 3)),
            dict(indentor=self.impact[0]),
            dict(time=self.time[:, None]),
            dict(mesh_batch=self.mesh_batch[:-1]),
            dict(time_batch=self.time_batch[:-1]),
            dict(mesh_batch=self.mesh_batch.float()),
            dict(time_batch=self.time_batch.float()),
            dict(mesh_batch=torch.full_like(self.mesh_batch, -1)),
            dict(time_batch=torch.full_like(self.time_batch, 2)),
            dict(batch_size=3),
            dict(batch_size=0),
        ]
        for overrides in cases:
            with self.subTest(overrides=list(overrides)):
                with self.assertRaisesRegex(ValueError, r"\S"):
                    self.forward(**overrides)

    def test_rejects_nonfinite_inputs(self):
        for field, original in (("mesh", self.mesh), ("indentor", self.impact), ("time", self.time)):
            for invalid in (float("nan"), float("inf")):
                value = original.clone()
                value.reshape(-1)[0] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, r"\S"):
                        self.forward(**{field: value})


class ExistingTimePreprocessingTests(unittest.TestCase):
    def test_acceleration_csv_preserves_configured_stride_and_truncation(self):
        # Import lazily so standalone architecture checks have no tracking or
        # preprocessing dependency beyond PyTorch and NumPy.
        from utils.utils import Config, DataPreprocessor

        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                data_format="legacy", prediction_target="acceleration",
                output_dir=str(Path(directory) / "run"), acceleration_dir=directory,
            )
            self.assertEqual(config.time_subsample_stride, Config.time_subsample_stride)
            stride = config.time_subsample_stride
            time = np.linspace(0.0, 0.025, 5 * stride + 3).astype(np.float32)
            acceleration = (7.0 + np.arange(len(time)) ** 2).astype(np.float32)
            pd.DataFrame({"Time": time, "A(in g)": acceleration}).to_csv(
                Path(directory) / "HoodImpactor_1_SAE1000.csv", index=False
            )
            preprocessor = DataPreprocessor(config)
            sampled_time, sampled_acceleration = preprocessor.load_acceleration_history(1)
            np.testing.assert_allclose(sampled_time, time[::stride], rtol=1e-6, atol=1e-9)
            np.testing.assert_array_equal(sampled_acceleration, acceleration[::stride])
            self.assertEqual(len(sampled_time), len(time[::stride]))

            config.time_subsample_stride = 3
            config.max_train_time = float(time[4 * stride])
            sampled_time, sampled_acceleration = preprocessor.load_acceleration_history(1)
            keep = time <= config.max_train_time
            np.testing.assert_allclose(sampled_time, time[keep][::3], rtol=1e-6, atol=1e-9)
            np.testing.assert_array_equal(sampled_acceleration, acceleration[keep][::3])


class TrainingRoundTripTests(unittest.TestCase):
    def test_train_reload_and_predict_keep_sampling_units_and_training_only_scalers(self):
        from train_mesh_impact_history import CHECKPOINT_NAME, HistoryPredictor, main
        from utils.utils import Config

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_dir = root / "meshes"
            acceleration_dir = root / "acceleration"
            mesh_dir.mkdir()
            acceleration_dir.mkdir()
            run_dir = root / "trained"
            raw_times = np.linspace(0.0, 0.025, 101, dtype=np.float32)
            meshes, impacts, accelerations, doe = [], [], [], []
            for run in range(1, 7):
                design = (run - 1) // 2
                coordinate = np.linspace(-2.0, 2.0, 6 + run, dtype=np.float32)
                mesh = np.column_stack((
                    coordinate + 15.0 * design,
                    np.sin(coordinate) + 10.0 * design,
                    3.0 + 0.2 * coordinate + 0.1 * run,
                )).astype(np.float32)
                impact = np.asarray([0.2 * run + 15.0 * design, -0.4 * run + 10.0 * design], dtype=np.float32)
                acceleration = (
                    15.0 + 3.0 * run + (10.0 + run) * np.sin(np.pi * raw_times / 0.025) ** 2
                ).astype(np.float32)
                meshes.append(mesh)
                impacts.append(impact)
                accelerations.append(acceleration)
                pd.DataFrame(mesh, columns=["X1", "X2", "X3"]).to_csv(
                    mesh_dir / f"HoodImpactor_{run}_COORD.csv", index=False
                )
                pd.DataFrame({"Time": raw_times, "A(in g)": acceleration}).to_csv(
                    acceleration_dir / f"HoodImpactor_{run}_SAE1000.csv", index=False
                )
                doe.append({"Run_Number": run, "Indentor X Position": impact[0], "Indentor Y Position": impact[1]})
            doe_path = root / "DOE.csv"
            pd.DataFrame(doe).to_csv(doe_path, index=False)

            metrics = main([
                "--data-format", "legacy", "--mesh-geometry-dir", str(mesh_dir),
                "--doe-path", str(doe_path), "--acceleration-dir", str(acceleration_dir),
                "--num-samples", "6", "--samples-per-design", "2",
                "--test-designs", "2", "--val-designs", "1", "--epochs", "1",
                "--batch-size", "2", "--width", "16", "--num-heads", "2",
                "--num-latents", "4", "--latent-layers", "1", "--temporal-layers", "1",
                "--dropout", "0", "--device", "cpu", "--wandb-mode", "disabled",
                "--output-dir", str(run_dir),
            ])
            self.assertEqual(set(metrics), {"mse", "rmse", "mae", "r2"})
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            for artifact in (
                CHECKPOINT_NAME, "config.json", "splits.json", "scalers.joblib",
                "prediction_times.npy", "training_history.json", "training_history.csv",
                "metrics.json", "test_acceleration_histories.csv",
            ):
                self.assertTrue((run_dir / artifact).is_file(), artifact)

            saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["architecture"]["name"], "MeshImpactHistoryNet")
            self.assertEqual(saved["preprocessing"]["time_subsample_stride"], Config.time_subsample_stride)
            self.assertEqual(saved["preprocessing"]["acceleration_units"], "g")
            self.assertEqual(saved["training"]["initialization"], "from_scratch")
            splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(splits["train"]["run_numbers"], [1, 2])
            self.assertEqual(splits["validation"]["run_numbers"], [3, 4])
            self.assertEqual(splits["test"]["run_numbers"], [5, 6])
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history["train_losses"]), 1)
            self.assertTrue(np.isfinite(history["best_val_loss"]))

            predictor = HistoryPredictor.from_run(run_dir)
            stride = Config.time_subsample_stride
            expected_times = raw_times[::stride]
            np.testing.assert_allclose(predictor.time_points, expected_times, rtol=1e-6, atol=1e-9)
            np.testing.assert_array_equal(predictor.time_points, np.load(run_dir / "prediction_times.npy"))
            # Held-out designs have deliberately shifted inputs and targets.
            # Check every fitted scaler against only design 0's samples.
            preprocessor = predictor.preprocessor
            for scaler, training_values in (
                (preprocessor.mesh_scaler, np.concatenate(meshes[:2]).astype(np.float64)),
                (preprocessor.indentor_scaler, np.asarray(impacts[:2], dtype=np.float64)),
                (preprocessor.time_scaler, np.tile(expected_times, 2).astype(np.float64).reshape(-1, 1)),
                (preprocessor.accel_scaler, np.concatenate([a[::stride] for a in accelerations[:2]]).astype(np.float64).reshape(-1, 1)),
            ):
                np.testing.assert_allclose(scaler.mean_, training_values.mean(axis=0), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(scaler.scale_, training_values.std(axis=0), rtol=1e-6, atol=1e-7)

            exported = pd.read_csv(run_dir / "test_acceleration_histories.csv")
            self.assertEqual(len(exported), 2 * len(expected_times))
            for run in (5, 6):
                rows = exported[exported["run_number"] == run]
                prediction_g = predictor.predict(meshes[run - 1], impacts[run - 1])
                self.assertEqual(prediction_g.shape, expected_times.shape)
                self.assertTrue(np.isfinite(prediction_g).all())
                np.testing.assert_allclose(rows["time"], expected_times, rtol=1e-6, atol=1e-9)
                np.testing.assert_allclose(rows["acceleration_true_g"], accelerations[run - 1][::stride], rtol=1e-6)
                np.testing.assert_allclose(prediction_g, rows["acceleration_pred_g"], rtol=2e-5, atol=2e-5)

            # Supplying the saved grid must not subsample it a second time.
            explicit_prediction = predictor.predict(meshes[4], impacts[4], sampled_time_points=predictor.time_points)
            np.testing.assert_allclose(explicit_prediction, predictor.predict(meshes[4], impacts[4]))
            checkpoint = torch.load(run_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
            for key, value in predictor.model.state_dict().items():
                torch.testing.assert_close(value, checkpoint["model_state_dict"][key])
            # Verify physical g units directly against the normalized model output.
            mesh_tensor = torch.from_numpy(preprocessor.transform_mesh(meshes[4]))
            impact_tensor = torch.from_numpy(preprocessor.transform_indentor(impacts[4])).unsqueeze(0)
            time_tensor = torch.from_numpy(preprocessor.transform_time(predictor.time_points))
            with torch.no_grad():
                normalized_prediction = predictor.model(
                    mesh_tensor, torch.zeros(len(mesh_tensor), dtype=torch.long), impact_tensor,
                    time_tensor, torch.zeros(len(time_tensor), dtype=torch.long), batch_size=1,
                ).numpy()
            expected_g = normalized_prediction * preprocessor.accel_scaler.scale_[0] + preprocessor.accel_scaler.mean_[0]
            np.testing.assert_allclose(explicit_prediction, expected_g, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
