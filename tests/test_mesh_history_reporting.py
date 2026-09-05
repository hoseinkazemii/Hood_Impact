"""Regression checks for local training reports and explicit online tracking."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import mesh_impact_history_reporting as reporting
import train_mesh_impact_history as training


class TrainingHistoryReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mesh_history_reporting_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.history = {
            "train_losses": [0.8, 0.4, 0.2],
            "val_losses": [0.9, 0.5, 0.6],
            "best_val_loss": 0.5,
        }

    def test_plot_is_readable_png_for_single_or_multiple_epochs_without_figure_leaks(self):
        for length in (1, 3):
            history = {key: values[:length] for key, values in self.history.items() if isinstance(values, list)}
            figures_before = plt.get_fignums()
            with self.subTest(epochs=length):
                path = reporting.export_training_history_plot(history, self.root)
                self.assertEqual(path, self.root / "training_history.png")
                pixels = mpimg.imread(path)
                self.assertGreater(pixels.shape[0], 100)
                self.assertGreater(pixels.shape[1], 100)
                self.assertGreater(float(np.std(pixels)), 0.01)
                self.assertEqual(plt.get_fignums(), figures_before)

    def test_plot_rejects_empty_mismatched_or_nonfinite_histories(self):
        cases = [
            {"train_losses": [], "val_losses": []},
            {"train_losses": [0.5], "val_losses": [0.6, 0.4]},
            {"train_losses": [float("nan")], "val_losses": [0.6]},
            {"train_losses": [0.5], "val_losses": [float("inf")]},
        ]
        for history in cases:
            with self.subTest(history=history), self.assertRaises(ValueError):
                reporting.export_training_history_plot(history, self.root)

    def test_load_json_preserves_losses_and_saved_best_value(self):
        (self.root / "training_history.json").write_text(json.dumps(self.history), encoding="utf-8")
        loaded = reporting.load_history(self.root)
        self.assertEqual(loaded["train_losses"], self.history["train_losses"])
        self.assertEqual(loaded["val_losses"], self.history["val_losses"])
        self.assertEqual(loaded["best_val_loss"], self.history["best_val_loss"])

    def test_load_csv_when_json_is_absent(self):
        (self.root / "training_history.csv").write_text(
            "epoch,train_mse_normalized,validation_mse_normalized\n"
            "1,0.8,0.9\n2,0.4,0.5\n3,0.2,0.6\n", encoding="utf-8",
        )
        loaded = reporting.load_history(self.root)
        self.assertEqual(loaded["train_losses"], self.history["train_losses"])
        self.assertEqual(loaded["val_losses"], self.history["val_losses"])

    def test_missing_history_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            reporting.load_history(self.root)

    def test_cli_exports_existing_run_without_uploading_by_default(self):
        (self.root / "training_history.json").write_text(json.dumps(self.history), encoding="utf-8")
        with mock.patch.object(reporting, "upload_saved_run") as upload:
            reporting.main([str(self.root)])
        upload.assert_not_called()
        self.assertTrue((self.root / "training_history.png").is_file())


class SavedRunRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mesh_history_recovery_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.history = {"train_losses": [0.8, 0.4, 0.2], "val_losses": [0.9, 0.5, 0.6]}
        self.config = {
            "architecture": {"name": "MeshImpactHistoryNet"},
            "wandb": {"mode": "disabled", "project": "original-project", "run_name": "saved-run"},
        }
        for name, payload in (
            ("training_history.json", self.history), ("config.json", self.config),
            ("metrics.json", {"mae": 3.4, "rmse": 5.6, "r2": None}),
        ):
            (self.root / name).write_text(json.dumps(payload), encoding="utf-8")
        self.run = mock.Mock()
        self.run.id = "recovered123"
        self.run.project = "original-project"
        self.run.entity = "actual-team"
        self.run.url = "https://wandb.ai/actual-team/original-project/runs/recovered123"
        self.run.summary = {}

    def test_recovery_uploads_saved_losses_and_metrics_with_provenance(self):
        import wandb

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            wandb, "init", return_value=self.run,
        ) as initialize, mock.patch.object(wandb, "Image", return_value="mock-history-image"):
            receipt = reporting.upload_saved_run(self.root)
            kwargs = initialize.call_args.kwargs
            self.assertEqual(kwargs["mode"], "online")
            self.assertEqual(kwargs["project"], "original-project")
            self.assertEqual(kwargs["job_type"], "historical_backfill")
            self.assertEqual(kwargs["config"]["recovery"]["original_wandb_mode"], "disabled")
            logged = [call.args[0] for call in self.run.log.call_args_list]
            for epoch, row in enumerate(logged[:3], start=1):
                self.assertEqual(row, {
                    "epoch": epoch, "train/loss": self.history["train_losses"][epoch - 1],
                    "val/loss": self.history["val_losses"][epoch - 1],
                })
            self.assertEqual(logged[-1]["test/mae"], 3.4)
            self.assertEqual(logged[-1]["test/rmse"], 5.6)
            self.assertNotIn("test/r2", logged[-1])
            self.assertEqual(logged[-1]["training/history"], "mock-history-image")
            self.assertEqual(self.run.summary["best_epoch"], 2)
            self.assertTrue(self.run.summary["recovered_from_saved_results"])
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["url"], self.run.url)
            self.assertEqual(json.loads((self.root / "wandb_backfill.json").read_text(encoding="utf-8")), receipt)
            self.assertEqual(json.loads((self.root / "config.json").read_text(encoding="utf-8")), self.config)
            self.run.finish.assert_called_once()

            # Re-running recovery must not create a second online run.
            with self.assertRaisesRegex(ValueError, "already"):
                reporting.upload_saved_run(self.root)
            initialize.assert_called_once()

    def test_failed_upload_retains_incomplete_receipt_and_closes_run(self):
        import wandb

        self.run.log.side_effect = RuntimeError("upload failed")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(wandb, "init", return_value=self.run):
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                reporting.upload_saved_run(self.root)
        receipt = json.loads((self.root / "wandb_backfill.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "uploading")
        self.assertEqual(receipt["id"], self.run.id)
        self.run.finish.assert_called_once()


class WandbTrackingTests(unittest.TestCase):
    def test_python_defaults_enable_online_tracking(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = training.parse_args([])
        self.assertEqual(args.wandb_mode, "online")
        self.assertEqual(args.wandb_project, "hood-impact-mesh-attention")
        self.assertIsNone(args.wandb_entity)

    def test_environment_settings_and_explicit_cli_overrides(self):
        environment = {"WANDB_MODE": "offline", "WANDB_PROJECT": "env-project", "WANDB_ENTITY": "env-team"}
        with mock.patch.dict(os.environ, environment, clear=True):
            defaults = training.parse_args([])
            overrides = training.parse_args([
                "--wandb-mode", "online", "--wandb-project", "cli-project", "--wandb-entity", "cli-team",
            ])
        self.assertEqual((defaults.wandb_mode, defaults.wandb_project, defaults.wandb_entity),
                         ("offline", "env-project", "env-team"))
        self.assertEqual((overrides.wandb_mode, overrides.wandb_project, overrides.wandb_entity),
                         ("online", "cli-project", "cli-team"))

    def test_invalid_environment_mode_can_be_explicitly_overridden(self):
        with mock.patch.dict(os.environ, {"WANDB_MODE": "typo"}, clear=True):
            with self.assertRaises(SystemExit):
                training.parse_args([])
            self.assertEqual(training.parse_args(["--wandb-mode", "disabled"]).wandb_mode, "disabled")

    @staticmethod
    def make_args(mode="online"):
        return SimpleNamespace(wandb_mode=mode, wandb_project="test-project", wandb_entity="test-team", run_name=None)

    @staticmethod
    def make_run(mode="online"):
        run = mock.Mock()
        run.id = "abc123"
        run.name = "training-run"
        run.project = "test-project"
        run.entity = "test-team"
        run.settings = SimpleNamespace(mode=mode)
        run.url = "https://wandb.ai/test-team/test-project/runs/abc123"
        return run

    def test_online_initialization_records_destination_and_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args()
            initial_config = {"architecture": {"name": "MeshImpactHistoryNet"}}
            run = self.make_run()
            logger = mock.Mock()
            with mock.patch.object(training.wandb, "init", return_value=run) as initialize:
                actual = training.initialize_wandb(args, initial_config, root, logger)
            self.assertIs(actual, run)
            initialize.assert_called_once_with(
                project="test-project", entity="test-team", name=root.name,
                config=initial_config, dir=str(root), mode="online",
            )
            run.define_metric.assert_any_call("epoch")
            for metric in ("train/loss", "val/loss", "val/mae_g", "lr"):
                run.define_metric.assert_any_call(metric, step_metric="epoch")
            metadata = json.loads((root / "wandb_run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["id"], run.id)
            self.assertEqual(metadata["url"], run.url)
            self.assertEqual(metadata["mode"], "online")
            self.assertEqual(metadata["project"], run.project)
            self.assertEqual(metadata["entity"], run.entity)
            run.finish.assert_not_called()

    def test_local_modes_are_recorded_without_claiming_an_online_url(self):
        for mode in ("offline", "disabled"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self.make_run(mode)
                with mock.patch.object(training.wandb, "init", return_value=run):
                    training.initialize_wandb(self.make_args(mode), {}, root, mock.Mock())
                metadata = json.loads((root / "wandb_run.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["mode"], mode)
                self.assertIsNone(metadata["url"])

    def test_failed_online_initialization_surfaces_error_without_local_fallback(self):
        error = RuntimeError("authentication failed")
        with tempfile.TemporaryDirectory() as directory:
            logger = mock.Mock()
            with mock.patch.object(training.wandb, "init", side_effect=error) as initialize:
                with self.assertRaisesRegex(RuntimeError, "W&B initialization failed") as raised:
                    training.initialize_wandb(self.make_args(), {}, Path(directory), logger)
            self.assertIs(raised.exception.__cause__, error)
            initialize.assert_called_once()
            logger.error.assert_called_once()

    def test_mode_mismatch_fails_and_closes_unexpected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run("offline")
            with mock.patch.object(training.wandb, "init", return_value=run):
                with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                    training.initialize_wandb(self.make_args("online"), {}, Path(directory), mock.Mock())
            run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
