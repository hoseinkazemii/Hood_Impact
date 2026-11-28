"""The training-history plot and explicit online W&B tracking of the trainer."""

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

import train_mesh_impact_history as training


class TrainingHistoryPlotTests(unittest.TestCase):
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
                path = training.export_training_history_plot(history, self.root)
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
                training.export_training_history_plot(history, self.root)


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
            initialize.assert_called_once()
            kwargs = initialize.call_args.kwargs
            self.assertEqual({key: value for key, value in kwargs.items() if key != "settings"}, dict(
                project="test-project", entity="test-team", name=root.name,
                config=initial_config, dir=str(root), mode="online",
            ))
            self.assertEqual(kwargs["settings"].init_timeout, training.WANDB_INIT_TIMEOUT_SECONDS)
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

    def test_failed_online_initialization_retries_then_surfaces_error_without_local_fallback(self):
        error = RuntimeError("authentication failed")
        with tempfile.TemporaryDirectory() as directory:
            logger = mock.Mock()
            with mock.patch.object(training.wandb, "init", side_effect=error) as initialize, \
                    mock.patch.object(training.time, "sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "W&B initialization failed") as raised:
                    training.initialize_wandb(self.make_args(), {}, Path(directory), logger)
            self.assertIs(raised.exception.__cause__, error)
            self.assertEqual(initialize.call_count, training.WANDB_INIT_ATTEMPTS)
            self.assertTrue(all(call.kwargs["mode"] == "online" for call in initialize.call_args_list))
            self.assertEqual(sleep.call_count, training.WANDB_INIT_ATTEMPTS - 1)
            logger.error.assert_called_once()

    def test_slow_online_start_recovers_on_retry(self):
        # Job 3317059: the first contact with api.wandb.ai timed out.
        run = self.make_run()
        with tempfile.TemporaryDirectory() as directory:
            logger = mock.Mock()
            with mock.patch.object(training.wandb, "init", side_effect=[TimeoutError("timed out"), run]), \
                    mock.patch.object(training.time, "sleep"):
                self.assertIs(training.initialize_wandb(self.make_args(), {}, Path(directory), logger), run)
            logger.warning.assert_called_once()
            logger.error.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / "wandb_run.json").read_text())["mode"], "online")

    def test_local_modes_do_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(training.wandb, "init", side_effect=RuntimeError("disk")) as initialize, \
                    mock.patch.object(training.time, "sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    training.initialize_wandb(self.make_args("offline"), {}, Path(directory), mock.Mock())
            initialize.assert_called_once()
            sleep.assert_not_called()

    def test_mode_mismatch_fails_and_closes_unexpected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run("offline")
            with mock.patch.object(training.wandb, "init", return_value=run):
                with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                    training.initialize_wandb(self.make_args("online"), {}, Path(directory), mock.Mock())
            run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
