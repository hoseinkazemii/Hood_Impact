"""Chunked teacher forcing (``teacher_forcing_chunks``) in the mesh history networks."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mesh_change_metrics import time_window_metrics
from mesh_impact_history import MeshImpactHistoryNet
from utils.utils import Trainer


class TeacherForcingModelTests(unittest.TestCase):
    KWARGS = dict(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0)
    STEPS = 9  # three chunks of three

    def setUp(self):
        torch.manual_seed(21)
        self.model = MeshImpactHistoryNet(**self.KWARGS, teacher_forcing_chunks=3).eval()
        with torch.no_grad():
            # Nonzero history weights, as after training, so every path is live.
            for parameter in self.model.history_embedding.parameters():
                parameter.normal_()
        self.mesh = torch.randn(14, 3)
        self.mesh_batch = torch.tensor([0] * 6 + [1] * 8)
        self.impact = torch.tensor([[0.15, -0.4], [-0.7, 0.3]])
        self.time = torch.linspace(-1, 1, self.STEPS).repeat(2)
        self.time_batch = torch.arange(2).repeat_interleave(self.STEPS)
        self.history = torch.randn(2 * self.STEPS)

    def predict(self, model=None, **kwargs):
        model = self.model if model is None else model
        with torch.no_grad():
            return model(self.mesh, self.mesh_batch, self.impact, self.time, self.time_batch,
                         batch_size=2, **kwargs).reshape(2, self.STEPS)

    def test_a_chunk_never_reads_its_own_or_later_accelerations(self):
        reference = self.predict(history=self.history)
        for chunk in range(3):
            with self.subTest(chunk=chunk):
                changed = self.history.reshape(2, self.STEPS).clone()
                changed[:, 3 * chunk:3 * chunk + 3] += 5.0
                output = self.predict(history=changed.flatten())
                difference = (output - reference).abs()
                self.assertEqual(difference[:, :3 * chunk + 3].max().item(), 0.0)
                if chunk < 2:
                    self.assertGreater(difference[:, 3 * chunk + 3:].max().item(), 1e-4)

    def test_rollout_is_teacher_forcing_on_its_own_predictions(self):
        rollout = self.predict()
        torch.testing.assert_close(self.predict(history=rollout.flatten()), rollout, rtol=1e-5, atol=1e-6)
        self.assertGreater((self.predict(history=self.history) - rollout)[:, 3:].abs().max().item(), 1e-4)

    def test_last_chunk_may_be_shorter(self):
        # 9 times in 2 chunks: 5 + 4; the second reads times 0-3 shifted by 5.
        torch.manual_seed(21)
        model = MeshImpactHistoryNet(**self.KWARGS, teacher_forcing_chunks=2).eval()
        reference = self.predict(model, history=self.history)
        changed = self.history.reshape(2, self.STEPS).clone()
        changed[:, 4] += 5.0  # the first chunk's last time is read by no one
        torch.testing.assert_close(self.predict(model, history=changed.flatten()), reference, rtol=0, atol=0)
        torch.testing.assert_close(self.predict(model, history=self.predict(model).flatten()), self.predict(model),
                                   rtol=1e-5, atol=1e-6)

    def test_history_embedding_learns_from_the_loss(self):
        model = self.model.train()
        prediction = model(self.mesh, self.mesh_batch, self.impact, self.time, self.time_batch,
                           batch_size=2, history=self.history)
        prediction.square().mean().backward()
        self.assertGreater(model.history_embedding[0].weight.grad.abs().sum().item(), 0)

    def test_off_by_default_and_invalid_settings(self):
        plain = MeshImpactHistoryNet(**self.KWARGS).eval()
        self.assertEqual(plain.teacher_forcing_chunks, 0)
        self.assertFalse(hasattr(plain, "history_embedding"))
        self.assertFalse(any(key.startswith("history_") for key in plain.state_dict()))
        with self.assertRaisesRegex(ValueError, "teacher_forcing_chunks > 0"):
            self.predict(plain, history=self.history)
        for chunks in (1, -2, True, 2.0):
            with self.subTest(chunks=chunks), self.assertRaisesRegex(ValueError, "teacher_forcing_chunks"):
                MeshImpactHistoryNet(**self.KWARGS, teacher_forcing_chunks=chunks)
        with self.assertRaisesRegex(ValueError, "decoder='temporal'"):
            MeshImpactHistoryNet(**self.KWARGS, decoder="mesh_only", teacher_forcing_chunks=3)

    def test_rejects_bad_history_and_uneven_time_grids(self):
        with self.assertRaisesRegex(ValueError, "history must be finite"):
            self.predict(history=self.history[:-1])
        with self.assertRaisesRegex(ValueError, "history must be finite"):
            self.predict(history=self.history.masked_fill(torch.arange(18) == 4, float("nan")))
        with self.assertRaisesRegex(ValueError, "same number of times"):
            with torch.no_grad():
                self.model(self.mesh, self.mesh_batch, self.impact, self.time[:-1], self.time_batch[:-1], batch_size=2)


class RecordingModel(nn.Module):
    """Records whether each call received the true history."""

    teacher_forcing_chunks = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, mesh, mesh_batch, indentor, time, time_batch, batch_size=None, history=None):
        self.calls.append(history is not None)
        return self.scale * time


class TeacherForcingTrainerTests(unittest.TestCase):
    def test_training_feeds_the_truth_and_validation_rolls_out(self):
        batch = {"mesh": torch.zeros(3, 3), "mesh_batch": torch.zeros(3, dtype=torch.long),
                 "indentor": torch.zeros(1, 2), "time": torch.linspace(0, 1, 4),
                 "time_batch": torch.zeros(4, dtype=torch.long), "acceleration": torch.ones(4), "batch_size": 1}
        config = SimpleNamespace(device=torch.device("cpu"), learning_rate=1e-3, weight_decay=0.0, num_epochs=1,
                                 prediction_target="acceleration")
        preprocessor = SimpleNamespace(inverse_transform_acceleration=lambda values: values)
        model = RecordingModel()
        trainer = Trainer(model, [batch], [batch], config, preprocessor)
        trainer.train_epoch()
        trainer.validate()
        trainer.validate(teacher_forced=True)
        self.assertEqual(model.calls, [True, False, True])


class TimeWindowMetricTests(unittest.TestCase):
    def test_windows_follow_the_chunks_and_score_the_location_mean(self):
        times = [np.arange(5) * 1e-3] * 4

        def split(runs, curves):
            return SimpleNamespace(run_numbers=runs, time_arrays=times[:len(runs)]), np.ravel(curves)

        # Two locations per design; training designs 0 and 1, test design 2.
        train_curves = np.array([[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [4, 4, 4, 4, 4]], float)
        train_set, train_targets = split([1, 2, 3, 4], train_curves)
        test_curves = np.array([[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]], float)
        test_set, targets = split([5, 6], test_curves)
        predictions = test_curves + np.array([[0, 0, 0, 3, 3], [0, 0, 0, 3, 3]])
        report = time_window_metrics((train_set, train_targets, train_targets), (test_set, predictions, targets),
                                     samples_per_design=2, num_windows=2, teacher_forced=np.ravel(test_curves))
        self.assertEqual(report["window_points"], [3, 2])
        self.assertEqual(report["window_ms"], [[0.0, 2.0], [3.0, 4.0]])
        self.assertEqual(report["model_rmse_g"], [0.0, 3.0])
        self.assertEqual(report["teacher_forced_rmse_g"], [0.0, 0.0])
        # Location means are 1 and 3, exactly the test curves.
        self.assertEqual(report["location_mean_rmse_g"], [0.0, 0.0])
        self.assertEqual(report["num_test_curves_with_location_mean"], 2)
        no_forcing = time_window_metrics((train_set, train_targets, train_targets), (test_set, predictions, targets),
                                         samples_per_design=2, num_windows=2)
        self.assertIsNone(no_forcing["teacher_forced_rmse_g"])


if __name__ == "__main__":
    unittest.main()
