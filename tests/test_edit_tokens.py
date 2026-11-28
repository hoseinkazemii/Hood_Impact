"""Edit tokens (design-6 position + displacement) and the geometry-only local-edit pipeline.

The network reads geometry and the impact location only. Design 6's simulated
curves are never an input; they are read after training, as the reference the
edit effects in local_edit_metrics.json are measured from.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mesh_design_change import STATIC_CHANNELS, DesignChangeHistoryNet, DesignChangeTable, change_displacement_rms
from mesh_impact_history import MeshImpactHistoryNet, saved_model_kwargs
from toy_local_edits import training_args, write_dataset

KWARGS = dict(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0,
              neighborhood_layers=0)


class EditTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.meshes = [torch.randn(12, 3), torch.randn(12, 3), torch.randn(12, 3)]
        static = np.zeros((3, 4, len(STATIC_CHANNELS)), np.float32)
        static[:, :, :3] = np.random.default_rng(0).uniform(-200, 200, size=(3, 4, 3))
        static[1, :3, 3:6] = [[0.0, 0.0, 8.0], [3.0, -4.0, 0.0], [0.0, 0.0, -4.0]]  # design 1 moves 3 nodes
        self.table = DesignChangeTable([0, 1, 2], static, [4, 3, 0])  # design 2: no change nodes
        for design, mesh in enumerate(self.meshes):
            self.table.register(mesh, design)
        self.impact = torch.tensor([[0.2, -0.3], [0.2, -0.3], [-0.5, 0.4]])

    def model(self, cls=DesignChangeHistoryNet, **kwargs):
        torch.manual_seed(11)
        model = cls(**KWARGS, **kwargs).eval()
        model.set_coordinate_scalers([10, -20, 5], [100, 50, 25], [0, 0], [100, 100])
        if cls is DesignChangeHistoryNet:
            model.design_changes = self.table
        return model

    def predict(self, model):
        mesh_batch = torch.cat([torch.full((len(mesh),), b) for b, mesh in enumerate(self.meshes)])
        time = torch.linspace(-1, 1, 6).repeat(3)
        return model(torch.cat(self.meshes), mesh_batch, self.impact, time,
                     torch.arange(3).repeat_interleave(6), batch_size=3)

    def test_tokens_are_reference_position_and_scaled_displacement(self):
        model = self.model(design_change_features="edit")
        self.assertEqual(model.change_embedding[0].in_features, 6)
        static = torch.as_tensor(self.table.static[1:2])
        with self.assertRaisesRegex(RuntimeError, "displacement scale"):
            model.change_features(static)
        model.set_change_displacement_scale(5.0)
        features = model.change_features(static)
        torch.testing.assert_close(features[..., :3], (static[..., :3] - torch.tensor([10.0, -20, 5]))
                                   / torch.tensor([100.0, 50, 25]))
        torch.testing.assert_close(features[..., 3:], static[..., 3:6] / 5.0)

    def test_null_token_adds_nothing(self):
        backbone, changed = self.model(MeshImpactHistoryNet), self.model(design_change_features="edit")
        changed.set_change_displacement_scale(5.0)
        with torch.no_grad():
            for readout in changed.change_readouts:
                readout.output.weight.normal_()
        changed.load_state_dict(backbone.state_dict(), strict=False)
        difference = (self.predict(changed) - self.predict(backbone)).reshape(3, -1).abs().amax(dim=1)
        self.assertEqual(difference[2].item(), 0.0)
        self.assertGreater(difference[1].item(), 1e-3)

    def test_displacement_scale_is_the_rms_length_of_moved_nodes(self):
        changes = {design: {"static": self.table.static[design, :count]}
                   for design, count in enumerate(self.table.counts)}
        self.assertAlmostEqual(change_displacement_rms(changes, [0, 1, 2]), float(np.sqrt((64 + 25 + 16) / 3)))
        with self.assertRaisesRegex(ValueError, "move no change node"):
            change_displacement_rms(changes, [0, 2])


class RemovedReferenceResponseTests(unittest.TestCase):
    def test_model_has_no_response_input(self):
        model = DesignChangeHistoryNet(**KWARGS, design_change_features="edit")
        self.assertFalse(any("reference" in key for key in model.state_dict()))
        with self.assertRaises(TypeError):
            MeshImpactHistoryNet(**KWARGS, reference_response=True)

    def test_saved_runs_that_read_the_response_are_refused(self):
        modern = {"width": 16, "impact_conditioning": "film", "time_encoding": "fourier",
                  "pooling_query_residual": False, "pool_impactor_nodes": False}
        self.assertNotIn("reference_response",
                         saved_model_kwargs({**modern, "reference_response": False}, format_version=1))
        with self.assertRaisesRegex(ValueError, "removed on 2026-11-26"):
            saved_model_kwargs({**modern, "reference_response": True}, format_version=1)


def train(root, run_dir):
    from train_mesh_impact_history import main

    main(training_args(root, run_dir, "--epochs", "1", "--dropout", "0", "--design-change-features", "edit"))


class LocalEditTrainingTest(unittest.TestCase):
    def test_geometry_only_training_never_reads_the_base_response(self):
        from local_edits_data import LocalEditGeometry
        from train_mesh_impact_history import HistoryPredictor, parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--reference-response", "true"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = []
            for gain in (1.0, 3.0):  # the same campaign except for design 6's simulated curves
                root = write_dataset(directory / f"data_{gain}", base_gain=gain)
                run_dir = directory / f"run_{gain}"
                train(root, run_dir)
                saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                self.assertNotIn("reference_response", saved["architecture"]["kwargs"])
                self.assertNotIn("reference_response", saved)
                self.assertFalse((run_dir / "reference_responses.npz").exists())
                self.assertAlmostEqual(saved["design_change"]["edit_displacement_scale_mm"],
                                       float(np.sqrt((4 + 9 + 1) / 3)), places=5)

                predictor = HistoryPredictor.from_run(run_dir)
                geometry, _, _ = LocalEditGeometry.from_manifest(root / "manifest_local_edits.csv",
                                                                 root / "inp_files")
                # 250 mm: an impact point no design was simulated at.
                run_predictions = np.stack([predictor.predict(geometry.mesh(4), [x, 0.5]) for x in (200.0, 250.0)])
                self.assertEqual(run_predictions.shape, (2, 8))
                self.assertTrue(np.isfinite(run_predictions).all())
                predictions.append(run_predictions)
            np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
