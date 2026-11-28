"""The DesignChangeHistoryNet decoder path, its change table and saved-run compatibility."""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mesh_design_change import (
    LEGACY_STATIC_CHANNELS, STATIC_CHANNELS, DesignChangeHistoryNet, DesignChangeTable, saved_design_change_kwargs,
)
from mesh_impact_history import MeshImpactHistoryNet


class DesignChangeHistoryNetTests(unittest.TestCase):
    KWARGS = dict(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0,
                  neighborhood_layers=0)

    def setUp(self):
        torch.manual_seed(5)
        self.meshes = [torch.randn(12, 3), torch.randn(12, 3), torch.randn(12, 3)]
        static = np.zeros((3, 4, len(STATIC_CHANNELS)), np.float32)
        static[:, :, :3] = np.random.default_rng(0).uniform(-200, 200, size=(3, 4, 3))
        static[1, :3, 5] = [8.0, 12.0, 4.0]  # design 1 raises three nodes
        # Design 0 lists change nodes that did not move; design 2 has no change nodes at all.
        self.table = DesignChangeTable([0, 1, 2], static, [4, 3, 0])
        for design, mesh in enumerate(self.meshes):
            self.table.register(mesh, design)
        self.impact = torch.tensor([[0.2, -0.3], [0.2, -0.3], [-0.5, 0.4]])

    def model(self, cls=DesignChangeHistoryNet, **kwargs):
        torch.manual_seed(11)
        model = cls(**self.KWARGS, **kwargs).eval()
        model.set_coordinate_scalers([0, 0, 0], [1, 1, 1], [0, 0], [100, 100])
        if cls is DesignChangeHistoryNet:
            model.design_changes = self.table
            if model.design_change_features == "edit":
                model.set_change_displacement_scale(5.0)
        return model

    def predict(self, model, meshes=None, impact=None):
        meshes = self.meshes if meshes is None else meshes
        mesh_batch = torch.cat([torch.full((len(mesh),), b) for b, mesh in enumerate(meshes)])
        time = torch.linspace(-1, 1, 6).repeat(len(meshes))
        time_batch = torch.arange(len(meshes)).repeat_interleave(6)
        return model(torch.cat(meshes), mesh_batch, self.impact if impact is None else impact,
                     time, time_batch, batch_size=len(meshes))

    def test_starts_as_the_backbone_with_identical_backbone_weights(self):
        for features in ("edit", "xyz"):
            backbone, changed = self.model(MeshImpactHistoryNet), self.model(design_change_features=features)
            missing = changed.load_state_dict(backbone.state_dict(), strict=False).missing_keys
            self.assertTrue(missing and all(key.startswith("change_") for key in missing))
            torch.manual_seed(11)
            fresh = dict(DesignChangeHistoryNet(**self.KWARGS, design_change_features=features).named_parameters())
            for name, value in backbone.named_parameters():
                torch.testing.assert_close(fresh[name], value, rtol=0, atol=0)
            torch.testing.assert_close(self.predict(changed), self.predict(backbone), rtol=0, atol=0)

    def readout_difference(self, **kwargs):
        """Per-design max |change model - backbone| once the readouts are switched on."""
        backbone, changed = self.model(MeshImpactHistoryNet), self.model(**kwargs)
        with torch.no_grad():
            for readout in changed.change_readouts:
                readout.output.weight.normal_()
        changed.load_state_dict(backbone.state_dict(), strict=False)
        return (self.predict(changed) - self.predict(backbone)).reshape(3, -1).abs().amax(dim=1)

    def test_edit_tokens_leave_a_design_without_change_nodes_unchanged(self):
        # The null token's value is zero, so only designs with change nodes move.
        difference = self.readout_difference(design_change_features="edit")
        self.assertEqual(difference[2].item(), 0.0)
        self.assertGreater(difference[0].item(), 1e-3)
        self.assertGreater(difference[1].item(), 1e-3)

    def test_xyz_tokens_read_every_design(self):
        # No gate: a design without change nodes reads the learned null token.
        difference = self.readout_difference(design_change_features="xyz")
        self.assertTrue((difference > 1e-3).all(), difference)

    def test_change_path_learns_from_the_acceleration_loss(self):
        model = self.model().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            self.predict(model).square().mean().backward()
            if step == 0:
                self.assertGreater(model.change_readouts[0].output.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertGreater(model.change_embedding[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(self.predict(model)).all())

    def test_unknown_hood_and_missing_table_are_errors(self):
        model = self.model()
        with self.assertRaisesRegex(KeyError, "not in the design-change table"):
            self.predict(model, meshes=[torch.randn(12, 3)], impact=self.impact[:1])
        model.design_changes = None
        with self.assertRaisesRegex(RuntimeError, "no design_changes table"):
            self.predict(model)

    def test_xyz_tokens_are_each_change_nodes_normalized_xyz(self):
        model = self.model(design_change_features="xyz")
        model.set_coordinate_scalers([10, -20, 5], [100, 50, 25], [0, 0], [100, 100])
        self.assertEqual(model.change_embedding[0].in_features, 3)
        static = torch.as_tensor(self.table.static[1:2])
        expected = (static[..., :3] + static[..., 3:6] - torch.tensor([10.0, -20, 5])) / torch.tensor([100.0, 50, 25])
        torch.testing.assert_close(model.change_features(static), expected)

    def test_invalid_change_features_are_rejected(self):
        for features in ("xy", "impact_relative"):
            with self.assertRaisesRegex(ValueError, "design_change_features"):
                DesignChangeHistoryNet(**self.KWARGS, design_change_features=features)

    def test_table_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            self.table.save(path)
            loaded = DesignChangeTable.load(path)
        self.assertTrue(loaded.same_as(self.table))
        self.assertEqual(loaded.lookup(self.meshes[1]), 1)

    def test_tables_saved_with_the_three_gap_channels_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            legacy = np.zeros((*self.table.static.shape[:2], len(LEGACY_STATIC_CHANNELS)), np.float32)
            legacy[..., :len(STATIC_CHANNELS)] = self.table.static
            keys = sorted(self.table.index_of)
            np.savez(path, designs=self.table.designs, static=legacy, counts=self.table.counts,
                     channels=np.array(LEGACY_STATIC_CHANNELS), digests=np.array(keys, dtype=str),
                     digest_index=np.array([self.table.index_of[key] for key in keys], dtype=np.int64))
            self.assertTrue(DesignChangeTable.load(path).same_as(self.table))
            # Nonzero gap channels came from the 1704 near-clone set, which this code no longer reads.
            legacy[0, 0, 6] = 1.0
            np.savez(path, designs=self.table.designs, static=legacy, counts=self.table.counts,
                     channels=np.array(LEGACY_STATIC_CHANNELS), digests=np.array(keys, dtype=str),
                     digest_index=np.array([self.table.index_of[key] for key in keys], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "stores channels"):
                DesignChangeTable.load(path)


class SavedRunTests(unittest.TestCase):
    """Configs of earlier runs rebuild when every retired option held its one remaining value."""

    LOCAL_EDIT_RUN = {  # architecture kwargs of run 3654261 (2026-11-21)
        "width": 128, "num_heads": 4, "num_latents": 256, "local_latents": None, "latent_layers": 3,
        "temporal_layers": 2, "dropout": 0.1, "decoder": "temporal", "impact_conditioning": "film",
        "time_encoding": "fourier", "fourier_num_frequencies": 6, "fourier_min_frequency_hz": 20.0,
        "fourier_max_frequency_hz": 640.0, "neighborhood_layers": 2, "neighborhood_k": 256,
        "neighborhood_scale_mm": 20.0, "neighborhood_chunk_size": 1024, "impactor_nodes": 0,
        "pooling_query_residual": False, "pool_impactor_nodes": False, "geometry_film": "none",
        "geometry_fourier_octaves": 5, "teacher_forcing_chunks": 0, "design_change_layers": 0,
        "design_change_fourier_octaves": 2, "design_change_features": "xyz",
        "design_change_displacement_gate": False,
    }

    def test_local_edit_runs_rebuild(self):
        kwargs = saved_design_change_kwargs(self.LOCAL_EDIT_RUN, format_version=1)
        backbone = set(inspect.signature(MeshImpactHistoryNet.__init__).parameters) - {"self"}
        self.assertEqual(set(kwargs) - {"design_change_features"}, backbone)
        model = DesignChangeHistoryNet(**{**kwargs, "width": 16, "num_latents": 4})
        self.assertEqual(model.local_latents, 2)  # None: half the tokens local, as before 2026-11-22
        self.assertEqual(model.design_change_features, "xyz")

    def test_runs_that_used_a_removed_option_are_refused(self):
        for key, value in (("decoder", "mesh_only"), ("impact_conditioning", "legacy_additive"),
                           ("time_encoding", "legacy_five"), ("geometry_film", "latent"),
                           ("teacher_forcing_chunks", 4), ("impactor_nodes", 286),
                           ("pool_impactor_nodes", True), ("design_change_layers", 1),
                           ("design_change_displacement_gate", True), ("reference_response", True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "removed|reference_response"):
                saved_design_change_kwargs({**self.LOCAL_EDIT_RUN, key: value}, format_version=1)
        without_features = dict(self.LOCAL_EDIT_RUN)
        del without_features["design_change_features"]  # before 2026-10-26: impact-relative tokens
        with self.assertRaisesRegex(ValueError, "impact-relative"):
            saved_design_change_kwargs(without_features, format_version=1)


if __name__ == "__main__":
    unittest.main()
