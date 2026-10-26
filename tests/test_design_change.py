"""Change-node detection and the DesignChangeHistoryNet decoder path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from design_change_nodes import STATIC_CHANNELS, near_clone_changes
from mesh_design_change import DesignChangeHistoryNet, DesignChangeTable
from mesh_impact_history import MeshImpactHistoryNet


def geometry(ids, xyz):
    return {"ids": np.asarray(ids), "xyz": np.asarray(xyz, float),
            "gap": 20.0 - np.asarray(xyz, float)[:, 2], "panel": np.ones(len(ids), dtype=int)}


class NearCloneChangeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(-100, 100, size=(20, 3))
        one, two = base.copy(), base.copy()
        one[[2, 3], 2] += 5.0
        two[7, 2] += 3.0
        morph = base + [1.0, 0.0, 0.0]
        self.geometries = {0: geometry(range(20), base), 1: geometry(range(20), one),
                           2: geometry(range(20), two), 3: geometry(range(20), morph),
                           4: geometry(range(100, 120), base)}

    def test_row_rule_union_and_base_displacement(self):
        changes, report = near_clone_changes(self.geometries, max_nodes=10)
        self.assertEqual(report["clusters"][0], {"designs": [0, 1, 2], "base": 0, "change_nodes": 3})
        for design in (0, 1, 2):
            self.assertEqual(changes[design]["node_ids"].tolist(), [2, 3, 7])
        static = changes[1]["static"]
        np.testing.assert_allclose(static[:, 3:6], [[0, 0, 5], [0, 0, 5], [0, 0, 0]])
        np.testing.assert_allclose(static[:, 7], [-5, -5, 0])
        np.testing.assert_array_equal(changes[0]["static"][:, [3, 4, 5, 7]], 0)
        # The morph moves every node and a different mesh shares no IDs.
        for design in (3, 4):
            self.assertEqual(len(changes[design]["node_ids"]), 0)
            self.assertEqual(changes[design]["base"], design)

    def test_rejects_cluster_without_lowest_member(self):
        # Design 1 now lowers node 2 and raises node 3, so 0 and 1 each dip below the other.
        self.geometries[1]["xyz"][2, 2] -= 10.0
        with self.assertRaisesRegex(ValueError, "no member is lowest"):
            near_clone_changes(self.geometries, max_nodes=10)


class DesignChangeHistoryNetTests(unittest.TestCase):
    KWARGS = dict(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0)

    def setUp(self):
        torch.manual_seed(5)
        self.meshes = [torch.randn(12, 3), torch.randn(12, 3), torch.randn(12, 3)]
        static = np.zeros((3, 4, len(STATIC_CHANNELS)), np.float32)
        static[:, :, :3] = np.random.default_rng(0).uniform(-200, 200, size=(3, 4, 3))
        static[1, :3, 5] = [8.0, 12.0, 4.0]  # design 1 raises three nodes
        static[1, :3, 7] = [-8.0, -12.0, -4.0]
        # Design 0 is its cluster's base; design 2 has no change nodes at all.
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
        return model

    def predict(self, model, meshes=None, impact=None):
        meshes = self.meshes if meshes is None else meshes
        mesh_batch = torch.cat([torch.full((len(mesh),), b) for b, mesh in enumerate(meshes)])
        time = torch.linspace(-1, 1, 6).repeat(len(meshes))
        time_batch = torch.arange(len(meshes)).repeat_interleave(6)
        return model(torch.cat(meshes), mesh_batch, self.impact if impact is None else impact,
                     time, time_batch, batch_size=len(meshes))

    def test_starts_as_the_backbone_with_identical_backbone_weights(self):
        backbone, changed = self.model(MeshImpactHistoryNet), self.model()
        missing = changed.load_state_dict(backbone.state_dict(), strict=False).missing_keys
        self.assertTrue(missing and all(key.startswith("change_") for key in missing))
        torch.manual_seed(11)
        fresh = dict(DesignChangeHistoryNet(**self.KWARGS).named_parameters())
        for name, value in backbone.named_parameters():
            torch.testing.assert_close(fresh[name], value, rtol=0, atol=0)
        torch.testing.assert_close(self.predict(changed), self.predict(backbone), rtol=0, atol=0)

    def test_only_displaced_designs_leave_the_backbone(self):
        backbone, changed = self.model(MeshImpactHistoryNet), self.model()
        with torch.no_grad():
            for readout in changed.change_readouts:
                readout.output.weight.normal_()
        changed.load_state_dict(backbone.state_dict(), strict=False)
        difference = (self.predict(changed) - self.predict(backbone)).reshape(3, -1).abs().amax(dim=1)
        self.assertEqual(difference[0].item(), 0.0)  # base: zero displacement
        self.assertEqual(difference[2].item(), 0.0)  # no change nodes
        self.assertGreater(difference[1].item(), 1e-3)

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

    def test_tokens_are_each_change_nodes_normalized_xyz(self):
        model = self.model()
        model.set_coordinate_scalers([10, -20, 5], [100, 50, 25], [0, 0], [100, 100])
        self.assertEqual(model.change_embedding[0].in_features, 3)
        static = torch.as_tensor(self.table.static[1:2])
        features = model.change_features(static, torch.zeros(1, 2))
        expected = (static[..., :3] + static[..., 3:6] - torch.tensor([10.0, -20, 5])) / torch.tensor([100.0, 50, 25])
        torch.testing.assert_close(features, expected)

    def test_saved_runs_keep_their_impact_relative_tokens(self):
        from mesh_design_change import saved_design_change_kwargs
        old = saved_design_change_kwargs({**self.KWARGS, "design_change_fourier_octaves": 5})
        self.assertEqual(old["design_change_features"], "impact_relative")
        self.assertEqual(DesignChangeHistoryNet(**old).change_embedding[0].in_features, 30)
        new = saved_design_change_kwargs({**self.KWARGS, "design_change_features": "xyz"})
        self.assertEqual(new["design_change_features"], "xyz")
        with self.assertRaisesRegex(ValueError, "design_change_features"):
            DesignChangeHistoryNet(**self.KWARGS, design_change_features="xy")

    def test_table_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            self.table.save(path)
            loaded = DesignChangeTable.load(path)
        self.assertTrue(loaded.same_as(self.table))
        self.assertEqual(loaded.lookup(self.meshes[1]), 1)


if __name__ == "__main__":
    unittest.main()
