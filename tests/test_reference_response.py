"""Reference-response output (a_ref + s * network) and edit tokens (reference position + displacement)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from design_change_nodes import STATIC_CHANNELS
from mesh_design_change import DesignChangeHistoryNet, DesignChangeTable, change_displacement_rms
from mesh_impact_history import MeshImpactHistoryNet
from reference_response import ReferenceResponseTable, residual_rms

KWARGS = dict(width=16, num_heads=2, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0)
TIMES = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
CURVES = np.stack([np.sin(3 * TIMES), np.cos(2 * TIMES) - 0.5]).astype(np.float32)
LOADS = np.array([[10.0, 20.0], [30.0, -5.0]])


class ReferenceTableTests(unittest.TestCase):
    def setUp(self):
        self.table = ReferenceResponseTable(LOADS, TIMES, CURVES, labels=[4, 9])

    def test_lookup_matches_within_tolerance_only(self):
        np.testing.assert_array_equal(self.table.lookup([[30.2, -5.1], [10.0, 20.0]]), [1, 0])
        with self.assertRaisesRegex(KeyError, "No reference response"):
            self.table.lookup([[11.0, 20.0]])
        with self.assertRaisesRegex(ValueError, "within the matching tolerance"):
            ReferenceResponseTable([[0.0, 0.0], [0.3, 0.0]], TIMES, CURVES)

    def test_interpolation_is_exact_on_the_grid_and_linear_between(self):
        times = torch.tensor([[TIMES[0], TIMES[3], 0.5 * (TIMES[3] + TIMES[4]), TIMES[-1]]] * 2)
        valid = torch.tensor([[True] * 4, [True, True, True, False]])
        values = self.table.curves_at(torch.tensor([0, 1]), times, valid)
        expected = np.stack([np.interp(row.numpy(), TIMES, CURVES[k]) for k, row in enumerate(times)])
        expected[1, 3] = 0.0  # padding
        np.testing.assert_allclose(values.numpy(), expected, rtol=0, atol=1e-6)
        self.assertEqual(values[0, 1].item(), CURVES[0, 3])
        np.testing.assert_allclose(self.table.curve([10.0, 20.0], times[0].numpy()), expected[0], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "outside the reference grid"):
            self.table.curves_at(torch.tensor([0]), torch.tensor([[1.5]]), torch.tensor([[True]]))

    def test_round_trip_and_residual_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            self.table.save(Path(directory) / "table.npz")
            loaded = ReferenceResponseTable.load(Path(directory) / "table.npz")
        self.assertTrue(loaded.same_as(self.table))
        runs = [CURVES[0] + 0.3, CURVES[1] - 0.4 * TIMES]
        rms = residual_rms(self.table, LOADS, [TIMES, TIMES], runs)
        self.assertAlmostEqual(rms, float(np.sqrt(np.mean(np.concatenate([np.full(7, 0.3), 0.4 * TIMES]) ** 2))),
                               places=6)


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.meshes = [torch.randn(10, 3), torch.randn(10, 3)]
        self.table = ReferenceResponseTable(LOADS, TIMES, CURVES)

    def model(self, cls=MeshImpactHistoryNet, **kwargs):
        torch.manual_seed(3)
        model = cls(**KWARGS, **kwargs).eval()
        model.set_coordinate_scalers([0, 0, 0], [1, 1, 1], [0, 0], [10, 10])
        if getattr(model, "reference_response", False):
            model.reference_table = self.table
        return model

    def predict(self, model, times=TIMES, loads=LOADS):
        times = torch.as_tensor(times)
        mesh_batch = torch.cat([torch.full((len(mesh),), b) for b, mesh in enumerate(self.meshes)])
        indentor = torch.as_tensor(loads, dtype=torch.float32) / 10.0
        return model(torch.cat(self.meshes), mesh_batch, indentor, times.repeat(2),
                     torch.arange(2).repeat_interleave(len(times)), batch_size=2).reshape(2, -1)

    def test_untrained_model_returns_the_reference(self):
        torch.testing.assert_close(self.predict(self.model(reference_response=True)), torch.as_tensor(CURVES),
                                   rtol=0, atol=1e-7)

    def test_output_is_reference_plus_scaled_network_that_reads_the_reference(self):
        model = self.model(reference_response=True)
        with torch.no_grad():
            model.acceleration_head[-1].weight.normal_()
        reference = torch.as_tensor(CURVES)
        delta = self.predict(model) - reference
        model.set_reference_delta_scale(3.0)
        torch.testing.assert_close(self.predict(model) - reference, 3 * delta, rtol=1e-5, atol=1e-6)
        # The network's own output changes when only the reference curve changes.
        model.set_reference_delta_scale(1.0)
        model.reference_table = ReferenceResponseTable(LOADS, TIMES, CURVES + 1.0)
        self.assertGreater((self.predict(model) - (reference + 1.0) - delta).abs().max().item(), 1e-4)

    def test_other_weights_and_keys_match_the_plain_network(self):
        plain, referenced = self.model(), self.model(reference_response=True)
        self.assertFalse(any("reference" in key for key in plain.state_dict()))
        fresh = dict(referenced.named_parameters())
        for name, value in plain.named_parameters():
            if not name.startswith("acceleration_head.3"):
                torch.testing.assert_close(fresh[name], value, rtol=0, atol=0)
        self.assertEqual(referenced.acceleration_head[-1].weight.abs().sum().item(), 0.0)
        self.assertIn("reference_delta_scale", referenced.state_dict())

    def test_reference_path_learns(self):
        model = self.model(reference_response=True).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        target = torch.as_tensor(CURVES) + 0.5 * torch.as_tensor(TIMES)
        for _ in range(2):
            optimizer.zero_grad()
            (self.predict(model) - target).square().mean().backward()
            optimizer.step()
        self.assertGreater(model.reference_embedding[0].weight.grad.abs().sum().item(), 0)

    def test_errors(self):
        model = self.model(reference_response=True)
        with self.assertRaisesRegex(KeyError, "No reference response"):
            self.predict(model, loads=LOADS + 5.0)
        model.reference_table = None
        with self.assertRaisesRegex(RuntimeError, "no reference_table"):
            self.predict(model)
        with self.assertRaisesRegex(ValueError, "teacher_forcing_chunks=0"):
            MeshImpactHistoryNet(**KWARGS, reference_response=True, teacher_forcing_chunks=2)
        with self.assertRaisesRegex(ValueError, "without reference_response"):
            self.model().set_reference_delta_scale(1.0)


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
            model.change_features(static, torch.zeros(1, 2))
        model.set_change_displacement_scale(5.0)
        features = model.change_features(static, torch.zeros(1, 2))
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

    def test_reference_and_edit_tokens_together_start_as_the_reference(self):
        model = self.model(design_change_features="edit", reference_response=True)
        model.set_change_displacement_scale(5.0)
        loads = (self.impact * 100.0).numpy().astype(np.float64)
        model.reference_table = ReferenceResponseTable(loads[1:], np.linspace(-1, 1, 6), np.ones((2, 6)) * [[1.0], [2.0]])
        torch.testing.assert_close(self.predict(model).reshape(3, -1)[1:], torch.tensor([[1.0] * 6, [2.0] * 6]))


# Inner panel = nodes 20-23 of this deck; the rest is headform and outer skin.
DECK = """*HEADING
*NODE
1, 0.0, 0.0, 90.0
2, 1.0, 0.0, 90.0
20, 0.0, 0.0, 0.0
10, 0.0, 0.0, 5.0
21, 1.0, 0.0, 0.0
3, 0.0, 1.0, 90.0
11, 1.0, 0.0, 5.0
22, 1.0, 1.0, 0.0
12, 1.0, 1.0, 5.0
23, 0.0, 1.0, 0.0
*ELEMENT, TYPE=S3R
100, 1, 2, 3
*ELEMENT, TYPE=S3R
200, 10, 11, 12
*ELEMENT, TYPE=S4
300, 20, 21, 22, 23
*ELSET, ELSET="Rigid Body1_2-1-3"
100
*ELSET, ELSET="Hood_Outer-1-2"
200
*ELSET, ELSET="Hood_Inner-1-2"
300
"""
RAW_TIMES = np.linspace(0.0, 0.007, 8)
CHANGES = {"N1": [(21, 1.0, 0.0, 0.0, 2.0)], "N2": [(22, 1.0, 1.0, 0.0, 3.0), (23, 0.0, 1.0, 0.0, 1.0)],
           "N3": [(20, 0.0, 0.0, 0.0, -1.5)], "N4": [(21, 1.0, 0.0, 0.0, -2.5)]}


def curve(design, location):
    shape = np.sin(np.pi * RAW_TIMES / RAW_TIMES[-1])
    return 50.0 * shape * (1.0 + 0.1 * location) + 4.0 * design * shape ** 2


def write_dataset(root):
    root = Path(root)
    for name in ("inp_files", "output_history_acc", "design_changes"):
        (root / name).mkdir(parents=True)
    (root / "inp_files" / "HoodImpact_1.inp").write_text(DECK)
    rows, run = [], 0
    for design, locations in [("D6", (1, 2, 3)), *((name, (1, 2)) for name in CHANGES)]:
        for location in locations:
            run += 1
            rows.append({"run": run, "design": design, "location": location, "role": "near",
                         "x_mm": 100.0 * location, "y_mm": 0.5, "dist_to_edit_mm": 10.0 * location})
            number = 0 if design == "D6" else int(design[1:])
            pd.DataFrame({"Time": RAW_TIMES, "A(in g)": curve(number, location)}).to_csv(
                root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "manifest_local_edits.csv", index=False)
    for design, changes in CHANGES.items():
        pd.DataFrame(changes, columns=["label", "x", "y", "z_design6", "dz"]).to_csv(
            root / "design_changes" / f"{design}_changed_nodes.csv", index=False)
    pd.DataFrame({"design": list(CHANGES), "kind": "single", "variant": "S",
                  "dz_max_mm": [2.0, 3.0, 0.0, 0.0], "dz_min_mm": [0.0, 0.0, -1.5, -2.5]}).to_csv(
        root / "local_edit_designs.csv", index=False)
    pd.DataFrame({"design": list(CHANGES), "site": ["s1", "s1", "s2", "s1"]}).to_csv(
        root / "local_edit_doe.csv", index=False)
    return root


class LocalEditTrainingTest(unittest.TestCase):
    def test_one_epoch_with_reference_and_edit_tokens(self):
        from local_edits_data import LocalEditGeometry
        from train_mesh_impact_history import HistoryPredictor, main

        with tempfile.TemporaryDirectory() as directory:
            root = write_dataset(Path(directory) / "data")
            run_dir = Path(directory) / "run"
            main(["--data-format", "localedits", "--inp-dir", str(root / "inp_files"),
                  "--impact-coords-path", str(root / "manifest_local_edits.csv"),
                  "--acceleration-dir", str(root / "output_history_acc"),
                  "--test-designs", "4", "--val-designs", "3", "--epochs", "1", "--batch-size", "2",
                  "--time-subsample-stride", "1", "--width", "16", "--num-heads", "2", "--num-latents", "4",
                  "--latent-layers", "1", "--temporal-layers", "1", "--dropout", "0",
                  "--fourier-max-frequency-hz", "400",
                  "--design-change-nodes", "true", "--design-change-features", "edit",
                  "--reference-response", "true", "--device", "cpu", "--wandb-mode", "disabled",
                  "--output-dir", str(run_dir)])
            saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["architecture"]["kwargs"]["reference_response"])
            reference = saved["reference_response"]
            self.assertEqual(reference["load_cases"], 3)
            # s = training RMS of (a - a_D6), normalized with the training-only scaler (N1, N2).
            train = np.concatenate([curve(d, loc) for d in (1, 2) for loc in (1, 2)])
            residual = np.concatenate([curve(d, loc) - curve(0, loc) for d in (1, 2) for loc in (1, 2)])
            self.assertAlmostEqual(reference["delta_scale_normalized"],
                                   float(np.sqrt(np.mean(residual ** 2))) / float(train.std()), places=4)
            self.assertAlmostEqual(saved["design_change"]["edit_displacement_scale_mm"],
                                   float(np.sqrt((4 + 9 + 1) / 3)), places=5)
            self.assertTrue((run_dir / "reference_responses.npz").is_file())

            predictor = HistoryPredictor.from_run(run_dir)
            geometry, _, _ = LocalEditGeometry.from_manifest(root / "manifest_local_edits.csv", root / "inp_files")
            prediction = predictor.predict(geometry.mesh(4), [200.0, 0.5])
            self.assertEqual(prediction.shape, (8,))
            self.assertTrue(np.isfinite(prediction).all())
            with self.assertRaisesRegex(KeyError, "No reference response"):
                predictor.predict(geometry.mesh(4), [250.0, 0.5])


if __name__ == "__main__":
    unittest.main()
