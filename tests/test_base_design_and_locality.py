"""Training on the base design and change locality (2026-11-28), on the hood and on the B1-B4 layout.

* Base design: a design with no change nodes is read through the null token
  alone; with "edit" tokens the readouts then add exactly nothing, so the base
  design is an ordinary training sample whose prediction is the backbone's.
* Change locality: each readout head lowers its attention score for a change
  node by w * r^2 (r = distance from the load point, w learned); an edit far
  from the load point then adds nothing. With w = 0 it is the plain readout.
* B1-B4: every benchmark family stores a "base" design with zero change nodes
  and edits as base position + displacement, so the base design is a sample of
  the same kind there; those problems have one fixed load, so locality stays off.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mesh_design_change import (
    STATIC_CHANNELS, DesignChangeHistoryNet, DesignChangeTable, change_displacement_rms, saved_design_change_kwargs,
)
from mesh_impact_history import MeshImpactHistoryNet
from toy_local_edits import training_args, write_dataset

KWARGS = dict(width=16, num_heads=4, num_latents=4, latent_layers=1, temporal_layers=2, dropout=0.0,
              neighborhood_layers=0)
PILOT = Path(r"D:\Papers\Hood_Impact\LocalEditBenchmarks\pilot")


def randomize_readouts(model, seed=3):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for readout in model.change_readouts:
            readout.output.weight.copy_(torch.randn(readout.output.weight.shape, generator=generator))


class ChangeLocalityTests(unittest.TestCase):
    """Design 1 moves three nodes near (300, 0) mm; design 0 is the base (no change nodes)."""

    def setUp(self):
        torch.manual_seed(5)
        self.meshes = [torch.randn(12, 3), torch.randn(12, 3)]
        static = np.zeros((2, 3, len(STATIC_CHANNELS)), np.float32)
        static[1, :, :3] = [[300.0, 0.0, 0.0], [305.0, 2.0, 0.0], [295.0, -2.0, 0.0]]
        static[1, :, 5] = [8.0, 6.0, 4.0]
        self.table = DesignChangeTable([0, 1], static, [0, 3])
        for design, mesh in enumerate(self.meshes):
            self.table.register(mesh, design)

    def model(self, cls=DesignChangeHistoryNet, **kwargs):
        torch.manual_seed(11)
        model = cls(**KWARGS, **kwargs).eval()
        # Mesh and impact both in units of 100 mm about the origin.
        model.set_coordinate_scalers([0, 0, 0], [100, 100, 100], [0, 0], [100, 100])
        if cls is DesignChangeHistoryNet:
            model.design_changes = self.table
            model.set_change_displacement_scale(6.0)
        return model

    def predict(self, model, design, impact_mm):
        mesh = self.meshes[design]
        impact = torch.tensor([impact_mm], dtype=torch.float32) / 100.0
        time = torch.linspace(-1, 1, 6)
        return model(mesh, torch.zeros(len(mesh), dtype=torch.long), impact, time,
                     torch.zeros(len(time), dtype=torch.long), batch_size=1)

    def pair(self, log_rate):
        """(backbone, locality model with randomized readouts and every rate = exp(log_rate))."""
        backbone, local = self.model(MeshImpactHistoryNet), self.model(change_locality=True)
        local.load_state_dict(backbone.state_dict(), strict=False)
        randomize_readouts(local)
        with torch.no_grad():
            local.change_log_rate.fill_(log_rate)
        return backbone, local

    def test_rates_start_spread_over_heads_and_load_off_by_default(self):
        model = self.model(change_locality=True)
        self.assertEqual(tuple(model.change_log_rate.shape), (2, 4))  # (readouts, heads)
        torch.testing.assert_close(model.change_log_rate[0].exp(), torch.logspace(math.log10(0.5), math.log10(32.0), 4))
        plain = self.model()
        self.assertFalse(plain.change_locality)
        self.assertIsNone(plain.change_log_rate)
        self.assertNotIn("change_log_rate", plain.state_dict())
        # Saved configs without the key (every run before 2026-11-28) rebuild without locality.
        for version in (1, 2):
            kwargs = saved_design_change_kwargs({"design_change_features": "edit", **({} if version == 2 else {
                "impact_conditioning": "film", "time_encoding": "fourier", "pooling_query_residual": False,
                "pool_impactor_nodes": False, "design_change_layers": 0,
                "design_change_displacement_gate": False})}, format_version=version)
            self.assertFalse(DesignChangeHistoryNet(**KWARGS, **kwargs).change_locality)
        with self.assertRaises(ValueError):
            DesignChangeHistoryNet(**KWARGS, change_locality="yes")

    def test_far_edit_adds_nothing_and_a_near_edit_acts(self):
        backbone, local = self.pair(math.log(50.0))
        with torch.no_grad():
            near = (self.predict(local, 1, [300.0, 0.0]) - self.predict(backbone, 1, [300.0, 0.0])).abs().max()
            far = (self.predict(local, 1, [-300.0, 0.0]) - self.predict(backbone, 1, [-300.0, 0.0])).abs().max()
        self.assertGreater(near.item(), 1e-3)
        self.assertLess(far.item(), 1e-6)
        # Without locality (rate 0) the same edit acts at the far impact too: the leak this removes.
        backbone, plain = self.pair(-math.inf)
        with torch.no_grad():
            leak = (self.predict(plain, 1, [-300.0, 0.0]) - self.predict(backbone, 1, [-300.0, 0.0])).abs().max()
        self.assertGreater(leak.item(), 1e-3)

    def test_zero_rate_is_the_readout_without_locality(self):
        plain = self.model()
        randomize_readouts(plain)
        local = self.model(change_locality=True)
        local.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            local.change_log_rate.fill_(-math.inf)
            for impact in ([300.0, 0.0], [-300.0, 50.0]):
                torch.testing.assert_close(self.predict(local, 1, impact), self.predict(plain, 1, impact),
                                           rtol=0, atol=1e-6)

    def test_base_design_is_exactly_the_backbone(self):
        backbone, local = self.pair(math.log(2.0))
        with torch.no_grad():
            for impact in ([300.0, 0.0], [-300.0, 0.0]):
                self.assertEqual((self.predict(local, 0, impact) - self.predict(backbone, 0, impact)).abs().max().item(), 0.0)

    def test_rates_are_learned(self):
        _, local = self.pair(math.log(2.0))
        local.train()
        self.predict(local, 1, [320.0, 40.0]).square().mean().backward()
        gradient = local.change_log_rate.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_distance_uses_the_load_point_coordinates_in_mesh_units(self):
        model = self.model(change_locality=True)
        static = torch.as_tensor(self.table.static[1:2])
        indentor = torch.tensor([[3.0, 0.0]])  # (300, 0) mm through the impact scaler
        expected = ((static[0, :, :2] - torch.tensor([300.0, 0.0])) / 100.0).square().sum(-1)
        torch.testing.assert_close(model.change_distance_squared(static, indentor)[0], expected)


class TrainingOnTheBaseDesignTest(unittest.TestCase):
    def test_toy_run_with_base_design_and_locality(self):
        from local_edits_data import LocalEditGeometry
        from train_mesh_impact_history import HistoryPredictor, main, parse_args, resolve_splits

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, run_dir = write_dataset(directory / "data"), directory / "run"
            main(training_args(root, run_dir, "--epochs", "2", "--train-on-base-design", "true",
                               "--change-locality", "true"))

            samples = pd.read_csv(run_dir / "local_edit_samples.csv")
            base = samples[samples["design"] == "D6"]
            self.assertEqual(base["sample"].tolist(), [1, 2, 3])
            self.assertEqual(set(base["split"]), {"train"})
            splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(splits["train"]["design_ids"], [0, 1, 2])
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["data"]["local_edits"]["train_on_base_design"])
            self.assertTrue(config["architecture"]["kwargs"]["change_locality"])

            # Edit scores and baselines see edited designs only; design 6's fit is reported apart.
            report = json.loads((run_dir / "local_edit_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["splits"]["train"]["designs"], ["N1", "N2"])
            self.assertEqual(report["splits"]["train"]["runs"], 4)
            self.assertEqual(report["splits"]["train"]["base_design_fit"]["runs"], 3)
            self.assertNotIn("base_design_fit", report["splits"]["test"])
            self.assertNotIn("D6", set(pd.read_csv(run_dir / "local_edit_runs.csv")["design"]))
            table = DesignChangeTable.load(run_dir / "design_change_nodes.npz")
            self.assertEqual(int(table.counts[list(table.designs).index(0)]), 0)

            # Geometry and impact in, nothing else: design 6 and an edit predict from the saved run.
            predictor = HistoryPredictor.from_run(run_dir)
            geometry, _, _ = LocalEditGeometry.from_manifest(root / "manifest_local_edits.csv", root / "inp_files")
            for design in (0, 4):
                self.assertTrue(np.isfinite(predictor.predict(geometry.mesh(design), [150.0, 0.5])).all())

            # A resume keeps the option; design 6 can never be held out.
            resumed = parse_args(["--resume-from", str(run_dir), "--output-dir", str(directory / "resumed"),
                                  "--device", "cpu", "--wandb-mode", "disabled"])
            self.assertTrue(resumed.train_on_base_design)
            self.assertTrue(resumed.change_locality)
            with self.assertRaisesRegex(ValueError, "cannot be held out"):
                resolve_splits({"run_numbers": [1, 143, 285]}, [0], [1])

    def test_switch_values(self):
        from train_mesh_impact_history import parse_args
        with tempfile.TemporaryDirectory() as directory:
            base = ["--device", "cpu", "--output-dir", directory, "--wandb-mode", "disabled"]
            args = parse_args(base)
            self.assertEqual((args.train_on_base_design, args.change_locality), (False, False))
            args = parse_args(base + ["--train-on-base-design", "true", "--change-locality", "false"])
            self.assertEqual((args.train_on_base_design, args.change_locality), (True, False))
            with self.assertRaises(SystemExit):
                parse_args(base + ["--change-locality", "maybe"])


@unittest.skipUnless(PILOT.is_dir(), "B1-B4 pilot data not on this machine")
class BenchmarkLayoutTest(unittest.TestCase):
    """The pilot families (base + 5 edits each) read as designs of this model, with one fixed load."""

    def family(self, folder):
        designs = sorted(path for path in folder.iterdir() if (path / f"{path.name}_mesh.npz").is_file())
        designs.sort(key=lambda path: path.name != "base")
        meshes, changes = [], {}
        for number, path in enumerate(designs):
            saved = np.load(path / f"{path.name}_mesh.npz")
            pad = lambda array: np.pad(array, ((0, 0), (0, 3 - array.shape[1])))  # noqa: E731  2D -> z = 0
            meshes.append(pad(saved["xyz"]).astype(np.float32))
            changes[number] = {"static": np.concatenate([pad(saved["change_base_xyz"]),
                                                         pad(saved["change_displacement"])], axis=1)}
        return [path.name for path in designs], meshes, changes

    def test_base_is_a_design_without_change_nodes_and_edits_act(self):
        for folder in sorted(PILOT.iterdir()):
            if not folder.is_dir():
                continue
            with self.subTest(benchmark=folder.name):
                names, meshes, changes = self.family(folder)
                self.assertEqual(names[0], "base")
                self.assertEqual(len(changes[0]["static"]), 0)
                self.assertTrue(all(len(changes[n]["static"]) > 0 for n in range(1, len(names))))
                stacked = np.concatenate(meshes)
                mean, scale = stacked.mean(0), np.maximum(stacked.std(0), 1e-6)
                torch.manual_seed(2)
                backbone = MeshImpactHistoryNet(**KWARGS).eval()
                model = DesignChangeHistoryNet(**KWARGS).eval()  # change_locality off: one fixed load
                model.load_state_dict(backbone.state_dict(), strict=False)
                randomize_readouts(model)
                for net in (backbone, model):
                    net.set_coordinate_scalers(mean, scale, [0.0, 0.0], [1.0, 1.0])
                model.design_changes = DesignChangeTable.from_changes(changes)
                model.set_change_displacement_scale(change_displacement_rms(changes, list(changes)))
                normalized = [torch.as_tensor((mesh - mean) / scale) for mesh in meshes]
                for number, mesh in enumerate(normalized):
                    model.design_changes.register(mesh, number)
                load = torch.zeros(1, 2)  # the same (absent) load location for every design
                time = torch.linspace(-1, 1, 5)
                with torch.no_grad():
                    for number, mesh in enumerate(normalized):
                        args = (mesh, torch.zeros(len(mesh), dtype=torch.long), load, time,
                                torch.zeros(len(time), dtype=torch.long))
                        difference = (model(*args, batch_size=1) - backbone(*args, batch_size=1)).abs().max().item()
                        if number == 0:
                            self.assertEqual(difference, 0.0)
                        else:
                            self.assertGreater(difference, 1e-4)


if __name__ == "__main__":
    unittest.main()
