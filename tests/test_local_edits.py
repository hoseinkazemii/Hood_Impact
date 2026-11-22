"""Local-edit dataset: sample numbering, design-6 exclusion, meshes, change tables and metrics."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from local_edit_metrics import local_edit_report
from local_edits_data import (
    LocalEditGeometry, base_reference_curves, load_design_changes, load_local_edit_data,
    sample_design, sample_location, sample_number,
)
from mesh_design_change import register_design_changes
from train_mesh_impact_history import build_config, parse_args
from utils.utils import DataPreprocessor

# Headform (1-3), inner panel (20-23) and outer skin (10-12), interleaved.
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
TIMES = np.linspace(0.0, 0.007, 8)
# run, design, location; design 6 has 3 locations here, the edits 2 each.
RUNS = [(1, "D6", 1), (2, "D6", 2), (3, "D6", 3), (4, "N1", 1), (5, "N1", 2), (6, "N2", 1), (7, "N2", 2)]
CHANGES = {"N1": [(21, 1.0, 0.0, 0.0, 2.0)], "N2": [(22, 1.0, 1.0, 0.0, 3.0), (23, 0.0, 1.0, 0.0, 1.0)]}


def curve(run):
    return 50.0 * np.sin(np.pi * TIMES / TIMES[-1]) * (1.0 + 0.1 * run) + run


def write_dataset(root):
    root = Path(root)
    (root / "inp_files").mkdir()
    (root / "output_history_acc").mkdir()
    (root / "design_changes").mkdir()
    (root / "inp_files" / "HoodImpact_1.inp").write_text(DECK)
    pd.DataFrame([{"run": run, "design": design, "location": location, "role": "near",
                   "x_mm": 0.5 + location, "y_mm": 0.5, "dist_to_edit_mm": 10.0 * location}
                  for run, design, location in RUNS]).to_csv(root / "manifest_local_edits.csv", index=False)
    for run, _, _ in RUNS:
        pd.DataFrame({"Time": TIMES, "A(in g)": curve(run)}).to_csv(
            root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    for design, rows in CHANGES.items():
        pd.DataFrame(rows, columns=["label", "x", "y", "z_design6", "dz"]).to_csv(
            root / "design_changes" / f"{design}_changed_nodes.csv", index=False)
    # Both edits on one site, both shallower: N1 is N2's sibling.
    pd.DataFrame({"design": ["N1", "N2"], "kind": "single", "variant": "S", "dz_max_mm": [2.0, 3.0],
                  "dz_min_mm": [0.0, 0.0]}).to_csv(root / "local_edit_designs.csv", index=False)
    pd.DataFrame({"design": ["N1", "N2"], "site": ["s1", "s1"]}).to_csv(root / "local_edit_doe.csv", index=False)
    return root


def preprocessor_for(root, stride=1):
    config = SimpleNamespace(
        data_format="localedits", prediction_target="acceleration", mesh_nodes="inner_panel",
        inp_dir=str(root / "inp_files"), impact_coords_path=str(root / "manifest_local_edits.csv"),
        acceleration_dir=str(root / "output_history_acc"), time_subsample_stride=stride, max_train_time=None)
    return DataPreprocessor(config)


class LocalEditDataTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = write_dataset(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_sample_numbers_put_one_location_in_one_slot(self):
        self.assertEqual(sample_number(54, 9), 54 * 142 + 9)
        self.assertEqual((sample_design(54 * 142 + 9), sample_location(54 * 142 + 9)), (54, 9))
        self.assertEqual((sample_design(142), sample_location(142)), (0, 142))
        with self.assertRaises(ValueError):
            sample_number(1, 143)

    def test_loader_skips_design_6_and_adds_dz(self):
        data = DataPreprocessor.load_all_data(preprocessor_for(self.root))
        self.assertEqual(data["run_numbers"], [143, 144, 285, 286])
        self.assertEqual(data["source_runs"], [4, 5, 6, 7])
        base = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        expected_n1, expected_n2 = base.copy(), base.copy()
        expected_n1[1, 2] = 2.0
        expected_n2[2, 2], expected_n2[3, 2] = 3.0, 1.0
        np.testing.assert_array_equal(data["mesh_geometries"][0], expected_n1)
        np.testing.assert_array_equal(data["mesh_geometries"][2], expected_n2)
        # One shared, read-only mesh per design.
        self.assertIs(data["mesh_geometries"][0], data["mesh_geometries"][1])
        self.assertFalse(data["mesh_geometries"][0].flags.writeable)
        np.testing.assert_allclose(data["indentor_positions"][1], [2.5, 0.5])
        np.testing.assert_allclose(data["accelerations"][3], curve(7), rtol=1e-6)

    def test_base_reference_and_include_base(self):
        processor = preprocessor_for(self.root, stride=2)
        curves, times = base_reference_curves(processor)
        self.assertEqual(sorted(curves), [1, 2, 3])
        np.testing.assert_allclose(times, TIMES[::2], rtol=1e-6)
        np.testing.assert_allclose(curves[2], curve(2)[::2], rtol=1e-6)
        data = load_local_edit_data(processor, include_base=True)
        self.assertEqual(data["run_numbers"][:3], [1, 2, 3])

    def test_change_tables_register_against_the_meshes(self):
        changes, report = load_design_changes(self.root / "manifest_local_edits.csv", self.root / "inp_files", [1, 2])
        np.testing.assert_array_equal(changes[2]["rows"], [2, 3])
        np.testing.assert_array_equal(changes[2]["node_ids"], [22, 23])
        np.testing.assert_allclose(changes[2]["static"][:, :3], [[1, 1, 0], [0, 1, 0]])
        np.testing.assert_allclose(changes[2]["static"][:, 5], [3.0, 1.0])
        self.assertEqual(report["designs"]["N2"]["change_nodes"], 2)
        data = load_local_edit_data(preprocessor_for(self.root))
        model = SimpleNamespace(impactor_nodes=0, design_changes=None)
        summary = register_design_changes(model, changes, data["mesh_geometries"],
                                          [sample_design(s) for s in data["run_numbers"]], lambda mesh: mesh)
        self.assertEqual(summary["registered_hoods"], 2)
        self.assertEqual(model.design_changes.lookup(np.asarray(data["mesh_geometries"][2])), 1)

    def test_change_table_must_match_the_base_deck(self):
        pd.DataFrame([(21, 1.0, 0.0, 0.5, 2.0)], columns=["label", "x", "y", "z_design6", "dz"]).to_csv(
            self.root / "design_changes" / "N1_changed_nodes.csv", index=False)
        geometry, _, _ = LocalEditGeometry.from_manifest(self.root / "manifest_local_edits.csv",
                                                         self.root / "inp_files")
        with self.assertRaisesRegex(ValueError, "not design 6"):
            geometry.mesh(1)

    def test_report_scores_effects_against_design_6(self):
        processor = preprocessor_for(self.root)
        data = load_local_edit_data(processor)
        curves, times = base_reference_curves(processor)

        def split(indices, predictions):
            dataset = SimpleNamespace(run_numbers=[data["run_numbers"][i] for i in indices],
                                      time_arrays=[data["time_arrays"][i] for i in indices])
            truth = np.concatenate([data["accelerations"][i] for i in indices])
            return dataset, np.concatenate(predictions), truth

        n2_truth = [data["accelerations"][i] for i in (2, 3)]
        n2_base = [curves[1], curves[2]]
        cases = {"perfect": (n2_truth, 1.0), "copy_d6": (n2_base, 0.0)}
        for label, (prediction, expected) in cases.items():
            with self.subTest(label):
                report, runs, _ = local_edit_report({
                    "train": split((0, 1), [data["accelerations"][0], data["accelerations"][1]]),
                    "validation": split((2,), prediction[:1]),
                    "test": split((2, 3), prediction),
                }, curves, times, self.root / "manifest_local_edits.csv")
                test = report["splits"]["test"]
                self.assertAlmostEqual(test["difference_r2"], expected, places=6)
                # Sibling copy = design 6 + N1's true effect at the same location.
                effect_n1 = [data["accelerations"][i] - curves[i + 1] for i in (0, 1)]
                effect_n2 = [n2_truth[i] - n2_base[i] for i in (0, 1)]
                manual = 1 - sum(np.square(a - b).sum() for a, b in zip(effect_n1, effect_n2)) / sum(
                    np.square(b).sum() for b in effect_n2)
                self.assertAlmostEqual(test["baselines"]["sibling_copy"]["difference_r2"], manual, places=5)
                self.assertEqual(test["baselines"]["sibling_copy"]["runs"], 2)
                self.assertEqual(len(test["windows_difference_r2"]), 3)
                self.assertEqual(sorted(runs["split"].unique()), ["test", "train", "validation"])


class LocalEditConfigTest(unittest.TestCase):
    def test_localedits_needs_inner_panel_and_xyz_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            base = ["--data-format", "localedits", "--device", "cpu", "--output-dir", directory,
                    "--wandb-mode", "disabled"]
            config = build_config(parse_args(base + ["--design-change-nodes", "true"]))
            self.assertEqual((config.samples_per_design, config.num_samples, config.mesh_nodes),
                             (142, 1440, "inner_panel"))
            with self.assertRaisesRegex(ValueError, "inner_panel"):
                build_config(parse_args(base + ["--mesh-nodes", "all"]))
            with self.assertRaisesRegex(ValueError, "xyz"):
                build_config(parse_args(base + ["--design-change-nodes", "true",
                                                "--design-change-features", "impact_relative"]))


if __name__ == "__main__":
    unittest.main()
