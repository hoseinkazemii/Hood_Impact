"""Inner-panel meshes: only the hood inner panel's nodes reach the network."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_mesh_impact_history import build_config, parse_args, validate_location_shortcuts
from utils.utils import DataPreprocessor, mesh_node_ids

# Headform (1, 2, 3), outer skin (10-12), inner panel (20-23) and hinge (30-32)
# nodes, interleaved so the loader must keep the file order of the inner rows.
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
30, 5.0, 5.0, 0.0
31, 6.0, 5.0, 0.0
32, 6.0, 6.0, 0.0
*ELEMENT, TYPE=S3R
100, 1, 2, 3
200, 10, 11, 12
400, 30, 31, 32
*ELEMENT, TYPE=S4
300, 20, 21, 22, 23
*ELSET, ELSET="Rigid Body1_2-1-3"
100
*ELSET, ELSET="Hood_Outer-1-2"
200
*ELSET, ELSET="Hood_Inner-1-2"
300
*ELSET, ELSET="Hinges-1-2"
400
"""


class InnerPanelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.inp_dir = Path(self.directory.name)
        (self.inp_dir / "HoodImpact_1.inp").write_text(DECK)

    def tearDown(self):
        self.directory.cleanup()

    def load(self, mesh_nodes):
        config = SimpleNamespace(data_format="euroncap1704", inp_dir=str(self.inp_dir),
                                 mesh_nodes=mesh_nodes, samples_per_design=1)
        return DataPreprocessor(config).load_mesh_geometry(1)

    def test_inner_panel_keeps_only_inner_nodes_in_file_order(self):
        np.testing.assert_array_equal(self.load("inner_panel"),
                                      [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertEqual(mesh_node_ids(self.inp_dir / "HoodImpact_1.inp", "inner_panel").tolist(),
                         [20, 21, 22, 23])

    def test_all_keeps_the_whole_node_block(self):
        self.assertEqual(len(self.load("all")), 13)
        self.assertEqual(mesh_node_ids(self.inp_dir / "HoodImpact_1.inp").tolist()[:3], [1, 2, 20])
        with self.assertRaisesRegex(ValueError, "mesh_nodes"):
            self.load("outer")


class MeshNodesCliTests(unittest.TestCase):
    def test_inner_panel_is_the_default_and_drops_the_headform_rows(self):
        args = parse_args(["--impactor-nodes", "286"])
        self.assertEqual(args.mesh_nodes, "inner_panel")
        self.assertEqual(args.impactor_nodes, 0)
        validate_location_shortcuts(args.impactor_nodes, args.pool_impactor_nodes, args.mesh_nodes)

    def test_whole_deck_keeps_the_headform_holdout(self):
        args = parse_args(["--mesh-nodes", "all", "--impactor-nodes", "286"])
        self.assertEqual(args.impactor_nodes, 286)
        with self.assertRaisesRegex(ValueError, "location shortcut"):
            validate_location_shortcuts(0, False, "all")

    def test_csv_meshes_default_to_all_and_cannot_select_the_inner_panel(self):
        self.assertEqual(parse_args(["--data-format", "legacy"]).mesh_nodes, "all")
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--data-format", "legacy", "--mesh-nodes", "inner_panel", "--output-dir", directory])
            with self.assertRaisesRegex(ValueError, "--mesh-nodes all"):
                build_config(args)


if __name__ == "__main__":
    unittest.main()
