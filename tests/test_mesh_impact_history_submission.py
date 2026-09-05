"""Checks for complete 1704-job inputs and the launcher's preflight contract."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import preflight_mesh_impact_history_1704 as preflight


class SubmissionSplitTests(unittest.TestCase):
    def test_complete_mapping_and_design_boundaries(self):
        splits = preflight.validate_splits([2], [10])
        self.assertEqual(len(splits["train"]["run_numbers"]), 1420)
        self.assertEqual(splits["test"]["run_numbers"], list(range(285, 427)))
        self.assertEqual(splits["validation"]["run_numbers"], list(range(1421, 1563)))
        for group in splits.values():
            for run in group["run_numbers"]:
                self.assertIn((run - 1) // 142, group["design_ids"])
        all_runs = [run for group in splits.values() for run in group["run_numbers"]]
        self.assertEqual(sorted(all_runs), list(range(1, 1705)))

    def test_multiple_held_out_designs(self):
        splits = preflight.validate_splits([0, 1, 2, 3], [4, 5])
        self.assertEqual(len(splits["train"]["run_numbers"]), 852)
        self.assertEqual(len(splits["test"]["run_numbers"]), 568)
        self.assertEqual(len(splits["validation"]["run_numbers"]), 284)

    def test_rejects_invalid_or_empty_splits(self):
        cases = [([], [10]), ([2], []), ([2], [2]), ([-1], [10]),
                 ([12], [10]), ([2, 2], [10]), ([True], [10]),
                 (list(range(11)), [11])]
        for test, validation in cases:
            with self.subTest(test=test, validation=validation):
                with self.assertRaises(ValueError):
                    preflight.validate_splits(test, validation)


class SubmissionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="mesh_history_preflight_")
        cls.root = Path(cls.temporary.name)
        (cls.root / "inp_files").mkdir()
        (cls.root / "output_history_acc").mkdir()
        for run in range(1, 1705):
            (cls.root / "inp_files" / f"HoodImpact_{run}.inp").write_text("*NODE\n1, 0, 0, 0\n2, 1, 2, 3\n*ELEMENT\n")
            (cls.root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv").write_text("Time,A(in g)\n0,1\n")
        cls.coords = pd.DataFrame({"X1": np.arange(1704), "X2": np.arange(1704) / 2})
        cls.coords_path = cls.root / "ImpactCoords_1704.csv"
        cls.coords.to_csv(cls.coords_path, index=False)
        cls.history_path = cls.root / "output_history_acc" / "HoodImpact_1_SAE1000_interp1000.csv"
        cls.history = pd.DataFrame({"Time": np.arange(1000) * 0.0001, "A(in g)": np.arange(1000) * 0.25})
        cls.history.to_csv(cls.history_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def tearDown(self):
        self.coords.to_csv(self.coords_path, index=False)
        self.history.to_csv(self.history_path, index=False)

    def test_accepts_xy_only_without_hic_or_generation_metadata(self):
        root, xy = preflight.validate_dataset(self.root)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(xy.shape, (1704, 2))
        np.testing.assert_array_equal(xy[-1], [1703, 851.5])

    def test_rejects_missing_exact_run_even_with_same_file_count(self):
        original = self.root / "inp_files" / "HoodImpact_1704.inp"
        wrong = original.with_name("HoodImpact_1705.inp")
        original.rename(wrong)
        try:
            with self.assertRaisesRegex(ValueError, "Missing file: .*HoodImpact_1704.inp"):
                preflight.validate_dataset(self.root)
        finally:
            wrong.rename(original)

    def test_rejects_empty_history(self):
        self.history_path.write_text("")
        with self.assertRaisesRegex(ValueError, "Empty file: .*HoodImpact_1_SAE1000_interp1000.csv"):
            preflight.validate_dataset(self.root)

    def test_rejects_bad_coordinate_table(self):
        bad_numeric = self.coords.astype(object)
        bad_numeric.loc[17, "X1"] = "invalid"
        bad_finite = self.coords.copy()
        bad_finite.loc[17, "X2"] = np.inf
        for coords in (self.coords.iloc[:-1], self.coords.drop(columns="X2"), bad_numeric, bad_finite):
            with self.subTest(rows=len(coords), columns=list(coords.columns)):
                coords.to_csv(self.coords_path, index=False)
                with self.assertRaises(ValueError):
                    preflight.validate_dataset(self.root)

    def test_first_run_uses_shared_configured_stride_and_cutoff(self):
        xy = self.coords.to_numpy(dtype=np.float32)
        for stride, cutoff in ((16, None), (7, 0.0201), (1, None)):
            with self.subTest(stride=stride, cutoff=cutoff):
                with mock.patch.object(preflight.Config, "time_subsample_stride", stride), mock.patch.object(
                    preflight.Config, "max_train_time", cutoff
                ):
                    data, source_count, cutoff_count = preflight.validate_first_run(self.root, xy)
                source = self.history["Time"].to_numpy(dtype=np.float32)
                acceleration = self.history["A(in g)"].to_numpy(dtype=np.float32)
                selected = np.ones(len(source), dtype=bool) if cutoff is None else source <= cutoff
                self.assertEqual(source_count, 1000)
                self.assertEqual(cutoff_count, int(selected.sum()))
                np.testing.assert_array_equal(data["time_arrays"][0], source[selected][::stride])
                np.testing.assert_array_equal(data["accelerations"][0], acceleration[selected][::stride])

    def test_rejects_nonmonotonic_source_before_subsampling_hides_it(self):
        history = self.history.copy()
        history.loc[1, "Time"] = history.loc[0, "Time"]
        history.to_csv(self.history_path, index=False)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            preflight.validate_first_run(self.root, self.coords.to_numpy(dtype=np.float32))

    def test_rejects_invalid_configured_stride(self):
        for stride in (0, -1, True, 1.5):
            with self.subTest(stride=stride), mock.patch.object(preflight.Config, "time_subsample_stride", stride):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    preflight.validate_first_run(self.root, self.coords.to_numpy(dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
