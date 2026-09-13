"""Checks for complete 1704-job inputs and the launcher's preflight contract."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import preflight_mesh_impact_history_1704 as preflight
import train_mesh_impact_history as training


class SubmissionSplitTests(unittest.TestCase):
    def test_defaults_match_hardest_existing_split_and_ablation(self):
        for parse in (preflight.parse_args, training.parse_args):
            args = parse([])
            self.assertEqual(args.decoder, "mesh_only")
            self.assertEqual(args.test_designs, [10, 11])
            self.assertEqual(args.val_designs, [5])
        splits = preflight.validate_splits([10, 11], [5])
        self.assertEqual(splits["train"]["design_ids"], [0, 1, 2, 3, 4, 6, 7, 8, 9])
        self.assertEqual(len(splits["train"]["run_numbers"]), 1278)
        self.assertEqual(splits["validation"]["run_numbers"], list(range(711, 853)))
        self.assertEqual(splits["test"]["run_numbers"], list(range(1421, 1705)))
        held_clusters, warnings = preflight.validate_cluster_holdout([10, 11], [5])
        self.assertEqual(held_clusters, ["D"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("cluster B", warnings[0])

    def test_rejects_test_geometry_clones_in_training(self):
        for test, validation in (([2], [10]), ([11], [5])):
            with self.subTest(test=test, validation=validation):
                with self.assertRaisesRegex(ValueError, "near-clones in training"):
                    preflight.validate_cluster_holdout(test, validation)

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


class SubmissionLauncherTests(unittest.TestCase):
    """Run the real launchers with local stand-ins for Slurm and modules."""

    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash unavailable")
        temporary = tempfile.TemporaryDirectory(prefix="mesh_launcher_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("HOOD_MESH_")}
        self.env["SLURM_SUBMIT_DIR"] = self.shell_path(self.repo)
        venv_bin = self.root / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "activate").write_text(":\n", encoding="utf-8")
        self.env["HOOD_MESH_VENV"] = self.shell_path(venv_bin.parent)
        for name in ("module", "nvidia-smi"):
            self.stub(name, ":\n")
        # Reproduce job 3139891: srun cannot start a task, although the batch
        # shell itself can execute Python on its allocated GPU.
        self.stub("srun", "echo 'srun: error: task 0 launch failed: Error configuring interconnect' >&2\nexit 89\n")
        self.stub("python", """printf 'PYTHON'; printf ' <%s>' "$@"
printf ' CUDA_VISIBLE_DEVICES=%s\\n' "${CUDA_VISIBLE_DEVICES-unset}"
case "$2" in
    preflight_mesh_impact_history_1704.py) exit "${MESH_TEST_PREFLIGHT_STATUS:-0}" ;;
    train_mesh_impact_history.py) exit "${MESH_TEST_TRAIN_STATUS:-0}" ;;
esac
""")
        self.env.update(MESH_TEST_PREFLIGHT_STATUS="0", MESH_TEST_TRAIN_STATUS="0")
        self.stub("sbatch", """printf 'DECODER=%s\\nTEST=%s\\nVAL=%s\\nLEAK=%s\\n' \\
    "$HOOD_MESH_DECODER" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_ALLOW_CLONE_LEAK"
printf 'ARG=%s\\n' "$@"
""")

    @staticmethod
    def shell_path(path):
        text = path.as_posix()
        return f"/{text[0].lower()}{text[2:]}" if os.name == "nt" else text

    def stub(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def launch(self, script, extra_args=(), expected_returncode=0):
        result = subprocess.run(
            [self.bash, "-c", 'export PATH="$1:$PATH"; shift; bash "$@"',
             "launcher-test", self.shell_path(self.bin_dir),
             self.shell_path(self.repo / script), *extra_args],
            env=self.env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, expected_returncode, result.stdout + result.stderr)
        return result.stdout

    def test_ablation_launcher_pins_hardest_split_despite_old_environment(self):
        self.env.update(HOOD_MESH_DECODER="temporal", HOOD_MESH_TEST_DESIGNS="2",
                        HOOD_MESH_VAL_DESIGNS="10", HOOD_MESH_ALLOW_CLONE_LEAK="1")
        output = self.launch("submit_mesh_impact_history_1704_no_temporal.sh", ["--time=00:10:00"])
        for expected in ("DECODER=mesh_only", "TEST=10 11", "VAL=5", "LEAK=0",
                         "ARG=--time=00:10:00", "run_mesh_impact_history_1704.sbatch"):
            self.assertIn(expected, output)

    def test_cluster_launcher_defaults_to_hardest_ablation(self):
        output = self.launch("submit_mesh_impact_history_1704_clusterD.sh")
        for expected in ("DECODER=mesh_only", "TEST=10 11", "VAL=5"):
            self.assertIn(expected, output)

    def test_batch_runs_directly_with_broken_srun_and_preserves_gpu_decoder_and_split(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-allocated-by-slurm"
        for decoder in ("", "temporal"):
            with self.subTest(decoder=decoder):
                self.env["HOOD_MESH_DECODER"] = decoder
                output = self.launch("run_mesh_impact_history_1704.sbatch")
                commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
                self.assertEqual(len(commands), 2, output)
                self.assertIn("preflight_mesh_impact_history_1704.py", commands[0])
                self.assertIn("train_mesh_impact_history.py", commands[1])
                for command in commands:
                    self.assertIn(f"<--decoder> <{decoder or 'mesh_only'}>", command)
                    self.assertIn("<--test-designs> <10> <11> <--val-designs> <5>", command)
                    self.assertIn("CUDA_VISIBLE_DEVICES=GPU-allocated-by-slurm", command)
                self.assertIn("Finished. Checkpoints, metrics and acceleration histories:", output)

    def test_failed_direct_preflight_stops_before_training(self):
        self.env["MESH_TEST_PREFLIGHT_STATUS"] = "17"
        output = self.launch("run_mesh_impact_history_1704.sbatch", expected_returncode=17)
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 1, output)
        self.assertIn("preflight_mesh_impact_history_1704.py", commands[0])
        self.assertNotIn("Training command:", output)
        self.assertNotIn("Finished.", output)

    def test_direct_training_failure_is_the_batch_job_exit_status(self):
        self.env["MESH_TEST_TRAIN_STATUS"] = "23"
        output = self.launch("run_mesh_impact_history_1704.sbatch", expected_returncode=23)
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 2, output)
        self.assertIn("train_mesh_impact_history.py", commands[1])
        self.assertNotIn("Finished.", output)


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
