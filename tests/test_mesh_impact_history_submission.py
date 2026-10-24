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
    def test_defaults_hold_out_whole_cluster_B_and_keep_temporal_model(self):
        for parse in (preflight.parse_args, training.parse_args):
            args = parse([])
            self.assertEqual(args.decoder, "temporal")
            self.assertEqual(args.test_designs, [4, 5])
            self.assertEqual(args.val_designs, [11])
            self.assertEqual(args.time_encoding, "fourier")
            self.assertEqual(args.fourier_num_frequencies, 6)
        splits = preflight.validate_splits([4, 5], [11])
        self.assertEqual(splits["train"]["design_ids"], [0, 1, 2, 3, 6, 7, 8, 9, 10])
        self.assertEqual(len(splits["train"]["run_numbers"]), 1278)
        self.assertEqual(splits["validation"]["run_numbers"], list(range(1563, 1705)))
        self.assertEqual(splits["test"]["run_numbers"], list(range(569, 853)))
        held_clusters, warnings = preflight.validate_cluster_holdout([4, 5], [11])
        self.assertEqual(held_clusters, ["B"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("cluster D", warnings[0])

    def test_rejects_test_geometry_clones_in_training(self):
        for test, validation in (([2], [10]), ([11], [5]), ([4], [11]), ([5], [11])):
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
    train_mesh_impact_history.py|train_temporal_deeponet_1704.py) exit "${MESH_TEST_TRAIN_STATUS:-0}" ;;
esac
""")
        self.env.update(MESH_TEST_PREFLIGHT_STATUS="0", MESH_TEST_TRAIN_STATUS="0")
        self.stub("sbatch", """printf 'DECODER=%s\\nTEST=%s\\nVAL=%s\\nLEAK=%s\\n' \\
    "$HOOD_MESH_DECODER" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_ALLOW_CLONE_LEAK"
printf 'LOCAL=%s\\nK=%s\\nDIFFERENCE=%s\\n' "$HOOD_MESH_NEIGHBORHOOD_LAYERS" "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT"
printf 'RESUME=%s\\n' "$HOOD_MESH_RESUME_FROM"
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

    def test_temporal_deeponet_submission_forwards_slurm_options(self):
        output = self.launch("submit_temporal_deeponet_1704.sh", ["--time=00:30:00"])
        for expected in ("ARG=--time=00:30:00", "run_temporal_deeponet_1704.sbatch",
                         "temporal_deeponet_1704_%j.out", "ARG=--export=ALL"):
            self.assertIn(expected, output)

    def test_temporal_deeponet_ignores_old_model_exports_and_preserves_gpu(self):
        self.env.update(HOOD_MESH_RESUME_FROM="old-run", HOOD_MESH_TEST_DESIGNS="2",
                        HOOD_MESH_VAL_DESIGNS="10", HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="1",
                        HOOD_MESH_EPOCHS="900", CUDA_VISIBLE_DEVICES="GPU-allocated-by-slurm",
                        HOOD_TD_EPOCHS="7", HOOD_TD_SEED="19", HOOD_TD_DATA_ROOT="/data/with spaces")
        output = self.launch("run_temporal_deeponet_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        for expected in ("train_temporal_deeponet_1704.py", "<--test-designs> <4> <5> <--val-designs> <11>",
                         "<--epochs> <7>", "<--seed> <19>", "<--data-root> </data/with spaces>",
                         "CUDA_VISIBLE_DEVICES=GPU-allocated-by-slurm", "<--device> <cuda>"):
            self.assertIn(expected, command)
        for unwanted in ("--resume-from", "--design-difference-weight", "--decoder", "900"):
            self.assertNotIn(unwanted, command)

    def test_temporal_deeponet_failure_propagates_to_slurm(self):
        self.env["MESH_TEST_TRAIN_STATUS"] = "23"
        output = self.launch("run_temporal_deeponet_1704.sbatch", expected_returncode=23)
        self.assertNotIn("Finished.", output)

    def test_ablation_launcher_pins_hardest_split_despite_old_environment(self):
        self.env.update(HOOD_MESH_DECODER="temporal", HOOD_MESH_TEST_DESIGNS="2",
                        HOOD_MESH_VAL_DESIGNS="10", HOOD_MESH_ALLOW_CLONE_LEAK="1")
        output = self.launch("submit_mesh_impact_history_1704_no_temporal.sh", ["--time=00:10:00"])
        for expected in ("DECODER=mesh_only", "TEST=10 11", "VAL=5", "LEAK=0",
                         "ARG=--time=00:10:00", "run_mesh_impact_history_1704.sbatch"):
            self.assertIn(expected, output)

    def test_cluster_launcher_defaults_to_hardest_temporal_model(self):
        output = self.launch("submit_mesh_impact_history_1704_clusterD.sh")
        for expected in ("DECODER=temporal", "TEST=10 11", "VAL=5"):
            self.assertIn(expected, output)

    def test_cluster_B_launcher_overrides_old_split_but_accepts_model_options(self):
        self.env.update(HOOD_MESH_TEST_DESIGNS="10 11", HOOD_MESH_VAL_DESIGNS="5",
                        HOOD_MESH_ALLOW_CLONE_LEAK="1", HOOD_MESH_NEIGHBORHOOD_LAYERS="2")
        output = self.launch("submit_mesh_impact_history_1704_clusterB.sh", ["--time=08:00:00"])
        for expected in ("DECODER=temporal", "TEST=4 5", "VAL=11", "LEAK=0", "LOCAL=2",
                         "ARG=--job-name=mesh_hist_clusterB", "ARG=--time=08:00:00"):
            self.assertIn(expected, output)

    def test_local_sensitivity_launcher_pins_both_changes_and_cluster_B_split(self):
        self.env.update(HOOD_MESH_DECODER="mesh_only", HOOD_MESH_TEST_DESIGNS="2",
                        HOOD_MESH_VAL_DESIGNS="10", HOOD_MESH_ALLOW_CLONE_LEAK="1",
                        HOOD_MESH_NEIGHBORHOOD_LAYERS="0", HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="0")
        output = self.launch("submit_mesh_impact_history_1704_local_sensitivity.sh", ["--time=00:30:00"])
        for expected in ("DECODER=temporal", "TEST=4 5", "VAL=11", "LEAK=0",
                         "LOCAL=2", "K=16", "ARG=--time=08:00:00", "ARG=--time=00:30:00"):
            self.assertIn(expected, output)
        self.assertLess(output.index("ARG=--time=08:00:00"), output.index("ARG=--time=00:30:00"))

    def test_hic_filtered_launcher_and_training_argument(self):
        for threshold in ("10", "20"):
            if threshold == "20":
                self.env["HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT"] = threshold
            output = self.launch("submit_mesh_impact_history_1704_hic_filtered_k256.sh")
            for expected in ("LOCAL=2", "K=256", "TEST=4 5", "VAL=11",
                             f">= {threshold}%", "ARG=--job-name=mesh_hist_hic_filtered_k256"):
                self.assertIn(expected, output)
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertNotIn("--hic-range-threshold-percent", commands[0])
        self.assertIn("<--hic-range-threshold-percent> <20>", commands[1])

    def test_attention_submission_and_post_training_export(self):
        self.stub("sbatch", 'printf "EXPORT=%s\\nHIC=%s\\nSPLIT=%s\\nSUMMARY=%s\\nRUNS=%s\\n" "$HOOD_MESH_EXPORT_ATTENTION" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" "$HOOD_MESH_ATTENTION_SPLIT" "$HOOD_MESH_ATTENTION_SUMMARY_ONLY" "$HOOD_MESH_ATTENTION_RUNS"; printf "ARG=%s\\n" "$@"\n')
        self.env.update(HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="10", HOOD_MESH_ATTENTION_SPLIT="train",
                        HOOD_MESH_ATTENTION_SUMMARY_ONLY="1", HOOD_MESH_ATTENTION_RUNS="572")
        output = self.launch("submit_mesh_impact_history_1704_attention_export.sh")
        self.assertIn("EXPORT=1", output)
        for expected in ("HIC=30", "SPLIT=test", "SUMMARY=0",
                         "RUNS=577 622 670 693 710 719 764 812 835 852"):
            self.assertIn(expected, output)
        self.assertIn("ARG=--time=08:00:00", output)
        self.env.update(HOOD_MESH_EXPORT_ATTENTION="1", HOOD_MESH_ATTENTION_SPLIT="test",
                        HOOD_MESH_ATTENTION_RUNS="572 714", HOOD_MESH_ATTENTION_SUMMARY_ONLY="1")
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 3)
        self.assertIn("train_mesh_impact_history.py", commands[1])
        self.assertIn("export_mesh_pooling_attention.py", commands[2])
        for expected in ("<--split> <test>", "<--summary-only>", "<--runs> <572> <714>"):
            self.assertIn(expected, commands[2])
        self.env["MESH_TEST_TRAIN_STATUS"] = "23"
        failed = self.launch("run_mesh_impact_history_1704.sbatch", expected_returncode=23)
        self.assertNotIn("export_mesh_pooling_attention.py", failed)

    def test_fresh_cluster_B_launchers_cannot_silently_resume_an_old_split(self):
        self.env["HOOD_MESH_RESUME_FROM"] = "old_cluster_D_run"
        for script in ("submit_mesh_impact_history_1704_clusterB.sh",
                       "submit_mesh_impact_history_1704_local_sensitivity.sh"):
            output = self.launch(script, expected_returncode=2)
            self.assertNotIn("ARG=", output)

    def test_resume_launcher_passes_source_and_dependency_with_8_hours(self):
        source = self.root / "old run with spaces"
        source.mkdir()
        checkpoint = source / "hood_impact_best_model.pt"
        checkpoint.touch()
        for selected in (source, checkpoint):
            output = self.launch("submit_mesh_impact_history_1704_local_sensitivity_resume.sh",
                                 [self.shell_path(selected), "--dependency=afterany:12345"])
            for expected in ("ARG=--time=08:00:00", "ARG=--dependency=afterany:12345",
                             "ARG=--job-name=mesh_hist_local_resume", f"RESUME={self.shell_path(selected)}"):
                self.assertIn(expected, output)

    def test_resume_launcher_rejects_missing_source_without_submitting(self):
        for arguments in ([], [self.shell_path(self.root / "missing")]):
            output = self.launch("submit_mesh_impact_history_1704_local_sensitivity_resume.sh",
                                 arguments, expected_returncode=2)
            self.assertNotIn("ARG=", output)

    def test_resume_batch_uses_saved_settings_instead_of_stale_training_exports(self):
        source = self.shell_path(self.root / "old run")
        self.env.update(HOOD_MESH_RESUME_FROM=source, HOOD_MESH_EPOCHS="999",
                        HOOD_MESH_WIDTH="256", HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="0",
                        HOOD_MESH_NEIGHBOR_CACHE_DIR="/shared/neighbor-graphs")
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 2)
        self.assertIn(f"<--resume-from> <{source}>", commands[0])
        self.assertIn(f"<--resume-from> <{source}>", commands[1])
        for command in commands:
            self.assertIn("<--neighbor-cache-dir> </shared/neighbor-graphs>", command)
        for option in ("--epochs", "--width", "--design-difference-weight", "--test-designs"):
            self.assertNotIn(f"<{option}>", commands[1])
        self.assertIn("<--output-dir>", commands[1])
        self.assertNotIn("Test designs:", output)
        self.assertIn("split and architecture restored", output)

    def test_new_options_reach_both_direct_preflight_and_training(self):
        self.env.update(HOOD_MESH_NEIGHBORHOOD_LAYERS="2", HOOD_MESH_NEIGHBORHOOD_K="16",
                        HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="1.0", HOOD_MESH_BATCH_SIZE="8",
                        HOOD_MESH_IMPACT_CONDITIONING="film",
                        HOOD_MESH_TIME_ENCODING="fourier", HOOD_MESH_FOURIER_NUM_FREQUENCIES="8",
                        HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="10", HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="800",
                        HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4", HOOD_MESH_MAX_TRAIN_TIME="0.02",
                        HOOD_MESH_NEIGHBOR_CACHE_DIR="/shared/neighbor-graphs",
                        HOOD_MESH_NEIGHBORHOOD_SCALE_MM="20", HOOD_MESH_NEIGHBORHOOD_CHUNK_SIZE="1024",
                        HOOD_MESH_POOLING_QUERY_RESIDUAL="false", HOOD_MESH_POOL_IMPACTOR_NODES="false")
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertNotIn("--design-difference-weight", command)
            self.assertIn("<--pooling-query-residual> <false>", command)
            self.assertIn("<--pool-impactor-nodes> <false>", command)
            for expected in ("<--neighborhood-layers> <2>", "<--neighborhood-k> <16>",
                             "<--batch-size> <8>", "<--impact-conditioning> <film>",
                             "<--time-encoding> <fourier>", "<--fourier-num-frequencies> <8>",
                             "<--fourier-min-frequency-hz> <10>", "<--fourier-max-frequency-hz> <800>",
                             "<--time-subsample-stride> <4>", "<--max-train-time> <0.02>",
                             "<--neighbor-cache-dir> </shared/neighbor-graphs>",
                             "<--neighborhood-scale-mm> <20>", "<--neighborhood-chunk-size> <1024>"):
                self.assertIn(expected, command)

    def test_fourier_launcher_pins_encoding_and_retains_the_filtered_k256_experiment(self):
        self.stub("sbatch", """printf 'TIME=%s\\nBANDS=%s\\nFMIN=%s\\nFMAX=%s\\nHIC=%s\\nK=%s\\nTEST=%s\\nVAL=%s\\nNAME=%s\\nIMPACT=%s\\n' \\
    "$HOOD_MESH_TIME_ENCODING" "$HOOD_MESH_FOURIER_NUM_FREQUENCIES" "$HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ" \\
    "$HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" "$HOOD_MESH_NEIGHBORHOOD_K" \\
    "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_RUN_NAME" "$HOOD_MESH_IMPACT_CONDITIONING"
printf 'ARG=%s\\n' "$@"
""")
        self.env.update(HOOD_MESH_TIME_ENCODING="legacy_five", HOOD_MESH_IMPACT_CONDITIONING="legacy_additive",
                        HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30")
        output=self.launch("submit_mesh_impact_history_1704_fourier_k256.sh", ["--time=12:00:00"])
        for expected in ("TIME=fourier", "IMPACT=film", "BANDS=6", "FMIN=20", "FMAX=640", "HIC=30", "K=256",
                         "TEST=4 5", "VAL=11", "NAME=mesh_history_clusterB_k256_fourier_hic30pct",
                         "ARG=--job-name=mesh_hist_fourier_k256", "ARG=--time=12:00:00"):
            self.assertIn(expected,output)

    def test_dense_fourier_launcher_uses_all_locations_and_250_queries(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'STRIDE=%s\\nHIC=%s\\nNAME=%s\\nTIME=%s\\nBATCH=%s\\nTRACKING=%s\\n' \\
    "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" \\
    "$HOOD_MESH_RUN_NAME" "$HOOD_MESH_TIME_ENCODING" "$HOOD_MESH_BATCH_SIZE" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        self.env.update(HOOD_MESH_TIME_SUBSAMPLE_STRIDE="16", HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30")
        output = self.launch("submit_mesh_impact_history_1704_fourier_k256_t250.sh")
        for expected in ("STRIDE=4", "HIC=0", "TIME=fourier", "BATCH=8",
                         "TRACKING=online",
                         "NAME=mesh_history_clusterB_k256_fourier_hic0pct_t250",
                         "ARG=--job-name=mesh_hist_fourier_k256_t250", "ARG=--time=24:00:00"):
            self.assertIn(expected, output)
        self.env["WANDB_MODE"] = "offline"
        self.assertIn("TRACKING=offline", self.launch("submit_mesh_impact_history_1704_fourier_k256_t250.sh"))

    def test_no_shortcut_launcher_closes_both_paths_and_otherwise_matches_the_20261004_run(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'RESIDUAL=%s\\nPOOL_HEADFORM=%s\\nIMPACTOR=%s\\nSTRIDE=%s\\nHIC=%s\\nCUTOFF=[%s]\\n' \\
    "$HOOD_MESH_POOLING_QUERY_RESIDUAL" "$HOOD_MESH_POOL_IMPACTOR_NODES" "$HOOD_MESH_IMPACTOR_NODES" \\
    "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" "$HOOD_MESH_MAX_TRAIN_TIME"
printf 'K=%s\\nLAYERS=%s\\nTIME=%s\\nIMPACT=%s\\nTEST=%s\\nVAL=%s\\nEPOCHS=%s\\nBATCH=%s\\nNAME=%s\\nTRACKING=%s\\n' \\
    "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_NEIGHBORHOOD_LAYERS" "$HOOD_MESH_TIME_ENCODING" \\
    "$HOOD_MESH_IMPACT_CONDITIONING" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" \\
    "$HOOD_MESH_EPOCHS" "$HOOD_MESH_BATCH_SIZE" "$HOOD_MESH_RUN_NAME" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        # Stale exports from other experiments must not leak into the comparison.
        self.env.update(HOOD_MESH_POOLING_QUERY_RESIDUAL="true", HOOD_MESH_POOL_IMPACTOR_NODES="true",
                        HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4", HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30",
                        HOOD_MESH_MAX_TRAIN_TIME="0.02", HOOD_MESH_NEIGHBORHOOD_K="16",
                        HOOD_MESH_IMPACTOR_NODES="0")
        output = self.launch("submit_mesh_impact_history_1704_no_location_shortcuts.sh")
        for expected in ("RESIDUAL=false", "POOL_HEADFORM=false", "IMPACTOR=286", "STRIDE=16", "HIC=0",
                         "CUTOFF=[]", "K=256", "LAYERS=2", "TIME=fourier", "IMPACT=film", "TEST=4 5",
                         "VAL=11", "EPOCHS=100", "BATCH=8", "TRACKING=online",
                         "NAME=mesh_history_clusterB_k256_fourier_hic0pct_no_shortcuts",
                         "ARG=--job-name=mesh_hist_no_shortcuts", "ARG=--time=24:00:00"):
            self.assertIn(expected, output)
        self.assertIn("mesh_history_1704_no_location_shortcuts_%j.out", output)
        # Our job name, wall time and log names come after the chained defaults, so sbatch uses them.
        arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
        self.assertGreater(arguments.index("--time=24:00:00"), arguments.index("--time=08:00:00"))
        self.assertGreater(arguments.index("--job-name=mesh_hist_no_shortcuts"),
                           arguments.index("--job-name=mesh_hist_clusterB_k256"))

    def test_location_holdout_launcher_adds_the_split_to_the_no_shortcut_settings(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'HOLDOUT=%s\\nRESIDUAL=%s\\nPOOL_HEADFORM=%s\\nIMPACTOR=%s\\nSTRIDE=%s\\nHIC=%s\\n' \\
    "$HOOD_MESH_LOCATION_HOLDOUT" "$HOOD_MESH_POOLING_QUERY_RESIDUAL" "$HOOD_MESH_POOL_IMPACTOR_NODES" \\
    "$HOOD_MESH_IMPACTOR_NODES" "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT"
printf 'K=%s\\nTIME=%s\\nTEST=%s\\nVAL=%s\\nEPOCHS=%s\\nNAME=%s\\nTRACKING=%s\\n' \\
    "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_TIME_ENCODING" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" \\
    "$HOOD_MESH_EPOCHS" "$HOOD_MESH_RUN_NAME" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        self.env.update(HOOD_MESH_LOCATION_HOLDOUT="/stale/split.json", HOOD_MESH_POOLING_QUERY_RESIDUAL="true",
                        HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4", HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30",
                        HOOD_MESH_TEST_DESIGNS="10 11")
        output = self.launch("submit_mesh_impact_history_1704_location_holdout.sh")
        split = self.shell_path(self.repo / "location_splits" / "euroncap1704_checkerboard.json")
        for expected in (f"HOLDOUT={split}", "RESIDUAL=false", "POOL_HEADFORM=false", "IMPACTOR=286",
                         "STRIDE=16", "HIC=0", "K=256", "TIME=fourier", "TEST=4 5", "VAL=11", "EPOCHS=100",
                         "NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_location_holdout",
                         "TRACKING=online", "ARG=--job-name=mesh_hist_location_holdout", "ARG=--time=16:00:00"):
            self.assertIn(expected, output)
        arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
        self.assertGreater(arguments.index("--time=16:00:00"), arguments.index("--time=08:00:00"))
        # The batch job passes the split to training only; preflight checks designs, not locations.
        self.env["HOOD_MESH_LOCATION_HOLDOUT"] = "/split dir/split.json"
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertNotIn("--location-holdout", commands[0])
        self.assertIn("<--location-holdout> </split dir/split.json>", commands[1])

    def test_random70_launcher_adds_the_per_design_split_to_the_no_shortcut_settings(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'HOLDOUT=%s\\nFILM=%s\\nOCTAVES=[%s]\\nRESIDUAL=%s\\nPOOL_HEADFORM=%s\\n' \\
    "$HOOD_MESH_LOCATION_HOLDOUT" "$HOOD_MESH_GEOMETRY_FILM" "$HOOD_MESH_GEOMETRY_FOURIER_OCTAVES" \\
    "$HOOD_MESH_POOLING_QUERY_RESIDUAL" "$HOOD_MESH_POOL_IMPACTOR_NODES"
printf 'IMPACTOR=%s\\nSTRIDE=%s\\nHIC=%s\\nK=%s\\nTEST=%s\\nVAL=%s\\nNAME=%s\\nTRACKING=%s\\n' \\
    "$HOOD_MESH_IMPACTOR_NODES" "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" \\
    "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_RUN_NAME" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        self.env.update(HOOD_MESH_LOCATION_HOLDOUT="/stale/split.json", HOOD_MESH_GEOMETRY_FILM="all",
                        HOOD_MESH_GEOMETRY_FOURIER_OCTAVES="5", HOOD_MESH_POOLING_QUERY_RESIDUAL="true",
                        HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4", HOOD_MESH_TEST_DESIGNS="10 11")
        output = self.launch("submit_mesh_impact_history_1704_random70.sh")
        split = self.shell_path(self.repo / "location_splits" / "euroncap1704_random70_seed42.json")
        for expected in (f"HOLDOUT={split}", "FILM=none", "OCTAVES=[]", "RESIDUAL=false", "POOL_HEADFORM=false",
                         "IMPACTOR=286", "STRIDE=16", "HIC=0", "K=256", "TEST=4 5", "VAL=11",
                         "NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_random70", "TRACKING=online",
                         "ARG=--job-name=mesh_hist_random70", "ARG=--time=16:00:00"):
            self.assertIn(expected, output)
        self.assertIn("mesh_history_1704_random70_%j.out", output)
        arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
        self.assertGreater(arguments.index("--time=16:00:00"), arguments.index("--time=08:00:00"))

    def test_geometry_film_launcher_adds_only_the_injection_to_the_no_shortcut_settings(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'FILM=%s\\nOCTAVES=%s\\nRESIDUAL=%s\\nPOOL_HEADFORM=%s\\nIMPACTOR=%s\\n' \\
    "$HOOD_MESH_GEOMETRY_FILM" "$HOOD_MESH_GEOMETRY_FOURIER_OCTAVES" "$HOOD_MESH_POOLING_QUERY_RESIDUAL" \\
    "$HOOD_MESH_POOL_IMPACTOR_NODES" "$HOOD_MESH_IMPACTOR_NODES"
printf 'STRIDE=%s\\nHIC=%s\\nCUTOFF=[%s]\\nHOLDOUT=[%s]\\nK=%s\\nLAYERS=%s\\nTIME=%s\\nIMPACT=%s\\n' \\
    "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" "$HOOD_MESH_MAX_TRAIN_TIME" \\
    "$HOOD_MESH_LOCATION_HOLDOUT" "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_NEIGHBORHOOD_LAYERS" \\
    "$HOOD_MESH_TIME_ENCODING" "$HOOD_MESH_IMPACT_CONDITIONING"
printf 'TEST=%s\\nVAL=%s\\nEPOCHS=%s\\nBATCH=%s\\nSEED=%s\\nNAME=%s\\nTRACKING=%s\\n' \\
    "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_EPOCHS" "$HOOD_MESH_BATCH_SIZE" \\
    "$HOOD_MESH_SEED" "$HOOD_MESH_RUN_NAME" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        # Stale exports from other experiments must not leak into the comparison.
        self.env.update(HOOD_MESH_POOLING_QUERY_RESIDUAL="true", HOOD_MESH_POOL_IMPACTOR_NODES="true",
                        HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4", HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30",
                        HOOD_MESH_MAX_TRAIN_TIME="0.02", HOOD_MESH_LOCATION_HOLDOUT="/stale/split.json",
                        HOOD_MESH_NEIGHBORHOOD_K="16", HOOD_MESH_TEST_DESIGNS="10 11",
                        HOOD_MESH_GEOMETRY_FOURIER_OCTAVES="9")
        output = self.launch("submit_mesh_impact_history_1704_geometry_film.sh")
        for expected in ("FILM=all", "OCTAVES=5", "RESIDUAL=false", "POOL_HEADFORM=false", "IMPACTOR=286",
                         "STRIDE=16", "HIC=0", "CUTOFF=[]", "HOLDOUT=[]", "K=256", "LAYERS=2", "TIME=fourier",
                         "IMPACT=film", "TEST=4 5", "VAL=11", "EPOCHS=100", "BATCH=8", "SEED=42",
                         "NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_geometry_film_all",
                         "TRACKING=online", "ARG=--job-name=mesh_hist_geometry_film", "ARG=--time=24:00:00"):
            self.assertIn(expected, output)
        self.assertIn("mesh_history_1704_geometry_film_%j.out", output)
        arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
        self.assertGreater(arguments.index("--time=24:00:00"), arguments.index("--time=08:00:00"))
        self.assertGreater(arguments.index("--job-name=mesh_hist_geometry_film"),
                           arguments.index("--job-name=mesh_hist_clusterB_k256"))
        self.env["HOOD_MESH_GEOMETRY_FILM"] = "latent"
        output = self.launch("submit_mesh_impact_history_1704_geometry_film.sh")
        self.assertIn("FILM=latent", output)
        self.assertIn("NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_geometry_film_latent", output)
        for rejected in ("none", "decoder"):
            self.env["HOOD_MESH_GEOMETRY_FILM"] = rejected
            output = self.launch("submit_mesh_impact_history_1704_geometry_film.sh", expected_returncode=2)
            self.assertNotIn("ARG=", output)
        # The batch job hands the mode to both preflight and training.
        self.env.update(HOOD_MESH_GEOMETRY_FILM="all", HOOD_MESH_GEOMETRY_FOURIER_OCTAVES="5")
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("<--geometry-film> <all>", command)
            self.assertIn("<--geometry-fourier-octaves> <5>", command)
        self.assertIn("Geometry memory FiLM: all", output)

    def test_design_change_launcher_adds_only_the_change_path_to_the_no_shortcut_settings(self):
        self.env.pop("WANDB_MODE", None)
        self.stub("sbatch", """printf 'CHANGE=%s\\nCHANGE_LAYERS=%s\\nCHANGE_OCTAVES=%s\\nFILM=%s\\n' \\
    "$HOOD_MESH_DESIGN_CHANGE_NODES" "$HOOD_MESH_DESIGN_CHANGE_LAYERS" "$HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES" \\
    "$HOOD_MESH_GEOMETRY_FILM"
printf 'RESIDUAL=%s\\nPOOL_HEADFORM=%s\\nIMPACTOR=%s\\nSTRIDE=%s\\nHIC=%s\\nCUTOFF=[%s]\\nHOLDOUT=[%s]\\n' \\
    "$HOOD_MESH_POOLING_QUERY_RESIDUAL" "$HOOD_MESH_POOL_IMPACTOR_NODES" "$HOOD_MESH_IMPACTOR_NODES" \\
    "$HOOD_MESH_TIME_SUBSAMPLE_STRIDE" "$HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT" "$HOOD_MESH_MAX_TRAIN_TIME" \\
    "$HOOD_MESH_LOCATION_HOLDOUT"
printf 'K=%s\\nLAYERS=%s\\nTIME=%s\\nIMPACT=%s\\nTEST=%s\\nVAL=%s\\nEPOCHS=%s\\nBATCH=%s\\nSEED=%s\\n' \\
    "$HOOD_MESH_NEIGHBORHOOD_K" "$HOOD_MESH_NEIGHBORHOOD_LAYERS" "$HOOD_MESH_TIME_ENCODING" \\
    "$HOOD_MESH_IMPACT_CONDITIONING" "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_EPOCHS" \\
    "$HOOD_MESH_BATCH_SIZE" "$HOOD_MESH_SEED"
printf 'NAME=%s\\nTRACKING=%s\\n' "$HOOD_MESH_RUN_NAME" "$WANDB_MODE"
printf 'ARG=%s\\n' "$@"
""")
        # Stale exports from other experiments must not leak into the comparison.
        self.env.update(HOOD_MESH_GEOMETRY_FILM="all", HOOD_MESH_POOLING_QUERY_RESIDUAL="true",
                        HOOD_MESH_POOL_IMPACTOR_NODES="true", HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4",
                        HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="30", HOOD_MESH_MAX_TRAIN_TIME="0.02",
                        HOOD_MESH_LOCATION_HOLDOUT="/stale/split.json", HOOD_MESH_NEIGHBORHOOD_K="16",
                        HOOD_MESH_TEST_DESIGNS="10 11")
        output = self.launch("submit_mesh_impact_history_1704_design_change_nodes.sh")
        for expected in ("CHANGE=true", "CHANGE_LAYERS=2", "CHANGE_OCTAVES=5", "FILM=none", "RESIDUAL=false",
                         "POOL_HEADFORM=false", "IMPACTOR=286", "STRIDE=16", "HIC=0", "CUTOFF=[]", "HOLDOUT=[]",
                         "K=256", "LAYERS=2", "TIME=fourier", "IMPACT=film", "TEST=4 5", "VAL=11", "EPOCHS=100",
                         "BATCH=8", "SEED=42",
                         "NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_nodes",
                         "TRACKING=online", "ARG=--job-name=mesh_hist_design_change", "ARG=--time=14:00:00"):
            self.assertIn(expected, output)
        self.assertIn("mesh_history_1704_design_change_nodes_%j.out", output)
        arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
        self.assertGreater(arguments.index("--time=14:00:00"), arguments.index("--time=08:00:00"))
        # The batch job hands the change path to both preflight and training.
        self.env.update(HOOD_MESH_DESIGN_CHANGE_NODES="true", HOOD_MESH_DESIGN_CHANGE_LAYERS="2",
                        HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="5")
        output = self.launch("run_mesh_impact_history_1704.sbatch")
        commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("<--design-change-nodes> <true>", command)
            self.assertIn("<--design-change-layers> <2>", command)
            self.assertIn("<--design-change-fourier-octaves> <5>", command)
        self.assertIn("Design-change nodes: true", output)

    def test_design_change_sweep_variants_pin_only_their_own_settings(self):
        self.stub("sbatch", """printf 'CHANGE=%s\\nLAYERS=%s\\nOCTAVES=%s\\nDROPOUT=%s\\nWD=%s\\n' \\
    "$HOOD_MESH_DESIGN_CHANGE_NODES" "$HOOD_MESH_DESIGN_CHANGE_LAYERS" "$HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES" \\
    "$HOOD_MESH_DROPOUT" "$HOOD_MESH_WEIGHT_DECAY"
printf 'EPOCHS=%s\\nBATCH=%s\\nLR=%s\\nSEED=%s\\nK=%s\\nTEST=%s\\nVAL=%s\\nNAME=%s\\n' \\
    "$HOOD_MESH_EPOCHS" "$HOOD_MESH_BATCH_SIZE" "$HOOD_MESH_LR" "$HOOD_MESH_SEED" "$HOOD_MESH_NEIGHBORHOOD_K" \\
    "$HOOD_MESH_TEST_DESIGNS" "$HOOD_MESH_VAL_DESIGNS" "$HOOD_MESH_RUN_NAME"
printf 'ARG=%s\\n' "$@"
""")
        # Stale exports from other experiments must not leak into the sweep.
        self.env.update(HOOD_MESH_DROPOUT="0.4", HOOD_MESH_WEIGHT_DECAY="0.3", HOOD_MESH_LR="0.01",
                        HOOD_MESH_SEED="7", HOOD_MESH_EPOCHS="5", HOOD_MESH_RUN_NAME="stale",
                        HOOD_MESH_DESIGN_CHANGE_LAYERS="9", HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="9")
        variants = {"small": ("0", "2", "0.1", "0.00001", "mesh_hist_dc_small"),
                    "regularized": ("2", "5", "0.2", "0.05", "mesh_hist_dc_reg"),
                    "small_regularized": ("0", "2", "0.2", "0.05", "mesh_hist_dc_small_reg")}
        for variant, (layers, octaves, dropout, decay, job) in variants.items():
            with self.subTest(variant=variant):
                output = self.launch(f"submit_mesh_impact_history_1704_design_change_{variant}.sh")
                for expected in ("CHANGE=true", f"LAYERS={layers}", f"OCTAVES={octaves}", f"DROPOUT={dropout}",
                                 f"WD={decay}", "EPOCHS=100", "BATCH=8", "LR=0.0003", "SEED=42", "K=256",
                                 "TEST=4 5", "VAL=11",
                                 f"NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_{variant}",
                                 f"mesh_history_1704_design_change_{variant}_%j.out"):
                    self.assertIn(expected, output)
                arguments = [line[4:] for line in output.splitlines() if line.startswith("ARG=")]
                job_names = [argument for argument in arguments if argument.startswith("--job-name=")]
                self.assertEqual(job_names[-1], f"--job-name={job}")
        output = self.launch("submit_mesh_impact_history_1704_design_change_sweep.sh")
        self.assertEqual(output.count("CHANGE=true"), 3)
        for variant in variants:
            self.assertIn(f"NAME=mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_{variant}", output)

    def test_batch_runs_directly_with_broken_srun_and_preserves_gpu_decoder_and_split(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-allocated-by-slurm"
        for decoder in ("", "temporal", "mesh_only"):
            with self.subTest(decoder=decoder):
                self.env["HOOD_MESH_DECODER"] = decoder
                output = self.launch("run_mesh_impact_history_1704.sbatch")
                commands = [line for line in output.splitlines() if line.startswith("PYTHON")]
                self.assertEqual(len(commands), 2, output)
                self.assertIn("preflight_mesh_impact_history_1704.py", commands[0])
                self.assertIn("train_mesh_impact_history.py", commands[1])
                for command in commands:
                    self.assertIn(f"<--decoder> <{decoder or 'temporal'}>", command)
                    self.assertIn("<--test-designs> <4> <5> <--val-designs> <11>", command)
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

    def test_explicit_dense_preprocessing_overrides_shared_default(self):
        args = preflight.parse_args(["--time-subsample-stride", "4"])
        data, source_count, cutoff_count = preflight.validate_first_run(
            self.root, self.coords.to_numpy(dtype=np.float32), args)
        self.assertEqual((source_count, cutoff_count, len(data["time_arrays"][0])), (1000, 1000, 250))
        np.testing.assert_array_equal(data["accelerations"][0], self.history["A(in g)"].to_numpy()[::4])

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


class ImpactorBoundaryTests(unittest.TestCase):
    """The held-out block must be measured against the files, not assumed."""

    def write_runs(self, root, moving):
        (root / "inp_files").mkdir()
        (root / "output_history_acc").mkdir()
        for run in (1, 2):
            rows = []
            for node in range(6):
                shift = run * 50 if node in moving else 0
                rows.append(f"{node + 1}, {node + shift}, 0.0, 0.0")
            body = chr(10).join(rows)
            (root / "inp_files" / f"HoodImpact_{run}.inp").write_text(f"*NODE{chr(10)}{body}{chr(10)}*ELEMENT{chr(10)}")

    def test_accepts_a_leading_block_that_matches_the_moving_nodes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_runs(root, moving={0, 1})
            preflight.validate_impactor_boundary(root, 2)

    def test_rejects_a_count_that_does_not_match_the_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_runs(root, moving={0, 1})
            for wrong in (1, 3):
                with self.assertRaisesRegex(ValueError, "does not match the data"):
                    preflight.validate_impactor_boundary(root, wrong)

    def test_rejects_moving_nodes_that_are_not_the_leading_block(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_runs(root, moving={3, 4})
            with self.assertRaisesRegex(ValueError, "does not match the data"):
                preflight.validate_impactor_boundary(root, 2)

    def test_launcher_and_preflight_agree_on_the_1704_headform(self):
        launcher = Path("submit_mesh_impact_history_1704_local_sensitivity.sh").read_text()
        self.assertIn('export HOOD_MESH_IMPACTOR_NODES="286"', launcher)
        self.assertIn("IMPACTOR_NODES:impactor-nodes",
                      Path("run_mesh_impact_history_1704.sbatch").read_text())
        for parse in (preflight.parse_args, training.parse_args):
            self.assertEqual(parse([]).impactor_nodes, 0)
            self.assertEqual(parse(["--impactor-nodes", "286"]).impactor_nodes, 286)

    def test_location_shortcut_switches_parse_true_false_and_default_off(self):
        sbatch = Path("run_mesh_impact_history_1704.sbatch").read_text()
        for name in ("POOLING_QUERY_RESIDUAL:pooling-query-residual", "POOL_IMPACTOR_NODES:pool-impactor-nodes"):
            self.assertIn(name, sbatch)
        for parse in (preflight.parse_args, training.parse_args):
            defaults = parse([])
            self.assertIs(defaults.pooling_query_residual, False)
            self.assertIs(defaults.pool_impactor_nodes, False)
            for text, expected in (("true", True), ("false", False), ("1", True), ("0", False)):
                with self.subTest(parser=parse.__module__, text=text):
                    parsed = parse(["--pooling-query-residual", text, "--pool-impactor-nodes", text])
                    self.assertIs(parsed.pooling_query_residual, expected)
                    self.assertIs(parsed.pool_impactor_nodes, expected)
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                parse(["--pool-impactor-nodes", "maybe"])

    def test_headform_must_be_identified_unless_pooling_it_on_purpose(self):
        with self.assertRaisesRegex(ValueError, "location shortcut"):
            training.validate_location_shortcuts(0, False)
        training.validate_location_shortcuts(286, False)
        training.validate_location_shortcuts(0, True)
        with mock.patch("builtins.print"), mock.patch("sys.stderr"):
            self.assertEqual(preflight.main(["--impactor-nodes", "0"]), 1)
        archived = Path("submit_mesh_impact_history_1704_no_temporal.sh").read_text()
        self.assertIn('export HOOD_MESH_POOL_IMPACTOR_NODES="true"', archived)
