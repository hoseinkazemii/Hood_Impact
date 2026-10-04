"""Execute the correction launchers with local stand-ins; never submit a job."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


BASELINE_FILES = (
    "hood_impact_best_model.pt", "config.json", "scalers.joblib",
    "splits.json", "prediction_times.npy",
)
SCRIPTS = ("submit_mesh_geometry_correction_1704.sh", "run_mesh_geometry_correction_1704.sbatch")


class GeometryCorrectionSubmissionTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash unavailable")
        temporary = tempfile.TemporaryDirectory(prefix="geometry correction launcher ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "project with spaces"
        self.repo.mkdir()
        source = Path(__file__).resolve().parents[1]
        for script in SCRIPTS:
            shutil.copy2(source / script, self.repo / script)
        self.bin = self.root / "mock bin"
        self.bin.mkdir()
        self.baseline = self.repo / "runs/mesh_impact_history/20260930_204155_3283115_1704"
        self.create_baseline(self.baseline)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("HOOD_MESH_", "HOOD_CORRECTION_", "CORRECTION_TEST_", "WANDB_", "SLURM_"))}
        self.env["SLURM_SUBMIT_DIR"] = self.shell_path(self.repo)
        venv_bin = self.root / "venv with spaces" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "activate").write_text(":\n", encoding="utf-8", newline="\n")
        self.env["HOOD_MESH_VENV"] = self.shell_path(venv_bin.parent)
        self.stub("module", ":\n")
        self.stub("nvidia-smi", ":\n")
        self.stub("git", "printf 'mock-commit\\n'\n")
        self.stub("python", "printf 'PYTHON_ARG=%s\\n' \"$@\"\n"
                  "printf 'GPU=%s\\n' \"${CUDA_VISIBLE_DEVICES:-unset}\"\n"
                  "exit \"${CORRECTION_TEST_TRAIN_STATUS:-0}\"\n")
        self.stub("sbatch", "printf 'MOCK_SUBMISSION\\n'\n"
                  "printf 'BASELINE=%s\\n' \"${HOOD_CORRECTION_BASELINE_RUN}\"\n"
                  "printf 'SBATCH_ARG=%s\\n' \"$@\"\n"
                  "exit \"${CORRECTION_TEST_SUBMIT_STATUS:-0}\"\n")
        self.stub("srun", "echo 'ERROR: extra srun step used' >&2\nexit 89\n")

    @staticmethod
    def shell_path(path):
        value = Path(path).as_posix()
        return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

    @staticmethod
    def create_baseline(path):
        path.mkdir(parents=True)
        for name in BASELINE_FILES:
            (path / name).write_text("mock artifact\n", encoding="utf-8")

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def launch(self, script, extra_args=(), returncode=0):
        result = subprocess.run(
            [self.bash, "-c", 'export PATH="$1:$PATH"; shift; bash "$@"',
             "correction-launcher-test", self.shell_path(self.bin),
             self.shell_path(self.repo / script), *extra_args],
            env=self.env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, returncode, result.stdout + result.stderr)
        return result

    @staticmethod
    def python_args(output):
        return [line.removeprefix("PYTHON_ARG=") for line in output.splitlines()
                if line.startswith("PYTHON_ARG=")]

    def test_bash_syntax_and_lf_line_endings(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                path = self.repo / name
                self.assertNotIn(b"\r", path.read_bytes())
                result = subprocess.run([self.bash, "-n", self.shell_path(path)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrapper_restores_default_baseline_and_forwards_flags_without_splitting(self):
        result = self.launch(SCRIPTS[0], ("--time=00:30:00", "--comment=geometry correction trial"))
        arguments = [line.removeprefix("SBATCH_ARG=") for line in result.stdout.splitlines()
                     if line.startswith("SBATCH_ARG=")]
        self.assertIn(f"BASELINE={self.shell_path(self.baseline)}", result.stdout)
        self.assertIn(f"--chdir={self.shell_path(self.repo)}", arguments)
        self.assertIn("--export=ALL", arguments)
        self.assertEqual(arguments[-3:], ["--time=00:30:00", "--comment=geometry correction trial",
                                          self.shell_path(self.repo / SCRIPTS[1])])

    def test_wrapper_accepts_alternate_baseline_with_spaces(self):
        alternate = self.root / "alternate baseline run"
        self.create_baseline(alternate)
        self.env["HOOD_CORRECTION_BASELINE_RUN"] = self.shell_path(alternate)
        result = self.launch(SCRIPTS[0])
        self.assertIn(f"BASELINE={self.shell_path(alternate)}", result.stdout)

    def test_each_required_artifact_is_checked_before_submission(self):
        for artifact in BASELINE_FILES:
            with self.subTest(artifact=artifact):
                path = self.baseline / artifact
                path.unlink()
                try:
                    result = self.launch(SCRIPTS[0], returncode=2)
                    self.assertIn(artifact, result.stderr)
                    self.assertNotIn("MOCK_SUBMISSION", result.stdout)
                finally:
                    path.write_text("mock artifact\n", encoding="utf-8")

    def test_submission_failure_propagates(self):
        self.env["CORRECTION_TEST_SUBMIT_STATUS"] = "19"
        self.launch(SCRIPTS[0], returncode=19)

    def test_job_defaults_match_python_cli_and_preserve_slurm_gpu(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-allocated-by-slurm"
        result = self.launch(SCRIPTS[1])
        arguments = self.python_args(result.stdout)
        self.assertEqual(arguments[:2], ["-u", "train_mesh_geometry_correction.py"])
        expected = {"--baseline-run": self.shell_path(self.baseline), "--epochs": "300",
                    "--batch-size": "64", "--lr": "1e-3", "--weight-decay": "0", "--seed": "42",
                    "--device": "cuda", "--wandb-project": "hood-impact-geometry-correction"}
        for option, value in expected.items():
            self.assertEqual(arguments[arguments.index(option) + 1], value)
        self.assertIn("GPU=GPU-allocated-by-slurm", result.stdout)
        # Parsing the actual mocked launch catches misspelled/unsupported flags.
        from train_mesh_geometry_correction import parse_args
        cpu_arguments = arguments[2:]
        cpu_arguments[cpu_arguments.index("--device") + 1] = "cpu"
        parsed = parse_args(cpu_arguments)
        self.assertEqual(parsed.epochs, 300)
        self.assertEqual(parsed.batch_size, 64)

    def test_job_forwards_new_settings_paths_and_ignores_old_experiment_exports(self):
        self.env.update(HOOD_CORRECTION_EPOCHS="7", HOOD_CORRECTION_BATCH_SIZE="4",
                        HOOD_CORRECTION_WIDTH="48", HOOD_CORRECTION_LAYERS="2",
                        HOOD_CORRECTION_ANCHOR_SPACING_MM="20", HOOD_CORRECTION_PROFILE_K="3",
                        HOOD_CORRECTION_GEOMETRY_NOISE_FLOOR_MM="0.2",
                        HOOD_CORRECTION_FOURIER_NUM_FREQUENCIES="4",
                        HOOD_CORRECTION_FOURIER_MIN_FREQUENCY_HZ="10",
                        HOOD_CORRECTION_FOURIER_MAX_FREQUENCY_HZ="80",
                        HOOD_CORRECTION_BASELINE_CHUNK_SIZE="64", HOOD_CORRECTION_NO_PLOTS="1",
                        HOOD_MESH_NEIGHBOR_CACHE_DIR="/cache/with spaces",
                        HOOD_CORRECTION_LR="0.002", HOOD_CORRECTION_WEIGHT_DECAY="0.01",
                        HOOD_CORRECTION_SEED="19", HOOD_CORRECTION_RUN_NAME="trial with spaces",
                        HOOD_CORRECTION_OUTPUT_DIR="/results/output with spaces",
                        HOOD_MESH_DATA_ROOT="/data/input with spaces", WANDB_MODE="disabled",
                        WANDB_ENTITY="team with spaces", HOOD_MESH_EPOCHS="999",
                        HOOD_MESH_TEST_DESIGNS="0 1", HOOD_MESH_DECODER="mesh_only",
                        HOOD_MESH_RESUME_FROM="old experiment", HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="1")
        result = self.launch(SCRIPTS[1])
        arguments = self.python_args(result.stdout)[2:]
        expected = {"--epochs": "7", "--batch-size": "4", "--correction-width": "48",
                    "--correction-layers": "2", "--anchor-spacing-mm": "20", "--profile-k": "3",
                    "--geometry-noise-floor-mm": "0.2", "--fourier-num-frequencies": "4",
                    "--fourier-min-frequency-hz": "10", "--fourier-max-frequency-hz": "80",
                    "--baseline-neighborhood-chunk-size": "64", "--neighbor-cache-dir": "/cache/with spaces",
                    "--lr": "0.002", "--weight-decay": "0.01", "--seed": "19",
                    "--run-name": "trial with spaces", "--output-dir": "/results/output with spaces",
                    "--data-root": "/data/input with spaces", "--wandb-mode": "disabled",
                    "--wandb-entity": "team with spaces"}
        for option, value in expected.items():
            self.assertEqual(arguments[arguments.index(option) + 1], value)
        for unwanted in ("--resume-from", "--test-designs", "--decoder", "--design-difference-weight", "999"):
            self.assertNotIn(unwanted, arguments)
        self.assertIn("--no-plots", arguments)
        from train_mesh_geometry_correction import parse_args
        arguments[arguments.index("--device") + 1] = "cpu"
        parsed = parse_args(arguments)
        self.assertEqual(parsed.correction_width, 48)
        self.assertEqual(parsed.fourier_num_frequencies, 4)

    def test_direct_job_checks_missing_baseline_before_python_or_module(self):
        (self.baseline / "splits.json").unlink()
        self.stub("module", "echo 'MODULE_WAS_CALLED'\n")
        result = self.launch(SCRIPTS[1], returncode=2)
        self.assertIn("splits.json", result.stderr)
        self.assertNotIn("PYTHON_ARG", result.stdout)
        self.assertNotIn("MODULE_WAS_CALLED", result.stdout)

    def test_training_failure_propagates_without_success_message(self):
        self.env["CORRECTION_TEST_TRAIN_STATUS"] = "23"
        result = self.launch(SCRIPTS[1], returncode=23)
        self.assertNotIn("Finished.", result.stdout)


if __name__ == "__main__":
    unittest.main()
