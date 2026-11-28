"""run_mesh_change_local_edits_delta.sbatch passes options the trainer accepts, equal to its defaults."""
import re
import shlex
from pathlib import Path

from train_mesh_impact_history import MODEL_DEFAULTS, parse_args

LAUNCHER = Path(__file__).resolve().parents[1] / "run_mesh_change_local_edits_delta.sbatch"


def launcher_arguments(**variables):
    """The launcher's fresh-run ARGS array, with its shell variables substituted."""
    block = re.search(r"\n    ARGS=\(\n(.*?)\n    \)\n", LAUNCHER.read_text(encoding="utf-8"), re.S).group(1)
    block = block.replace('"${COMMON[@]}"', "")
    values = {"DATA_ROOT": "/data", "TEST_DESIGN": "54", "VAL_DESIGN": "55", "CHANGE_FEATURES": "edit",
              "TRAIN_ON_BASE": "false", "CHANGE_LOCALITY": "false", "SEED": "42", **variables}
    return shlex.split(re.sub(r"\$\{(\w+)\}", lambda match: values[match.group(1)], block))


def test_launcher_options_parse_and_equal_the_trainer_defaults(tmp_path):
    common = ["--device", "cpu", "--wandb-mode", "disabled", "--output-dir", str(tmp_path)]
    args = parse_args(launcher_arguments() + common)
    assert {key: getattr(args, key) for key in MODEL_DEFAULTS} == MODEL_DEFAULTS
    assert (args.test_designs, args.val_designs) == ([54], [55])
    assert (args.epochs, args.batch_size, args.lr, args.weight_decay, args.seed, args.time_subsample_stride) == (
        100, 8, 3e-4, 1e-5, 42, 16)
    assert args.impact_coords_path == "/data/manifest_local_edits.csv"
    assert parse_args(launcher_arguments(CHANGE_FEATURES="xyz") + common).design_change_features == "xyz"
    # The launcher's defaults are the best geometry-only model (run 24537698): both new options off.
    assert (args.train_on_base_design, args.change_locality) == (False, False)
    switched = parse_args(launcher_arguments(TRAIN_ON_BASE="true", CHANGE_LOCALITY="true", SEED="43") + common)
    assert (switched.train_on_base_design, switched.change_locality, switched.seed) == (True, True, 43)
