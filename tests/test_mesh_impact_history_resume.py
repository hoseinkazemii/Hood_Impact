"""Resume actual optimization after interruption, including legacy checkpoints and configs."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toy_local_edits import training_args, write_dataset
from train_mesh_impact_history import (
    CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, DataPreprocessor, main, parse_args,
)
from utils.utils import Trainer


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("toy") / "data")


def run_args(dataset, path, *extra):
    return training_args(dataset, path, "--epochs", "4", "--dropout", "0.1", *extra)


def resume_args(source, output):
    return ["--resume-from", str(source), "--output-dir", str(output), "--device", "cpu", "--wandb-mode", "disabled"]


def interrupt_after_two_epochs(dataset, path, extra_args=()):
    original = Trainer.save_checkpoint

    def save_then_interrupt(trainer, filename):
        original(trainer, filename)
        if Path(filename).name == LAST_CHECKPOINT_NAME and len(trainer.train_losses) == 2:
            raise RuntimeError("simulated job timeout")

    with mock.patch.object(Trainer, "save_checkpoint", save_then_interrupt):
        with pytest.raises(RuntimeError, match="simulated job timeout"):
            main(run_args(dataset, path, *extra_args))


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_training_state_and_finishes_original_target(tmp_path, dataset, legacy):
    source, resumed, baseline = (tmp_path / name for name in ("source", "resumed", "baseline"))
    interrupt_after_two_epochs(dataset, source)
    selected = source / (CHECKPOINT_NAME if legacy else LAST_CHECKPOINT_NAME)
    checkpoint = torch.load(selected, weights_only=True)
    completed = len(checkpoint["train_losses"])
    if legacy:
        for key in ("completed_epochs", "torch_rng_state", "cuda_rng_states", "numpy_rng_state",
                    "python_rng_state"):
            checkpoint.pop(key, None)
        torch.save(checkpoint, selected)
        (source / LAST_CHECKPOINT_NAME).unlink()
    source_bytes = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}

    original_load = Trainer.load_checkpoint

    def load_and_verify(trainer, filename):
        restored = original_load(trainer, filename)
        assert len(trainer.train_losses) == completed
        assert trainer.scheduler.state_dict() == checkpoint["scheduler_state_dict"]
        assert trainer.optimizer.state_dict()["param_groups"] == checkpoint["optimizer_state_dict"]["param_groups"]
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model_state_dict"][key], atol=0, rtol=0)
        for index, state in trainer.optimizer.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, checkpoint["optimizer_state_dict"]["state"][index][key], atol=0, rtol=0)
        return restored

    with mock.patch.object(Trainer, "load_checkpoint", load_and_verify), mock.patch.object(
            DataPreprocessor, "fit_scalers", side_effect=AssertionError("Resume must reuse saved scalers")):
        main(resume_args(source, resumed))
    assert source_bytes == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    history = json.loads((resumed / "training_history.json").read_text())
    assert len(history["train_losses"]) == len(history["val_losses"]) == 4
    assert history["train_losses"][:completed] == checkpoint["train_losses"]
    saved = json.loads((resumed / "config.json").read_text())
    assert saved["training"]["resumed_after_epoch"] == completed
    assert saved["training"]["initialization"] == "checkpoint"
    assert saved["architecture"]["kwargs"]["neighborhood_layers"] == 2
    assert saved["training"]["loss"] == "normalized_acceleration_mse"
    assert saved["preprocessing"]["time_subsample_stride"] == 1
    for name in ("test_acceleration_histories.csv", "local_edit_metrics.json", "design_change_nodes.npz"):
        assert (resumed / name).is_file(), name
    final = torch.load(resumed / LAST_CHECKPOINT_NAME, weights_only=True)
    assert final["completed_epochs"] == final["scheduler_state_dict"]["last_epoch"] == 4
    if not legacy:
        # Fresh checkpoint resume must reproduce uninterrupted optimization,
        # including dropout RNG, shuffled batches.
        main(run_args(dataset, baseline))
        expected = torch.load(baseline / LAST_CHECKPOINT_NAME, weights_only=True)
        assert final["train_losses"] == expected["train_losses"]
        assert final["val_losses"] == expected["val_losses"]
        for key, value in final["model_state_dict"].items():
            torch.testing.assert_close(value, expected["model_state_dict"][key], atol=0, rtol=0)


def test_format_1_config_of_an_earlier_run_resumes(tmp_path, dataset):
    """Runs started before 2026-11-27 saved every option of the time; they still resume."""
    source, resumed = tmp_path / "source", tmp_path / "resumed"
    interrupt_after_two_epochs(dataset, source)
    path = source / "config.json"
    saved = json.loads(path.read_text())
    saved["format_version"] = 1
    saved["architecture"]["kwargs"].update(
        decoder="temporal", impact_conditioning="film", time_encoding="fourier", impactor_nodes=0,
        pooling_query_residual=False, pool_impactor_nodes=False, geometry_film="none",
        geometry_fourier_octaves=5, teacher_forcing_chunks=0, design_change_layers=0,
        design_change_fourier_octaves=2, design_change_displacement_gate=False)
    path.write_text(json.dumps(saved))
    assert parse_args(["--resume-from", str(source)]).design_change_features == "edit"
    main(resume_args(source, resumed))
    restored = json.loads((resumed / "config.json").read_text())
    assert restored["format_version"] == 2
    assert "decoder" not in restored["architecture"]["kwargs"]

    saved["architecture"]["kwargs"]["teacher_forcing_chunks"] = 3
    path.write_text(json.dumps(saved))
    with pytest.raises(SystemExit):
        parse_args(["--resume-from", str(source)])


def test_resume_rejects_changed_settings_splits_and_existing_output(tmp_path, dataset):
    source = tmp_path / "source"
    interrupt_after_two_epochs(dataset, source)
    for option, value in (("--epochs", "100"), ("--time-subsample-stride", "16"),
                          ("--neighborhood-layers", "0"), ("--design-change-features", "xyz"),
                          ("--test-designs", "3")):
        with pytest.raises(SystemExit):
            parse_args(["--resume-from", str(source), option, value])
    with pytest.raises(FileExistsError):
        main(["--resume-from", str(source), "--output-dir", str(source), "--device", "cpu"])
    config = SimpleNamespace(inp_dir=str(dataset / "inp_files"),
                             impact_coords_path=str(dataset / "manifest_local_edits.csv"),
                             acceleration_dir=str(dataset / "output_history_acc"),
                             time_subsample_stride=1, max_train_time=None)
    changed = DataPreprocessor(config).load_all_data()
    changed["run_numbers"] = list(changed["run_numbers"])
    changed["run_numbers"][0] = 3  # Shift a training run to another design.
    with mock.patch.object(DataPreprocessor, "load_all_data", return_value=changed):
        with pytest.raises(ValueError, match="differ from the saved resume split"):
            main(resume_args(source, tmp_path / "changed"))


def test_resume_keeps_its_saved_split_and_settings(tmp_path, dataset):
    source = tmp_path / "source"
    interrupt_after_two_epochs(dataset, source)
    assert parse_args([]).test_designs == [54]
    resumed = parse_args(["--resume-from", str(source), "--batch-size", "2"])
    assert (resumed.test_designs, resumed.val_designs) == ([4], [3])
    assert resumed.resume_splits["train"]["design_ids"] == [1, 2]
    assert (resumed.epochs, resumed.width, resumed.neighborhood_k) == (4, 16, 3)
    assert resumed.inp_dir == str((dataset / "inp_files").resolve())


def test_checkpoint_write_failure_preserves_previous_file(tmp_path):
    path = tmp_path / LAST_CHECKPOINT_NAME
    path.write_bytes(b"previous complete checkpoint")
    trainer = mock.Mock()
    trainer.checkpoint_state.return_value = {}

    def fail_write(state, filename):
        Path(filename).write_bytes(b"partial next checkpoint")
        raise OSError("disk write interrupted")

    with mock.patch("torch.save", side_effect=fail_write), pytest.raises(OSError, match="interrupted"):
        Trainer.save_checkpoint(trainer, str(path))
    assert path.read_bytes() == b"previous complete checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()


def test_resume_rejects_former_combined_objective(tmp_path, dataset):
    source = tmp_path / "source"
    interrupt_after_two_epochs(dataset, source)
    path = source / "config.json"
    saved = json.loads(path.read_text())
    saved["training"]["loss"] = "normalized_acceleration_mse_plus_design_difference_mse"
    path.write_text(json.dumps(saved))
    with pytest.raises(SystemExit):
        parse_args(["--resume-from", str(source)])


def test_removed_options_are_not_accepted():
    for option, value in (("--design-difference-weight", "0"), ("--data-format", "euroncap1704"),
                          ("--teacher-forcing-chunks", "3"), ("--decoder", "mesh_only"),
                          ("--design-change-nodes", "true"), ("--mesh-nodes", "all")):
        with pytest.raises(SystemExit):
            parse_args([option, value])


def test_resume_recovers_timeout_between_best_and_latest_writes(tmp_path, dataset):
    source = tmp_path / "source"
    original = Trainer.save_checkpoint

    def interrupt_after_best(trainer, filename):
        original(trainer, filename)
        if Path(filename).name == CHECKPOINT_NAME and len(trainer.train_losses) == 2:
            raise RuntimeError("timeout between checkpoint writes")

    with mock.patch.object(Trainer, "save_checkpoint", interrupt_after_best), mock.patch.object(
            Trainer, "validate", side_effect=[(2.0, 1.0), (1.0, 1.0)]):
        with pytest.raises(RuntimeError, match="between checkpoint writes"):
            main(run_args(dataset, source))
    assert len(torch.load(source / LAST_CHECKPOINT_NAME, weights_only=True)["train_losses"]) == 1
    args = parse_args(["--resume-from", str(source)])
    assert Path(args.resume_from).name == CHECKPOINT_NAME
    resumed = tmp_path / "resumed"
    main(resume_args(source, resumed))
    saved = json.loads((resumed / "config.json").read_text())
    assert saved["training"]["resumed_after_epoch"] == 2
