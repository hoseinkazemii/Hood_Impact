"""Resume actual optimization after interruption, including legacy checkpoints."""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

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


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(14)
    data = {key: [] for key in ("run_numbers", "mesh_geometries", "indentor_positions", "time_arrays", "accelerations")}
    times = np.linspace(0, .025, 5, dtype=np.float32)
    for design in (0, 4, 6, 5, 10, 11):
        for location in range(2):
            data["run_numbers"].append(design * 2 + location + 1)
            data["mesh_geometries"].append((rng.normal(size=(9 + location, 3)) + design).astype(np.float32))
            data["indentor_positions"].append([location * 10., location * 3.])
            data["time_arrays"].append(times.copy())
            data["accelerations"].append((20 + design * 3 + (10 + location) * np.sin(times * 100)).astype(np.float32))
    data["indentor_positions"] = np.asarray(data["indentor_positions"], dtype=np.float32)
    return data


def training_args(path, weight=1):
    return ["--data-format", "euroncap1704", "--num-samples", "12", "--samples-per-design", "2",
            "--epochs", "4", "--batch-size", "4", "--width", "8", "--num-heads", "2", "--num-latents", "2",
            "--latent-layers", "1", "--temporal-layers", "1", "--neighborhood-layers", "2", "--neighborhood-k", "3",
            "--impactor-nodes", "2", "--dropout", "0.1", "--design-difference-weight", str(weight),
            "--device", "cpu", "--wandb-mode", "disabled", "--output-dir", str(path)]


def interrupt_after_two_epochs(path, weight=1):
    original = Trainer.save_checkpoint

    def save_then_interrupt(trainer, filename):
        original(trainer, filename)
        if Path(filename).name == LAST_CHECKPOINT_NAME and len(trainer.train_losses) == 2:
            raise RuntimeError("simulated job timeout")

    with mock.patch.object(Trainer, "save_checkpoint", save_then_interrupt):
        with pytest.raises(RuntimeError, match="simulated job timeout"):
            main(training_args(path, weight))


@pytest.mark.parametrize("weight,legacy", [(0, False), (1, False), (1, True)])
def test_resume_preserves_training_state_and_finishes_original_target(tmp_path, toy_data, weight, legacy):
    source, resumed, baseline = (tmp_path / name for name in ("source", "resumed", "baseline"))
    with mock.patch.object(DataPreprocessor, "load_all_data", return_value=toy_data):
        interrupt_after_two_epochs(source, weight)
        selected = source / (CHECKPOINT_NAME if legacy else LAST_CHECKPOINT_NAME)
        checkpoint = torch.load(selected, weights_only=True)
        completed = len(checkpoint["train_losses"])
        if legacy:
            for key in ("completed_epochs", "torch_rng_state", "cuda_rng_states", "numpy_rng_state",
                        "python_rng_state", "train_mse_losses", "train_difference_losses", "train_pair_counts"):
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
            main(["--resume-from", str(source), "--output-dir", str(resumed), "--device", "cpu", "--wandb-mode", "disabled"])
        assert source_bytes == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
        history = json.loads((resumed / "training_history.json").read_text())
        assert len(history["train_losses"]) == len(history["val_losses"]) == 4
        assert history["train_losses"][:completed] == checkpoint["train_losses"]
        saved = json.loads((resumed / "config.json").read_text())
        assert saved["training"]["resumed_after_epoch"] == completed
        assert saved["training"]["initialization"] == "checkpoint"
        assert saved["architecture"]["kwargs"]["neighborhood_layers"] == 2
        assert saved["training"]["design_difference_weight"] == weight
        assert (resumed / "test_acceleration_histories.csv").is_file()
        final = torch.load(resumed / LAST_CHECKPOINT_NAME, weights_only=True)
        assert final["completed_epochs"] == final["scheduler_state_dict"]["last_epoch"] == 4
        if legacy:
            assert history["train_mse_losses"][:completed] == [None] * completed
            assert all(np.isfinite(history["train_mse_losses"][completed:]))
            frame = pd.read_csv(resumed / "training_history.csv")
            assert frame.train_mse_normalized.iloc[:completed].isna().all()
            assert frame.epoch.tolist() == [1, 2, 3, 4]
        else:
            # Fresh checkpoint resume must reproduce uninterrupted optimization,
            # including dropout RNG, shuffled batches and matched sampler epoch.
            main(training_args(baseline, weight))
            expected = torch.load(baseline / LAST_CHECKPOINT_NAME, weights_only=True)
            assert final["train_losses"] == expected["train_losses"]
            assert final["val_losses"] == expected["val_losses"]
            for key, value in final["model_state_dict"].items():
                torch.testing.assert_close(value, expected["model_state_dict"][key], atol=0, rtol=0)


def test_resume_rejects_changed_settings_splits_and_existing_output(tmp_path, toy_data):
    source = tmp_path / "source"
    with mock.patch.object(DataPreprocessor, "load_all_data", return_value=toy_data):
        interrupt_after_two_epochs(source)
        for option, value in (("--epochs", "100"), ("--neighborhood-layers", "0"), ("--design-difference-weight", "0")):
            with pytest.raises(SystemExit):
                parse_args(["--resume-from", str(source), option, value])
        with pytest.raises(FileExistsError):
            main(["--resume-from", str(source), "--output-dir", str(source), "--device", "cpu"])
        changed = dict(toy_data)
        changed["run_numbers"] = list(toy_data["run_numbers"])
        changed["run_numbers"][0] = 3  # Shift a run to a different training design.
        with mock.patch.object(DataPreprocessor, "load_all_data", return_value=changed):
            with pytest.raises(ValueError, match="differ from the saved resume split"):
                main(["--resume-from", str(source), "--output-dir", str(tmp_path / "changed"),
                      "--device", "cpu", "--wandb-mode", "disabled"])


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


def test_resume_recovers_timeout_between_best_and_latest_writes(tmp_path, toy_data):
    source = tmp_path / "source"
    original = Trainer.save_checkpoint

    def interrupt_after_best(trainer, filename):
        original(trainer, filename)
        if Path(filename).name == CHECKPOINT_NAME and len(trainer.train_losses) == 2:
            raise RuntimeError("timeout between checkpoint writes")

    with mock.patch.object(DataPreprocessor, "load_all_data", return_value=toy_data):
        with mock.patch.object(Trainer, "save_checkpoint", interrupt_after_best), mock.patch.object(
                Trainer, "validate", side_effect=[(2.0, 1.0), (1.0, 1.0)]):
            with pytest.raises(RuntimeError, match="between checkpoint writes"):
                main(training_args(source))
        assert len(torch.load(source / LAST_CHECKPOINT_NAME, weights_only=True)["train_losses"]) == 1
        args = parse_args(["--resume-from", str(source)])
        assert Path(args.resume_from).name == CHECKPOINT_NAME
        resumed = tmp_path / "resumed"
        main(["--resume-from", str(source), "--output-dir", str(resumed), "--device", "cpu", "--wandb-mode", "disabled"])
        saved = json.loads((resumed / "config.json").read_text())
        assert saved["training"]["resumed_after_epoch"] == 2


def test_resume_preflight_reads_saved_experiment_and_dataset_location(tmp_path, toy_data):
    from preflight_mesh_impact_history_1704 import parse_args as parse_preflight_args

    source = tmp_path / "source"
    with mock.patch.object(DataPreprocessor, "load_all_data", return_value=toy_data):
        interrupt_after_two_epochs(source)
    # The preflight is specifically for the full 1704 dataset. Only parsing is
    # exercised here; the disk fixture intentionally contains tiny meshes.
    path = source / "config.json"
    saved = json.loads(path.read_text())
    saved["data"].update(num_samples=1704, samples_per_design=142, inp_dir=str(tmp_path / "data" / "inp_files"))
    path.write_text(json.dumps(saved))
    args = parse_preflight_args(["--resume-from", str(source), "--batch-size", "1", "--neighborhood-layers", "0"])
    assert args.batch_size == 4
    assert args.neighborhood_layers == 2
    assert args.impactor_nodes == 2
    assert args.design_difference_weight == 1
    assert Path(args.data_root) == tmp_path / "data"
