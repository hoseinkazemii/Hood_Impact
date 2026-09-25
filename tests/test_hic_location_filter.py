import numpy as np
import pandas as pd
import pytest

from hic_location_filter import analyze, filter_data, summarize
from train_mesh_impact_history import build_config, parse_args, resolve_splits


def test_inclusive_threshold_zero_mean_and_counts():
    table, report = summarize([[95, 96, 0], [105, 104, 0]], 10)
    assert table.range_percent.tolist() == [10, 8, 0]
    assert report["selected_locations"] == [1]
    assert report["threshold_counts"]["10"] == 1
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            summarize([[1], [2]], invalid)


def test_same_locations_in_every_split_preserving_original_run_ids():
    runs = list(range(1, 1705))
    data = {"run_numbers": runs, "indentor_positions": np.zeros((1704, 2)),
            "accelerations": [[run] for run in runs]}
    filtered = filter_data(data, [2, 142])
    config = build_config(parse_args(["--data-format", "euroncap1704", "--device", "cpu"]))
    splits = resolve_splits(filtered, config, [4, 5], [11])
    assert [len(splits[key]["run_numbers"]) for key in ("train", "validation", "test")] == [18, 2, 4]
    for group in splits.values():
        assert {(run - 1) % 142 + 1 for run in group["run_numbers"]} == {2, 142}
    assert filtered["accelerations"] == [[run] for run in filtered["run_numbers"]]
    assert filtered["indentor_positions"].shape == (24, 2)
    with pytest.raises(ValueError, match="No qualifying"):
        filter_data(data, [])


def test_full_history_hic_and_missing_history_failure(monkeypatch):
    def history(path, **kwargs):
        run = int(path.name.split("_")[1])
        design, location = divmod(run - 1, 142)
        amplitude = 100 + (design * 10 if location == 0 else 0)
        return pd.DataFrame({"Time": [0, .005, .010, .015], "A(in g)": [amplitude] * 4})
    monkeypatch.setattr(pd, "read_csv", history)
    table, report = analyze("unused", 10)
    assert report["selected_locations"] == [1]
    assert table.loc[0, "hic_design_0"] == pytest.approx(.015 * 100**2.5)
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing simulation")
    monkeypatch.setattr(pd, "read_csv", missing)
    with pytest.raises(FileNotFoundError):
        analyze("unused", 10)


def test_training_filters_before_scaling_and_exports_only_selected(tmp_path, monkeypatch):
    import json
    import torch
    import train_mesh_impact_history as training
    import hic_location_filter
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    data = {key: [] for key in ("run_numbers", "mesh_geometries", "indentor_positions",
                               "time_arrays", "accelerations")}
    for run in range(1, 1705):
        location = (run - 1) % 142 + 1
        data["run_numbers"].append(run)
        data["mesh_geometries"].append(rng.normal(size=(5, 3)).astype(np.float32))
        data["indentor_positions"].append([location, 0])
        data["time_arrays"].append(np.linspace(0, .02, 3, dtype=np.float32))
        data["accelerations"].append(np.array([1, 2, 3], dtype=np.float32) * (1 if location == 1 else 10000))
    data["indentor_positions"] = np.asarray(data["indentor_positions"], dtype=np.float32)
    monkeypatch.setattr(training.DataPreprocessor, "load_all_data", lambda self: data)
    table, report = summarize(np.array([[95] + [100] * 141, [105] + [100] * 141]), 10)
    monkeypatch.setattr(hic_location_filter, "analyze", lambda *args: (table, report))
    original = training.create_data_loaders
    def checked(filtered, *args, **kwargs):
        assert len(filtered["run_numbers"]) == 12
        assert all(np.max(a) == 3 for a in filtered["accelerations"])
        return original(filtered, *args, **kwargs)
    monkeypatch.setattr(training, "create_data_loaders", checked)
    try:
        training.main(["--data-format", "euroncap1704", "--device", "cpu", "--epochs", "1",
                       "--batch-size", "4", "--width", "8", "--num-heads", "2", "--num-latents", "2",
                       "--latent-layers", "1", "--temporal-layers", "1", "--neighborhood-layers", "2",
                       "--neighborhood-k", "256", "--hic-range-threshold-percent", "10",
                       "--wandb-mode", "disabled", "--output-dir", str(tmp_path / "run")])
        saved = json.loads((tmp_path / "run/config.json").read_text())
        assert saved["location_filter"]["selected_locations"] == [1]
        splits = json.loads((tmp_path / "run/splits.json").read_text())
        assert splits["test"]["run_numbers"] == [569, 711]
        restored = training.parse_args(["--resume-from", str(tmp_path / "run")])
        assert restored.hic_range_threshold_percent == 10
        assert restored.resume_config["location_filter"]["selected_locations"] == [1]
    finally:
        torch.set_num_threads(previous)
