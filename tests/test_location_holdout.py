import json

import numpy as np
import pandas as pd
import pytest

from location_holdout import DEFAULT_SPLIT_FILE, apply_split, checkerboard, load_split
from train_mesh_impact_history import build_config, parse_args, resolve_splits


def lattice():
    rows = [(w, k) for w in range(2, 6) for k in range(-2, 3)]
    return pd.DataFrame({"loc": np.arange(1, len(rows) + 1),
                         "X1": [100.0 * w for w, _ in rows], "X2": [155 + 100.0 * k for _, k in rows],
                         "wadh_mm": [100.0 * w for w, _ in rows], "col_k": [k for _, k in rows]})


def full_data(report, samples=142):
    runs = list(range(1, 12 * samples + 1))
    xy = [report["coordinates_mm"][str((run - 1) % samples + 1)] for run in runs]
    return {"run_numbers": runs, "indentor_positions": np.asarray(xy, dtype=np.float32),
            "accelerations": [[run] for run in runs]}


def test_checkerboard_alternates_and_keeps_mirror_images_together():
    grid = lattice()
    report = checkerboard(grid)
    test = set(report["test_locations"])
    by_cell = {(round(r.wadh_mm / 100), r.col_k): r.loc for r in grid.itertuples()}
    for (w, k), location in by_cell.items():
        assert (location in test) == ((w + k) % 2 == 1)
        assert (location in test) == (by_cell[(w, -k)] in test)
        for neighbour in ((w + 1, k), (w, k + 1)):
            if neighbour in by_cell:
                assert (location in test) != (by_cell[neighbour] in test)
    off_lattice = grid.assign(wadh_mm=grid.wadh_mm + 50)
    with pytest.raises(ValueError, match="lattice"):
        checkerboard(off_lattice)


def test_committed_split_is_71_and_71_on_the_1704_grid():
    report = load_split()
    seen, unseen = report["train_validation_locations"], report["test_locations"]
    assert len(seen) == len(unseen) == 71
    assert sorted(seen + unseen) == list(range(1, 143))
    assert report["file"] == str(DEFAULT_SPLIT_FILE)


def test_test_designs_see_only_unseen_locations():
    report = load_split()
    data = apply_split(full_data(report), report, [4, 5])
    config = build_config(parse_args(["--data-format", "euroncap1704", "--device", "cpu"]))
    splits = resolve_splits(data, config, [4, 5], [11])
    assert [len(splits[key]["run_numbers"]) for key in ("train", "validation", "test")] == [639, 71, 142]
    location = {key: {(run - 1) % 142 + 1 for run in group["run_numbers"]} for key, group in splits.items()}
    assert location["train"] == location["validation"] == set(report["train_validation_locations"])
    assert location["test"] == set(report["test_locations"])
    assert data["accelerations"] == [[run] for run in data["run_numbers"]]


def test_rejects_impact_coordinates_that_do_not_match_the_location_ids():
    report = load_split()
    data = full_data(report)
    data["indentor_positions"][700] += [0, 100]
    with pytest.raises(ValueError, match="Run 701: impact XY"):
        apply_split(data, report, [4, 5])


def test_rejects_overlapping_lists(tmp_path):
    report = json.loads(DEFAULT_SPLIT_FILE.read_text())
    report["test_locations"].append(report["train_validation_locations"][0])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="disjoint"):
        load_split(path)


def test_training_saves_the_holdout_and_resume_restores_it(tmp_path, monkeypatch):
    import torch
    import train_mesh_impact_history as training
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    report = load_split()
    rng = np.random.default_rng(42)
    data = full_data(report)
    data["mesh_geometries"] = [rng.normal(size=(5, 3)).astype(np.float32) for _ in data["run_numbers"]]
    data["time_arrays"] = [np.linspace(0, .02, 3, dtype=np.float32) for _ in data["run_numbers"]]
    data["accelerations"] = [np.array([1, 2, 3], dtype=np.float32) for _ in data["run_numbers"]]
    monkeypatch.setattr(training.DataPreprocessor, "load_all_data", lambda self: data)
    try:
        training.main(["--data-format", "euroncap1704", "--device", "cpu", "--epochs", "2",
                       "--batch-size", "64", "--width", "8", "--num-heads", "2", "--num-latents", "2",
                       "--latent-layers", "1", "--temporal-layers", "1", "--impactor-nodes", "2",
                       "--location-holdout", str(DEFAULT_SPLIT_FILE),
                       "--wandb-mode", "disabled", "--output-dir", str(tmp_path / "run")])
        saved = json.loads((tmp_path / "run/config.json").read_text())
        assert saved["location_holdout"]["test_locations"] == report["test_locations"]
        assert (tmp_path / "run/location_holdout.json").is_file()
        splits = json.loads((tmp_path / "run/splits.json").read_text())
        assert len(splits["test"]["run_numbers"]) == 142
        assert {(run - 1) % 142 + 1 for run in splits["test"]["run_numbers"]} == set(report["test_locations"])
        restored = training.parse_args(["--resume-from", str(tmp_path / "run")])
        assert restored.location_holdout == str(DEFAULT_SPLIT_FILE)
    finally:
        torch.set_num_threads(previous)
