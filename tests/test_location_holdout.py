import json

import numpy as np
import pandas as pd
import pytest

from location_holdout import DEFAULT_SPLIT_FILE, SPLIT_DIR, apply_split, checkerboard, load_split, per_design_random

RANDOM70_FILE = SPLIT_DIR / "euroncap1704_random70_seed42.json"
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


@pytest.mark.parametrize("split_file", [DEFAULT_SPLIT_FILE, RANDOM70_FILE], ids=["checkerboard", "random70"])
def test_training_saves_the_holdout_and_resume_restores_it(tmp_path, monkeypatch, split_file):
    import torch
    import train_mesh_impact_history as training
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    report = load_split(split_file)
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
                       "--location-holdout", str(split_file),
                       "--wandb-mode", "disabled", "--output-dir", str(tmp_path / "run")])
        saved = json.loads((tmp_path / "run/config.json").read_text())
        assert saved["location_holdout"]["scheme"] == report["scheme"]
        assert (tmp_path / "run/location_holdout.json").is_file()
        # The final epoch's checkpoint is scored beside the best-validation one.
        for name in ("metrics.json", "test_acceleration_histories.csv", "train_acceleration_histories.csv",
                     "design_difference_r2.json"):
            assert (tmp_path / "run/last_checkpoint" / name).is_file()
        splits = json.loads((tmp_path / "run/splits.json").read_text())
        if report["scheme"] == "checkerboard":
            assert len(splits["test"]["run_numbers"]) == 142
            assert {(run - 1) % 142 + 1 for run in splits["test"]["run_numbers"]} == set(report["test_locations"])
        else:
            assert saved["location_holdout"]["design_locations"] == report["design_locations"]
            assert [len(splits[key]["run_numbers"]) for key in ("train", "validation", "test")] == [891, 99, 198]
        restored = training.parse_args(["--resume-from", str(tmp_path / "run")])
        assert restored.location_holdout == str(split_file)
    finally:
        torch.set_num_threads(previous)


def test_per_design_random_draws_an_independent_reproducible_subset_per_design():
    grid = lattice()
    report = per_design_random(grid, num_designs=12, fraction=0.7, seed=7)
    subsets = [report["design_locations"][str(design)] for design in range(12)]
    assert all(len(kept) == 14 and len(set(kept)) == 14 for kept in subsets)
    assert len({tuple(kept) for kept in subsets}) > 1
    assert per_design_random(grid, 12, 0.7, 7) == report
    assert per_design_random(grid, 12, 0.7, 8) != report
    with pytest.raises(ValueError, match="fraction"):
        per_design_random(grid, fraction=0)


def test_committed_random70_split_keeps_99_locations_on_every_design():
    report = load_split(RANDOM70_FILE)
    assert (report["scheme"], report["fraction"], report["seed"]) == ("per_design_random", 0.7, 42)
    assert sorted(report["design_locations"]) == sorted(str(design) for design in range(12))
    assert all(len(kept) == 99 for kept in report["design_locations"].values())
    assert report["coordinates_mm"] == load_split()["coordinates_mm"]


def test_random70_keeps_each_designs_own_locations_and_the_design_split():
    report = load_split(RANDOM70_FILE)
    data = apply_split(full_data(report), report, [4, 5])
    config = build_config(parse_args(["--data-format", "euroncap1704", "--device", "cpu"]))
    splits = resolve_splits(data, config, [4, 5], [11])
    assert [len(splits[key]["run_numbers"]) for key in ("train", "validation", "test")] == [891, 99, 198]
    assert splits["test"]["design_ids"] == [4, 5] and splits["validation"]["design_ids"] == [11]
    for run in data["run_numbers"]:
        design, location = divmod(run - 1, 142)
        assert location + 1 in report["design_locations"][str(design)]
    report = {**report, "design_locations": {k: v for k, v in report["design_locations"].items() if k != "7"}}
    with pytest.raises(ValueError, match="design 7 has no location list"):
        apply_split(full_data(report), report, [4, 5])


def test_rejects_repeated_or_unknown_locations_in_a_random_split(tmp_path):
    report = json.loads(RANDOM70_FILE.read_text())
    for bad in ([1, 1, 2], [0, 5], []):
        report["design_locations"]["3"] = bad
        path = tmp_path / "split.json"
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="design 3"):
            load_split(path)
