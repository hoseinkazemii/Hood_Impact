import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gpu_multiscale_hic_v6 as v6


def manifest():
    runs = np.arange(1, 1705)
    return pd.DataFrame({
        "run": runs, "design": (runs - 1) // 142, "loc": (runs - 1) % 142 + 1,
        "X1": runs * 0.1, "X2": 0.0, "X3": 700.0,
        "source_run": (runs - 1) % 600 + 1,
    })


def write_manifest(root, frame):
    folder = root / "HoodImpact_1704_EuroNCAP"
    folder.mkdir(exist_ok=True)
    frame.to_csv(folder / "manifest_1704.csv", index=False)


def test_merged_ids_paths_and_split():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_manifest(root, manifest().sample(frac=1, random_state=42))
        metadata = v6.build_metadata(root, "1704")
        assert metadata.iloc[-1].source_run == 1704
        assert metadata.iloc[-1].original_source_run != 1704
        assert v6.history_path(root, "euroncap142", 1704).name == "HoodImpact_1704_SAE1000_interp1000.csv"
        assert v6.representative_deck(root, 11, "euroncap142").name == "HoodImpact_1563.inp"
        train, val, test = v6.split_indices(metadata, 10, 11)
        assert (len(train), len(val), len(test)) == (1420, 142, 142)
        assert set(metadata.iloc[train].design) == set(range(10))
        assert not (set(train) & set(val) or set(train) & set(test))


@pytest.mark.parametrize("column,value", [("run", 2), ("design", 0.5), ("loc", 142), ("X1", np.nan)])
def test_invalid_manifest_rejected(column, value):
    with tempfile.TemporaryDirectory() as directory:
        frame = manifest()
        frame[column] = frame[column].astype(float)
        frame.loc[0, column] = value
        write_manifest(Path(directory), frame)
        with pytest.raises(ValueError):
            v6.build_metadata(Path(directory), "1704")


def test_142_location_normalization_and_evaluation():
    frame = manifest().rename(columns={"loc": "location"})
    frame["source"] = "euroncap142"
    samples = len(frame)
    hic = (frame.location + frame.design * 10).to_numpy(np.float32)
    hic[frame.design >= 10] += 1e6
    times = np.tile(np.linspace(0, 0.025, 10), (samples, 1))
    data = v6.PreparedData(
        frame, np.ones((samples, 3, len(v6.MAP_CHANNELS), 2, 2), np.float32),
        np.ones((samples, len(v6.SCALAR_FEATURES)), np.float32),
        times, np.ones_like(times) * 10, hic, "test-1704",
    )
    train, val, _ = v6.split_indices(frame, 10, 11)
    normalizer = v6.TrainingNormalizer().fit(data, train)
    np.testing.assert_allclose(normalizer.euroncap_hic_mean, np.arange(1, 143) + 45)
    assert normalizer.industry_hic_mean.size == 0
    normalized = normalizer.transform_hic(hic[val], frame.iloc[val])
    np.testing.assert_allclose(normalizer.inverse_hic(normalized, frame.iloc[val]), hic[val])
    prediction = {"sample_indices": val, "hic_direct": hic[val], "acceleration": data.accelerations[val]}
    metrics, _ = v6.detailed_metrics(data, prediction, 0.015)
    v6.add_baseline_metrics(metrics, data, val, normalizer.hic_baseline(frame.iloc[val]))
    assert set(metrics) == {"combined", "euroncap142"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "normalization.npz"
        normalizer.save(path)
        with np.load(path, allow_pickle=False) as saved:
            assert saved["euroncap_hic_mean"].shape == (142,)
