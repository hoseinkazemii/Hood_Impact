"""Head-centred, location-free network: origin at the headform, structure channels, mirroring."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from mesh_design_change import DesignChangeHistoryNet, DesignChangeTable
from mesh_design_clusters import GEOMETRY_CLUSTERS
from mesh_impact_history import HEAD_FRAME_SCALE_MM, MeshImpactHistoryNet

MESH_MEAN, MESH_SCALE = np.array([10.0, -5.0, 3.0]), np.array([2.0, 3.0, 0.5])
HEADFORM = 3


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(cls=MeshImpactHistoryNet, **kwargs):
    torch.manual_seed(5)
    options = dict(width=16, num_heads=2, num_latents=6, latent_layers=1, temporal_layers=1, dropout=0.0,
                   neighborhood_layers=1, neighborhood_k=3, impactor_nodes=HEADFORM,
                   impact_conditioning="head_relative")
    model = cls(**{**options, **kwargs})
    model.set_coordinate_scalers(MESH_MEAN, MESH_SCALE, MESH_MEAN[:2], MESH_SCALE[:2])
    return model.eval()


def hood_mm(seed=0, nodes=15):
    """Raw (nodes, 3) mesh in mm: the headform rows first, above a scattered hood."""
    generator = np.random.default_rng(seed)
    hood = generator.normal(size=(nodes - HEADFORM, 3)) * [120.0, 120.0, 15.0]
    head = np.array([[20.0, 10.0, 90.0], [40.0, 10.0, 95.0], [30.0, 30.0, 92.0]])
    return np.vstack((head, hood))


def normalize(raw_mm, channels=None):
    mesh = (raw_mm - MESH_MEAN) / MESH_SCALE
    if channels is not None:
        mesh = np.hstack((mesh, channels))
    return torch.as_tensor(mesh, dtype=torch.float32)


def predict(model, mesh, impact=(0.0, 0.0)):
    times = torch.tensor([-.8, -.1, .7])
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long), torch.tensor([impact], dtype=torch.float32),
                 times, torch.zeros(3, dtype=torch.long))


def test_head_relative_model_has_no_location_parameters_and_keeps_local_tokens():
    model = make_model()
    for name in ("impact_embedding", "impact_film"):
        assert not hasattr(model, name)
    assert model.num_global_latents == 3
    expected = torch.logspace(math.log10(0.5), math.log10(32.0), 3).log()
    torch.testing.assert_close(model.local_log_precision.detach(), expected)
    raw = hood_mm()
    mesh = normalize(raw)
    _, _, _, _, bias = model._mesh_pooling_inputs(mesh, torch.zeros(2), None)
    radius = np.linalg.norm(raw[:, :2] - raw[:HEADFORM, :2].mean(axis=0), axis=1) / HEAD_FRAME_SCALE_MM
    local = -expected.exp()[:, None] * torch.as_tensor(radius**2, dtype=torch.float32)
    torch.testing.assert_close(bias[3:, HEADFORM:], local[:, HEADFORM:])
    assert (bias[:3, HEADFORM:] == 0).all()
    assert torch.isneginf(bias[:, :HEADFORM]).all()


def test_prediction_ignores_the_impact_input_and_absolute_position_but_not_the_head_position():
    model = make_model()
    raw = hood_mm()
    first = predict(model, normalize(raw), (0.3, -0.4))
    torch.testing.assert_close(first, predict(model, normalize(raw), (5.0, 2.0)), atol=0, rtol=0)
    # Moving hood and head together changes nothing.
    torch.testing.assert_close(first, predict(model, normalize(raw + [350.0, -120.0, 40.0])), atol=2e-5, rtol=2e-5)
    # Moving the head over the hood (another impact location) does.
    moved = raw.copy()
    moved[:HEADFORM, :2] += 80.0
    assert not torch.allclose(first, predict(model, normalize(moved)), atol=1e-4)


def test_head_relative_settings_are_validated():
    with pytest.raises(ValueError, match="impactor_nodes"):
        make_model(impactor_nodes=0)
    with pytest.raises(ValueError, match="impactor_nodes"):
        make_model(pool_impactor_nodes=True)
    with pytest.raises(ValueError, match="geometry_film"):
        make_model(geometry_film="latent")
    with pytest.raises(ValueError, match="mirror_augmentation needs"):
        make_model(impact_conditioning="film", mirror_augmentation=True)
    with pytest.raises(ValueError, match="mirror_odd_channels"):
        make_model(node_extra_features=2, mirror_odd_channels=(2,))


def change_table(raw_mm, mirrored_mm, scaled):
    """Designs 0 (the hood) and 1 (its mirror image), each with four change nodes."""
    rows = np.array([3, 5, 8, 11])
    static = np.zeros((2, 4, 9), np.float32)
    static[0, :, :3] = raw_mm[rows]
    static[0, :, 3:6] = [[0.5, 1.5, 9.0], [0.0, -2.0, 11.0], [1.0, 0.5, 8.0], [0.0, 0.0, 10.0]]
    static[0, :, 6:8] = [[20.0, -9.0]] * 4
    static[1] = static[0]
    static[1, :, 1] *= -1  # base y
    static[1, :, 4] *= -1  # displacement y
    table = DesignChangeTable([0, 1], static, [4, 4])
    table.register(scaled[0][HEADFORM:], 0)
    table.register(scaled[1][HEADFORM:], 1)
    return table


def test_mirror_augmentation_equals_feeding_the_mirror_image():
    torch.manual_seed(0)
    channels = np.random.default_rng(1).normal(size=(15, 2))
    raw = hood_mm()
    mirrored = raw * [1.0, -1.0, 1.0]
    mirrored_channels = channels * [1.0, -1.0]
    original, image = normalize(raw, channels), normalize(mirrored, mirrored_channels)
    model = make_model(DesignChangeHistoryNet, node_extra_features=2, mirror_odd_channels=(1,),
                       mirror_augmentation=True, design_change_layers=1, design_change_fourier_octaves=2)
    for readout in model.change_readouts:  # Zero at initialization; make the change path count.
        nn.init.normal_(readout.output.weight, std=0.3)
    model.design_changes = change_table(raw, mirrored, (original, image))

    expected = predict(model, image)  # eval: never mirrored
    model.train()
    model.mirror_flags = lambda batch_size: [True] * batch_size
    torch.testing.assert_close(predict(model, original), expected, atol=2e-5, rtol=2e-5)
    model.mirror_flags = lambda batch_size: [False] * batch_size
    assert not torch.allclose(predict(model, original), expected, atol=1e-4)


def test_mirror_flags_only_while_training_with_augmentation():
    model = make_model(mirror_augmentation=True)
    assert model.mirror_flags(64) == [False] * 64
    model.train()
    torch.manual_seed(2)
    flags = model.mirror_flags(64)
    assert 0 < sum(flags) < 64
    assert make_model().train().mirror_flags(4) == [False] * 4


def test_change_tokens_read_offsets_from_the_headform_centroid_in_3d():
    model = make_model(DesignChangeHistoryNet, design_change_layers=1, design_change_fourier_octaves=2)
    static = torch.zeros(1, 2, 9)
    static[0, :, :3] = torch.tensor([[100.0, -50.0, 10.0], [0.0, 0.0, 0.0]])
    origin = torch.tensor([[20.0, 30.0, 90.0]])
    features = model.change_features(static, origin)
    assert features.shape[-1] == model.change_embedding[0].in_features == 11 + 4 * 2
    torch.testing.assert_close(features[0, :, :3], (static[0, :, :3] - origin) / 250.0)
    flipped = model.change_features(static, origin, torch.tensor([True]))
    torch.testing.assert_close(flipped[0, :, 1], -features[0, :, 1])


# --------------------------------------------------------------------------- data
HEAD_IDS = (901, 902, 903, 904)


def cluster_member(design):
    for index, members in enumerate(GEOMETRY_CLUSTERS.values()):
        if design in members:
            return index, list(members).index(design)
    raise ValueError(design)


def write_deck(path, design, head_xy):
    """Small 1704-style deck: the headform (high IDs) first, then hood nodes out of ID order.

    Designs of one geometry cluster share node IDs; member m raises inner node 21 by 0.3 m mm.
    """
    cluster, member = cluster_member(design)
    hx, hy = head_xy
    head = [(901, hx - .5, hy - .5, 9.0), (902, hx + .5, hy - .5, 9.0),
            (903, hx + .5, hy + .5, 9.0), (904, hx - .5, hy + .5, 9.0)]
    outer = [(11, 0, 0, 5), (12, 3, 0, 5), (13, 3, 3, 5), (14, 0, 3, 5)]
    inner = [(21, 0, 0, 2 + .3 * member), (22, 3, 0, 2), (23, 3, 3, 2), (24, 0, 3, 2)]
    inner += [(25 + i, .5 + i, 1.5, 2.2 + .1 * cluster) for i in range(cluster)]
    hinges = [(31, 5, 5, 1), (32, 6, 5, 1), (33, 6, 6, 1), (34, 5, 6, 1)]
    elements = {"Rigid Body1_2-1-3": [(1, *HEAD_IDS)], "Hood_Outer-1-2": [(2, 11, 12, 13, 14)],
                "Hood_Inner-1-2": [(3, 21, 22, 23, 24)] + [(4 + i, 21, 22, 25 + i, 24) for i in range(cluster)],
                "Hinges-1-2": [(30, 31, 32, 33, 34)]}
    nodes = head + (outer + inner + hinges)[::-1]
    lines = ["*HEADING", "*NODE"] + [f"{n}, {x}, {y}, {z}" for n, x, y, z in nodes]
    lines.append("*ELEMENT, TYPE=S4")
    lines += [", ".join(map(str, element)) for group in elements.values() for element in group]
    for name, group in elements.items():
        lines += [f'*ELSET, ELSET="{name}"', ", ".join(str(element[0]) for element in group)]
    lines.append(f'*RIGID BODY, REF NODE={HEAD_IDS[0]}, ELSET="Rigid Body1_2-1-3"')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return np.array([node for node, *_ in nodes]), np.array([xyz for _, *xyz in nodes], dtype=np.float32)


def test_structure_channels_join_the_mesh_by_node_id(tmp_path):
    from design_difference_data import PANELS, load_design_geometry
    from utils.utils import STRUCTURE_CHANNELS, Config, DataPreprocessor

    inp_dir = tmp_path / "inp_files"
    inp_dir.mkdir()
    ids, xyz = write_deck(inp_dir / "HoodImpact_2.inp", design=0, head_xy=(1.5, 1.5))
    write_deck(inp_dir / "HoodImpact_1.inp", design=0, head_xy=(1.0, 1.0))  # read for the design geometry
    config = Config(data_format="euroncap1704", inp_dir=str(inp_dir), output_dir=str(tmp_path / "out"))
    config.node_features = "structure"
    mesh = DataPreprocessor(config).load_mesh_geometry(2)
    assert mesh.shape == (len(ids), 3 + len(STRUCTURE_CHANNELS))
    np.testing.assert_array_equal(mesh[:, :3], xyz)
    assert not mesh[:4, 3:].any()  # Headform rows.

    geometry = load_design_geometry(0, tmp_path)
    row_of = {node: row for row, node in enumerate(geometry["ids"])}
    rows = [row_of[node] for node in ids[4:]]
    for index, name in enumerate(STRUCTURE_CHANNELS[:3]):
        np.testing.assert_array_equal(mesh[4:, 3 + index], geometry["panel"][rows] == PANELS.index(name))
    np.testing.assert_allclose(mesh[4:, 6:9], geometry["normal"][rows], rtol=1e-6)
    np.testing.assert_allclose(mesh[4:, 9], np.log1p(geometry["gap"][rows] / 10.0), rtol=1e-6)
    assert mesh[4:, 3:6].sum(axis=1).tolist() == [1.0] * (len(ids) - 4)

    preprocessor = DataPreprocessor(config)
    preprocessor.fit_scalers([mesh], np.zeros((1, 2)))
    assert preprocessor.mesh_scaler.n_features_in_ == 3
    transformed = preprocessor.transform_mesh(mesh)
    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(transformed[:, 3:], mesh[:, 3:])
    np.testing.assert_allclose(transformed[:, :3], preprocessor.mesh_scaler.transform(mesh[:, :3]), rtol=1e-6)


def test_structure_channels_reject_an_unknown_hood_node(tmp_path):
    from utils.utils import Config, DataPreprocessor

    inp_dir = tmp_path / "inp_files"
    inp_dir.mkdir()
    write_deck(inp_dir / "HoodImpact_1.inp", design=0, head_xy=(1.0, 1.0))
    config = Config(data_format="euroncap1704", inp_dir=str(inp_dir), output_dir=str(tmp_path / "out"))
    config.node_features = "structure"
    processor = DataPreprocessor(config)
    processor.load_mesh_geometry(1)
    with pytest.raises(ValueError, match="no structure channels"):
        processor._structure_channels(1, np.array([901, 902, 903, 904, 11, 77, 12]))


def test_head_centred_training_end_to_end(tmp_path):
    from train_mesh_impact_history import HistoryPredictor, main
    from utils.utils import STRUCTURE_CHANNELS, Config, DataPreprocessor

    inp_dir, acceleration_dir = tmp_path / "inp_files", tmp_path / "acc"
    inp_dir.mkdir()
    acceleration_dir.mkdir()
    times = np.linspace(0, .025, 101)
    for design in range(12):
        for location, head_xy in ((1, (1.0, 1.0)), (2, (2.0, 1.5))):
            run = 142 * design + location
            write_deck(inp_dir / f"HoodImpact_{run}.inp", design, head_xy)
            acceleration = 60 + 5 * design + 10 * location + (40 + 3 * design) * np.sin(np.pi * times / .025) ** 2
            pd.DataFrame({"Time": times, "A(in g)": acceleration}).to_csv(
                acceleration_dir / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    coords = tmp_path / "ImpactCoords_1704.csv"
    pd.DataFrame({"X1": np.full(1704, 1.5), "X2": np.full(1704, 1.5), "X3": np.full(1704, 9.0)}).to_csv(
        coords, index=False)
    run_dir = tmp_path / "run"
    metrics = main([
        "--data-format", "euroncap1704", "--inp-dir", str(inp_dir), "--impact-coords-path", str(coords),
        "--acceleration-dir", str(acceleration_dir), "--locations", "1", "2",
        "--test-designs", "4", "5", "--val-designs", "11",
        "--impact-conditioning", "head_relative", "--node-features", "structure", "--mirror-augmentation", "true",
        "--design-change-nodes", "true", "--design-change-layers", "1", "--design-change-fourier-octaves", "2",
        "--impactor-nodes", "4", "--neighborhood-layers", "1", "--neighborhood-k", "3",
        "--epochs", "2", "--batch-size", "4", "--width", "16", "--num-heads", "2", "--num-latents", "4",
        "--latent-layers", "1", "--temporal-layers", "1", "--dropout", "0", "--time-subsample-stride", "4",
        "--device", "cpu", "--wandb-mode", "disabled", "--output-dir", str(run_dir),
    ])
    assert all(np.isfinite(value) for value in metrics.values())

    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    kwargs = saved["architecture"]["kwargs"]
    assert saved["architecture"]["name"] == "DesignChangeHistoryNet"
    assert kwargs["impact_conditioning"] == "head_relative" and kwargs["mirror_augmentation"] is True
    assert kwargs["node_extra_features"] == len(STRUCTURE_CHANNELS)
    assert kwargs["mirror_odd_channels"] == [STRUCTURE_CHANNELS.index("normal_y")]
    assert saved["preprocessing"]["node_features"] == "structure"
    assert saved["preprocessing"]["node_channels"] == list(STRUCTURE_CHANNELS)
    assert (run_dir / "design_change_nodes.npz").is_file()
    assert (run_dir / "design_difference_r2.json").is_file()
    train = pd.read_csv(run_dir / "train_acceleration_histories.csv")
    assert train.run_number.nunique() == 18  # Nine training designs at two locations.

    # Reload predicts the exported test history from the mesh with its structure channels.
    predictor = HistoryPredictor.from_run(run_dir)
    config = Config(data_format="euroncap1704", inp_dir=str(inp_dir), output_dir=str(tmp_path / "out"))
    config.node_features = "structure"
    run = 142 * 5 + 2
    mesh = DataPreprocessor(config).load_mesh_geometry(run)
    exported = pd.read_csv(run_dir / "test_acceleration_histories.csv")
    np.testing.assert_allclose(predictor.predict(mesh, np.array([9.9, -9.9])),
                               exported[exported.run_number == run].acceleration_pred_g, rtol=2e-5, atol=2e-5)
