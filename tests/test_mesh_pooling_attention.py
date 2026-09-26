import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mesh_impact_history import MeshImpactHistoryNet
from export_mesh_pooling_attention import capture_self_attention, export_case, main, read_nodes
from train_mesh_impact_history import HistoryPredictor
from utils.utils import DataPreprocessor


def model():
    torch.manual_seed(17)
    return MeshImpactHistoryNet(width=8, num_heads=2, num_latents=6, latent_layers=1,
                                temporal_layers=1, neighborhood_layers=2, neighborhood_k=3,
                                impactor_nodes=2, dropout=.1).eval()


def test_exported_attention_reconstructs_actual_pooling_and_chunks():
    m = model()
    xyz, impact = torch.randn(11, 3), torch.randn(2)
    with torch.no_grad():
        condition = m.impact_embedding(impact)
        tokens, q, k, v, bias = m._mesh_pooling_inputs(xyz, impact, condition)
        actual = m._encode_mesh(xyz, impact, condition)
        weights = torch.cat([w for _, w in m.iter_pooling_attention(xyz, impact, 2)], dim=1)
        full = torch.cat([w for _, w in m.iter_pooling_attention(xyz, impact, 6)], dim=1)
        torch.testing.assert_close(weights, full)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 6))
        expected_weights = torch.softmax(q @ k.transpose(-1, -2) / 2 + bias[None], dim=-1)
        torch.testing.assert_close(weights, expected_weights)
        reconstructed = tokens + m.mesh_output((weights @ v).transpose(0, 1).reshape(6, 8))
        torch.testing.assert_close(actual, reconstructed, atol=1e-6, rtol=1e-5)
    m.train()
    with pytest.raises(ValueError, match="eval"):
        list(m.iter_pooling_attention(xyz, impact))


def test_file_export_preserves_ids_summaries_and_full_weights(tmp_path):
    m = model()
    xyz = np.random.default_rng(7).normal(size=(11, 3)).astype(np.float32)
    predictor = HistoryPredictor(m, SimpleNamespace(
        transform_mesh=lambda x: x, transform_indentor=lambda x: x,
        transform_time=lambda x: x, inverse_transform_acceleration=lambda x: x),
        np.linspace(0, .025, 5, dtype=np.float32))
    ids = np.arange(11) * 10 + 123
    meta = export_case(predictor, ids, xyz, np.array([1., 2.]), tmp_path / "case", chunk_size=2)
    weights = np.load(tmp_path / "case/pooling_weights.npy")
    frame = pd.read_csv(tmp_path / "case/node_attention.csv")
    np.testing.assert_array_equal(frame.node_id, ids)
    np.testing.assert_allclose(frame.mean_all, weights.mean((0, 1)), rtol=1e-6)
    np.testing.assert_allclose(frame.mean_global, weights[:, :3].mean((0, 1)), rtol=1e-6)
    np.testing.assert_allclose(frame.mean_local, weights[:, 3:].mean((0, 1)), rtol=1e-6)
    assert frame.is_headform.sum() == 2
    assert meta["max_row_sum_error"] < 1e-6
    for kind, shape in (("tokens", (1, 2, 6, 6)), ("temporal", (1, 2, 5, 5))):
        matrix = np.load(tmp_path / "case" / f"{kind}_self_attention.npy")
        assert matrix.shape == shape
        np.testing.assert_allclose(matrix.sum(-1), 1, atol=1e-6)
    before = predictor.predict(xyz, np.array([1., 2.]))
    captured, during = capture_self_attention(predictor, xyz, np.array([1., 2.]))
    after = predictor.predict(xyz, np.array([1., 2.]))
    np.testing.assert_allclose(before, during, atol=1e-5)
    np.testing.assert_allclose(before, after, atol=1e-6)
    for module in m.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    export_case(predictor, ids, xyz, np.array([1., 2.]), tmp_path / "summary", full=False)
    assert not (tmp_path / "summary/pooling_weights.npy").exists()
    np.testing.assert_allclose(pd.read_csv(tmp_path / "summary/node_attention.csv").mean_all, frame.mean_all)


def test_cli_reloads_checkpoint_and_restricts_to_saved_split(tmp_path):
    m = model()
    inp = tmp_path / "data/inp_files"
    inp.mkdir(parents=True)
    xyz = np.random.default_rng(19).normal(size=(11, 3)).astype(np.float32)
    node_ids = list(range(51, 62))
    (inp / "HoodImpact_1.inp").write_text("*NODE\n" + "\n".join(
        f"{i},{x},{y},{z}" for i, (x, y, z) in zip(node_ids, xyz)) + "\n*ELEMENT, TYPE=S4\n")
    ids, parsed = read_nodes(inp / "HoodImpact_1.inp")
    np.testing.assert_array_equal(ids, node_ids)
    np.testing.assert_array_equal(parsed, xyz)
    pd.DataFrame({"X1": [1], "X2": [2]}).to_csv(tmp_path / "data/ImpactCoords_1704.csv", index=False)
    run = tmp_path / "trained"
    run.mkdir()
    preprocessor = DataPreprocessor(SimpleNamespace())
    preprocessor.fit_scalers([xyz], np.array([[1., 2.]]), [np.array([0., .01])], [np.array([1., 2.])])
    m.set_coordinate_scalers(preprocessor.mesh_scaler.mean_, preprocessor.mesh_scaler.scale_,
                             preprocessor.indentor_scaler.mean_, preprocessor.indentor_scaler.scale_)
    preprocessor.save_scalers(str(run / "scalers.joblib"))
    np.save(run / "prediction_times.npy", [0., .01])
    torch.save({"model_state_dict": m.state_dict()}, run / "hood_impact_best_model.pt")
    kwargs = dict(width=8, num_heads=2, num_latents=6, latent_layers=1, temporal_layers=1,
                  neighborhood_layers=2, neighborhood_k=3, impactor_nodes=2, dropout=.1)
    (run / "config.json").write_text(json.dumps({"format_version": 1,
        "architecture": {"name": "MeshImpactHistoryNet", "kwargs": kwargs}, "preprocessing": {},
        "data": {"data_format": "euroncap1704", "samples_per_design": 142,
                 "inp_dir": str(inp), "impact_coords_path": str(tmp_path / "data/ImpactCoords_1704.csv")}}))
    (run / "splits.json").write_text(json.dumps({"test": {"run_numbers": [1]}}))
    with pytest.raises(SystemExit):
        main([str(run), "--runs", "2"])
    main([str(run)])
    manifest = json.loads((run / "attention_export/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["run_numbers"] == [1]
    assert manifest["cases"][0]["shape_head_token_node"] == [2, 6, 11]
    with pytest.raises(SystemExit):
        main([str(run)])
