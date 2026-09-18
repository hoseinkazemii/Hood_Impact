"""Checks that the analysis interventions and statistics mean what they claim."""
import numpy as np
import torch

from analyze_impact_locality_1704 import (
    batched_hic, frozen_prior_predictions, rank_association,
    symmetric_cell_mismatch, local_mismatch,
)
from evaluate_mesh_impact_hic import hic15
from mesh_impact_history import MeshImpactHistoryNet


def test_batched_hic_matches_independent_pair_search_on_irregular_grid():
    times = np.array([0., .001, .004, .010, .013, .017, .025])
    histories = np.array([[0., 5., 40., 75., 200., 10., 2.], [-3., 2., 3., 5., 10., 20., 10.]])
    expected = [hic15(times, a)[0] for a in histories]
    np.testing.assert_allclose(batched_hic(times, histories), expected, rtol=1e-12)


def test_geometry_detects_local_change_without_node_correspondence():
    x, y = np.meshgrid(np.arange(-4,5)*20., np.arange(-4,5)*20.)
    original = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    changed = original.copy()
    changed[(abs(x.ravel())<=20)&(abs(y.ravel())<=20),2] = 8.
    xy, squared = symmetric_cell_mismatch(original, changed[::-1])
    near, far = local_mismatch(xy, squared, np.array([[0.,0.], [70.,70.]]), [30.])[:,0]
    assert near > far + 3
    _, unchanged = symmetric_cell_mismatch(original, original[::-1])
    np.testing.assert_array_equal(unchanged, 0)


def test_location_control_removes_a_shared_location_pattern():
    locations = np.arange(20.)
    geometry = np.stack([locations+offset for offset in [0,30,100]])
    response = np.stack([locations*scale for scale in [1,2,3]])
    assert rank_association(geometry, response) > .999
    assert abs(rank_association(geometry, response, location_control=True)) < 1e-10


def test_factor_one_reproduces_model_and_other_factors_act_without_weight_changes():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    model = MeshImpactHistoryNet(width=16, num_heads=2, num_latents=6,
                                latent_layers=1, temporal_layers=1, neighborhood_layers=1,
                                neighborhood_k=3, neighborhood_chunk_size=5,
                                impactor_nodes=2, dropout=0).eval()
    mesh, impact, times = torch.randn(15,3), torch.randn(2), torch.linspace(-1,1,9)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    actual = frozen_prior_predictions(model,mesh,impact,times,[0.,1.,4.])
    with torch.no_grad():
        expected = model(mesh,torch.zeros(len(mesh),dtype=torch.long),impact[None],
                         times,torch.zeros(len(times),dtype=torch.long))
    torch.testing.assert_close(actual[1],expected)
    assert not torch.allclose(actual[0],actual[2])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
