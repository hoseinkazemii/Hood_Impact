import numpy as np

from analyze_hic30_feature_response import (
    neighborhood_features, curve_gram, feature_associations, choose_features,
    prediction_weights, weights_mse, evaluate_folds,
)


def test_plane_features_and_order_invariance():
    xy=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)
    points=np.column_stack([xy,7+2*xy[:,0]-3*xy[:,1]])
    actual=neighborhood_features(points)
    np.testing.assert_allclose(actual,[7,np.sqrt(13),2,-3,0],atol=1e-10)
    np.testing.assert_allclose(neighborhood_features(points[::-1]),actual)


def test_unchanged_center_can_have_different_context_features():
    points=np.array([[0,0,0],[-1,-1,0],[-1,1,0],[1,-1,0],[1,1,0]],float)
    changed=points.copy()
    changed[1:,2]=3
    np.testing.assert_equal(changed[0],points[0])
    assert neighborhood_features(changed)[0]!=neighborhood_features(points)[0]


def test_full_curve_correlation_known_signal_and_constant_feature():
    x=np.arange(7.)
    features=np.column_stack([x,np.ones(7)])[:,None,:]
    curves=(x[:,None,None]*np.array([1.,-2.,3.])[None,None,:])+100
    r2=feature_associations(features,curve_gram(curves),np.arange(7))
    np.testing.assert_allclose(r2[:,0],[[1.]])
    assert np.isnan(r2[:,1]).all()
    shifted=curves+np.array([4.,90.,-2.])
    np.testing.assert_allclose(feature_associations(features,curve_gram(shifted),np.arange(7)),r2,equal_nan=True)


def test_gram_error_matches_explicit_curve_predictions():
    rng=np.random.default_rng(62)
    features=rng.normal(size=(8,3,4))
    curves=rng.normal(size=(8,2,17))
    train=np.array([0,1,2,3,4,5]); test=np.array([6,7])
    gram=curve_gram(curves)
    chosen,_=choose_features(features,gram,train)
    weights=prediction_weights(features,train,test,chosen)
    np.testing.assert_allclose(weights.sum(axis=-1),1)
    assert np.count_nonzero(weights[...,test])==0
    predictions=np.einsum('slhd,dlt->slht',weights,curves)
    expected=np.mean((predictions-curves[test].transpose(1,0,2)[None])**2,axis=-1)
    np.testing.assert_allclose(weights_mse(weights,gram,test),expected,rtol=1e-10,atol=1e-10)


def test_feature_selection_does_not_read_held_out_responses():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(8,3,4))
    y=rng.normal(size=(8,2,17))
    train=np.arange(6)
    original=choose_features(x,curve_gram(y),train)
    y[6:]=1e6*rng.normal(size=y[6:].shape)
    updated=choose_features(x,curve_gram(y),train)
    np.testing.assert_array_equal(original[0],updated[0])
    np.testing.assert_allclose(original[1],updated[1])


def test_constant_features_have_zero_predictive_gain():
    rng=np.random.default_rng(11)
    features=np.ones((6,2,3))
    curves=rng.normal(size=(6,2,12))
    result=evaluate_folds(features,curve_gram(curves),[[i] for i in range(6)],'constant')
    np.testing.assert_allclose(result['skill'],0,atol=1e-12)
    np.testing.assert_allclose(result['selected_skill'],0,atol=1e-12)


def test_held_out_predictions_generalize_linear_curve_signal():
    x=np.linspace(-1,1,12)
    features=x[:,None,None]
    curves=x[:,None,None]*np.array([1.,2.,-1.])[None,None,:]+20
    result=evaluate_folds(features,curve_gram(curves),[[i] for i in range(12)],'linear')
    assert result['skill'].item()>.98
