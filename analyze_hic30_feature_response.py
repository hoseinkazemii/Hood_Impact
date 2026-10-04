"""Scan geometry-context features against between-design acceleration differences.

Every structural reference node is evaluated, without a node-motion or
impact-distance filter. Features describe both panels around fixed physical XY
sites. Correlations are descriptive; separate design/family holdouts test a
simple feature-to-curve predictor. The production neural network is untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from abaqus_scripts.inp_geom import Deck
from analyze_hic30_node_response import clean_json
from mesh_design_clusters import GEOMETRY_CLUSTERS

ROOT = Path(__file__).resolve().parent
COUNTS = (32, 256)
FEATURES = [f'{panel}_{name}_{k}' for panel in ('inner', 'outer')
            for k in COUNTS for name in ('height', 'height_spread', 'slope_X1', 'slope_X2', 'nonplanarity')]
FEATURES += ['inner_nearest_XY_distance', 'outer_nearest_XY_distance', 'panel_separation_32', 'panel_separation_256']
DESCRIPTIONS = {name: name.replace('_', ' ') for name in FEATURES}
PRECISION = np.array([.0001 if 'slope' in n else .01 for n in FEATURES])
FAMILIES = tuple(tuple(v) for v in GEOMETRY_CLUSTERS.values())


def neighborhood_features(points):
    """Five physical descriptors, invariant to mesh point ordering.

    Height is mean Z, spread is Z standard deviation, slopes describe a least
    squares plane, nonplanarity is RMS residual about that plane. Slopes are
    undefined for degenerate XY neighborhoods and cause an explicit failure.
    """
    mean = points.mean(axis=-2)
    c = points - mean[..., None, :]
    cov = np.einsum('...ki,...kj->...ij', c, c) / points.shape[-2]
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a*d-b*b
    if np.any(det <= 1e-10):
        raise ValueError('Degenerate XY neighborhood')
    u, v = cov[..., 0, 2], cov[..., 1, 2]
    sx, sy = (d*u-b*v)/det, (a*v-b*u)/det
    residual = np.maximum(0, cov[..., 2, 2]-sx*u-sy*v)
    return np.stack([mean[..., 2], np.sqrt(cov[..., 2, 2]), sx, sy, np.sqrt(residual)], axis=-1)


def extract_features(data_dir, xyz, impacts):
    """Features at all reference-node sites plus all impact sites, for 12 designs."""
    queries = np.concatenate([xyz[:, :2], impacts])
    output = np.empty((12, len(queries), len(FEATURES)), dtype=np.float64)
    radii = np.empty((12, len(queries), 2, len(COUNTS)), dtype=np.float32)
    nearest_ids = np.empty((12, len(queries), 2), dtype=np.int64)
    for design in range(12):
        deck = Deck(data_dir/'inp_files'/f'HoodImpact_{142*design+1}.inp')
        head = deck.impactor_node_ids()
        for panel, elset in enumerate(('Hood_Inner-1-2', 'Hood_Outer-1-2')):
            ids = np.asarray(sorted(deck.elset_nodes(elset)-head))
            points = np.array([deck.nodes[int(n)] for n in ids])
            tree = cKDTree(points[:, :2])
            for start in range(0, len(queries), 512):
                stop = min(start+512, len(queries))
                distance, ix = tree.query(queries[start:stop], k=max(COUNTS), workers=4)
                nearest_ids[design, start:stop, panel] = ids[ix[:, 0]]
                output[design, start:stop, 20+panel] = distance[:, 0]
                for j, k in enumerate(COUNTS):
                    output[design, start:stop, panel*10+j*5:panel*10+j*5+5] = neighborhood_features(points[ix[:, :k]])
                    radii[design, start:stop, panel, j] = distance[:, k-1]
        for j in range(len(COUNTS)):
            output[design, :, 22+j] = output[design, :, 10+j*5]-output[design, :, j*5]
        print(f'Geometry context: design {design+1}/12; all {len(xyz):,} node sites included.', flush=True)
    output = np.round(output/PRECISION)*PRECISION
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite geometric feature')
    return output, radii, nearest_ids


def curve_gram(curves):
    """Design-by-design inner products of complete location-specific curves."""
    return np.einsum('dlt,elt->lde', curves, curves)/curves.shape[-1]


def centered_training_gram(gram, train):
    sub = gram[:, train][:, :, train]
    return sub-sub.mean(axis=1, keepdims=True)-sub.mean(axis=2, keepdims=True)+sub.mean(axis=(1, 2), keepdims=True)


def feature_associations(features, gram, train):
    """Return squared full-curve correlation, shape sites x features x locations.

    Equivalent to the fraction of between-design curve variance explained by
    separate intercept+one-feature least-squares fits at every time sample.
    Undefined for constant features or constant response curves.
    """
    x = features[train].transpose(1, 2, 0)
    x = x-x.mean(axis=-1, keepdims=True)
    norm2 = np.sum(x*x, axis=-1)
    xc = np.divide(x, np.sqrt(norm2[..., None]), out=np.zeros_like(x), where=norm2[..., None]>1e-14)
    g = centered_training_gram(gram, train)
    total = np.trace(g, axis1=1, axis2=2)
    # A small design Gram avoids forming predictions at 1,000 time samples
    # for every node, feature and impact location. This is algebraically exact.
    outer = (xc[..., :, None]*xc[..., None, :]).reshape(-1, len(train)**2)
    explained = (outer @ g.reshape(len(g), -1).T).reshape(x.shape[:2]+(len(g),))
    result = np.divide(explained, total, out=np.full_like(explained, np.nan), where=total>1e-12)
    result[norm2<=1e-14] = np.nan
    return np.clip(result, 0, 1)


def choose_features(features, gram, train):
    r2 = feature_associations(features, gram, train)
    indices = np.argmax(np.nan_to_num(r2, nan=-1), axis=1)
    best = np.take_along_axis(r2, indices[:, None, :], axis=1)[:, 0, :]
    return indices, best


def prediction_weights(features, train, test, chosen):
    """Linear feature-to-curve probe; feature choice uses training curves only.

    Fixed ridge penalty after training-only standardization shrinks the slope
    by n/(n+1). Constant training features predict the training mean curve.
    Returns (site, location, test-design, source-design) weights.
    """
    x = features.transpose(1, 2, 0)
    x = np.take_along_axis(x, chosen[..., None], axis=1)
    mean = x[..., train].mean(axis=-1)
    centered = x[..., train]-mean[..., None]
    ss = np.sum(centered**2, axis=-1)
    beta = np.divide((x[..., test]-mean[..., None])*(len(train)/(len(train)+1)),
                     ss[..., None], out=np.zeros(x.shape[:2]+(len(test),)), where=ss[..., None]>1e-14)
    w = np.zeros(x.shape[:2]+(len(test), features.shape[0]))
    w[..., train] = 1/len(train)+beta[..., None]*centered[..., None, :]
    return w


def weights_mse(weights, gram, test):
    residual = weights.copy()
    for i, d in enumerate(test):
        residual[..., i, d] -= 1
    return np.maximum(0, np.einsum('sltd,lde,slte->slt', residual, gram, residual, optimize=True))


def evaluate_folds(features, gram, folds, label, map_sites=None):
    """Per-site held-out error, with feature selection repeated within folds.

    Also evaluate a region+feature selected using training associations only.
    The latter estimates the selection procedure, not a fixed winning site.
    """
    n, sites, _ = features.shape
    if map_sites is None:
        map_sites = sites
    locations = len(gram)
    error = np.zeros((sites, locations))
    baseline = np.zeros(locations)
    selected_error = np.zeros(locations)
    selections = []
    all_designs = np.arange(n)
    for fold, test in enumerate(folds):
        test = np.asarray(test)
        train = np.setdiff1d(all_designs, test)
        mean_weights = np.zeros((1, locations, len(test), n))
        mean_weights[..., train] = 1/len(train)
        baseline += weights_mse(mean_weights, gram, test)[0].sum(axis=-1)
        fold_best = np.full(locations, -np.inf)
        fold_site = np.zeros(locations, dtype=int)
        fold_feature = np.zeros(locations, dtype=int)
        fold_error = np.zeros(locations)
        for start in range(0, sites, 512):
            stop = min(start+512, sites)
            local = features[:, start:stop]
            chosen, strength = choose_features(local, gram, train)
            weights = prediction_weights(local, train, test, chosen)
            loss = weights_mse(weights, gram, test).sum(axis=-1)
            error[start:stop] += loss
            strength = np.nan_to_num(strength, nan=-1)
            strength[max(0, map_sites-start):] = -2
            at = strength.argmax(axis=0)
            value = strength[at, np.arange(locations)]
            improved = value > fold_best
            fold_best[improved] = value[improved]
            fold_site[improved] = (start+at)[improved]
            fold_feature[improved] = chosen[at, np.arange(locations)][improved]
            fold_error[improved] = loss[at, np.arange(locations)][improved]
        selected_error += fold_error
        selections.append(dict(test_designs=test.tolist(), selected_sites=fold_site.tolist(), selected_features=fold_feature.tolist()))
        print(f'{label}: fold {fold+1}/{len(folds)} complete.', flush=True)
    skill = 1-error/baseline
    return dict(skill=skill, mse=error/n, baseline_mse=baseline/n,
                selected_skill=1-selected_error/baseline, selections=selections)


def best_distinct(values, xy, count=3, spacing=100):
    candidates = np.flatnonzero(np.isfinite(values))
    candidates = candidates[np.argsort(-values[candidates], kind='stable')]
    chosen = []
    for node in candidates:
        if all(np.linalg.norm(xy[node]-xy[other])>=spacing for other in chosen):
            chosen.append(int(node))
            if len(chosen) == count:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/hic30_node_response_maps/analysis_arrays.npz')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'Data/HoodImpact_1704_EuroNCAP')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/hic30_feature_response_maps')
    parser.add_argument('--reuse-features', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--compute-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if not args.render_only:
        with threadpool_limits(limits=4):
            compute(args)
    if not args.compute_only:
        from render_hic30_feature_response import render
        render(args.output_dir, args.data_dir)


def compute(args):
    out = args.output_dir
    source = dict(np.load(args.source, allow_pickle=False))
    xyz, ids = source['ref_xyz'], source['ref_ids']
    if args.reuse_features:
        saved = np.load(out/'geometry_features.npz', allow_pickle=False)
        features, radii, nearest = saved['features'], saved['radii'], saved['nearest_ids']
        np.testing.assert_array_equal(saved['ref_ids'], ids)
        np.testing.assert_array_equal(saved['locations'], source['locations'])
    else:
        features, radii, nearest = extract_features(args.data_dir, xyz, source['impacts'])
        np.savez_compressed(out/'geometry_features.npz', features=features, radii=radii, nearest_ids=nearest,
                            feature_names=np.asarray(FEATURES), ref_ids=ids, locations=source['locations'])
    # Center curves before Gram construction for numerically stable errors.
    # Prediction weights sum to one, so subtracting the same curve from all
    # designs changes neither training correlations nor held-out errors.
    full = source['full']
    centered = full-full.mean(axis=0, keepdims=True)
    gram = curve_gram(centered)
    indices = []; strength = []
    feature_scores = np.empty((len(xyz), len(FEATURES), len(gram)), dtype=np.float32)
    for start in range(0, len(xyz), 512):
        stop = min(start+512, len(xyz))
        r2 = feature_associations(features[:, start:stop], gram, np.arange(12))
        best = np.argmax(np.nan_to_num(r2, nan=-1), axis=1)
        indices.append(best)
        strength.append(np.sqrt(np.take_along_axis(r2, best[:, None, :], axis=1)[:, 0]))
        feature_scores[start:stop] = np.sqrt(r2)
    indices, strength = np.concatenate(indices), np.concatenate(strength)
    print('Direct feature/curve correlations complete; starting held-out checks.', flush=True)
    design = evaluate_folds(features, gram, [[i] for i in range(12)], 'Held-out design', len(xyz))
    family = evaluate_folds(features, gram, FAMILIES, 'Held-out family', len(xyz))
    # Impact sites are included as a predeclared baseline, not as map candidates.
    impact_design = np.diag(design['skill'][len(xyz):])
    impact_family = np.diag(family['skill'][len(xyz):])
    np.savez_compressed(out/'feature_response_arrays.npz',
        ref_ids=ids, ref_xyz=xyz, ref_part=source['ref_part'], locations=source['locations'],
        impacts=source['impacts'], range_percent=source['range_percent'], times=source['times'],
        association=strength, best_feature=indices, feature_associations=feature_scores,
        design_skill=design['skill'][:len(xyz)], family_skill=family['skill'][:len(xyz)],
        design_baseline_mse=design['baseline_mse'], family_baseline_mse=family['baseline_mse'],
        selection_design_skill=design['selected_skill'], selection_family_skill=family['selected_skill'],
        impact_design_skill=impact_design, impact_family_skill=impact_family)
    candidates = []; summary = []; values = []
    (out/'node_scores').mkdir(exist_ok=True)
    for il, loc in enumerate(source['locations']):
        frame = pd.DataFrame(dict(node_id=ids, X1_mm=xyz[:, 0], X2_mm=xyz[:, 1], X3_mm=xyz[:, 2],
            association=strength[:, il], best_feature=[FEATURES[f] for f in indices[:, il]],
            held_out_design_skill=design['skill'][:len(xyz), il], held_out_family_skill=family['skill'][:len(xyz), il],
            advantage_over_impact_features_family=family['skill'][:len(xyz), il]-impact_family[il]))
        frame.to_csv(out/'node_scores'/f'location_{loc:03d}.csv.gz', index=False)
        chosen = best_distinct(strength[:, il], xyz[:, :2])
        for rank, node in enumerate(chosen, 1):
            row = frame.iloc[node].to_dict()
            row.update(location=int(loc), rank=rank, node_index=node)
            candidates.append(row)
            f = indices[node, il]
            for d in range(12):
                values.append(dict(location=int(loc), rank=rank, reference_node_id=int(ids[node]), design=d,
                    feature=FEATURES[f], feature_value=features[d, node, f],
                    nearest_inner_node=int(nearest[d, node, 0]), nearest_outer_node=int(nearest[d, node, 1]),
                    inner_radius_32_mm=float(radii[d, node, 0, 0]), inner_radius_256_mm=float(radii[d, node, 0, 1]),
                    outer_radius_32_mm=float(radii[d, node, 1, 0]), outer_radius_256_mm=float(radii[d, node, 1, 1])))
        top = chosen[0]
        summary.append(dict(location=int(loc), hic_range_percent=source['range_percent'][il],
            best_association=strength[top, il], candidate_node_id=int(ids[top]), candidate_feature=FEATURES[indices[top, il]],
            candidate_design_skill=design['skill'][top, il], candidate_family_skill=family['skill'][top, il],
            best_node_design_skill=np.nanmax(design['skill'][:len(xyz), il]),
            best_node_family_skill=np.nanmax(family['skill'][:len(xyz), il]),
            impact_design_skill=impact_design[il], impact_family_skill=impact_family[il],
            training_selected_region_design_skill=design['selected_skill'][il],
            training_selected_region_family_skill=family['selected_skill'][il]))
    pd.DataFrame(candidates).to_csv(out/'top_regions.csv', index=False)
    pd.DataFrame(values).to_csv(out/'candidate_feature_values.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'location_summary.csv', index=False)
    metadata = dict(feature_names=FEATURES, feature_precision=PRECISION, neighborhood_nodes=COUNTS,
        locations=source['locations'], num_sites=len(xyz), source=str(args.source.resolve()),
        feature_scope='Both structural panels at every reference XY site; no changed-node or impact-distance filter.',
        families=GEOMETRY_CLUSTERS, design_selections=design['selections'], family_selections=family['selections'],
        definition='sqrt(fraction of between-design full-curve variance explained by a single geometric feature); largest among 24 predefined features.',
        cv_definition='1 - squared held-out curve error / squared held-out training-mean-curve error; feature chosen on training data only.',
        units='Heights, spread, roughness, distance, separation: mm; slopes: mm/mm.',
        limitations=['Descriptive associations are not causal importance or current-network attribution.',
            'All 12 designs and 46 outcome-filtered locations are exploratory; map peaks are selected from many sites.',
            'Held-out checks use a simple linear feature probe, not all possible nonlinear feature combinations.',
            'Unchanged center coordinates are allowed; identical context features cannot distinguish designs.',
            'Nearest-neighbor neighborhoods can contain remeshing/sampling effects; separation is a mean-height proxy, not measured contact clearance.',
            'The same surrounding information can be represented at several nodes; red centers are not unique physical causes.'])
    (out/'analysis_metadata.json').write_text(json.dumps(clean_json(metadata), indent=2), encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
