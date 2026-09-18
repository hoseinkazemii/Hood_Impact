"""Matched-location geometry/response diagnostics and a frozen distance-prior sweep.

This is an exploratory analysis, not a causal FE perturbation experiment or a
retrained architecture comparison. Outputs keep held-out cluster D identifiable.
Run with --skip-model for the simulation-only figure, or --plot-only to redraw.
"""
from __future__ import annotations

import argparse
import ast
from itertools import combinations
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from abaqus_scripts.inp_geom import Deck
from mesh_impact_history import MeshImpactHistoryNet

ROOT = Path(__file__).resolve().parent
RADII = np.array([25, 50, 75, 100, 150, 200, 300, 400, 600, 800], float)
FACTORS = np.array([0, .25, .5, 1, 2, 4], float)
COLORS = ["#2878b5", "#db7137", "#278368", "#7857a1"]


def cluster_mapping():
    """Read the authoritative literal without importing the training stack."""
    tree = ast.parse((ROOT / "mesh_design_sensitivity.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GEOMETRY_CLUSTERS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("GEOMETRY_CLUSTERS literal not found")


def batched_hic(times, acceleration, window=.015):
    """Exact trapezoidal HIC over all admissible pairs on a shared time grid."""
    times = np.asarray(times, float)
    a = np.asarray(acceleration, float)
    if a.shape[-1] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Acceleration and strictly increasing times must align")
    integral = np.concatenate((np.zeros((*a.shape[:-1], 1)), np.cumsum(
        .5 * (a[..., 1:] + a[..., :-1]) * np.diff(times), axis=-1)), axis=-1)
    best = np.zeros(a.shape[:-1])
    for lag in range(1, len(times)):
        duration = times[lag:] - times[:-lag]
        valid = duration <= window
        if not valid.any():
            break
        mean = (integral[..., lag:] - integral[..., :-lag])[..., valid] / duration[valid]
        best = np.maximum(best, (np.maximum(mean, 0)**2.5 * duration[valid]).max(axis=-1))
    return best


def row_correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denom = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1), denom, out=np.zeros_like(denom), where=denom > 1e-12)


def rank_association(geometry, response, location_control=False):
    """Equal-pair Spearman association; optional pair + location rank residuals.

    Inputs are (cluster pairs, matched impact locations). No independence or
    significance claim is made for these dependent observations.
    """
    x, y = rankdata(geometry, axis=1), rankdata(response, axis=1)
    if location_control:
        x = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()
        y = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
        return float(row_correlation(x.ravel(), y.ravel()))
    return float(row_correlation(x, y).mean())


def symmetric_cell_mismatch(first, second, cell_mm=20.):
    """Symmetric 3D nearest-node mismatch, aggregated into equal-area XY cells.

    This point-cloud descriptor retains remeshing effects; it is not a measured
    displacement or a point-to-surface distance. Equal XY cell weighting limits
    domination by densely meshed regions and preserves multiple shell layers.
    """
    distances = np.concatenate((cKDTree(second).query(first)[0], cKDTree(first).query(second)[0]))
    xy = np.concatenate((first[:, :2], second[:, :2]))
    cells, inverse = np.unique(np.floor(xy / cell_mm).astype(int), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    squared = np.bincount(inverse, weights=distances**2) / count
    return (cells + .5) * cell_mm, squared


def local_mismatch(xy, squared, impacts, radii=RADII):
    distances = np.linalg.norm(impacts[:, None, :] - xy[None, :, :], axis=-1)
    output = []
    for radius in radii:
        inside = distances <= radius
        if np.any(inside.sum(1) == 0):
            raise ValueError(f"No geometry cell inside radius {radius}")
        output.append(np.sqrt((inside * squared).sum(1) / inside.sum(1)))
    return np.stack(output, axis=-1)


def read_simulation(data_dir, out_dir):
    clusters = cluster_mapping()
    impacts = pd.read_csv(data_dir / "ImpactCoords_1704.csv")[["X1", "X2"]].to_numpy().reshape(12, 142, 2)
    if not np.allclose(impacts, impacts[:1], atol=.001, rtol=0):
        raise ValueError("Impact XY differs between designs")
    histories, times = [], None
    for run in range(1, 1705):
        frame = pd.read_csv(data_dir / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv",
                            usecols=["Time", "A(in g)"])
        t, a = frame["Time"].to_numpy(), frame["A(in g)"].to_numpy()
        if times is None:
            times = t
        if t.shape != times.shape or not np.allclose(t, times, atol=1e-10, rtol=0):
            raise ValueError(f"Time grid mismatch at run {run}")
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite history at run {run}")
        histories.append(a)
    acc = np.asarray(histories).reshape(12, 142, -1)
    print("Loaded and aligned all 1,704 full-resolution histories", flush=True)
    hic = batched_hic(times, acc)
    cluster_acc = np.stack([acc[list(m)].mean(0) for m in clusters.values()])
    cluster_hic = np.stack([hic[list(m)].mean(0) for m in clusters.values()])
    clouds, decks, audit, neighbor_scales = [], [], [], []
    for name, members in clusters.items():
        design = members[0]
        deck = Deck(data_dir / "inp_files" / f"HoodImpact_{142*design+1}.inp")
        impactor = deck.impactor_node_ids()
        if set(list(deck.nodes)[:286]) != impactor:
            raise ValueError(f"Headform node order differs for cluster {name}")
        shell_ids = set(n for element in deck.elems.values() for n in element) - impactor
        _, cloud = deck.coords(shell_ids)
        clouds.append(cloud)
        decks.append(deck)
        audit.append(dict(cluster=name, representative_design=design, shell_nodes=len(cloud), headform_nodes=len(impactor)))
        structural = np.array(list(deck.nodes.values()))[286:]
        neighbor_radius = cKDTree(structural).query(structural, k=16)[0][:, -1]
        neighbor_scales.append(dict(cluster=name, design=design,
                                    p10_mm=float(np.percentile(neighbor_radius,10)),
                                    median_mm=float(np.median(neighbor_radius)),
                                    p90_mm=float(np.percentile(neighbor_radius,90))))
    pairs = list(combinations(range(4), 2))
    geometry, cell_xy, cell_mse, aligned = [], [], [], []
    for i, j in pairs:
        xy, sq = symmetric_cell_mismatch(clouds[i], clouds[j])
        geometry.append(local_mismatch(xy, sq, impacts[0]))
        cell_xy.append(xy)
        cell_mse.append(sq)
        # Independent descriptor for same-topology pairs: exact node-ID motion.
        if (i, j) in ((0, 1), (2, 3)):
            a, b = decks[i], decks[j]
            if a.elems != b.elems:
                raise ValueError("Same-family element connectivity changed")
            ids = sorted(set(n for e in a.elems.values() for n in e) - a.impactor_node_ids())
            pa = np.array([a.nodes[n] for n in ids])
            pb = np.array([b.nodes[n] for n in ids])
            cells, inv = np.unique(np.floor((pa[:, :2]+pb[:, :2]) / 40).astype(int), axis=0, return_inverse=True)
            sq2 = np.bincount(inv, weights=((pa-pb)**2).sum(1)) / np.bincount(inv)
            aligned.append(local_mismatch((cells+.5)*20, sq2, impacts[0]))
    geometry = np.array(geometry)
    delta_acc = np.array([cluster_acc[i]-cluster_acc[j] for i, j in pairs])
    curve_difference = np.sqrt(np.mean(delta_acc**2, axis=-1))
    hic_difference = np.array([np.abs(cluster_hic[i]-cluster_hic[j]) for i, j in pairs])
    rows = []
    for pair_index, (i, j) in enumerate(pairs):
        for loc in range(142):
            row = dict(pair="ABCD"[i]+"ABCD"[j], location=loc+1, involves_test_D=j == 3,
                       impact_x_mm=impacts[0, loc, 0], impact_y_mm=impacts[0, loc, 1],
                       history_difference_rms_g=curve_difference[pair_index, loc],
                       absolute_hic_difference=hic_difference[pair_index, loc])
            row.update({f"geometry_rms_{r:g}mm": geometry[pair_index, loc, k] for k, r in enumerate(RADII)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "matched_geometry_response.csv", index=False)
    # Cell coordinates differ by pair; save a single long geometry table.
    pd.concat([pd.DataFrame(dict(pair=p, x_mm=xy[:, 0], y_mm=xy[:, 1], mismatch_squared_mm2=sq))
               for p, xy, sq in zip(range(6), cell_xy, cell_mse)]).to_csv(out_dir / "geometry_cells.csv", index=False)
    np.savez_compressed(out_dir / "simulation_arrays.npz", times=times, acc=acc, hic=hic,
                        cluster_acc=cluster_acc, cluster_hic=cluster_hic, impacts=impacts[0],
                        geometry=geometry, aligned_geometry=np.array(aligned), curve_difference=curve_difference,
                        hic_difference=hic_difference, delta_acc=delta_acc)
    (out_dir / "geometry_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "neighborhood_physical_scales.json").write_text(json.dumps(neighbor_scales, indent=2), encoding="utf-8")
    print("Computed six cluster-pair geometry comparisons; headform excluded by element sets", flush=True)


def simulation_summary(arrays, out_dir):
    g, response, hic = arrays["geometry"], arrays["curve_difference"], arrays["hic_difference"]
    pairs = list(combinations(range(4), 2))
    train = [p for p, (_, j) in enumerate(pairs) if j != 3]
    test = [p for p, (_, j) in enumerate(pairs) if j == 3]
    rows = []
    for k, radius in enumerate(RADII):
        rows.append(dict(radius_mm=radius,
                         history_all_pairs=rank_association(g[:, :, k], response),
                         history_training_pairs=rank_association(g[train, :, k], response[train]),
                         history_D_pairs=rank_association(g[test, :, k], response[test]),
                         history_pair_and_location_controlled=rank_association(g[:, :, k], response, True),
                         hic_all_pairs=rank_association(g[:, :, k], hic),
                         hic_pair_and_location_controlled=rank_association(g[:, :, k], hic, True),
                         aligned_AB=float(row_correlation(rankdata(arrays["aligned_geometry"][0, :, k]), rankdata(response[0]))),
                         aligned_CD=float(row_correlation(rankdata(arrays["aligned_geometry"][1, :, k]), rankdata(response[-1])))))
    scale = pd.DataFrame(rows)
    scale.to_csv(out_dir / "radius_associations.csv", index=False)
    time_rows = []
    for lo, hi in zip([0, 5, 10, 15, 20], [5, 10, 15, 20, 25.01]):
        selected = (arrays["times"] >= lo / 1000) & (arrays["times"] < hi / 1000)
        interval_difference = np.sqrt(np.mean(arrays["delta_acc"][:, :, selected]**2, axis=-1))
        for k, radius in enumerate(RADII):
            time_rows.append(dict(start_ms=lo, end_ms=min(hi, 25), radius_mm=radius,
                                  pair_and_location_controlled=rank_association(g[:, :, k], interval_difference, True)))
    pd.DataFrame(time_rows).to_csv(out_dir / "time_radius_associations.csv", index=False)
    # Descriptive geometry quartiles are set separately within each cluster pair.
    k = int(np.flatnonzero(RADII == 100)[0])
    ranks = rankdata(g[:, :, k], axis=1)
    quartile = np.minimum(((ranks - 1) * 4 / 142).astype(int), 3)
    qrows = []
    for p, (i, j) in enumerate(pairs):
        for q in range(4):
            selected = quartile[p] == q
            qrows.append(dict(pair="ABCD"[i]+"ABCD"[j], geometry_quartile=q+1,
                              mean_history_difference_g=response[p, selected].mean(),
                              mean_hic_difference=hic[p, selected].mean()))
    pd.DataFrame(qrows).to_csv(out_dir / "geometry_quartiles.csv", index=False)
    cluster_acc = arrays["cluster_acc"]
    clusters = cluster_mapping()
    # Within-cluster scatter is descriptive, not asserted to be pure solver noise.
    within = np.stack([np.mean((arrays["acc"][list(m)] - cluster_acc[i])**2, axis=(0, 2))
                       for i, m in enumerate(clusters.values())])
    across = np.sqrt(np.mean((cluster_acc - cluster_acc.mean(0))**2, axis=(0, 2)))
    summary = dict(clusters={n:list(m) for n,m in clusters.items()}, locations=142,
                   between_cluster_history_rms_median_g=float(np.median(across)),
                   within_cluster_history_rms_median_g=float(np.median(np.sqrt(within.mean(0)))),
                   association_at_100mm=scale.loc[scale.radius_mm == 100].iloc[0].to_dict(),
                   time_association_at_100mm=[r for r in time_rows if r["radius_mm"] == 100],
                   exploratory_max_history_association=scale.loc[scale.history_all_pairs.idxmax()].to_dict(),
                   caveats=["Observational whole-design comparisons do not identify causal node importance.",
                            "Only four geometry clusters; six pairs share clusters and are not independent.",
                            "Nearest-node mismatch includes remeshing and ignores material/connectivity differences.",
                            "Matched node-ID displacement corroboration uses the A-B and C-D families only.",
                            "Spatially adjacent impact locations are dependent; no independent-sample p-values reported.",
                            "D-containing analyses are exploratory test-set diagnostics, not hyperparameter selection."])
    (out_dir / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return scale


def frozen_prior_predictions(model, mesh, impact, times, factors=FACTORS):
    """Reuse node encoding; vary ONLY the pooling distance-score multiplier.

    A factor of zero removes only the explicit distance bias. Impact features,
    neighborhood attention, learned content attention and all weights remain.
    Each variant decodes separately to match single-sample baseline arithmetic.
    """
    import torch
    from torch.nn import functional as F
    with torch.inference_mode():
        condition = model.impact_embedding(impact[None])[0]
        features = model.node_features(mesh, impact)
        nodes = model.node_embedding(features)
        if model.neighborhood_blocks:
            nodes = model._encode_neighborhoods(mesh, nodes)
        tokens = model.latent_queries + condition[None]
        w, heads, count = model.width // model.num_heads, model.num_heads, model.num_latents
        query = model.mesh_query(model.query_norm(tokens)).reshape(count, heads, w).transpose(0, 1)
        key, value = model.mesh_key_value(nodes).chunk(2, dim=-1)
        key = key.reshape(-1, heads, w).transpose(0, 1)
        value = value.reshape(-1, heads, w).transpose(0, 1)
        precision = torch.cat((model.local_log_precision.new_zeros(model.num_global_latents),
                               F.softplus(model.local_log_precision)))
        spatial_bias = -precision[:, None] * features[:, 5][None]
        tf = torch.stack((times, times.square(), times.pow(3), times.tanh(), torch.exp(-times.square())), dim=-1)
        tq = model.time_embedding(tf)[None] + condition[None, None]
        mask = torch.zeros(1, len(times), dtype=torch.bool, device=times.device)
        predictions = []
        for factor in factors:
            update = F.scaled_dot_product_attention(query[None], key[None], value[None],
                                                    attn_mask=(spatial_bias * float(factor))[None, None], dropout_p=0.)
            update = update.squeeze(0).transpose(0, 1).reshape(count, model.width)
            memory = (tokens + model.mesh_output(update))[None]
            for block in model.latent_blocks:
                memory = block(memory)
            memory = model.memory_norm(memory)
            decoded = tq
            for block in model.decoder_blocks:
                decoded = block(decoded, memory, mask)
            predictions.append(model.acceleration_head(decoded)[0, :, 0])
        return torch.stack(predictions)


def score_predictions(prediction, truth, predicted_hic, true_hic):
    return dict(history_rmse_g=float(np.sqrt(np.mean((prediction-truth)**2))),
                hic_r2=float(1-np.sum((predicted_hic-true_hic)**2)/np.sum((true_hic-true_hic.mean())**2)),
                hic_rmse=float(np.sqrt(np.mean((predicted_hic-true_hic)**2))),
                hic_mape_percent=float(100*np.mean(np.abs(predicted_hic-true_hic)/true_hic)))


def model_diagnostic(data_dir, run_dir, out_dir, arrays, device):
    import joblib
    import torch
    from torch.nn import functional as F
    from visualize_design_sensitivity_1704 import load_nodes
    config = json.loads((run_dir / "config.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())
    model = MeshImpactHistoryNet(**config["architecture"]["kwargs"])
    checkpoint = torch.load(run_dir / "hood_impact_best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    scalers = joblib.load(run_dir / "scalers.joblib")
    times = np.load(run_dir / "prediction_times.npy").astype(np.float32)
    times_tensor = torch.as_tensor(scalers["time"].transform(times[:, None]).ravel(), device=device)
    impacts = pd.read_csv(data_dir / "ImpactCoords_1704.csv")[["X1", "X2"]].to_numpy(np.float32)
    all_predictions, runs, split_names = [], [], []
    started = time.perf_counter()
    check_errors = []
    for split, source_split in (("val", "validation"), ("test", "test")):
        for design in splits[source_split]["design_ids"]:
            for location in range(142):
                run = 142*design + location + 1
                # Read actual decks, including exact headform position, rather than reconstructing.
                raw = load_nodes(data_dir / "inp_files", run).astype(np.float32)
                mesh = torch.as_tensor(scalers["mesh"].transform(raw), device=device)
                impact = torch.as_tensor(scalers["indentor"].transform(impacts[run-1:run])[0], device=device)
                prediction = frozen_prior_predictions(model, mesh, impact, times_tensor)
                if location in (0, 141):
                    with torch.inference_mode():
                        ordinary = model(mesh, torch.zeros(len(mesh), dtype=torch.long, device=device), impact[None],
                                         times_tensor, torch.zeros(len(times), dtype=torch.long, device=device))
                    error = float((prediction[3]-ordinary).abs().max().cpu())
                    if error > 2e-5:
                        raise ValueError(f"Factor=1 failed baseline equivalence: {error}")
                    check_errors.append(error)
                normalized = prediction.cpu().numpy()
                physical = scalers["accel"].inverse_transform(normalized.reshape(-1, 1)).reshape(normalized.shape)
                all_predictions.append(physical)
                runs.append(run)
                split_names.append(split)
                if len(runs) % 30 == 0 or len(runs) == 1:
                    print(f"Frozen prior sweep: {len(runs)}/426 impacts, six factors, {time.perf_counter()-started:.0f}s", flush=True)
    predictions = np.asarray(all_predictions)
    run_indices = np.array(runs)-1
    full_truth = arrays["acc"].reshape(1704, -1)[run_indices]
    full_hic = arrays["hic"].ravel()[run_indices]
    sampled_truth = np.stack([np.interp(times, arrays["times"], a) for a in full_truth])
    predicted_hic = batched_hic(times, predictions)
    np.savez_compressed(out_dir / "frozen_prior_arrays.npz", runs=runs, splits=split_names, times=times,
                        factors=FACTORS, prediction=predictions, truth=sampled_truth, true_hic=full_hic,
                        predicted_hic=predicted_hic)
    rows = []
    for split in ("val", "test"):
        sel = np.array(split_names) == split
        for f, factor in enumerate(FACTORS):
            rows.append(dict(split=split, distance_multiplier=factor, n=int(sel.sum()),
                             **score_predictions(predictions[sel, f], sampled_truth[sel], predicted_hic[sel, f], full_hic[sel])))
    # Location-only baseline: average actual TRAINING histories at that same XY.
    train_designs = splits["train"]["design_ids"]
    mean_train = arrays["acc"][train_designs].mean(0)
    baseline = np.stack([np.interp(times, arrays["times"], a) for a in mean_train])
    baseline_hic = batched_hic(times, baseline)
    location_indices = run_indices % 142
    baseline_rows = []
    for split in ("val", "test"):
        sel = np.array(split_names) == split
        locs = location_indices[sel]
        baseline_rows.append(dict(split=split, **score_predictions(baseline[locs], sampled_truth[sel],
                                                                 baseline_hic[locs], full_hic[sel])))
    pd.DataFrame(rows).to_csv(out_dir / "frozen_prior_metrics.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out_dir / "location_only_baseline.csv", index=False)
    precision = F.softplus(model.local_log_precision).detach().cpu().numpy()
    mesh_scale = model.mesh_scale.detach().cpu().numpy()
    prior = pd.DataFrame(dict(token=np.arange(1, len(precision)+1), learned_precision=precision,
                             initial_precision=np.geomspace(.5,32,len(precision)),
                             e_fold_x_mm=mesh_scale[0]/np.sqrt(precision), e_fold_y_mm=mesh_scale[1]/np.sqrt(precision)))
    prior.to_csv(out_dir / "learned_distance_prior.csv", index=False)
    # Conditional resampling of matched test locations, keeping the two designs together.
    # This is not a confidence interval over unseen clusters or training seeds.
    test_sel = np.array(split_names) == "test"
    errors = (predicted_hic[test_sel] - full_hic[test_sel, None])**2
    errors = errors.reshape(len(splits["test"]["design_ids"]), 142, len(FACTORS)).mean(0)
    rng = np.random.default_rng(42)
    # Coarse XY blocks retain some local spatial dependence.
    block_xy = np.floor(arrays["impacts"] / 300).astype(int)
    _, block = np.unique(block_xy, axis=0, return_inverse=True)
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    changes = []
    for _ in range(2000):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        means = np.sqrt(errors[indices].mean(0))
        changes.append(means-means[3])
    interval = np.percentile(changes, [2.5,97.5], axis=0)
    pd.DataFrame(dict(distance_multiplier=FACTORS, hic_rmse_change_lower=interval[0],
                      hic_rmse_change_upper=interval[1])).to_csv(out_dir / "conditional_spatial_bootstrap.csv", index=False)
    report = dict(run=str(run_dir), architecture=config["architecture"], training=config["training"],
                  factors=FACTORS.tolist(), max_normalized_factor_one_equivalence_error=max(check_errors),
                  bootstrap_spatial_blocks=len(groups),
                  caveats=["Inference-only intervention; weights were trained with multiplier 1.",
                           "Zero removes the explicit pooling distance penalty, not other impact information.",
                           "HIC scores use full-resolution simulations; history RMSE uses the checkpoint time grid.",
                           "Validation design 5 shares cluster B with training design 4.",
                           "Test D curves are diagnostic; do not select hyperparameters on D.",
                           "Bootstrap conditions on this one held-out cluster and fixed checkpoint."])
    (out_dir / "model_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.17)
    ax.set_axisbelow(True)


def plot_simulation(out_dir, arrays, scale):
    plt.rcParams.update({"font.size":10, "axes.titlesize":12, "axes.labelsize":10, "font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    xy = arrays["impacts"]
    cell = pd.read_csv(out_dir / "geometry_cells.csv")
    background = cell.groupby(["x_mm", "y_mm"]).mismatch_squared_mm2.mean().reset_index()
    mismatch = np.sqrt(background.mismatch_squared_mm2.to_numpy())
    ax = axes[0,0]
    dots = ax.scatter(background.x_mm, background.y_mm, c=mismatch, s=3, cmap="magma_r",
                      vmin=0, vmax=np.percentile(mismatch,98), rasterized=True)
    ax.scatter(xy[:,0], xy[:,1], facecolors="none", edgecolors="#278368", s=12, linewidths=.5)
    fig.colorbar(dots, ax=ax, label="RMS nearest-node mismatch (mm)", shrink=.8)
    ax.set(title="A  Where the four hood geometries differ", xlabel="X (mm)", ylabel="Y (mm)", aspect="equal")
    ax = axes[0,1]
    ca = arrays["cluster_acc"]
    spread = np.sqrt(np.mean((ca-ca.mean(0))**2, axis=(0,2)))
    ax.scatter(background.x_mm, background.y_mm, c="#dedede", s=1, rasterized=True)
    dots=ax.scatter(xy[:,0],xy[:,1],c=spread,s=40,cmap="viridis",edgecolors="white",linewidths=.3)
    fig.colorbar(dots, ax=ax, label="Between-cluster history spread (g RMS)", shrink=.8)
    ax.set(title="B  Where acceleration differs across designs", xlabel="X (mm)",ylabel="Y (mm)",aspect="equal")
    ax=axes[0,2]
    for col, label, color, ls in (("history_all_pairs","History: all six pairs",COLORS[0],"-"),
                                  ("hic_all_pairs","HIC: all six pairs",COLORS[1],"-"),
                                  ("history_pair_and_location_controlled","History: location also controlled",COLORS[0],"--"),
                                  ("hic_pair_and_location_controlled","HIC: location also controlled",COLORS[1],"--")):
        ax.plot(RADII,scale[col],color=color,ls=ls,marker="o",ms=3,label=label)
    ax.axhline(0,color="black",lw=.7)
    ax.set(xlabel="Geometry neighborhood radius (mm)",ylabel="Rank association",title="C  Does nearby geometry track response differences?")
    ax.legend(fontsize=8,loc="best")
    ax=axes[1,0]
    ax.plot(RADII,scale.history_training_pairs,"o-",color=COLORS[2],label="A-B, A-C, B-C (training clusters)")
    ax.plot(RADII,scale.history_D_pairs,"s-",color=COLORS[3],label="A-D, B-D, C-D (includes test D)")
    ax.plot(RADII,scale.aligned_AB,"--",color=COLORS[1],label="A-B: corresponding-node displacement")
    ax.plot(RADII,scale.aligned_CD,"--",color="#777777",label="C-D: corresponding-node displacement")
    ax.axhline(0,color="black",lw=.7)
    ax.set(xlabel="Geometry neighborhood radius (mm)",ylabel="History rank association",title="D  Check cluster dependence and remeshing")
    ax.legend(fontsize=8,loc="best")
    ax=axes[1,1]
    quartiles=pd.read_csv(out_dir / "geometry_quartiles.csv")
    for i,(pair,group) in enumerate(quartiles.groupby("pair")):
        ax.plot(group.geometry_quartile,group.mean_history_difference_g,marker="o",alpha=.8,label=pair)
    ax.set(xticks=[1,2,3,4],xticklabels=["Least","2","3","Most"], xlabel="Local geometry difference quartile (100 mm)",
           ylabel="Mean matched-pair history difference (g RMS)",title="E  More local change: larger response change?")
    ax.legend(ncol=3,fontsize=8)
    ax=axes[1,2]
    bins=np.array([0,5,10,15,20,25.01])/1000
    heat=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        sel=(arrays["times"]>=lo)&(arrays["times"]<hi)
        response=np.sqrt(np.mean(arrays["delta_acc"][:,:,sel]**2,axis=-1))
        heat.append([rank_association(arrays["geometry"][:,:,k],response,True) for k in range(len(RADII))])
    im=ax.imshow(heat,origin="lower",aspect="auto",cmap="RdBu_r",vmin=-.6,vmax=.6)
    ax.set(xticks=range(len(RADII)),xticklabels=[f"{r:g}" for r in RADII],yticks=range(5),
           yticklabels=["0–5","5–10","10–15","15–20","20–25"],xlabel="Geometry neighborhood radius (mm)",
           ylabel="History interval (ms)",title="F  Does the relevant scale change over time?")
    fig.colorbar(im,ax=ax,label="Pair + location controlled rank association",shrink=.8)
    for ax in axes.flat:
        style(ax)
    axes[1,2].grid(False)
    fig.suptitle("Do nodes near the impact carry information about design-dependent response?\n"
                 "1,704 simulations · 142 matched impact locations · four equally weighted geometry clusters",fontsize=16)
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    fig.text(.02,.018,"Exploratory association, not causal node importance. Six cluster pairs are dependent. Geometry excludes the headform; 20 mm XY cells limit mesh-density bias.\n"
             "Solid curves average within-pair Spearman correlations. Dashed curves in C also remove shared location effects. D-containing results must not select tuning parameters.",fontsize=9,color="#444444")
    for ext in ("png","pdf"):
        fig.savefig(out_dir / f"impact_locality_simulation.{ext}",dpi=190,facecolor="white")
    plt.close(fig)


def plot_model(out_dir):
    metrics=pd.read_csv(out_dir / "frozen_prior_metrics.csv")
    baseline=pd.read_csv(out_dir / "location_only_baseline.csv")
    prior=pd.read_csv(out_dir / "learned_distance_prior.csv")
    fig,axes=plt.subplots(1,3,figsize=(17,6.3))
    fig.subplots_adjust(left=.055,right=.985,bottom=.25,top=.77,wspace=.30)
    for split,label,color in (("val","Validation B (near-clone of training)",COLORS[2]),("test","Held-out cluster D",COLORS[3])):
        selected=metrics[metrics.split==split]
        for ax,metric in zip(axes[:2],["hic_r2","history_rmse_g"]):
            ax.plot(range(len(FACTORS)),selected[metric],"o-",color=color,label=label,lw=2)
        base=baseline[baseline.split==split].iloc[0]
        axes[0].axhline(base.hic_r2,color=color,ls=":",alpha=.8)
        axes[1].axhline(base.history_rmse_g,color=color,ls=":",alpha=.8)
    for ax in axes[:2]:
        ax.axvline(3,color="#777777",ls="--",lw=1)
        ax.set(xticks=range(len(FACTORS)),xticklabels=["0\nno bias","0.25","0.5","1\ntrained","2","4"],
               xlabel="Multiplier of the learned distance penalty")
    axes[0].set(title="A  HIC accuracy vs full-resolution simulation",ylabel="HIC R² (higher is better)")
    axes[0].legend(fontsize=8,loc="best")
    axes[1].set(title="B  Acceleration-history accuracy",ylabel="History RMSE in g (lower is better)")
    axes[2].plot(prior.token,prior.e_fold_x_mm,"o-",color=COLORS[0],label="X semi-axis")
    axes[2].plot(prior.token,prior.e_fold_y_mm,"s-",color=COLORS[1],label="Y semi-axis")
    axes[2].set(title="C  The learned prior already covers many scales",xlabel="Impact-local pooling token",ylabel="Distance where prior falls to 1/e (mm)",xticks=[1,4,8,12,16])
    axes[2].legend(fontsize=9)
    for ax in axes:
        style(ax)
    fig.suptitle("Does stronger focus near the impact help the existing neighborhood model?\n"
                 "Frozen checkpoint · six distance penalties · 142 validation + 284 test impacts",fontsize=16)
    fig.text(.015,.035,"Only the explicit mesh-pooling distance bias changes; weights, local kNN blocks and impact inputs stay fixed. Stronger penalty = narrower focus.\n"
             "Dotted lines: location-only predictor averaging training histories at the same impact XY. This is an inference sensitivity test, not a retrained ablation.\n"
             "Token widths describe the geometric prior, not measured attention or causal importance. Test D must not be used to select the multiplier.",fontsize=9,color="#444444")
    for ext in ("png","pdf"):
        fig.savefig(out_dir / f"impact_locality_model.{ext}",dpi=190,facecolor="white")
    plt.close(fig)


def plot_history_examples(out_dir, arrays):
    """Examples selected by geometry alone, without selecting response extremes."""
    clusters = list(cluster_mapping().values())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.82, hspace=.55, wspace=.22)
    k = int(np.flatnonzero(RADII == 100)[0])
    examples = []
    for row, (pair_index, a, b) in enumerate(((0, 0, 1), (5, 2, 3))):
        order = np.argsort(arrays["geometry"][pair_index, :, k])
        for column, quantile in enumerate((.125, .875)):
            loc = int(order[round(quantile * (len(order)-1))])
            ax = axes[row, column]
            for c in (a, b):
                for design in clusters[c]:
                    ax.plot(arrays["times"]*1000, arrays["acc"][design, loc], color=COLORS[c], alpha=.22, lw=.8)
                ax.plot(arrays["times"]*1000, arrays["cluster_acc"][c, loc], color=COLORS[c], lw=2,
                        label=f"Cluster {'ABCD'[c]} mean")
            mismatch = float(arrays["geometry"][pair_index, loc, k])
            curve = float(arrays["curve_difference"][pair_index, loc])
            hic = float(arrays["hic_difference"][pair_index, loc])
            ax.set(title=f"{'ABCD'[a]}–{'ABCD'[b]} · location {loc+1} · {'less' if column == 0 else 'more'} local change\n"
                         f"Geometry {mismatch:.2f} mm RMS; history Δ {curve:.1f} g RMS; HIC Δ {hic:.0f}",
                   xlabel="Time (ms)", ylabel="Acceleration (g)")
            ax.legend(fontsize=9)
            style(ax)
            examples.append(dict(pair="ABCD"[a]+"ABCD"[b], geometry_quantile=quantile,
                                 location=loc+1, geometry_rms_mm=mismatch,
                                 history_difference_rms_g=curve, absolute_hic_difference=hic))
    fig.suptitle("Actual simulated histories at matched impact locations\n"
                 "Examples at the 12.5th and 87.5th geometry percentiles within 100 mm", fontsize=16)
    fig.text(.03, .025, "Thin lines: individual designs. Thick lines: cluster means. Examples were selected by geometry rank only, not by response separation.\n"
             "A–B and C–D each share mesh connectivity. All-panel statistics, rather than these four examples, should guide interpretation.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"impact_locality_histories.{ext}", dpi=190, facecolor="white")
    plt.close(fig)
    pd.DataFrame(examples).to_csv(out_dir / "history_examples.csv", index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/mesh_impact_history/20260914_173055_3151837_1704")
    parser.add_argument("--out-dir",type=Path,default=ROOT/"figures/impact_locality_1704")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--skip-model",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--reuse-simulation",action="store_true")
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if not args.plot_only and not args.reuse_simulation:
        read_simulation(args.data_dir,args.out_dir)
    with np.load(args.out_dir/"simulation_arrays.npz") as archive:
        arrays={name:archive[name] for name in archive.files}
    scale=simulation_summary(arrays,args.out_dir)
    plot_simulation(args.out_dir,arrays,scale)
    plot_history_examples(args.out_dir,arrays)
    if not args.skip_model:
        if not args.plot_only:
            model_diagnostic(args.data_dir,args.run_dir,args.out_dir,arrays,args.device)
        plot_model(args.out_dir)
    print(scale.to_string(index=False),flush=True)
    print(f"Saved analysis in {args.out_dir}",flush=True)


if __name__ == "__main__":
    main()
