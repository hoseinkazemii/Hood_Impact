"""Check whether feature-response map peaks identify specific geometry.

The 12 designs form four near-clone geometry clusters. A feature that only
labels the cluster reaches a computable maximum association, so the script
reports that ceiling and how many sites tie at the top of each map. It then
tests whether any feature tracks differences between near-clones better than
relabeling which clone received which response. Only saved arrays are read.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analyze_hic30_feature_response import FEATURES, PRECISION, curve_gram, centered_training_gram
from analyze_hic30_node_response import clean_json
from mesh_design_clusters import GEOMETRY_CLUSTERS

ROOT = Path(__file__).resolve().parent
# Only A and C have more than two designs; a two-design family has a single
# within-family contrast, so relabeling it never changes a squared association.
PERMUTATION_FAMILIES = ('A', 'C')
TIE_TOLERANCE = 0.01
CLUSTER_COLORS = dict(A='#2a78d6', B='#eb6834', C='#1baf7a', D='#4a3aa7')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CURVE_LOCATIONS = (127, 16, 141)
FEATURE_GROUPS = dict(
    inner=[i for i, n in enumerate(FEATURES) if n.startswith('inner_') and 'nearest' not in n],
    outer=[i for i, n in enumerate(FEATURES) if n.startswith('outer_') and 'nearest' not in n],
    separation=[i for i, n in enumerate(FEATURES) if n.startswith('panel_separation')])


def indicator(clusters, n):
    z = np.zeros((n, len(clusters)))
    for j, members in enumerate(clusters):
        z[list(members), j] = 1
    return z


def single_feature_ceiling(gram):
    """Largest squared association any one scalar per design can reach.

    Equals the top eigenvalue share of the design-centered curve Gram: the
    feature would have to equal the first principal-component score.
    """
    g = centered_training_gram(gram, np.arange(gram.shape[-1]))
    return np.linalg.eigvalsh(g)[:, -1]/np.trace(g, axis1=1, axis2=2)


def cluster_label_ceiling(gram, clusters):
    """Largest squared association for a feature constant within every cluster."""
    n = gram.shape[-1]
    g = centered_training_gram(gram, np.arange(n))
    a = (np.eye(n)-1/n) @ indicator(clusters, n)
    w, v = np.linalg.eigh(a.T @ a)
    keep = w > 1e-9
    basis = a @ (v[:, keep]/np.sqrt(w[keep]))
    return np.linalg.eigvalsh(basis.T @ g @ basis)[:, -1]/np.trace(g, axis1=1, axis2=2)


def between_cluster_fraction(gram, clusters):
    """Share of between-design curve variance carried by the cluster means."""
    n = gram.shape[-1]
    g = centered_training_gram(gram, np.arange(n))
    z = indicator(clusters, n)
    p = z @ np.linalg.pinv(z)
    return np.trace(p @ g @ p, axis1=1, axis2=2)/np.trace(g, axis1=1, axis2=2)


def within_family(values, families):
    """Designs of the given families, concatenated, each centered on its family mean."""
    return np.concatenate([values[list(f)]-values[list(f)].mean(axis=0, keepdims=True) for f in families])


def within_family_directions(features, families, precision):
    """Unit vectors of family-centered features that vary within a family.

    Features are on a rounding grid, so any real difference moves some design
    at least a quarter step from its family mean.
    """
    w = within_family(features, families).transpose(1, 2, 0)
    site, feature = np.nonzero(np.abs(w).max(axis=-1) > .25*np.asarray(precision))
    x = w[site, feature]
    return x/np.linalg.norm(x, axis=1, keepdims=True), site, feature


def family_relabelings(families):
    """Every within-family relabeling, as index arrays into the concatenated order."""
    offsets = np.cumsum([0]+[len(f) for f in families])
    blocks = [itertools.permutations(range(a, b)) for a, b in zip(offsets[:-1], offsets[1:])]
    return np.array([np.concatenate(p) for p in itertools.product(*blocks)])


def squared_associations(directions, gram):
    """Squared association of unit design vectors, shape (vectors, locations)."""
    g = gram/np.trace(gram, axis1=1, axis2=2)[:, None, None]
    outer = (directions[:, :, None]*directions[:, None, :]).reshape(len(directions), -1)
    return outer @ g.reshape(len(g), -1).T


def permutation_scan(directions, gram, perms):
    """Best association per location against its within-family relabeling null.

    The identity relabeling is included, so the smallest p-value is 1/len(perms).
    The adjusted p-value is the single-step min-p correction for scanning all
    locations (Westfall and Young), which keeps their dependence.
    """
    observed = squared_associations(directions, gram)
    best = observed.max(axis=0)
    null = np.stack([squared_associations(directions[:, p], gram).max(axis=0) for p in perms])
    p_value = (null >= best-1e-12).mean(axis=0)
    null_p = (null[None] >= null[:, None]-1e-12).mean(axis=1)
    adjusted = (null_p.min(axis=1)[:, None] <= p_value+1e-12).mean(axis=0)
    return dict(best=best, best_index=observed.argmax(axis=0), null=null, p=p_value, adjusted=adjusted)


def tie_counts(association, best, xy, tolerance=TIE_TOLERANCE):
    """Sites within tolerance of each location's best score, and their XY reach."""
    rows = []
    for il in range(association.shape[1]):
        top = np.nanargmax(association[:, il])
        tied = np.flatnonzero(association[:, il] >= best[il]-tolerance)
        rows.append((len(tied), float(np.linalg.norm(xy[tied]-xy[top], axis=1).max())))
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/hic30_node_response_maps/analysis_arrays.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/hic30_feature_response_maps')
    args = parser.parse_args()
    out = args.output_dir
    source = dict(np.load(args.source, allow_pickle=False))
    maps = dict(np.load(out/'feature_response_arrays.npz', allow_pickle=False))
    saved = np.load(out/'geometry_features.npz', allow_pickle=False)
    np.testing.assert_array_equal(saved['ref_ids'], source['ref_ids'])
    np.testing.assert_array_equal(maps['locations'], source['locations'])
    sites = len(source['ref_ids'])
    # The saved feature bank appends the 46 impact sites after the node sites.
    features = saved['features'][:, :sites]
    clusters = list(GEOMETRY_CLUSTERS.values())
    full = source['full']
    gram = curve_gram(full-full.mean(axis=0, keepdims=True))
    any_ceiling = np.sqrt(single_feature_ceiling(gram))
    label_ceiling = np.sqrt(cluster_label_ceiling(gram, clusters))
    between = between_cluster_fraction(gram, clusters)
    association = maps['association']
    best = np.nanmax(association, axis=0)
    ties = tie_counts(association, best, source['ref_xyz'][:, :2])
    at_label_ceiling = (association >= label_ceiling-TIE_TOLERANCE).sum(axis=0)

    families = [GEOMETRY_CLUSTERS[k] for k in PERMUTATION_FAMILIES]
    directions, site, feature = within_family_directions(features, families, PRECISION)
    within_gram = curve_gram(within_family(full, families))
    perms = family_relabelings(families)
    print(f'Within-family scan: {len(directions):,} varying site-features at {len(np.unique(site)):,} sites; '
          f'{len(perms)} relabelings.', flush=True)
    scan = permutation_scan(directions, within_gram, perms)

    summary = pd.DataFrame(dict(location=source['locations'], hic_range_percent=source['range_percent'],
        between_shape_fraction=between, within_shape_fraction=1-between,
        any_feature_ceiling=any_ceiling, shape_label_ceiling=label_ceiling, observed_best_association=best,
        top5_cutoff=np.nanquantile(association, .95, axis=0),
        sites_within_001_of_best=ties[:, 0], tied_site_reach_mm=ties[:, 1], sites_at_shape_label_ceiling=at_label_ceiling,
        within_AC_best_association=np.sqrt(scan['best']),
        within_AC_null95_association=np.sqrt(np.quantile(scan['null'], .95, axis=0)),
        within_AC_p=scan['p'], within_AC_adjusted_p=scan['adjusted'],
        within_AC_best_node=source['ref_ids'][site[scan['best_index']]],
        within_AC_best_feature=[FEATURES[f] for f in feature[scan['best_index']]]))
    for name, columns in FEATURE_GROUPS.items():
        summary[f'best_{name}_association'] = np.nanmax(maps['feature_associations'][:, columns], axis=(0, 1))
    summary.to_csv(out/'identifiability_summary.csv', index=False)
    np.savez_compressed(out/'identifiability_arrays.npz', locations=source['locations'], within_null_max_r2=scan['null'],
                        within_best_r2=scan['best'], relabelings=perms)
    metadata = dict(
        shape_label_ceiling='Largest association (sqrt of variance fraction) reachable by any feature constant within each of the 4 geometry clusters.',
        any_feature_ceiling='Largest association any single scalar per design can reach (first principal-component share).',
        between_shape_fraction='Fraction of between-design full-curve variance explained by the 4 cluster-mean curves.',
        tie_tolerance=TIE_TOLERANCE, permutation_families=dict(zip(PERMUTATION_FAMILIES, families)),
        relabelings=len(perms), within_varying_site_features=len(directions), within_varying_sites=len(np.unique(site)),
        within_test='Features and curves centered within families A and C; best association over all varying site-features, '
                    'compared with the same maximum after every within-family relabeling of designs. Adjusted p uses '
                    'single-step min-p over the 46 locations.',
        limitations=['Only 8 designs inform the within-family test, so modest real effects can be missed.',
                     'Families B and D have one contrast each and are excluded from the relabeling test.',
                     'The 46 locations were selected with HIC from all 12 designs.'])
    (out/'identifiability_metadata.json').write_text(json.dumps(clean_json(metadata), indent=2), encoding='utf-8')
    render_summary(out, summary)
    render_curves(out, source)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.round(3).to_string(index=False))


def style(ax):
    ax.grid(color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#b9b8b2')
    ax.tick_params(colors=MUTED, labelsize=9)


def render_summary(out, summary):
    (out/'plots').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), layout='constrained')
    s = summary.sort_values('between_shape_fraction', ascending=False)
    ax = axes[0]
    x = np.arange(len(s))
    ax.bar(x, s.between_shape_fraction, width=.8, color='#2a78d6', edgecolor='white', linewidth=1, label='Between the 4 shapes')
    ax.bar(x, s.within_shape_fraction, bottom=s.between_shape_fraction, width=.8, color='#b7d3f6', edgecolor='white',
           linewidth=1, label='Between near-clones of one shape')
    ax.set(xticks=x, ylim=(0, 1.16), yticks=np.linspace(0, 1, 6), xlim=(-.8, len(s)-.2),
           ylabel='Share of design-to-design curve variance', xlabel='Impact location (sorted)')
    ax.set_xticklabels(s.location, rotation=90, fontsize=6.5)
    ax.axhline(np.median(summary.between_shape_fraction), color=INK, linewidth=.8)
    ax.text(len(s)-1, np.median(summary.between_shape_fraction)+.015,
            f'median {100*np.median(summary.between_shape_fraction):.0f}% between shapes', ha='right', fontsize=9)
    ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=9, handlelength=1.2, borderaxespad=.2)
    ax.set_title('What the acceleration differences are made of', loc='left', weight='bold')
    style(ax)
    ax.grid(axis='x', visible=False)

    ax = axes[1]
    lo, hi = .45, .9
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=.8)
    ax.scatter(summary.shape_label_ceiling, summary.observed_best_association, s=46, color='#2a78d6',
               edgecolors='white', linewidths=1.2, zorder=3)
    for r in summary.itertuples():
        if r.observed_best_association-r.shape_label_ceiling > .12:
            ax.annotate(f'location {r.location}', (r.shape_label_ceiling, r.observed_best_association), xytext=(7, -3),
                        textcoords='offset points', fontsize=8, color=MUTED)
    near = int((summary.observed_best_association-summary.shape_label_ceiling < .02).sum())
    ax.text(.47, .87, f'{near} of {len(summary)} locations: best site is within 0.02\n'
            'of what a pure shape label (A/B/C/D) scores.\n'
            'Points above the line add near-clone variation,\nwhich the right panel finds is chance-level.',
            va='top', fontsize=9)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', xlabel='Ceiling for a feature that only labels the shape',
           ylabel='Best association found on the map')
    ax.set_title('Map peaks sit at the shape-label ceiling', loc='left', weight='bold')
    style(ax)

    ax = axes[2]
    lo, hi = .45, .9
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=.8)
    ax.scatter(summary.within_AC_null95_association, summary.within_AC_best_association, s=46, color='#2a78d6',
               edgecolors='white', linewidths=1.2, zorder=3)
    ax.text(.47, .87, f'Relabeled clones reach the same score.\n'
            f'Smallest location p after scanning all 46: {summary.within_AC_adjusted_p.min():.2f}', va='top', fontsize=9)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal',
           xlabel='95th percentile after relabeling designs within A and C',
           ylabel='Best within-shape association found')
    ax.set_title('Near-clone differences: no feature beats chance', loc='left', weight='bold')
    style(ax)
    fig.savefig(out/'plots'/'why_the_maps_tie.png', dpi=160)
    plt.close(fig)


def render_curves(out, source):
    locations = list(source['locations'])
    t = 1000*source['times']
    fig, axes = plt.subplots(1, len(CURVE_LOCATIONS), figsize=(16, 4.8), layout='constrained', sharey=True)
    gram = curve_gram(source['full']-source['full'].mean(axis=0, keepdims=True))
    between = between_cluster_fraction(gram, list(GEOMETRY_CLUSTERS.values()))
    for ax, loc in zip(axes, CURVE_LOCATIONS):
        il = locations.index(loc)
        for name, members in GEOMETRY_CLUSTERS.items():
            for d in members:
                ax.plot(t, source['full'][d, il], color=CLUSTER_COLORS[name], linewidth=.8, alpha=.4)
        for name, members in GEOMETRY_CLUSTERS.items():
            ax.plot(t, source['full'][list(members), il].mean(axis=0), color=CLUSTER_COLORS[name], linewidth=2.2,
                    label=f'Shape {name} mean (designs {members[0]}–{members[-1]})')
        ax.set_title(f'Location {loc}: {100*between[il]:.0f}% of the spread is between shapes', loc='left',
                     weight='bold', fontsize=10.5)
        ax.set(xlabel='Time (ms)', xlim=(0, 25))
        style(ax)
    axes[0].set_ylabel('Headform acceleration (g)')
    axes[0].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('Same impact point, 12 designs: thick = mean of each hood shape, thin = its individual designs',
                 x=.01, ha='left', weight='bold', fontsize=13)
    fig.savefig(out/'plots'/'curves_by_shape.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
