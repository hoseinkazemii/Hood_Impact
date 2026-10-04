# Impact locality: simulation evidence and model sensitivity

Analysis date: 2026-09-17. Script: [analyze_impact_locality_1704.py](analyze_impact_locality_1704.py).

Nearby geometry contains information about acceleration differences, especially
early in the impact. The current trained model relies on its distance prior,
but making that prior substantially stronger hurts held-out HIC accuracy. These
results support retaining local geometry while investigating how the model learns
design differences; they do not establish that a narrower attention window will
improve a newly trained model.

## Figures and numerical outputs

- [Simulation evidence: PNG](figures/impact_locality_1704/impact_locality_simulation.png)
  / [PDF](figures/impact_locality_1704/impact_locality_simulation.pdf).
- [Frozen-model distance-prior sweep: PNG](figures/impact_locality_1704/impact_locality_model.png)
  / [PDF](figures/impact_locality_1704/impact_locality_model.pdf).
- [Actual matched-location history examples: PNG](figures/impact_locality_1704/impact_locality_histories.png)
  / [PDF](figures/impact_locality_1704/impact_locality_histories.pdf).
- All tables, intermediate arrays and diagnostic metadata are in
  `figures/impact_locality_1704/`. Generated outputs are ignored by Git under the
  repository's existing policy.

## What the simulations show

The 1,704 runs provide all combinations of 12 designs and 142 impact locations.
The authoritative clusters in `mesh_design_clusters.py` are A = 0–3,
B = 4–5, C = 6–9 and D = 10–11. All response time grids and impact XY coordinates
were checked for alignment before comparisons.

At each matched location, the analysis compares the RMS difference between two
cluster-mean acceleration histories with the local geometry mismatch between
their representative hoods. Each cluster and each of the six cluster pairs
receives equal weight. This avoids treating four nearly identical A designs as
four independent geometry families.

| Diagnostic at a 100 mm XY radius | Result |
|---|---:|
| History association, mean within-pair Spearman correlation | 0.474 |
| History association, training-cluster pairs A-B / A-C / B-C | 0.572 |
| History association, pairs involving held-out D | 0.376 |
| History rank association, pair and shared-location effects removed | 0.333 |
| HIC association, mean within-pair Spearman correlation | 0.370 |
| HIC rank association, pair and shared-location effects removed | 0.081 |

For the lowest local-geometry-difference quartile, the mean history difference
is 8.84 g RMS; for the highest quartile it is 18.47 g RMS, about 2.09 times as
large. Quartiles are defined separately inside each cluster pair. The middle
quartiles are not monotonic, and some pairs exhibit much weaker relationships.

The time dependence is informative. At 100 mm, the rank association after pair
and location controls is 0.478 during 0–5 ms, 0.175 during 5–10 ms, 0.106 during
10–15 ms, 0.154 during 15–20 ms, and 0.004 during 20–25 ms. Local mismatch is a
better descriptor of early response differences than late response differences
in this dataset. This does not establish that late response is caused only by
distant geometry.

The controlled whole-history association is about 0.38–0.39 at 200–300 mm,
falls to 0.09 at 600 mm, and becomes negative at 800 mm. Conversely, the raw
within-pair correlation has a maximum at 600 mm. The difference between those
curves is a reason not to declare the largest raw correlation an optimal radius:
shared spatial patterns explain part of the apparent relationship.

Matched-node displacement provides a separate check within the two mesh families.
At 100 mm its history correlation is 0.626 for A-B and 0.225 for C-D. Thus,
the association is not solely an artifact of comparing different node layouts,
but its strength is clearly cluster-dependent.

The actual-history examples are selected solely by local geometry rank, at
the 12.5th and 87.5th percentiles for A-B and C-D. Response extremes were not
used to choose the illustrated locations.

## What the trained network shows

The evaluated checkpoint is
`runs/mesh_impact_history/20260914_173055_3151837_1704/hood_impact_best_model.pt`.
It has two neighborhood blocks, k = 16, a temporal decoder and design-difference
loss weight 1. Its saved `impactor_nodes` is **0**, although the current launcher
sets 286. The analysis preserves the saved architecture.

The pooling bias was multiplied by m without modifying saved weights:

`bias = -m * learned_precision * [(dx / mesh_scale_x)^2 + (dy / mesh_scale_y)^2]`.

All six values were evaluated at all 142 validation impacts and 284 test impacts.
For m = 0, only the explicit pooling distance bias disappears. Node-minus-impact
features, impact embeddings, local neighborhood attention and learned content
attention remain. This is a frozen-model sensitivity experiment, not retraining
without locality. Six spot checks against the unmodified forward method produced
exactly equal normalized predictions for m = 1.

| Distance multiplier | Validation HIC R² | Test D HIC R² | Test D history RMSE (g) |
|---|---:|---:|---:|
| 0: explicit bias removed | 0.7825 | 0.5090 | 24.601 |
| 0.25: broader prior | 0.9177 | 0.7859 | 19.236 |
| 0.5: broader prior | 0.9356 | 0.7877 | 18.677 |
| 1: trained setting | 0.9558 | 0.7769 | 18.502 |
| 2: narrower prior | 0.9546 | 0.7802 | 18.423 |
| 4: narrower prior | 0.9367 | 0.7593 | 18.841 |
| Location-only training-history mean | 0.9216 | 0.7467 | 18.157 |

HIC is evaluated against the full-resolution simulation, using every admissible
window on each time grid, up to 15 ms. History RMSE uses the checkpoint's 63
sampled times. The location-only baseline averages the nine **training** designs'
histories at each XY; no validation or test labels enter that predictor. It is
applicable here because every test location also exists in training. It does not
demonstrate generalization to a new location.

The distance prior is useful to this checkpoint, but tightening it does not
resolve generalization. Factor 0.5 raises test HIC R² slightly while worsening
history RMSE; factor 2 has a tiny gain in both. Neither is compelling evidence
for changing the default. A conditional spatial block bootstrap (20 XY blocks
of 300 mm, 2,000 resamples, seed 42, both D designs kept together) gives these
95% intervals for the HIC RMSE change relative to m = 1:

- m = 0: +57.7 to +150.6 HIC units.
- m = 0.5: −16.5 to +7.0 HIC units.
- m = 2: −5.0 to +3.8 HIC units.
- m = 4: +1.5 to +15.6 HIC units.

These intervals condition on this one checkpoint and one test cluster. They
do not estimate uncertainty over new geometry clusters or training seeds.

The trained prior already spans many scales: its 1/e semi-axes range from
68–556 mm in X and 84–688 mm in Y. Their anisotropy comes from the XYZ scaler,
not an explicitly learned physical direction. The 16 precisions moved by at
most 5.8% from initialization. These are widths of the geometric prior; actual
attention also depends on learned query-key scores.

The structural 16-nearest-neighbor graph has a median outer-neighbor distance
of about 20.4–20.7 mm across the four representative hoods. That is a separate
scale from the impact-pooling prior. `neighborhood_scale_mm=20` normalizes the
relative-position features and does not set either neighborhood membership or
the impact-pooling radius.

## What can be concluded about the difference loss

Saved full-resolution HIC results give R² = 0.7603 for the temporal baseline
`20260912_005426_3134539_1704`, and R² = 0.7771 for the neighborhood-plus-difference-
loss run. Both use the same test designs 10 and 11. The corresponding HIC RMSEs
are 210.53 and 203.02, a reduction of about 3.6%.

That is a modest combined improvement. It cannot attribute the gain to the
design-difference loss, because the node encoder and training batches also
changed. The fresh local inference gives 0.7769 instead of the saved export's
0.7771; the earlier `design_comparison/fresh_test_metrics.json` already documents
small cluster-versus-local prediction differences. Comparisons within this new
sweep all use fresh inference under the same local environment.

## Next experiments suggested by these results

1. **Use whole-cluster validation for architecture selection.** Validation
   design 5 has a near-clone, design 4, in training. Its 0.956 HIC R² is a poor
   proxy for D's 0.777. Rotate a whole A/B/C cluster out during development,
   keeping clones together. Group-wise validation is supported directly by
   [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).
   D has now been inspected diagnostically; do not select a multiplier from its
   best point and present that score as an untouched final test.

2. **Separate the distance prior from the design-difference objective.** With
   neighborhood depth, headform treatment, optimizer and evaluation fixed, train
   a 2 × 2 ablation: pooling distance bias enabled/disabled and difference-loss
   weight 0/1. Retain matched batching in all arms to avoid an additional
   sampling confound. Repeat seeds for promising comparisons. The present sweep
   does not replace those retrained ablations.

3. **Use the location-only predictor as a required baseline.** A plausible
   architecture experiment is a location/time branch plus a geometry-dependent
   residual, `a(X,p,t) = a_base(p,t) + delta_a(X,p,t)`. The baseline competes on
   history RMSE, while the network improves HIC modestly; explicitly separating
   those jobs could make learning the design correction easier. This is a
   hypothesis to test, not a measured improvement.

4. **Test multiple physical geometry scales, while preserving global context.**
   For example, test patch summaries at 50, 150 and 300 mm with a time-dependent
   learned combination. These are candidate scales motivated by the exploratory
   curves, not established optima. The current local message-passing graph
   reaches roughly 20 mm in one hop, whereas the response associations extend
   much farther. Pooling already has broad support, so the question is whether
   richer patch features help, not simply whether a broader prior is available.
   A physical-mm parameterization would also make radii easier to interpret than
   the current scaler-dependent ellipses.

5. **Keep the headform setting consistent in new comparisons.** The current
   launcher excludes its 286 nodes from the local graph; the analyzed historical
   checkpoint does not. This is a third change requiring its own controlled
   comparison if performance is attributed to it.

These observations do not establish the causal importance of individual nodes.
The simulations change whole designs, and only four geometry clusters are
available. A causal physical test would change one local hood region while
holding the remaining geometry and simulation settings fixed, then rerun the
same impact. Deleting nodes or reading attention weights would not supply that
ground truth.

## Method details and checks

Hood geometry is identified from shell connectivity, excluding the actual
headform element set and rigid reference node. No tail slicing or interpolation
of mixed upper/lower shell heights is used. The older
`visualize_design_sensitivity_1704.py` geometry helper removes the trailing 286
nodes even though these decks put the headform first; it was not used for this
analysis's geometry selection.

Geometry representatives are designs 0, 4, 6 and 10. Each pair's mismatch is the
symmetric 3D nearest-node distance, averaged as squared distance within 20 mm XY
cells. Disk descriptors use the square root of the mean cell mismatch squared;
radius is measured in physical XY around the impact. This limits mesh-density
weighting but still includes remeshing effects. For A-B and C-D, exact equality
of element connectivity is verified and a corresponding-node displacement
descriptor provides the separate check in panel D. Differences in unmodeled
connectivity or contact are not separated by this analysis.

Responses use all designs within each cluster. The whole-history descriptor is
the RMS difference between cluster-mean curves. HIC differences use the absolute
difference of the cluster means of per-run HIC, not HIC of the mean curve.
Within each pair, geometry and response are ranked across the 142 locations.
Solid curves average their six Spearman correlations. Location-controlled curves
subtract both pair and location means from those ranks before correlating the
residuals. Those are different descriptive statistics, not causal estimates.

There are six dependent pairs, not 852 independent geometry experiments.
Adjacent impact locations are dependent as well. No independent-sample
significance tests are reported for the geometry correlations. Variation within
clusters is retained and is not automatically labeled pure solver noise.

The focused tests check the HIC calculation against an independent exhaustive
window implementation on irregular times, correct localization of a known
geometry change without node ordering, removal of a shared location pattern,
and equivalence of the unmodified forward pass with multiplier 1 while ensuring
the model state is unchanged. All four tests passed. Figures were visually
checked after rendering.

Reproduce with the existing environment:

```powershell
.\.venv\Scripts\python.exe analyze_impact_locality_1704.py
.\.venv\Scripts\python.exe -m pytest tests/test_impact_locality_analysis.py -q
```

Use `--skip-model` for simulation-only analysis, `--reuse-simulation` to reuse
the generated simulation arrays while rerunning the model sweep, and
`--plot-only` to redraw existing results. These reuse flags deliberately trust
the existing outputs; rerun without reuse after changing the dataset. The model
weights, training configuration and original result files are not modified.
