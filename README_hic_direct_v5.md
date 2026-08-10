# Direct HIC15 surrogate (V5)

`hic_direct_surrogate_v5.py` is a new, self-contained experiment for the 600
EuroNCAP runs. It does not import or modify the earlier DeepONet code and writes
to a new `runs/*_hic_direct_v5/` directory, so existing models remain intact.

## Why this path

The previous best run (`runs/20260719_014424`) predicts acceleration histories
at stride 16, selects its checkpoint by acceleration MSE, and computes HIC only
after training. Its mesh encoder also includes the 286 moving headform nodes,
drops impact X3, and globally max-pools roughly 39,000 absolute XYZ points. The
sparse hood/rib changes that distinguish nearby designs are consequently easy
to erase.

There is also a target-definition mismatch. `generation_metrics.csv` was
produced by a routine that checks only the longest admissible endpoint for each
start time. The correct HIC15 search must check every positive-duration window
up to 15 ms. Against HIC recomputed from the current 1,000-point histories, the
stored Euro600 values differ by about 78 HIC on average and by as much as 685.

V5 therefore:

1. recomputes HIC15 from every full-resolution history using every window up to
   15 ms;
2. uses tracked `manifest_600.csv` rather than the untracked
   `ImpactCoords_600.csv` or unordered `generation_metrics.csv`;
3. strips the rigid headform and parses one stationary hood mesh per design;
4. includes X1/X2/X3, surface height, lift, WAD, grid column, and multi-scale
   impact-centred hood/rib statistics;
5. transfers the 50-location profile of the closest training geometry and
   learns a regularized log-HIC residual from ordered geometry pairs; and
6. performs grouped design CV, refits on every non-test design, and only reads
   the reference-test targets when `--evaluate-test` is explicitly supplied.

This formulation matches the actual data regime: 12 independent designs and a
fixed 50-location factorial grid. It is deliberately a small-data retrieval +
random-forest surrogate rather than another high-capacity temporal network.

## Delta submission

The supplied launcher is the final-reference workflow requested for this
branch: it selects the residual shrinkage by grouped CV, freezes/refits the
model, and then passes `--evaluate-test` to report design 11. From the
repository root:

```bash
sbatch run_hic_direct_v5_600.sbatch
```

Defaults are validation design 10, test design 11, 768 trees for the final fit,
and 192 trees per grouped-CV fold. They can be overridden without editing code:

```bash
HOOD_VAL_DESIGN=9 HOOD_TEST_DESIGN=11 \
HOOD_HIC_TREES=1024 HOOD_HIC_CV_TREES=256 \
sbatch run_hic_direct_v5_600.sbatch
```

The first run caches canonical targets and geometry descriptors under
`runs/_hic_direct_v5_cache/`. Cache keys hash the input bytes and exact script
implementation; cache writes are atomic. Use `--rebuild-cache` to force a
recomputation. Run directories include microseconds plus a Slurm job ID or
random suffix, and an explicitly supplied non-empty output directory is never
overwritten.

During development or parameter checks, omit the evaluation flag. The script
then saves the CV artifacts and frozen model without computing design 11's HIC
targets or creating test artifacts:

```bash
python hic_direct_surrogate_v5.py --selection-mode group-cv
```

## Outputs

Each run records:

- `canonical_hic15_targets.csv`: development targets, plus reference-test
  targets only when `--evaluate-test` was requested;
- with `--selection-mode group-cv`, `group_cv_predictions.csv` and
  `group_cv_design_metrics.csv`: auditable out-of-fold development results;
- `validation_predictions.csv` and `model_selection.json`;
- `geometry_distance_matrix.csv` and `feature_importance.csv`;
- `hic_direct_v5_model.joblib`, `config.json`, and `train.log`.

With `--evaluate-test`, the run additionally records `test_predictions.csv`,
`test_metrics.json`, and `test_predicted_vs_true_hic.png`.

On the local reference data, a reduced configuration (256 final trees, 64 CV
trees) produced pooled grouped-CV R2 0.9702 and design-11 reference R2 0.9722
(RMSE 73.50, MAE 49.66) over the 50 EuroNCAP locations. Per-design CV is
important: design 10 is a hard fold (R2 0.8778), then becomes design 11's
uniquely close training neighbor after refit (geometry distance 0.874 mm versus
about 24 mm to the next design). Its profile alone gives design 11 R2 0.9554.
The learned residual raises that result, but the split is unusually favorable.

Design 11 has also been reported against earlier models, so it is a historical
held-out benchmark rather than a statistically untouched test set. Treat the
grouped out-of-design predictions and their per-design metrics as the primary
generalization evidence, and rotate the outer held-out design for a publication
estimate. These are same-location-grid, unseen-design results—not evidence of
generalization to new impact locations. The attached older score used 55
locations and a different stride-16 HIC target, so compare target definitions
as well as headline R2.

## Checks

Run the lightweight tests and dataset preflight with:

```bash
python -m unittest discover -s tests -v
python hic_direct_surrogate_v5.py --preflight-only
```
