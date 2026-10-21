# Common-mode + design-residual surrogate

This implements "The crucial output reformulation" from the literature review
(*Literature Review and Research Direction for Geometry-Sensitive Hood-Impact
Surrogate Modeling*, pp. 12-13). The code is in:

- `common_residual.py`: the model and the losses;
- `train_common_residual.py`: the trainer;
- `run_common_residual.sbatch`: the DeltaAI launcher.

## What it does

Each acceleration history is split into a part shared by all hoods at that
impact point and a part specific to the design:

```
a_bar_p(t)      = mean of a_{d,p}(t) over the training designs      (common curve)
delta a_{d,p}   = a_{d,p}(t) - a_bar_p(t)                            (design residual)
a_hat_{d,p}(t)  = common_p(t) + residual(G_d - G_ref, p)(t)
```

- **Common path.** Sees only the impact location. It uses the same decoder as MeshImpactHistoryNet, behind an impact token.
- **Residual path.** This is MeshImpactHistoryNet, fed the design's explicit deviation from a reference hood before any pooling.
  - Node rows are the reference hood's XYZ plus 12 channels per node: `dx, dy, dz, |dx|` (mm/10), the change in normal, the change in panel gap, and panel identity (outer / inner / hinge / other).
  - It uses the same configuration as the no-shortcut run `20261016`: k256 neighborhoods, Fourier time, and no location shortcut. There are no headform rows.

Loss, with every term normalized:

| term | definition | default weight |
|---|---|---|
| `abs` | \|\|a_hat - a\|\|^2 / sigma_a^2 | 1 |
| `common` | \|\|common - a_bar\|\|^2 / sigma_a^2. This keeps the two paths from trading off. | 1 |
| `delta` | \|\|residual - delta a\|\|^2 / s_r^2 | 1 |
| `pair` | For designs i < j at the same location in a batch: w_p \|\|(a_hat_i - a_hat_j) - (a_i - a_j)\|\|^2 / s_r^2 | 1 |
| `hic` | (log(1 + HIC_hat) - log(1 + HIC))^2, with a differentiable HIC over every window <= 15 ms | 0 (off) |

The pair weight is w_p = Var_d[HIC_{d,p}] / mean_p(Var), over the training designs,
clipped at 5. Locations where the design changes HIC weigh more, and none are
dropped. Each batch is one impact location with all nine training designs, so
every step has 36 same-location pairs.

### Differences from the review (all to avoid leakage or because of the mesh)

- **The common curve and the reference hoods use training designs only**, not all 12. Otherwise test curves would enter the target.
- **There is one reference hood per mesh family.** Designs 0-5 and 6-11 have different node sets. Each reference is the node-wise mean of that family's training designs: {0,1,2,3} and {6,7,8,9,10}.
- **Held-out designs are encoded as deviations from those references.** RMS deviation:

  | designs | RMS deviation from reference |
  |---|---|
  | 0-3 | 0.2-0.4 mm |
  | 6-9 | 4.8 mm |
  | 10, 11 | 19 mm |
  | 4, 5 | 23 mm |

## Run

```bash
mkdir -p runs/slurm
sbatch run_common_residual.sbatch                      # main run, ~12.5 h, W&B online
sbatch run_common_residual.sbatch --lambda-pair 0      # ablation: no pair loss
sbatch run_common_residual.sbatch --lambda-hic 0.1     # + HIC loss
sbatch run_common_residual.sbatch --resume runs/common_residual/<run>
```

Local smoke test (CPU or a small GPU):

```bash
python train_common_residual.py --max-locations 4 --epochs 2 --neighborhood-layers 0 \
    --num-latents 16 --width 32 --common-width 32 --wandb-mode disabled
```

## Outputs and how to read them

All outputs go to `runs/common_residual/<stamp>_<job>_<name>/`.

`metrics.json` reports each split (train, validation = 11, test = 4, 5):

| metric | meaning |
|---|---|
| `rmse_g`, `r2` | absolute acceleration error |
| `residual_r2` | how much of each design's deviation from the common curve is predicted. 0 = the common curve alone |
| `split_mean_residual_r2` | the same, averaged over the split's designs. For test this is the cluster-B shape effect without the near-clone part |
| `hic_r2`, `hic_mape_percent` | HIC15 on the 63-point grid |
| `baseline_common_only_*` | the geometry-free answer: training-design mean curve |
| `near_clone_difference_r2` | the approved difference R2 for train pairs, validation (10, 11) and test (4, 5) |

Other outputs:

- `design_difference_r2.json`: the same near-clone metric as every MeshImpactHistoryNet run, so the runs compare directly.
- `predictions.npz` (total, common, residual for all 12 designs) and `{train,test}_acceleration_histories.csv`.

Numbers to beat on the test designs (4, 5):

| | accel RMSE (g) | accel R2 | HIC R2 | near-clone diff R2 |
|---|---|---|---|---|
| Common curve only (no geometry) | 15.99 | 0.852 | 0.881 | 0 |
| No-shortcut MeshImpactHistoryNet `20261016` | 15.54 | 0.860 | 0.898 | ~0 |

The previous model is only 3% better than knowing nothing about geometry. The
residual path has to raise `residual_r2` and `split_mean_residual_r2` on the
test designs above 0 to show that it learned a geometry effect.

## Tests

```bash
python -m pytest tests/test_common_residual.py -q
```

The tests check that:

- the differentiable HIC equals `hic15.batched_hic`;
- the references and deviation features are built correctly;
- the losses vanish for a perfect prediction;
- the pair loss ignores a shared error and penalizes identical curves;
- `node_extra_features=0` leaves MeshImpactHistoryNet unchanged.
