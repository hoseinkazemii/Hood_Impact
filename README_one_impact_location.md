# One impact location: geometry-only test

Branch `feature/one-impact-location-test`. Jing asked to learn the HIC-variation
rank-1 location (137) alone: no impact-location branch, the outer skin removed,
the network reading only the inner frame, designs 4 and 5 in training, and one
design with a 100-125 g initial peak held out. The first step is to reproduce the
variation between the 11 training designs.

```bash
bash run_one_impact_location.sh                      # location 137, test design 7
HOOD_ONE_LOCATION=9 bash run_one_impact_location.sh  # another location
HOOD_ONE_TEST_DESIGNS="2" bash run_one_impact_location.sh
```

On the local Windows Python, wandb needs protobuf < 6 (see the launcher header,
`HOOD_PROTOBUF_COMPAT`).

## What changed

The network is still `MeshImpactHistoryNet`. Three switches remove the location:

| Option | Effect |
|---|---|
| `--impact-conditioning none` | No impact embedding and no FiLM on the pooling queries; the learned queries are constants. No radial prior either, so all tokens are global. `indentor` is still passed but never read. |
| `--mesh-parts inner` | Each mesh is the 286 headform rows (kept out of pooling, as before) plus the `Hood_Inner-1-2` nodes: 17,388 / 18,468 rows instead of 38,977 / 40,057. The outer skin and hinges are dropped. |
| `--locations 137` | Loads only the 12 runs at that location (any list of 1..142). |

Training changes for 11 samples:

* `--val-designs` with no IDs: no validation design. The training designs,
  evaluated in eval mode, are the `val/*` metrics and select the checkpoint; the
  test design is never seen. Saved as `training.checkpoint_selection`.
* Smaller network: width 64, 32 tokens, one latent, decoder and neighborhood
  layer (0.18 M parameters vs 1.59 M in `20261004_050747`), no dropout.
* 250 output times (stride 4) instead of 63, which resolve the 1 ms initial peak
  of cluster B. Full batch of 11 and 3000 epochs: one optimizer step per epoch.

## Outputs

Besides the usual run files, a `--locations` run writes
`train_acceleration_histories.csv`, `train_metrics.json` and `location_report/`
(`report_one_impact_location.py`, also runnable on its own):

* `location_137_all_designs.png/pdf`: every design, truth vs prediction;
* `location_137_designs_4_5.png/pdf`: the layout of the top-10 figures;
* `summary.csv`: per design, HIC15 and 0-3 ms peak (true and predicted), RMSE, R2;
* `report.json`: `train_variation_skill` (1 = each design's own curve, 0 = no
  design-to-design variation, the training mean for all), within-cluster
  difference R2, and the test design against two model-free references: the
  mean of its cluster siblings in training and the mean of all training designs.

## Reading the result

At location 137 the 12 runs are four geometry clusters. The 65% HIC spread is
cluster B (designs 4, 5: initial peak 194/215 g, HIC15 1210/1323), whose frame
lies 3 mm under the skin there (A 15.8, C 30.2, D 15.0 mm).

* Design 7 is the only design whose edit is under the head (3 mm away); its HIC15
  is 814 against 961-971 for its siblings 6, 8, 9. Copying the siblings is wrong
  for it, so it is the informative test. For any other 100-125 g design, the
  sibling mean is close to the answer and must be beaten before a test score
  says anything.
* The 4-5 difference here is mostly the solver time-step artifact: the 4->5 edit
  is 272 mm away and the two runs used 26.6k and 40.4k increments. A model that
  separates 4 and 5 in training reads the distant edit, so this is a fit to
  solver noise, not to local geometry.
* Within a design the frame is the same at every location, so one shared model
  cannot cover several locations without a location input. Run one model per
  location instead; mirror pairs (137/9, 127/43, 106/139) should agree.
