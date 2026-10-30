# Design-change nodes: a decoder path that pooling cannot average away

Branch `feature/design-change-node-injection`. The model is `DesignChangeHistoryNet` in
`mesh_design_change.py`, and the change nodes come from `design_change_nodes.py`.

## Why

`MeshImpactHistoryNet` pools about 38,700 hood nodes into 256 tokens. Near-clone designs differ at only
110-333 of those nodes, so their pooled memories are almost the same, and the network predicts the same
history for both. The probe of run `20261016_151923_3394500_1704` measured this: the relative near-clone
difference is 0.8% at the node embedding and 0.5% after pooling.

This experiment gives each changed node its own token, so the design difference reaches the decoder
without going through pooling.

## Change nodes

The rule is the row rule of the pair matrix (`mesh_design_pair_differences_1704.py`):

- Two designs are near-clones when they share node IDs and fewer than 300 nodes moved more than 0.1 mm
  between them. Larger counts (3.4k-17k) come from the morph between clusters, not from an edit.
- A design's change nodes are the union over all its near-clone partners.

| cluster | designs | base | change nodes |
|---|---|---|---|
| A | 0 1 2 3 | 0 | 188 |
| B | 4 5 | 4 | 110 |
| C | 6 7 8 9 | 6 | 333 |
| D | 10 11 | 10 | 192 |

Every member of a cluster has the same change nodes; only the coordinates there differ.

- **The edits add up.** Pair 0-1 (46 nodes) plus pair 0-2 (57) gives pair 1-2 (103), and cluster C works
  the same way.
- **Edits only raise nodes.** Every changed node rises (+Z) from 4 to 5 and from 10 to 11. So each
  cluster's base is the member lowest at its change nodes. The code checks this rule.
- **Displacement is measured from the base.** Each edit is 3-5 inner-panel patches raised about 10 mm
  (16-25 mm at most). That closes the inner-outer panel gap by up to 24 mm.
- **Held-out designs are used, but only their geometry.** Design 4's change nodes come from comparing its
  deck with design 5's. No held-out acceleration is read.

## Architecture

The backbone is `MeshImpactHistoryNet`, unchanged. The new path:

1. **One token per change node: its 3 coordinates.** Each token's input is the node's XYZ in this
   design, normalized as the backbone's mesh input (`--design-change-features xyz`, the default since
   2026-10-26). The runs in the results below used 18 or 30 inputs instead (`impact_relative`): the
   offset from the impact point and its distance, sin/cos of that offset, the displacement from the
   cluster base, the panel gap and its change, and an outer-panel flag. Saved runs keep their own setting.
2. **Self-attention over the change set: none by default.** `--design-change-layers N` adds N blocks over
   each design's change tokens plus one learned null token. The first run used 2; the sweep below found 0
   better, so each token now goes straight from its embedding to the readout.
3. **A second read in the decoder.** After every decoder block, the time queries cross-attend to the
   change tokens (`ChangeReadout`). The output projection has no bias and starts at zero, so with the same
   seed a new model starts as exactly the baseline network. Since 2026-10-30 this is plain attention
   (`--design-change-displacement-gate false`, the default): the tokens are already only the nodes that
   differ between near-clones, so the attention alone decides which matter, and every design, base
   included, reads its change nodes.

   The runs in the results below also scaled each node's value by its displacement from the cluster base
   / 10 mm and the null token's by zero (`--design-change-displacement-gate true`; saved runs keep it).
   There the base designs 0, 4, 6 and 10 always got exactly the backbone prediction, so the path learned
   only the effect of an edit.

The loss is the plain normalized acceleration MSE, as before. The model finds each hood by the digest of
its normalized coordinates, like the neighbor-graph cache. The table is saved as
`design_change_nodes.npz` beside the checkpoint; it is not part of the state dict.

## Mesh: inner panel only

Since 2026-10-26 the network sees only the hood inner panel (`--mesh-nodes inner_panel`, the default for
.inp decks). The loader keeps the nodes of the `Hood_Inner-1-2` elements, in deck order, and drops the
rest:

| node set | design 0 | kept |
|---|---|---|
| inner panel | 17,102 | yes |
| outer skin | 20,492 | no |
| hinges | 100 | no |
| fasteners / couplings | 997 | no |
| headform | 286 | no, so `--impactor-nodes` becomes 0 |

Designs 6-11 have 18,182 inner-panel nodes. Change nodes off the inner panel are dropped too: the 4
outer-skin nodes design 9 moves by 0.2 mm, so cluster C has 329 change nodes instead of 333. Preflight
checks that two impacts of one design give identical inner-panel meshes. Runs saved before this use
the whole deck (`all`) and still load that way.

The outer skin is identical within a cluster, but not between clusters. Designs 4/5 differ from 0, and
10/11 from 6, at about 9,000 outer-skin nodes, by up to 74 mm. Clusters B and D also differ from each
other by up to 18 mm. So there are three outer-skin shapes (A = C, B, D), and the test designs' shape (B)
is in no training design. The inner panel moves with the same morph (about 6,000 nodes, up to 80 mm), so
the cluster shape still reaches the network through it.

## Run (DeltaAI)

```bash
bash submit_mesh_impact_history_1704_design_change_nodes.sh
```

This is an A/B against `20261016_151923_3394500_1704`. The settings are the same:

- k256 neighborhoods, FiLM, Fourier time, all 142 locations, stride 16;
- no location shortcuts;
- test 4, 5 and validation 11;
- 100 epochs, batch 8, lr 3e-4, seed 42.

The changes are `--design-change-nodes true` with the change path at its defaults (0 layers, XYZ
tokens, no displacement gate), and inner-panel meshes. To repeat the 2026-10 sweep runs exactly, use the
`..._design_change_small/_regularized/_small_regularized.sh` wrappers, which pin `--mesh-nodes all`,
impact-relative tokens and the displacement gate.

### Long run (2026-10-30)

```bash
bash submit_mesh_impact_history_1704_design_change_long.sh
```

This is the best run so far, `20261027_055436_3458374_1704` (inner panel + XYZ tokens), with two
changes: no displacement gate, and 800 epochs instead of 100 (about 41 h of the 48 h DeltaAI limit at
3458374's 185 s per epoch; the cosine schedule spans all 800). The one-location test at location 137
separated the near-clones only after about 1500 full-batch epochs.

The best-validation (design 11) checkpoint is evaluated as usual. In 3458374 it came at epoch 60 of
100, so training also scores the final epoch's checkpoint in `<run>/last_checkpoint/` (every run does
this now): `metrics.json`, the train and test acceleration histories, and `design_difference_r2.json`.
For HIC on either one, run `python evaluate_mesh_impact_hic.py <dir> --data-dir Data/HoodImpact_1704_EuroNCAP`.
To beat, on test 4-5: the baseline's accel RMSE 15.54 g and HIC R2 0.898, and difference R2 > 0
(better than predicting the same history for both designs).

If the job times out, `hood_impact_last_model.pt` holds the last completed epoch. Continue it on the same
schedule with
`bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> --time=24:00:00`.

Before the dataset loads, the preflight finds the change nodes from the decks and runs a CUDA
forward/backward pass through the change path. The first run builds
`Data/HoodImpact_1704_EuroNCAP/design_difference_cache`. You can override `HOOD_MESH_DESIGN_CHANGE_LAYERS`,
`HOOD_MESH_DESIGN_CHANGE_FEATURES` and `HOOD_MESH_MESH_NODES`.

## Results (2026-10-25)

Four runs, all seed 42, 100 epochs, same split and backbone. They form a 2 x 2: change-path size (full =
2 layers, 5 octaves, 0.55 M parameters; small = 0 layers, 2 octaves, 0.15 M) by regularization (dropout 0.1
and weight decay 1e-5, or dropout 0.2 and weight decay 0.05). Run folders are under
`runs/mesh_impact_history/nodes_difference_attention/`; the comparison is in its `sweep_comparison/`.

Absolute accuracy (test = designs 4 and 5, best-validation checkpoint):

| run | change path | dropout / wd | best val MSE (epoch) | test accel RMSE (g) / R2 | test HIC R2 | test HIC MAPE |
|---|---|---|---|---|---|---|
| baseline `20261016_151923_3394500` | none | 0.1 / 1e-5 | **0.109** (78) | **15.54** / 0.860 | **0.898** | **6.3%** |
| reference `20261023_154041_3440305` | full | 0.1 / 1e-5 | 0.151 (89) | 16.05 / 0.851 | 0.879 | 7.1% |
| **small** `20261025_003920_3447060` | small | 0.1 / 1e-5 | 0.131 (85) | 15.57 / 0.860 | 0.860 | 7.3% |
| regularized `20261025_003920_3447061` | full | 0.2 / 0.05 | 0.176 (95) | 16.44 / 0.843 | 0.870 | 7.9% |
| small + reg. `20261025_004128_3447062` | small | 0.2 / 0.05 | 0.150 (94) | 16.18 / 0.848 | 0.844 | 8.9% |

Near-clone differences. Train = the 12 pairs in clusters A and C (1704 curves), test = 4 vs 5 (142).
Delta = mean |a_a - a_b| (g); error = mean |p - d| of the difference curve. A model that predicts no
difference scores error = Delta_GT (train 6.83 g, test 7.27 g).

| run | train Delta_pred | train error | train R2 | test Delta_pred | test error | test corr | test R2 |
|---|---|---|---|---|---|---|---|
| baseline | 0.12 | 6.83 | 0.00 | 0.06 | 7.26 | 0.03 | 0.00 |
| reference | 5.64 | **4.98** | **0.57** | 6.57 | 9.70 | 0.11 | -0.46 |
| **small** | 4.43 | 5.45 | 0.47 | 5.65 | **8.63** | **0.22** | **-0.20** |
| regularized | 5.87 | 5.83 | 0.40 | 7.11 | 10.07 | 0.13 | -0.49 |
| small + reg. | 4.19 | 6.15 | 0.31 | 6.75 | 9.81 | 0.14 | -0.43 |

What this shows:

- **The path works on designs it was trained on.** Every change-path run reproduces the training pairs'
  difference curves (train R2 0.31-0.57, 0.74 within 100 mm of the edit for the reference); the baseline
  predicts none.
- **It does not transfer to 4 vs 5.** Every run predicts a difference of about the right size but the wrong
  shape, so its error is above the no-difference score in every distance band, far from the edit included.
  Predicted 4-5 HIC gaps are 2-3x the true ones (mean |dHIC| 87-115 vs 41).
- **Small is the best of the four** and is the default: best validation, test acceleration equal to the
  baseline, and the least wrong test difference. Its test HIC R2 (0.860) is still below the baseline's.
- **Regularization made everything worse**, so the problem is not generic overfitting.
- **Likely cause:** each design's change set is unique, so the path can act as a design ID and memorize the
  whole training difference, including the far-field part that is the solver time-step artifact. The
  readout has no locality, so it can act at any distance from the impact.

For test pair 4-5 in these gated runs, design 4 is the base, so its prediction is the backbone's. Any
difference comes from design 5's edit.

## Limits

- **New hoods need registration.** Inference works only for hoods in the table. A new hood needs its change
  nodes found against a base, then `DesignChangeTable.register`.
- **Some tools don't support it yet.** `export_mesh_pooling_attention.py` and
  `infer_designs_mesh_impact_history.py` don't load this architecture. `HistoryPredictor.from_run` does.
- **Part of the 4-5 difference can't be learned.** Design 5's edit shrinks the solver time step. About a
  third of the 4-5 difference lies far from the edit and comes from that, not from geometry. Expect any
  gain at the near-edit locations first.

## Tests

```bash
python -m pytest tests/test_design_change.py tests/test_mesh_impact_history_submission.py -q
```
