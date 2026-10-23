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

1. **One token per change node.** Its inputs are:
   - the position relative to the impact point: XY offset and distance (/ 250 mm), and sin/cos of the
     offset at wavelengths 800, 400, 200, 100 and 50 mm;
   - the displacement from the cluster base (/ 10 mm) and its length;
   - the base inner-outer gap and its change (/ 10 mm);
   - an outer-panel flag.

   The impact-relative position is what can transfer to a held-out design. It encodes "a patch raised
   10 mm, 40 mm from the head" rather than a position on one hood.
2. **Self-attention over the change set.** Two blocks (`--design-change-layers`) run over each design's
   change tokens plus one learned null token.
3. **A second read in the decoder.** After every decoder block, the time queries cross-attend to the
   change tokens (`ChangeReadout`). Each node's value is scaled by its displacement / 10 mm, and the null
   token's value by zero. The output projection has no bias and starts at zero. As a result:
   - with the same seed, a new model starts as exactly the baseline network;
   - the base designs 0, 4, 6 and 10 always get exactly the backbone prediction;
   - the path learns only the effect of an edit.

The loss is the plain normalized acceleration MSE, as before. The model finds each hood by the digest of
its normalized coordinates, like the neighbor-graph cache. The table is saved as
`design_change_nodes.npz` beside the checkpoint; it is not part of the state dict.

## Run (DeltaAI)

```bash
bash submit_mesh_impact_history_1704_design_change_nodes.sh
```

This is an A/B against `20261016_151923_3394500_1704`. The settings are the same:

- k256 neighborhoods, FiLM, Fourier time, all 142 locations, stride 16;
- no location shortcuts;
- test 4, 5 and validation 11;
- 100 epochs, batch 8, lr 3e-4, seed 42.

The only change is `--design-change-nodes true`.

Before the dataset loads, the preflight finds the change nodes from the decks and runs a CUDA
forward/backward pass through the change path. The first run builds
`Data/HoodImpact_1704_EuroNCAP/design_difference_cache`. You can override `HOOD_MESH_DESIGN_CHANGE_LAYERS`
and `HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES`.

## What to compare

| | test accel RMSE (g) | test accel R2 | test HIC R2 | near-clone diff R2 train / test |
|---|---|---|---|---|
| baseline `20261016_151923_3394500_1704` | 15.54 | 0.860 | 0.898 | 0.002 / -0.001 |
| common residual `20261021_001529_3426410` | 18.17 | 0.809 | 0.622 | 0.35 / -0.04 |

Read these files from the new run:

- `design_difference_r2.json`: the approved near-clone difference R2.
- `metrics.json` and `hic_metrics.json`: absolute accuracy, which should not get worse.
- `design_change_nodes.json`: the clusters, bases and registration check.

For test pair 4-5, design 4 is the base, so its prediction is the backbone's. Any difference comes from
design 5's edit.

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
