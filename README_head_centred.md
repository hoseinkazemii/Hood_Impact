# Head-centred network: no impact-location input

Branch `feature/head-centred-no-location`, from `feature/design-change-node-injection`; it also carries the
one-impact-location work (README_one_impact_location.md). The model is `DesignChangeHistoryNet` with
`impact_conditioning="head_relative"`.

## Why

- The location-conditioned models learn location → history and give near-identical designs the same curve,
  even on their own training runs.
- The single-location test removed the location input. At location 137, though, no training design has a
  frame change under the head, so design 7, the only one that has, could not be predicted.
- Here the location input is removed and the origin moves to the head instead. The same hood then looks
  different from every impact point.
- Across all 142 locations, every edited design has locations with its change right under the head:

| design | 1 | 2 | 3 | 5 | 7 | 8 | 9 | 11 |
|---|---|---|---|---|---|---|---|---|
| locations with the change within 30 mm of the head | 1 | 3 | 3 | 5 | 3 | 1 | 3 | 3 |
| within 80 mm | 6 | 6 | 9 | 16 | 10 | 10 | 12 | 17 |

  In head-centred coordinates all of these look alike: a change near the origin.

## What the network reads

- **Origin: the headform centroid,** the mean of the first 286 mesh rows.
  - In the 1704 set this centroid is the impact location (ImpactCoords X1-X3, within 0.4 mm), and the
    headform is only translated between runs.
  - The network computes the origin itself; `indentor` is validated but never read.
  - Moving the hood and head together leaves the prediction unchanged.
- **Node features:**
  - the offset from the head / 250 mm, and the horizontal distance;
  - with `--node-features structure`, also:
    - the panel (outer, inner, hinge as 0/1);
    - the unit surface normal;
    - log(1 + gap / 10 mm), where the gap is the distance from an outer node to the inner panel, and from any
      other node to the outer panel.
  - These channels come from each design's deck (`design_difference_data.load_design_geometry`, cached in
    `design_difference_cache`) and are joined by node ID. The headform rows get zeros.
- **On real data at location 137:** the skin is 85 mm below the head centre in every design. The frame lies
  16.3 mm under the skin in design 0, but only 5.2 mm in design 4. That gap is what makes cluster B's HIC high
  there, and it is now a direct input.

## Architecture

The network is `DesignChangeHistoryNet` (README_design_change_nodes.md) with:

1. **Node embedding:** an MLP of the 11 node features above.
2. **kNN neighborhood attention ×2** (k 256). Unchanged: it already reads only offsets between nodes.
3. **Pooling into 256 tokens:**
   - 128 global tokens;
   - 128 local tokens whose radial prior is centred on the head, with initial widths 44-354 mm.
   - The prior's precision is stored as log(p). The older inverse-softplus storage left the sharp tokens
     frozen (diagnosed on 2026-09-27).
   - There is no impact embedding and no FiLM; the learned queries are constants.
4. **Latent mixing ×3** and a temporal decoder ×2, as before.
5. **Change-node tokens** (the node-difference path). Their position is now the 3-D offset from the head
   centroid instead of the XY offset from the indentor. The decoder reads them after each block.
6. **Mirror augmentation** (`--mirror-augmentation true`): half of each training batch is mirrored left-right.
   - Mirroring negates the y offsets, the neighborhood offsets, normal_y, and the change offsets and
     displacements.
   - The 1704 impacts are purely vertical (initial velocity -11.1 m/s along z), so a mirrored run is a
     valid run with the same history.
   - Neighbor graphs and change-table lookups use the unmirrored coordinates.

The `head_relative` mode requires `--impactor-nodes 286` and the headform kept out of pooling, and it rejects
`--geometry-film`, whose encoder reads absolute XYZ.

## Run (DeltaAI)

```bash
bash submit_mesh_impact_history_1704_head_centred.sh
```

This is an A/B against the design-change-nodes run `20261023_154041_3440305_1704`, with the same settings:

- k256 ×2 neighborhoods, Fourier time, all 142 locations, stride 16;
- test 4, 5 and validation 11;
- 100 epochs, batch 8, lr 3e-4, seed 42, plain MSE.

It changes three things: head-relative input, structure channels and mirroring. For ablations, set
`HOOD_MESH_NODE_FEATURES=xyz`, `HOOD_MESH_MIRROR_AUGMENTATION=false` or `HOOD_MESH_DESIGN_CHANGE_NODES=false`.

The launcher skips `submit_mesh_impact_history_1704_fourier_k256.sh`, which pins conditioning to film, and
sets its Fourier values itself. Preflight loads the first run with its structure channels and runs a CUDA
forward/backward pass of the head-centred change model; it passed on the local RTX 5050.

Running locally works, but a full epoch takes about 30 min on the RTX 5050 (about 1.4 s per run at full
size). Use Delta for the 100-epoch run.

## What to compare

| run | test accel RMSE (g) | test accel R2 | test HIC R2 | near-clone diff R2 train / test |
|---|---|---|---|---|
| baseline `20261016_151923_3394500_1704` | 15.54 | 0.860 | 0.898 | 0.002 / -0.001 |
| design-change nodes `20261023_154041_3440305_1704` | | | | 0.57 / -0.46 |

Read `design_difference_r2.json`, `test_design_sensitivity.json`, `metrics.json` and `hic_metrics.json` of the
new run.

The first question is whether the test near-clone difference (4-5) rises above zero. The design-change run
fitted its training edits but got 4-5 wrong.

## Limits

- **Test designs are pinned.** The launcher chain fixes test 4, 5 and validation 11. To test design 7 (its
  edit is under location 137), run the trainer directly. Its siblings 6, 8 and 9 then stay in training,
  which the preflight cluster check flags unless `HOOD_MESH_ALLOW_CLONE_LEAK=1`.
- **Inference needs full input.** It needs the headform rows first, plus the structure channels, as
  `DataPreprocessor.load_mesh_geometry` returns them. A new hood needs its deck for the channels and its
  registration in the design-change table.
- **Unsupported tools.** `export_mesh_pooling_attention.py` and `infer_designs_mesh_impact_history.py` do not
  load this architecture. `HistoryPredictor.from_run` does.
- **Solver-noise ceiling.** Part of every near-clone difference is the solver time-step artifact, so it
  cannot be learned from geometry.

## Tests

```bash
python -m pytest tests/test_head_centred.py -q
```
