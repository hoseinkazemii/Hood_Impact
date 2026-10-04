# Change-region attention for acceleration histories

`MeshChangeAttentionNet` gives regions that differ between training designs a
dedicated path into the acceleration predictor. A region can be far from the
impact point. The model still receives the full hood and impact coordinates,
and every training location still contributes ordinary acceleration MSE.
Improved design sensitivity is an experimental objective, not a measured result
of this implementation. Training optimizes plain acceleration MSE.

## Why anchors rank within design families

The twelve designs are not twelve independent shapes. They are four near-clone
families, and the two kinds of geometry difference differ by more than an order
of magnitude. Measured on the nine training designs of the default split:

| difference | share of hood affected | magnitude |
| --- | --- | --- |
| between families (base panel and rib layout) | 75 % of candidates | p99 14.6 mm, max 24.3 mm |
| within a family (the nudges, dimples and holes) | 0.4 - 4.7 % of candidates | max 8.6 - 12.3 mm |

Ranking candidates by pooled across-design variation therefore spends every
anchor on the base-shape difference: a first fit placed all 32 anchors at
16.8 - 21.7 mm scores, none on a within-family feature. That is the wrong target.
Sibling designs differ only by the local features, so a model given only
base-shape anchors has no path to tell them apart and keeps predicting one
history per impact location.

Scoring within families instead reduces 30072 "changed" candidates to the 964
that genuinely move between siblings, and places all 32 anchors on those sites
at 5.1 - 10.0 mm. Families come from the training meshes themselves, not from a
stored cluster table, so no held-out membership is assumed. Training designs
group as `[[0,1,2,3],[6,7,8,9],[10]]`, with within-family pairs differing over
0.58 % of points at most and cross-family pairs over 21.7 % - a 37x margin, so
the partition does not depend on a finely tuned threshold. Design 10 forms a
single-design family and contributes no anchor scores, only reference geometry.

`--change-scope all_designs` restores pooled ranking for an ablation.

## Architecture

1. **Training-only geometry atlas.** Remove the leading 286 rigid headform nodes,
   use one hood mesh per training design, and compare the unregistered point
   clouds in their common physical coordinate frame. Nearest-surface geometry
   differences identify spatially separated change regions. Anchors rank by
   pooled *within-family* variance, over families holding at least two training
   designs, so the base-shape difference between families cannot absorb them.
   `--min-change-mm` (0.5 mm) keeps near-zero numerical differences from padding
   the anchor list once real sites run out. Neither impact coordinates nor
   acceleration/HIC labels choose these regions. Defaults are 32 region anchors,
   10 mm candidate spacing and 30 mm anchor separation.
2. **Change-centered patches.** For each input hood, gather the 256 nearest hood
   nodes to each fixed atlas anchor. This selection is independent of impact
   location. Node displacements and radial distance profiles relative to the
   frozen training reference describe the current design, including absent
   material around an anchor. That reference is the mean over all training
   designs, so it sits between families; the common part still cancels when two
   siblings are compared, but a per-family reference would sharpen the features.
   Different meshes need not share node IDs.
3. **Neighborhood attention in each patch.** Two shared neighborhood attention
   blocks encode each patch, followed by pooling to a region token. The anchor
   and its relationship to the impact point condition the region representation.
4. **Context and fusion.** Retain the original full-mesh global and
   impact-conditioned local attention tokens. Combine those tokens with the
   change-region tokens and mix them using latent self-attention. This keeps a
   path for common geometry and low-variance impact locations.
5. **History decoder.** Time queries cross-attend to the fused geometry memory,
   then temporal self-attention and an MLP predict the acceleration history.
   Defaults retain width 128, 4 heads, 32 original latent tokens, 3 latent blocks
   and 2 temporal blocks. There is no design-ID embedding.

The existing neighborhood model actually updates hood nodes using physical
node-to-node neighborhoods across the mesh; its impact-local restriction comes
from part of the later token attention. The new branch changes which regions
receive dedicated representation before that compression.

HIC15 remains postprocessing of predicted acceleration in g, using time in
seconds and the same sampled grid as the targets. HIC is not a training target.

## Objective and split isolation

The loss is plain normalized acceleration MSE, `mean((a_pred - a_true)^2)`, with
shuffled batches. This experiment changes the architecture only, so any gain in
design sensitivity has to come from the change-region tokens rather than from a
term that rewards matching design-to-design differences. Same-location
difference accuracy is still *reported* after training by
`test_design_sensitivity.json`; it is a diagnostic, never an optimization target.

The default 1704-run split is train designs **0,1,2,3,6,7,8,9,10**, validation
design **11**, and test designs **4,5**. Splits are determined from run IDs before
opening meshes or histories. Training opens only train and validation files;
it does not inspect the existence, geometry, targets or predictions of test
runs. The atlas and all scalers are fitted on training data only. Validation
selects the best checkpoint by acceleration MSE. Test evaluation requires a
separate explicit command after training has finished.

Every training design's hood coordinates must be identical across its impact
locations after removal of the configured headform prefix. The loader fails
if this assumption is false, or if a required training/validation run is missing.
It does not silently drop runs. `data_access_audit.json` records accessed split
run IDs; `change_atlas.json` and `change_atlas.npz` record the frozen atlas,
including the derived families, the pairwise moved-fraction matrix and the
separation margin. Training warns when that margin falls below three, because a
partition without a real gap should be inspected rather than trusted.

## Run

Use the existing environment setup once on DeltaAI:

```bash
bash -l setup_mesh_impact_history_env.sh
bash submit_mesh_change_attention_1704.sh
```

The new launcher submits a 48-hour single-GPU job and defaults to 100 epochs,
256 nodes per change region and online W&B tracking. It does not call the old
all-dataset preflight. Submit a separate 16-neighbor run with:

```bash
HOOD_CHANGE_K=16 HOOD_MESH_RUN_NAME=mesh_change_k16 \
  bash submit_mesh_change_attention_1704.sh
```

`HOOD_MESH_DATA_ROOT`, `HOOD_MESH_OUTPUT_DIR`, `HOOD_MESH_EPOCHS`,
`HOOD_MESH_BATCH_SIZE` and the existing W&B environment variables work as usual.
New branch settings use `HOOD_CHANGE_ANCHORS`, `HOOD_CHANGE_K`,
`HOOD_CHANGE_LAYERS`, `HOOD_CHANGE_SCALE_MM`, `HOOD_CHANGE_CHUNK_SIZE`,
`HOOD_CHANGE_CANDIDATE_SPACING_MM`, `HOOD_CHANGE_ANCHOR_SEPARATION_MM`,
`HOOD_CHANGE_SCOPE`, `HOOD_CHANGE_MIN_CHANGE_MM`,
`HOOD_CHANGE_FAMILY_TOLERANCE_MM` and `HOOD_CHANGE_FAMILY_MAX_FRACTION`. Old neighborhood/decoder/split environment
variables do not change this launcher. Resume is explicitly unsupported for
this first version; start in a fresh output directory.

Direct training, with optional paths pointing to another dataset copy:

```bash
python train_mesh_change_attention.py --device cuda \
  --output-dir runs/mesh_change_attention/k256
```

The run saves best/latest checkpoints, training history CSV/JSON/PNG,
train-fitted scalers, sampled prediction times, architecture configuration,
the declared split and the fitted atlas. The trained checkpoint contains the
atlas tensors and coordinate normalization buffers, so inference never needs
the original training meshes.

After selecting a completed experiment using validation, explicitly evaluate
the frozen best checkpoint:

```bash
python train_mesh_change_attention.py \
  --evaluate-run runs/mesh_change_attention/k256 --device cuda
```

This loads only the saved test split with saved preprocessing and writes
`test_acceleration_histories.csv`, `metrics.json`, `test_hic15.csv`,
`test_design_sensitivity.json` and `test_evaluation.json`. The latter sensitivity
report includes same-location difference RMSE, a collapsed-predictor reference,
true/predicted spread across designs and per-location diagnostics. A spread
ratio near zero identifies design collapse; a ratio near one alone does not
prove accurate design differences, so read it together with difference RMSE.
Do not use final test diagnostics to select hyperparameters if the test set is
to remain an independent final comparison.

Existing raw-geometry inference works through the shared factory:

```python
from train_mesh_impact_history import HistoryPredictor

predictor = HistoryPredictor.from_run("runs/mesh_change_attention/k256", device="cpu")
acceleration_g = predictor.predict(mesh_xyz, impact_xy)
times_seconds = predictor.time_points
```

`mesh_xyz` keeps the original headform-first ordering expected by the dataset;
hood node IDs need not correspond across designs. An unseen design uses its
own input geometry at inference, but neither the atlas nor scalers are updated.

## Limits and interpretation

The atlas anchors the feature sites that vary among training siblings. A test
design whose distinguishing feature sits somewhere else, or whose family is not
represented in training at all, is not covered by any anchor; the full-mesh
context path still sees it, but with no dedicated representation. The default
split holds out an entire family (designs 4 and 5), so this is the live risk for
that split rather than a hypothetical one.

An atlas fitted on available training designs may miss a new change appearing
only in an unseen design or outside selected regions. The full-mesh context
path remains available, but it does not guarantee extrapolation. Geometry-only
point clouds also omit connectivity, thickness, material and boundary-condition
changes that are not represented in coordinates. Different point sampling can
look like geometric change. The method assumes meshes use a common aligned
coordinate frame and consistent physical units.

With only nine training designs, more architectural capacity cannot establish
generalization by itself. Compare ordinary acceleration accuracy, HIC15,
same-location difference accuracy, and low/high-variance location diagnostics
using validation for tuning. The current validation split has a single design,
so within-validation spread cannot be measured; a future predeclared split
with multiple validation designs can support that selection criterion.
