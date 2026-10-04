# Train and inspect the model's attention weights

Submit from the repository root on DeltaAI after syncing the updated code:

```bash
bash submit_mesh_impact_history_1704_attention_export.sh
```

This starts a fresh model with two neighborhood layers, k=256, temporal
decoding, ordinary acceleration MSE, and a pinned HIC variation threshold
of **30%**. The launcher requests one GPU for 48 hours, covering training and
post-training extraction. The best validation checkpoint is reloaded in eval
mode; extraction does not change the checkpoint or the training forward pass.

## What is exported

The launcher exports **10 test cases**: locations 9, 54, 102, 125 and 142 on
each of test designs 4 and 5. Original run numbers are
`577 622 670 693 710 719 764 812 835 852`. These five locations span the ordered
list of qualifying location IDs. All pass the 30% threshold.

At 30%, 46 of 142 locations remain: **414 training, 46 validation and 92 test
histories**. Training and test evaluation use that whole subset; only attention
extraction is limited to ten cases. This dedicated launcher pins the threshold,
test split, selected cases and full matrices even if older settings are exported.

Each case stores the pooling probabilities with shape
`[4 heads, 256 tokens, N nodes]`. Softmax is over all N nodes separately for
each head/token. The local-token spatial bias is included. These are the
attention weights **after both neighborhood layers, before latent mixing**.
It also exports token self-attention from every latent block and temporal
self-attention from every temporal decoder block, separately for every head.
These are the model's actual post-softmax weights, with dropout off. There is
no logarithm, division by uniform attention, clipping of saved values, or
additional rescaling. Neighborhood-edge and decoder cross-attention matrices
are not part of this export.

The full matrices are float32 NPY arrays, streamed in 16-token chunks so they
need not all reside in GPU memory simultaneously. Typical cases use about
160 MB each: roughly **1.6 GB for ten cases**, before CSVs and plots.
Download the entire `attention_export/` folder inside your training run.

```text
runs/mesh_impact_history/<run>/attention_export/
    manifest.json                   # checkpoint hash, selected runs, completion status
    run_0577/                       # example: design 4, location 9
        pooling_weights.npy         # float32 [head, token, node]
        tokens_self_attention.npy   # float32 [layer, head, query token, attended-to token]
        temporal_self_attention.npy # float32 [layer, head, query time, attended-to time]
        prediction_times_seconds.npy
        predicted_acceleration_g.npy
        node_ids.npy                # original Abaqus node IDs, aligned with matrix columns
        node_xyz_mm.npy             # original XYZ coordinates, aligned with matrix columns
        node_attention.csv          # per-node summaries, coordinates and headform flag
        top100_structural_nodes.csv # highest mean pooling attention among structural nodes
        hood_attention.png          # all/global/local top-view maps
        tokens_self_attention.png   # every layer and head, direct values
        temporal_self_attention.png # every layer and head, direct values
        metadata.json               # shape, impact XY, token banks, normalization check
```

The node CSV retains original mesh order. `node_index` is the zero-based matrix
column; `node_id` is the original Abaqus ID. Tokens 0–127 are global and 128–255
are local for the default 256-token model; metadata records actual boundaries.
Headform nodes remain in the weights and CSV. Plots hide their first 286 rows
so they do not obscure the hood, without renormalizing structural weights.

`mean_all` averages over heads and tokens. `mean_global` and `mean_local`
average over heads and the corresponding token bank. `mean_head_0`, etc.
average over tokens for each head. Each summary sums to approximately one
over all nodes. Averages may hide individual token/head specialization; inspect
the full weights to see that detail.

Heatmaps use the attention values directly on a linear color scale. A larger
number means more attention. A yellow star marks the impact location on hood
plots. The overview hood maps show the labelled averages; individual head/token
inspection shows that head/token's exact values.

With default settings, token self-attention has shape `[3, 4, 256, 256]` and
temporal self-attention has shape `[2, 4, 63, 63]`. Rows are queries and columns
are the tokens/times they attend to. Time coordinates are saved in seconds.
The matrices come from the actual forward pass via temporary export hooks;
normal training is unchanged. An older mesh-only decoder has no temporal
self-attention, so its temporal files are absent and metadata records that fact.

## Control output size

For exact node summaries and PNGs without the large pooling matrices (the
smaller token and temporal matrices are still exported):

```bash
python export_mesh_pooling_attention.py runs/mesh_impact_history/YOUR_RUN \
  --device cuda --summary-only --runs 577 622 670 693 710 719 764 812 835 852 \
  --output-dir runs/attention_summaries
```

For full matrices on selected original test run numbers:

```bash
python export_mesh_pooling_attention.py runs/mesh_impact_history/YOUR_RUN \
  --device cuda --runs 577 719 --output-dir runs/attention_two_cases
```

Those example runs are location 9 on test designs 4 and 5 and qualify at 30%.
Requested runs must exist in the saved split after filtering. To export another
split in a separate extraction, pass `--split train`, `validation`, or `all`.

## Export from an existing trained run

```bash
python export_mesh_pooling_attention.py runs/mesh_impact_history/YOUR_RUN --device cuda
# Selected cases, with relocated simulation data:
python export_mesh_pooling_attention.py runs/mesh_impact_history/YOUR_RUN \
  --data-root Data/HoodImpact_1704_EuroNCAP --runs 577 719 --device cuda
```

These commands use saved model settings, scalers and split membership. Older
MeshImpactHistoryNet checkpoints remain compatible. The exporter refuses to
overwrite an existing export directory; use `--output-dir NEW_DIRECTORY` for
a new extraction. If interrupted, `manifest.json` remains `in_progress`; only
`complete` indicates the entire selected set finished. A partial directory can
contain incomplete arrays and should not be treated as a completed export.

## Inspect after downloading

No source dataset or GPU is needed to inspect the exported files. The small
inspection script needs NumPy, pandas and Matplotlib, but not PyTorch:

```bash
python inspect_mesh_attention.py path/to/attention_export/run_0577 --head 0 --token 0
python inspect_mesh_attention.py path/to/attention_export/run_0577 --head 2 --token 128
# Token-to-token attention, first layer, first head:
python inspect_mesh_attention.py path/to/attention_export/run_0577 --kind tokens --layer 0 --head 0
# Time-to-time attention, first layer, first head:
python inspect_mesh_attention.py path/to/attention_export/run_0577 --kind temporal --layer 0 --head 0
```

Each command saves a PNG and CSV. Pooling CSVs rank nodes by weight; token and
temporal CSVs contain the selected layer/head matrix. Temporal CSV row and
column labels are times in seconds. The first two examples inspect a global
and local pooling token, respectively.

For direct numerical inspection:

```python
import numpy as np
weights = np.load("run_0577/pooling_weights.npy", mmap_mode="r")
ids = np.load("run_0577/node_ids.npy")
xyz = np.load("run_0577/node_xyz_mm.npy")
w = weights[0, 128]  # head 0, first local token
top = np.argsort(w)[-20:][::-1]
print(ids[top], xyz[top], w[top])
```

Pooling weights describe how the model aggregates node features; they are not
causal importance scores or attributions to a specific acceleration time.
Each structural node's features already incorporate two neighborhood layers,
so a large pooling weight also selects information received from its neighbors.
The exporter tests verify row normalization and reconstruct the actual pooled
representation from the exported weights and values, including radial bias.
