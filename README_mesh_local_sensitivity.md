# Neighborhood attention with ordinary acceleration MSE

The neighborhood model predicts acceleration histories from mesh XYZ, impact XY,
and requested times. Training uses only ordinary normalized acceleration MSE:

\[
\mathcal L = \frac{1}{\sum_b T_b}\sum_{b,t}
(\widehat y_{b,t}-y_{b,t})^2.
\]

The design-difference objective, paired sampler, specialized trainer, and weight
option have been removed. Training uses shuffled batches. There is no additional
HIC loss; HIC15 only selects locations for the filtered experiment.

## Submit on DeltaAI

From the repository root, after syncing the current code:

```bash
bash -l setup_mesh_impact_history_env.sh
# Two neighborhood layers, 256 neighbors, HIC variation >= 10%:
bash submit_mesh_impact_history_1704_hic_filtered_k256.sh
# All locations, with 256 neighbors:
bash submit_mesh_impact_history_1704_local_sensitivity_k256.sh
# All locations, with 16 neighbors:
bash submit_mesh_impact_history_1704_local_sensitivity.sh
```

The existing script names are retained. All use ordinary acceleration MSE.
The neighborhood launchers pin the temporal decoder, two neighborhood layers,
286 excluded headform nodes, 20 mm positional scaling, chunk size 1024, test
designs **4 and 5**, and validation design **11**. Training designs are
**0, 1, 2, 3, 6, 7, 8, 9, 10**. Unset `HOOD_MESH_RESUME_FROM` before a fresh run.

The k256 wrapper requests 24 hours on one GPU; the k16 wrapper requests 48 hours.
A trailing Slurm option overrides that limit. W&B uses the existing login and project.
The shared batch script runs Python directly in the Slurm allocation.

## HIC location selection


```bash
bash submit_mesh_impact_history_1704_hic_filtered_k256.sh
# Increase the threshold, for example to 20%:
HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT=20 bash submit_mesh_impact_history_1704_hic_filtered_k256.sh
```

This fresh experiment retains two neighborhood layers, 256 neighbors, temporal
decoding, ordinary acceleration MSE, test designs 4/5 and validation design 11.
The default threshold is **10%**, inclusive. Variation means
`100 * (max(HIC15) - min(HIC15)) / mean(HIC15)` across all 12 designs at a
matched location; it is a relative range, not a coefficient of variation.
An all-zero location has zero variation. HIC15 is recomputed from full
`SAE1000_interp1000` acceleration histories using every admissible trapezoidal
integration window up to 15 ms, before training time subsampling or truncation.

Local full-data counts:

| Threshold | Locations retained (of 142) |
|---|---:|
| 10% | 109 |
| 15% | 89 |
| 20% | 72 |
| 25% | 55 |
| 30% | 46 |
| 40% | 29 |
| 50% | 19 |

At 10%, the retained histories are **981 train / 109 validation / 218 test**.
The same location mask applies to every design before splitting and fitting
training scalers. All 1704 source runs must load successfully; an empty
selection is an error. Preflight still audits the full dataset and architecture;
the training log reports the actual filtered counts afterward.

Each run saves `hic_location_variation.csv` (all locations, per-design HIC,
relative range and selection), `hic_location_filter.json`, and the selection
inside `config.json`. `splits.json` contains only retained original run IDs.
Resume restores the saved selection and threshold. Existing launchers keep
their behavior when `HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT` is unset.
To recount without loading meshes or starting training:

```bash
python hic_location_filter.py --threshold 10
```

Reports default to `runs/hic_location_filter/`. Selection uses all designs,
including held-out response labels, as requested; evaluation therefore measures
performance on this response-selected subset, not an untouched population.
The B test designs are near-clones: their pairwise variation primarily measures
simulation scatter, so this test alone cannot establish generalization of
response differences between distinct geometry clusters. Compare predicted and
true design differences as well as overall R-squared when interpreting the run.

## Network

Current defaults, unless overridden through the shared batch script:

| Stage | Configuration |
|---|---|
| Node inputs | XYZ, impact-relative XY, squared XY distance: 6 features |
| Node embedding | 6 -> 128 -> 128, GELU, LayerNorm |
| Neighborhood encoder | 2 layers; 4 heads; k=256 in the k256 launchers |
| Neighborhood feedforward | 128 -> 256 -> 128 |
| Mesh pooling | 256 tokens: 128 global, 128 impact-local |
| Latent mixing | 3 self-attention blocks; 4 heads; feedforward 128 -> 512 -> 128 |
| Time features | t, t squared, t cubed, tanh(t), exp(-t squared), using normalized t |
| Time embedding | 5 -> 128 -> 128, plus impact embedding |
| Decoder | 2 blocks, each with mesh cross-attention and temporal self-attention |
| Acceleration head | LayerNorm, 128 -> 128 -> 1 |
| Dropout | 0.1 |

This configuration has **1,558,921 trainable parameters**. Neighbors and mesh
tokens are independent settings, even though both default to 256 in this run.

Neighborhood selection uses physical XYZ Euclidean distance, including self.
Each head learns feature attention plus a relative-position score and message.
Relative XYZ offsets are divided by 20 mm; this is a scaling factor, not a
cutoff radius. Two layers propagate information through two neighbor hops.

The leading 286 rigid-headform nodes bypass neighborhood attention but still
participate in pooling. Preflight checks their boundary against two runs.
Structural graphs are cached across impacts on an identical geometry. Chunking
and activation recomputation bound intermediate attention memory.

Global pooling queries have no distance bias. Local queries use a learned
negative squared-XY-distance bias around the impact. Both banks see all nodes.
Latent self-attention mixes these summaries before the decoder reads them.
Temporal attention is noncausal and couples the requested times; measured
accelerations never enter the decoder as inputs. The output is converted from
normalized acceleration to g using training-only scaler statistics.

The model consumes point coordinates, not element connectivity, thickness,
material properties or contact definitions. Design IDs are not network inputs.

## Optimization and artifacts

Defaults: AdamW, learning rate 0.0003, weight decay 0.00001, cosine scheduling,
gradient clipping at 1, seed 42, batch size 8, and 100 epochs. Time stride 16
subsamples the original histories. The filter uses full histories beforehand.
The best checkpoint minimizes validation acceleration MSE.

Scalers are fitted only on retained training examples. `config.json` records
`loss: normalized_acceleration_mse` and `batch_sampling: shuffle`.
`training_history.json`, CSV, and the loss plot contain ordinary train and
validation losses. Test exports contain only retained test runs.

## Resume

```bash
bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh RUN_DIRECTORY
```

Resume restores the saved architecture, split, HIC selection, scalers,
optimizer, scheduler, random-generator state when available, and total epoch
target. It writes a new output directory. Dataset paths can be relocated with
`HOOD_MESH_DATA_ROOT`. A checkpoint trained with the former combined objective
is rejected: start a fresh MSE experiment rather than mixing loss histories.
Historical checkpoints remain usable for inference with `HistoryPredictor`.

## Verification

Tests cover physical neighbor selection, permutation invariance, local
influence, chunked gradients, headform exclusion and caching, ordinary-MSE
training and reload, train-only scalers, filtered split membership, checkpoint
resume, and launcher forwarding. Geometry cluster metadata remains available
in `mesh_design_clusters.py` for preflight holdout checks and analysis tools.
