# GPU multi-scale HIC model (V6)

V6 is a new PyTorch experiment for the combined 660-sample dataset. It is
additive: it does not import or modify `temporal_deeponet.py`, `utils/utils.py`,
their launchers, or previous checkpoints.

## Why the architecture changed

The best 660-sample TemporalDeepONet encoded the entire absolute-coordinate
mesh with a global maximum before applying the impact location. That input also
contained the 286 translated rigid-headform nodes and omitted impact X3. The
resulting global vector frequently collapsed distinct hood designs. Training at
stride 16 retained only 63 of 1,000 history points, and checkpoint selection
used acceleration MSE rather than the requested HIC metric.

V6 instead uses a conventional, compact GPU convolutional model:

1. It removes the rigid headform through the Abaqus rigid-body element set and
   verifies that the two source corpora contain identical static geometry for
   each design.
2. It parses one hood per design and creates three impact-centered 48x48 maps
   covering local (±175 mm), regional (±400 mm), and whole-hood (±1,800 mm)
   structure. Each map retains occupancy, node density, minimum/maximum/mean
   structural height, dedicated inner-panel/rib maps, outer-skin height, and
   density/height channels for all 496 fastener connectors (992 endpoints).
3. A shared residual CNN encodes the three scales. Explicit X/Y/Z, fitted local
   surface height/normal, headform gap, and boundary descriptors are fused with
   those embeddings.
4. The primary head predicts a standardized correction to a training-only
   per-location HIC mean. Adding that correction back gives canonical HIC15.
   This residual-learning target lets the CNN focus on the smaller
   design/geometry interaction. A convolutional decoder simultaneously predicts
   all 1,000 acceleration values as an auxiliary physical task. The model has
   about 1.90 million trainable parameters.
5. Canonical HIC15 is recomputed from each full-resolution history by checking
   every positive-duration window up to 15 ms. Neither `generation_metrics.csv`
   nor the old stride-16 target is used.

The loss is standardized residual-HIC MSE plus small Smooth-L1 and
peak-weighted history losses. AdamW, mixed precision, gradient clipping,
warmup/cosine decay, and real patience-based early stopping are used. The
checkpoint is selected by validation HIC RMSE—not acceleration loss.

## Dataset and split

The default experiment uses all 660 simulations:

- `HoodImpact_60_IndustryLike`: 12 designs × 5 locations
- `HoodImpact_600_EuroNCAP`: 12 designs × 50 locations

Both complete directories must exist directly under `Data/`; transferring only
the 600-sample directory used by V5 is insufficient. V6 needs each directory's
`inp_files`, `output_history_acc`, and tracked coordinate/manifest CSV files.

It uses designs 0–9 for training, design 10 for validation, and design 11 as the
historical reference test. Samples are never randomly split. Normalization is
fitted only on training designs. Every export carries explicit design, source,
source-run, and location metadata, avoiding the old combined-run ID bug.

After validation chooses the epoch, V6 fits a fresh model and fresh normalizer
on designs 0–10 for exactly that many epochs. The original selection checkpoint
and validation report remain untouched. `final_model.pt` and
`final_normalization.npz` form the deployable pair. Both pre-refit and
post-refit design-11 results are reported so the effect of adding design 10 is
visible.

The residual target is specifically for same-grid, unseen-design prediction:
inference requires a known source and location from these 55 shared impact
locations. It must not be described as an arbitrary/off-grid HIC predictor.

Design 11 has been examined in previous experiments and is unusually close to
design 10, so it is not a statistically untouched test. The separate
IndustryLike-5 and EuroNCAP-50 metrics should be read alongside the pooled
score. For publication-level uncertainty, rotate the held-out design in
additional runs.

## DeltaAI environment and submission

DeltaAI is aarch64. Do not reuse a Delta x86 environment. Load the system
PyTorch stack and activate a DeltaAI-native environment before `sbatch`; the
batch job deliberately inherits that environment:

```bash
module reset
module load python/miniforge3_pytorch
conda activate base

# The shared environment is read-only. Create this once, or activate your
# existing DeltaAI-native environment. Replace <account> with your allocation.
python -m venv --system-site-packages /work/nvme/<account>/$USER/hood-v6
source /work/nvme/<account>/$USER/hood-v6/bin/activate
python -m pip install -r requirements_gpu_v6.txt

sbatch run_gpu_multiscale_hic_v6.sbatch
```

The launcher requests one GH200 allocation from `ghx4`, 16 CPU cores, and 64 GB
host memory. It fails before training if CUDA, data, geometry, or a model
forward/backward check is invalid. Defaults can be overridden without editing:

```bash
HOOD_V6_EPOCHS=250 HOOD_V6_BATCH_SIZE=24 \
sbatch run_gpu_multiscale_hic_v6.sbatch
```

## Outputs

Every run writes to a unique `runs/*_gpu_multiscale_hic_v6/` directory and
records configuration/git provenance, preprocessing fingerprint, normalizers,
model architecture, selection/final checkpoints, training/refit histories,
validation metrics and predictions, full predicted histories, and diagnostic
plots. The supplied launcher passes `--evaluate-test`, adding pre-refit and
post-refit reference-test artifacts. Omit that flag during architecture or
hyperparameter development. Pass `--no-refit-development` only for a diagnostic
selection-model run.

The main reported quantities are:

- direct-head canonical HIC15 R2/RMSE/MAE;
- HIC15 recomputed from the predicted 1,000-point history;
- pointwise acceleration metrics;
- separate IndustryLike-5 and EuroNCAP-50 scores; and
- a training-only location-mean baseline.

## Local checks

```bash
python -m unittest tests.test_gpu_multiscale_hic_v6 -v
python gpu_multiscale_hic_v6.py --preflight-only
```

On a CUDA workstation, a two-epoch end-to-end smoke run is available:

```bash
python gpu_multiscale_hic_v6.py \
  --require-cuda --smoke-test --evaluate-test \
  --output-dir runs/local_gpu_v6_smoke
```
