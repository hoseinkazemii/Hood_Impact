# Temporal DeepONet acceleration baseline on the 1704 local-change dataset

This experiment retrains the architecture from
`runs/acceleration_history_target_value/20260116_160228_best` on
`Data/HoodImpact_1704_EuroNCAP`: 12 designs x 142 impact locations.
It predicts acceleration histories in g. Weights and normalization statistics
are learned afresh; the January checkpoint and old dataset are not needed on Delta.

The January architecture is **not today's default** `HoodImpactNeuralOperator`.
Its weights have one `film_layer`, and its temporal trunk ends at width 256.
`temporal_deeponet.ACCELERATION_BEST_ARCH` and
`LegacyHoodImpactNeuralOperator` reproduce its 1,376,513 trainable parameters
and support a strict load of the original checkpoint:

- PointNet MLP: 3 -> 64 -> 128 -> 128 -> 128 -> 256 -> 128, global max pooling.
- One FiLM layer conditioned on impact XY; branch projection width 128.
- Eight Fourier frequencies, trunk MLP 17 -> 64 -> 128 -> 256.
- Five width-256 dilated convolution blocks, then projection to 128.
- Elementwise branch/trunk product; residual output MLP [256, 256].

The original two-FiLM class and its entry point remain available. The new
training entry point explicitly selects the January single-FiLM architecture.

## Submit on DeltaAI

From the existing repository checkout:

```bash
git pull --ff-only
bash submit_temporal_deeponet_1704.sh
```

This uses the same `.venv-mesh-history` environment and module as the attention
jobs. If that environment has not been created, first run
`bash -l setup_mesh_impact_history_env.sh`. The job requests one GPU, 16 CPUs,
96 GB RAM, and six hours on `ghx4`, account `bbqg-dtai-gh`.
Extra launcher arguments go to `sbatch`, for example `--time=12:00:00`.

```bash
# One-epoch pipeline check; still uses all 1704 runs and the complete split.
HOOD_TD_EPOCHS=1 bash submit_temporal_deeponet_1704.sh --time=00:30:00

# Explicitly match the epoch budget and seed of a comparison run.
HOOD_TD_EPOCHS=100 HOOD_TD_SEED=42 bash submit_temporal_deeponet_1704.sh

# Dataset and model forward/backward checks only, inside a GPU allocation.
python train_temporal_deeponet_1704.py --preflight-only --device cuda
```

W&B uses `WANDB_MODE` (default `online`), `WANDB_PROJECT` (default
`hood-impact-mesh-attention`), and optional `WANDB_ENTITY`. Existing login
credentials work. Use `WANDB_MODE=offline` for local tracking.

Infrastructure overrides: `HOOD_MESH_PYTORCH_MODULE`, `HOOD_MESH_VENV`,
`HOOD_MESH_OMP_THREADS`, and `HOOD_MESH_MKL_THREADS` work as in the attention jobs.
Data can be relocated with `HOOD_TD_DATA_ROOT` (falls back to `HOOD_MESH_DATA_ROOT`).
Training overrides use **HOOD_TD_**: `EPOCHS`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`,
`SEED`, `TIME_SUBSAMPLE_STRIDE`, `MAX_TRAIN_TIME`, `OUTPUT_DIR`, and `RUN_NAME`.
Old `HOOD_MESH_*` model/split/resume/loss settings do not alter this experiment.
The submission script fixes test designs 4,5 and validation design 11.

## Comparison protocol

| Setting | Default |
| --- | --- |
| Training designs | 0,1,2,3,6,7,8,9,10 (1,278 runs) |
| Validation design | 11 (142 runs) |
| Test designs | 4,5, whole geometry cluster B (284 runs) |
| Target / loss | Standardized acceleration / MSE |
| Time sampling | Full window, stride 16: 1,000 -> 63 time points |
| Mesh input | Complete INP node block, including rigid headform; no node subsampling |
| Scaling | StandardScaler fit only on training runs |
| Optimization | AdamW, lr 3e-4, weight decay 1e-5, cosine schedule, gradient clip 1 |
| Budget | 100 epochs, batch size 8, seed 42 |
| Checkpoint selection | Lowest validation MSE |

These preprocessing and optimization defaults match the current attention
baseline. January's architecture is retained, while the old 200-epoch training
budget is not the default. Set `HOOD_TD_EPOCHS=200` if that is the intended budget,
and use the same budget in the comparison runs.

Preflight requires every mesh/history pair and the impact-coordinate table,
checks the split for test clones in training, and runs the full Temporal
DeepONet forward/backward on a real sample. Training fails if any run is skipped.
No HIC target table is consumed. Validation design 11 has a near-clone (10) in
training, as in the current attention experiment; test cluster B stays held out.

The model has no neighborhood attention or design-difference loss. Comparing it
with the neighborhood experiment therefore compares complete methods when that
experiment uses the extra loss. Check each run's `config.json` and `splits.json`
before interpreting an architecture advantage. Repeated seeds and matched-location
geometry-sensitivity checks are needed to distinguish local-geometry learning
from favorable initialization or location-based prediction; one aggregate test
score does not establish that mechanism.

## Results and reload

Runs go to `runs/temporal_deeponet_1704/<timestamp>_<job>_clusterB/`; Slurm logs
go to `runs/slurm/temporal_deeponet_1704_<job>.out` and `.err`.
Each run saves configuration, exact splits, train-fitted scalers, sampled times,
best and last checkpoints, training curves/CSV/JSON, `metrics.json` (MSE in g^2,
RMSE/MAE in g, dimensionless R2), and `test_acceleration_histories.csv` using the same
columns as the attention runs. Existing run artifacts cannot be overwritten.
The launcher starts fresh training; it does not resume from an older checkpoint.

```python
from train_temporal_deeponet_1704 import TemporalDeepONetPredictor

predictor = TemporalDeepONetPredictor.from_run("runs/temporal_deeponet_1704/<run>")
acceleration_g = predictor.predict(mesh_xyz, impact_xy)
times_seconds = predictor.time_points
```

`mesh_xyz` is the complete raw mesh node array and `impact_xy` is the raw impact
position. Optional `sampled_time_points` are already-subsampled times; the
predictor does not apply the stride again. Use the saved grid for comparisons.
