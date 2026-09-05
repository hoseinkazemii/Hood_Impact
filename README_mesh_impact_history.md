# Mesh and impact location to acceleration history

`mesh_impact_history.py` defines the new `MeshImpactHistoryNet` from scratch.
It takes the irregular hood mesh node coordinates `(N, 3)` and the in-plane
impact location `(2,)`, and predicts a scalar acceleration history in **g**.
Mesh connectivity is not read: the geometry is an unordered set of all mesh
nodes, with no rasterization or regular mesh requirement. The network contains
no convolution layers and imports no previous model classes or checkpoints.

The architecture projects each node's coordinates and impact-relative geometry
into features. Impact-conditioned latent queries attend to all mesh nodes;
global queries and queries with learned distance priors describe the whole hood
and the impact neighborhood. Attention among the latent tokens combines their
information. Sampled output-time queries attend to those tokens, then temporal
self-attention couples the predicted history before a scalar output head.
The output times specify the fixed prediction grid; measured acceleration is
only a training target and is never a network input. Full mesh-to-mesh attention
is avoided by using a small latent set. Mesh node order does not define a spatial
axis, and variable node counts and history lengths are supported.

`train_mesh_impact_history.py` reuses only `utils/utils.py` configuration, data
preprocessing, design splitting, optimizer/training loop, and evaluation.
Coordinate scalers are fitted on training designs only; their statistics are
also stored in the model so the separately normalized mesh and impact remain
aligned when computing relative geometry.

## Train

Run from the repository root using an environment with PyTorch, NumPy, pandas,
scikit-learn, joblib, matplotlib, and W&B installed. W&B is disabled by default;
the shared trainer still requires the package to import.

Dependencies are listed in `requirements_mesh_impact_history.txt`. The local
`.venv` was verified with PyTorch 2.8, W&B 0.22 and Protobuf 6.33.6. Its Protobuf
pin fixes an existing W&B import failure with the system's Protobuf 7 without
changing system packages. On this Windows workspace, run the example commands
with `.\.venv\Scripts\python.exe` in place of `python`, or activate that environment.

```powershell
python train_mesh_impact_history.py --output-dir runs/mesh_impact_history/experiment_01
```

Defaults for data format, paths, sample count, epochs, batch size, learning rate,
and time subsampling come from the current `utils.utils.Config`. The current
default dataset is `industrylike`. Acceleration prediction is always selected.
The new architecture defaults are width 128, 4 attention heads, 32 latent tokens,
3 latent attention layers, 2 temporal layers, and dropout 0.1.

```powershell
python train_mesh_impact_history.py --data-format euroncap1704 --test-designs 2 --val-designs 10 --epochs 100 --batch-size 8 --lr 0.0003 --device cuda --output-dir runs/mesh_impact_history/euroncap_01
```

Design IDs are **zero based**: `(run_number - 1) // samples_per_design`.
The defaults reserve design 2 for testing and design 10 for validation.
For `industrylike`, one run is one design; for `euroncap1704`, each design has
142 runs; for `legacy`, each design has 50 runs. All other loaded designs train
the model. Validation/test designs must be disjoint, must have loaded runs,
and must leave a nonempty training set. Missing records are reported by the
shared preprocessor, and the exact successfully loaded split is saved.

Path overrides: `--inp-dir`, `--impact-coords-path`, `--acceleration-dir`,
and, for the legacy CSV format, `--mesh-geometry-dir` and `--doe-path`.
Additional options include `--num-samples`, `--samples-per-design`,
`--max-train-time`, `--weight-decay`, `--seed`, `--width`, `--num-heads`,
`--num-latents`, `--latent-layers`, `--temporal-layers`, and `--dropout`.
Run `python train_mesh_impact_history.py --help` for all options.
Use `--wandb-mode offline` for local W&B tracking or `--wandb-mode online`
for explicit online logging. Each training run needs a new output directory.

## Submit the 1704 dataset on DeltaAI

On the cluster, check out branch `agent/mesh-impact-history-1704`, then run from
the repository root:

```bash
git fetch origin
git switch agent/mesh-impact-history-1704
bash -l setup_mesh_impact_history_env.sh
bash submit_mesh_impact_history_1704.sh
```

The setup script creates a Linux-native `.venv-mesh-history` overlay using
DeltaAI's `python/miniforge3_pytorch/2.10.0` module and installs the project
dependencies. Run setup once. The job activates the same module and environment
inside its allocation. Set `HOOD_MESH_PYTORCH_MODULE` and `HOOD_MESH_VENV` for
both setup and submission if you use a different module or environment path.
For example, an environment on scratch can be selected with
`export HOOD_MESH_VENV=/work/nvme/<account>/$USER/hood-mesh-history`.

`submit_mesh_impact_history_1704.sh` creates `runs/slurm` before submitting
`run_mesh_impact_history_1704.sbatch`. The job requests one GPU, 16 CPU cores,
96 GB host memory, and six hours on `ghx4`, account `bbqg-dtai-gh`. Additional
submission-script arguments go to `sbatch`, so resources can be overridden:

```bash
# One epoch over all 1704 cases to check the complete pipeline first.
HOOD_MESH_EPOCHS=1 bash submit_mesh_impact_history_1704.sh --time=00:30:00

# Longer experiment with different held-out designs.
HOOD_MESH_EPOCHS=200 HOOD_MESH_BATCH_SIZE=8 \
HOOD_MESH_TEST_DESIGNS=7 HOOD_MESH_VAL_DESIGNS=3 \
bash submit_mesh_impact_history_1704.sh --time=08:00:00
```

The job always selects **euroncap1704**, 1,704 runs, acceleration prediction,
and 142 cases per design. Default test design 2 and validation design 10 give
**1,420 training / 142 validation / 142 test** cases. Epochs, batch size, learning
rate, and time stride inherit the current Python configuration (currently
100 epochs, batch 8, LR 0.0003, stride 16). There is no launcher time-stride
override. W&B defaults to disabled; set `WANDB_MODE=offline` or `online` to
enable tracking.

The 1704 dataset is excluded from Git and must already be available on the
cluster under `Data/HoodImpact_1704_EuroNCAP`. For a different location, set
`HOOD_MESH_DATA_ROOT=/path/to/HoodImpact_1704_EuroNCAP`. Preflight requires
the exact 1,704 `.inp` files, 1,704 acceleration CSVs, and
`ImpactCoords_1704.csv`; it does not require HIC targets. It checks the design
split, parses a sample using the configured stride, and runs a CUDA
forward/backward check before training. A local dataset check is also available:

```bash
python preflight_mesh_impact_history_1704.py --data-root Data/HoodImpact_1704_EuroNCAP
```

Multiple held-out designs are supported using quoted, space-separated IDs,
for example `HOOD_MESH_TEST_DESIGNS="0 1 2 3" HOOD_MESH_VAL_DESIGNS="4 5"`.
The existing dataset analysis identifies near-clone geometry groups
`[0,1,2,3]`, `[4,5]`, `[6,7,8,9]`, and `[10,11]`; hold out whole groups when
measuring generalization to different geometry.

Other optional environment overrides use the `HOOD_MESH_` prefix:
`LR`, `WEIGHT_DECAY`, `SEED`, `WIDTH`, `NUM_HEADS`, `NUM_LATENTS`, `LATENT_LAYERS`,
`TEMPORAL_LAYERS`, `DROPOUT`, `OUTPUT_DIR`, and `RUN_NAME`. Default result folders
are `runs/mesh_impact_history/<timestamp>_<jobid>_1704`; scheduler logs are
`runs/slurm/mesh_history_1704_<jobid>.out` and `.err`. The logs include the Git
commit, resolved interpreter, dataset path, and complete training command.

## Time grid and inputs

The existing preprocessing truncates at `Config.max_train_time` if configured,
then applies `time[::Config.time_subsample_stride]` and the identical slice to
acceleration **once**. The current stride is **16**, so a 1,000-point source
history produces 63 output points. No extra resampling or slicing is introduced.
The cutoff retains the shared preprocessing behavior, including its two-point
fallback when fewer than two source points fall before the cutoff.

The input mesh is `X1, X2, X3` in the source coordinate system. Impact is
`X1, X2`; the existing INP dataset preprocessing drops impact `X3` (the
drop/height axis). Inference uses the same coordinate units and frame.
The output time values retain the source CSV `Time` units (seconds for the
supplied datasets).

If records have different sampled time grids, training and test export preserve
each record's own grid. `prediction_times.npy` stores the already-subsampled
grid of the first training run for default inference; the reference run and
whether all loaded grids match are recorded in `config.json`. A mismatch also
produces a log warning.

## Predict from geometry and impact location

```python
import numpy as np
from train_mesh_impact_history import HistoryPredictor

predictor = HistoryPredictor.from_run("runs/mesh_impact_history/experiment_01", device="cpu")
mesh_xyz = np.load("new_hood_nodes.npy")  # (N, 3), physical coordinates
impact_xy = np.array([125.0, 350.0], dtype=np.float32)  # physical coordinates
acceleration_g = predictor.predict(mesh_xyz, impact_xy)
time = predictor.time_points  # already subsampled; len(time) == len(acceleration_g)
np.savetxt("prediction.csv", np.column_stack([time, acceleration_g]),
           delimiter=",", header="time,acceleration_g", comments="")
```

No measured response is needed for inference. For another run's output grid,
pass `sampled_time_points=already_subsampled_times` to `predict`. Those values
must already have the intended cutoff and stride; the predictor does not slice
them again. Because the temporal decoder couples time queries, use the complete
intended output grid together. Accuracy on new geometry or time ranges must be
established with held-out data.

## Saved results

- `config.json`: resolved preprocessing, data, training, and new architecture settings.
- `splits.json`: design IDs and original run numbers for each split.
- `scalers.joblib`: training-only normalization statistics.
- `prediction_times.npy`: default sampled output grid.
- `hood_impact_best_model.pt`: this new network's best validation checkpoint,
  including optimizer/scheduler state (the filename follows the shared trainer).
- `training_history.json`, `training_history.csv`: normalized train/validation MSE.
- `metrics.json`: test MSE in g squared, RMSE/MAE in g, and R2. Undefined metrics
  are JSON `null`.
- `test_acceleration_histories.csv`: `run_number`, `time`,
  `acceleration_true_g`, and `acceleration_pred_g` on each test run's sampled grid.
- `training.log`: run configuration summary, data/grid warnings, and final metrics;
  the shared trainer's epoch progress is printed to the console.

Training starts with fresh weights. Evaluation reloads the best validation
checkpoint from this run using `torch.load(..., weights_only=True)`.

## Validation

```powershell
python -m pytest tests/test_mesh_impact_history.py -q
```

Tests cover node-order invariance, variable batches and history lengths,
geometry/impact sensitivity, gradients, mixed precision, coordinate alignment,
the existing time stride, and a complete one-epoch training/checkpoint/inference
cycle with training-only scalers. A separate one-epoch CUDA smoke run on 12 real
industrylike cases completed with the default architecture and stride 16. This
checks execution; predictive accuracy still requires a full training run.
