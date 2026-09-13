# Mesh and impact location to acceleration history

For the **neighborhood encoder + design-sensitivity training** experiment on
the same cluster-D holdout, see [the architecture and training derivation](README_mesh_local_sensitivity.md).
Submit it with `bash submit_mesh_impact_history_1704_local_sensitivity.sh`.
The ordinary launcher and Python defaults still reproduce the temporal baseline.

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
self-attention couples the requested times before a scalar acceleration head.
Training defaults to the original `--decoder temporal` architecture, including
global/local mesh attention, latent-token attention, decoder cross-attention,
temporal self-attention and position-wise feedforward layers. The optional
`--decoder mesh_only` ablation removes only temporal self-attention.
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
scikit-learn, joblib, matplotlib, and W&B installed. W&B logs online by default
to project `hood-impact-mesh-attention`. Authenticate once in the training
environment with `python -m wandb login` before starting an online run.

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
3 latent attention layers, 2 decoder layers, dropout 0.1, and `decoder=temporal`.
The depth flag remains `--temporal-layers` for checkpoint compatibility; with
`mesh_only`, it counts mesh cross-attention blocks and adds no temporal attention.

```powershell
python train_mesh_impact_history.py --data-format euroncap1704 --test-designs 10 11 --val-designs 5 --decoder temporal --epochs 100 --batch-size 8 --lr 0.0003 --device cuda --output-dir runs/mesh_impact_history/euroncap_01
```

Design IDs are **zero based**: `(run_number - 1) // samples_per_design`.
The defaults reserve designs 10 and 11 for testing and design 5 for validation.
For `industrylike`, one run is one design; for `euroncap1704`, each design has
142 runs; for `legacy`, each design has 50 runs. All other loaded designs train
the model. Validation/test designs must be disjoint, must have loaded runs,
and must leave a nonempty training set. Missing records are reported by the
shared preprocessor, and the exact successfully loaded split is saved.

Path overrides: `--inp-dir`, `--impact-coords-path`, `--acceleration-dir`,
and, for the legacy CSV format, `--mesh-geometry-dir` and `--doe-path`.
Additional options include `--num-samples`, `--samples-per-design`,
`--max-train-time`, `--weight-decay`, `--seed`, `--width`, `--num-heads`,
`--num-latents`, `--latent-layers`, `--temporal-layers`, `--dropout`, and `--decoder`.
Run `python train_mesh_impact_history.py --help` for all options.
Tracking options inherit `WANDB_MODE`, `WANDB_PROJECT`, and `WANDB_ENTITY`;
explicit `--wandb-mode`, `--wandb-project`, and `--wandb-entity` take precedence.
Use `--wandb-mode offline` for local W&B tracking or `--wandb-mode disabled`
for a smoke test without tracking. Each training run needs a new output directory.
The run starts before dataset loading, so it appears promptly in the chosen
project. `training.log` and `wandb_run.json` record its actual team, project,
ID, mode, and online URL. Training/validation losses, validation MAE in g, and
learning rate use epoch as their horizontal axis; test metrics are recorded
after evaluating the best validation checkpoint. Initialization failures stop
training with a login/destination error instead of silently disabling tracking.

## Submit the 1704 dataset on DeltaAI

On the cluster, check out branch `agent/mesh-impact-history-1704`, then run from
the repository root:

```bash
git fetch origin
git switch agent/mesh-impact-history-1704
git pull --ff-only origin agent/mesh-impact-history-1704
bash -l setup_mesh_impact_history_env.sh
module load "${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}"
source "${HOOD_MESH_VENV:-.venv-mesh-history}/bin/activate"
python -m wandb login
bash submit_mesh_impact_history_1704_clusterD.sh
```

The setup script creates a Linux-native `.venv-mesh-history` overlay using
DeltaAI's `python/miniforge3_pytorch/2.10.0` module and installs the project
dependencies. Run setup once. The job activates the same module and environment
inside its allocation. Set `HOOD_MESH_PYTORCH_MODULE` and `HOOD_MESH_VENV` for
both setup and submission if you use a different module or environment path.
For example, an environment on scratch can be selected with
`export HOOD_MESH_VENV=/work/nvme/<account>/$USER/hood-mesh-history`.

`submit_mesh_impact_history_1704_clusterD.sh` creates `runs/slurm` before submitting
`run_mesh_impact_history_1704.sbatch`. The job requests one GPU, 16 CPU cores,
96 GB host memory, and six hours on `ghx4`, account `bbqg-dtai-gh`. Additional
submission-script arguments go to `sbatch`, so resources can be overridden:

```bash
# One epoch over all 1704 cases to check the complete pipeline first.
HOOD_MESH_EPOCHS=1 bash submit_mesh_impact_history_1704_clusterD.sh --time=00:30:00

# Longer experiment with different held-out designs.
HOOD_MESH_EPOCHS=200 HOOD_MESH_BATCH_SIZE=8 \
HOOD_MESH_TEST_DESIGNS="6 7 8 9" HOOD_MESH_VAL_DESIGNS="4 5" \
bash submit_mesh_impact_history_1704.sh --time=08:00:00
```

Preflight and training run directly as Python processes inside the allocated
single-node Slurm batch job, inheriting its `CUDA_VISIBLE_DEVICES`. Slurm sets
the GPU visibility for the batch script as described in its
[GPU management documentation](https://slurm.schedmd.com/gres.html#GPU_Management).
Both processes use the same activated environment; CUDA remains mandatory, and
a failed preflight stops the job before training. A training failure also
returns a nonzero batch exit status.

Job `3139891` stopped at the first `srun` with
`task 0 launch failed: Error configuring interconnect`, before Python started.
The earlier V6 jobs `3056355` and `3063195` successfully used the same PyTorch
module and GPU driver, so those logs do not indicate an environment version
mismatch. Direct Python avoids the failing extra Slurm step for this
single-process model. The `.out` file now prints `Launch: direct Python in
Slurm batch allocation` and `Starting dataset and CUDA preflight.` before the
Python checks. Pull the fix and resubmit with the submission command above;
no environment reinstall is needed for this launcher change.

The job always selects **euroncap1704**, 1,704 runs, acceleration prediction,
and 142 cases per design. Default test designs 10 and 11 and validation design 5
give **1,278 training / 142 validation / 284 test** cases. Epochs, batch size, learning
rate, and time stride inherit the current Python configuration (currently
100 epochs, batch 8, LR 0.0003, stride 16). There is no launcher time-stride
override. W&B defaults to online and uses the credentials saved by the login
command above. To select a team/project, submit with, for example:

```bash
WANDB_ENTITY=my-team WANDB_PROJECT=hood-impact-mesh-attention \
bash submit_mesh_impact_history_1704.sh
```

An existing `WANDB_MODE` environment setting is respected; use
`WANDB_MODE=online bash submit_mesh_impact_history_1704.sh` to explicitly
enable online logging, `WANDB_MODE=offline` for local tracking to sync later,
or `WANDB_MODE=disabled` for an intentionally untracked smoke test. An offline
run can be uploaded later using `python -m wandb sync /path/to/offline-run-*`.
Runs created with disabled tracking have no W&B event files to sync.

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

### Temporal-attention ablation on the hardest existing split

The dedicated `submit_mesh_impact_history_1704_no_temporal.sh` launcher pins
`decoder=mesh_only`, test designs **10,11** and validation design **5**, even if
older split/decoder settings are exported in the shell. Training includes all
142 impact locations for each of designs **0,1,2,3,4,6,7,8,9**. Cluster D appears
only in test, so validation-based checkpoint selection does not see it.
Validation design 5 still shares cluster B with training design 4; preflight
reports that limitation, matching the existing baseline exactly.

The saved `splits.json` and `metrics.json` identify the comparison runs:

| Run | Validation | Test | Acceleration R² | RMSE (g) |
| --- | --- | --- | --- | --- |
| `20260904_220022_3086980_1704` (test clones in training) | 10 | 2 | 0.92749 | 11.0542 |
| `20260912_004325_3134499_1704` | 10 | 11 | 0.83090 | 17.6786 |
| `20260912_005426_3134539_1704` (ablation baseline) | 5 | 10,11 | 0.82170 | 18.4398 |

Compare the new run against the last row. Architecture dimensions, preprocessing,
training loss, optimizer, seed and training schedule retain the baseline defaults.
The ablation has **1,131,281** parameters, versus **1,263,889** in the default
temporal model. This experiment measures the effect of removing temporal attention;
isolating the benefit of global or local attention individually would require
separate ablations.

The completed ablation `20260913_000210_3140017_1704` scored HIC15 R² **0.7360**
against full-resolution simulations, versus **0.7603** for the temporal baseline
`20260912_005426_3134539_1704` on the same 284 test impacts. HIC RMSE rose from
**210.53** to **220.95**, and predictions within ±10% fell from **62.0%** to
**56.0%**. Temporal attention is therefore restored as the training and launcher
default. The hardest split and direct Python launch fix are retained.

To submit the restored temporal model on this same split:

```bash
bash submit_mesh_impact_history_1704_clusterD.sh
```

The generic and cluster-D launchers also default to the hardest split and
`temporal`, but allow explicit environment overrides. The dedicated
`submit_mesh_impact_history_1704_no_temporal.sh` still pins the ablation for
reproduction, and saved `mesh_only` checkpoints remain loadable. Both preflight and
training receive the same resolved decoder. The decoder is recorded in
`config.json`, W&B configuration and training logs. Inference reloads that
decoder automatically; old configs without this field retain the original
temporal architecture and load their checkpoints strictly.

Other optional environment overrides use the `HOOD_MESH_` prefix:
`LR`, `WEIGHT_DECAY`, `SEED`, `WIDTH`, `NUM_HEADS`, `NUM_LATENTS`, `LATENT_LAYERS`,
`TEMPORAL_LAYERS`, `DROPOUT`, `DECODER`, `OUTPUT_DIR`, and `RUN_NAME`. Default result folders
are `runs/mesh_impact_history/<timestamp>_<jobid>_1704`; scheduler logs are
`runs/slurm/mesh_history_1704_<jobid>.out` and `.err` for the generic launcher,
or `mesh_history_1704_no_temporal_<jobid>.out` and `.err` for the ablation.
The logs include the Git
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
them again. For `temporal` checkpoints, use the complete intended output grid
together because that decoder couples time queries. For `mesh_only` checkpoints,
a time point's prediction is independent of the other requested times.
Accuracy on new geometry or time ranges must be
established with held-out data.

## Saved results

- `config.json`: resolved preprocessing, data, training, and new architecture settings.
- `splits.json`: design IDs and original run numbers for each split.
- `scalers.joblib`: training-only normalization statistics.
- `prediction_times.npy`: default sampled output grid.
- `hood_impact_best_model.pt`: this new network's best validation checkpoint,
  including optimizer/scheduler state (the filename follows the shared trainer).
- `training_history.json`, `training_history.csv`: normalized train/validation MSE.
- `training_history.png`: train/validation loss curves with the best validation
  epoch marked; also uploaded to W&B for online/offline tracking runs.
- `metrics.json`: test MSE in g squared, RMSE/MAE in g, and R2. Undefined metrics
  are JSON `null`.
- `test_acceleration_histories.csv`: `run_number`, `time`,
  `acceleration_true_g`, and `acceleration_pred_g` on each test run's sampled grid.
- `training.log`: run configuration summary, data/grid warnings, and final metrics;
  the shared trainer's epoch progress is printed to the console.
- `wandb_run.json`: resolved W&B mode, project, entity/team, run ID/name, and URL
  (`null` URL when tracking is offline or disabled).

Training starts with fresh weights. Evaluation reloads the best validation
checkpoint from this run using `torch.load(..., weights_only=True)`.

### Recover plots and W&B metrics from a completed run

The loss plot is now saved automatically after training, before test evaluation.
For an existing results folder, generate it without retraining:

```bash
python mesh_impact_history_reporting.py runs/mesh_impact_history/<run-folder>
```

Earlier launchers defaulted W&B to disabled. A disabled run has no W&B record
to synchronize, but its saved losses and final test metrics can be uploaded:

```bash
python mesh_impact_history_reporting.py runs/mesh_impact_history/<run-folder> --upload-wandb
```

Authenticate first using `python -m wandb login`, or provide `WANDB_API_KEY`
in the environment. Optional destination overrides are `--wandb-project`,
`--wandb-entity`, and `--run-name`. This creates a run labelled
`historical_backfill` with a `_recovered` name and saves its URL to
`wandb_backfill.json`. It uploads the original train/validation losses, final
test metrics, and training-history plot. Historical per-epoch validation MAE,
learning rate, and original timing were not saved and are not reconstructed.
The original config and checkpoint remain unchanged.

## Validation

```powershell
python -m pytest tests/test_mesh_impact_history.py tests/test_mesh_impact_history_submission.py tests/test_mesh_history_reporting.py -q
```

Tests cover node-order invariance, variable batches and history lengths,
geometry/impact sensitivity, gradients, mixed precision, coordinate alignment,
the existing time stride, and a complete one-epoch training/checkpoint/inference
cycle for both decoder variants with training-only scalers. Ablation checks cover
zero coupling between distinct time queries, CUDA gradients, legacy checkpoint
reload, the exact cluster-D split, and launchers with simulated Slurm commands.
The ablation also passed the real 1704-dataset preflight and a CUDA
forward/backward probe on its first 38,977-node mesh with 63 sampled times.
A previous one-epoch CUDA smoke run on 12 real industrylike cases used the
original temporal architecture and stride 16. These checks establish execution;
predictive accuracy still requires a full training run.
