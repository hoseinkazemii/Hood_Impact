# Direct geometry correction for acceleration histories

This experiment freezes a saved `MeshImpactHistoryNet` and adds a separate
coefficient-based correction to its acceleration prediction. Geometry information
in the new branch reaches the output directly instead of entering the baseline's
pooled token memory. The correction uses an MLP and fixed geometry measurements;
it adds no graph model, graph message passing, attention pooling, or local patch
decoder. The baseline retains its saved architecture and is evaluated without
dropout or weight updates.

The experiment implements the coefficient-based proposal:

\[
\hat a(G,p,t)=\hat a_{\mathrm{base}}(G,p,t)
             +\sum_{j=1}^{r}c_j(G)b_j(p,t).
\]

`c(G)` is a fixed, normalized geometry coefficient vector. An MLP learns
`b(p,t)`, the response functions conditioned on the impact coordinates and
Fourier time features. Only this response MLP is trained. The objective remains
ordinary normalized acceleration MSE, with HIC15 calculated after prediction.
There is no objective that rewards producing variance for its own sake.

## Geometry representation

The basis is fitted to the training designs only. The leading rigid-headform
nodes are excluded, using the saved baseline's headform-node setting. Part-specific
anchors are constructed from the training geometry union on a fixed physical
grid. At each anchor, the descriptor contains the signed XYZ offset to the nearest
node and a sorted profile of distances to the nearest `k` nodes of that part.
Part separation prevents an inner-panel measurement from switching to a nearby
outer-panel node. This measures geometry in its common physical coordinate frame
without assuming that different meshes share node IDs or row correspondence.
These are nearest-node measurements, so mesh density and remeshing can affect
them even if the physical surface changes little. They are not guaranteed
material-point correspondence.

Descriptors are centered on the training mean and scaled with a configurable
noise floor. SVD retains all numerically resolved training modes; there is no
99%-variance cutoff that deliberately discards small geometry modes. Coefficient
normalization is fitted on training designs only. Validation and test descriptors
are projected onto that fixed basis.

For a batch of `B` samples, rank `r`, and the saved 63-point prediction grid:

| Tensor | Shape |
| --- | --- |
| Geometry coefficients | `B x r` |
| Learned response functions at all queried times | `B x 63 x r` |
| Sum of coefficient times response | `B x 63` |
| Frozen baseline and final corrected history | `B x 63` |

The matrix product can equivalently be written `(1 x r) @ (r x 63)` per sample.
The geometry vector is shared across that design's impact locations; the learned
functions vary with the impact and time queries. The saved baseline need not use
Fourier time features: only the new correction branch uses the configured Fourier
features.

With nine training geometries, the centered basis has rank at most eight. Retaining
all modes preserves the representable training differences, but does not guarantee
generalization. The run records held-out geometry reconstruction errors to expose
changes outside the training span. No design-ID embedding is used.

On the default nine training geometries, the fitted basis has rank eight. A
geometry-only inspection found approximately 99.4% of descriptor-difference
energy for held-out designs 4/5 outside that span, versus approximately 7.5% for
validation design 11. This is a substantial limitation of the whole-family
holdout: a correction can only use the represented component. These values
describe descriptor reconstruction, not measured acceleration performance, and
the run recomputes and saves its own reconstruction diagnostics.

## Baseline and comparison cohort

The default source is:

```text
runs/mesh_impact_history/20260930_204155_3283115_1704
```

The checkpoint, scalers, exact split run numbers, impact-location selection and
prediction time grid are restored from this run. The default saved split uses
training designs `0,1,2,3,6,7,8,9,10`, validation design `11`, and test designs
`4,5`. This baseline contains 46 locations selected by its original HIC30 filter.
That earlier filter used labels from all twelve designs; this experiment preserves
the cohort for a matched comparison and does not claim the inherited selection
was independent of test labels.

New anchors, descriptor scaling, geometry modes and coefficient normalization
use training geometry only. Validation acceleration MSE selects the best correction
checkpoint. Test data is loaded for evaluation after checkpoint selection. The
frozen baseline predictions are prepared once per split and reused during
correction training, so the full baseline is not run again for every epoch.

## Submit on DeltaAI

The saved run must be present on the cluster with these files:

```text
hood_impact_best_model.pt
config.json
scalers.joblib
splits.json
prediction_times.npy
```

Use the existing mesh environment. If it has not been set up on that machine:

```bash
bash -l setup_mesh_impact_history_env.sh
```

Submit from the repository:

```bash
bash submit_mesh_geometry_correction_1704.sh
```

The launcher validates the baseline artifacts and submits one GPU, 16 CPUs,
96 GB RAM and eight hours on `ghx4`. Its defaults are 300 epochs, batch size 64,
learning rate `1e-3`, weight decay zero and seed 42. Slurm logs go to `runs/slurm`;
the run directory is `runs/mesh_geometry_correction/<timestamp>_<job_id>_1704`.

To use another saved baseline or different settings:

```bash
HOOD_CORRECTION_BASELINE_RUN=/path/to/saved/baseline \
HOOD_CORRECTION_EPOCHS=500 \
WANDB_MODE=offline \
    bash submit_mesh_geometry_correction_1704.sh --time=12:00:00
```

Additional `sbatch` options are forwarded as separate arguments. New training
settings use `HOOD_CORRECTION_*` variables, so old neighborhood, decoder and split
environment variables do not alter this experiment.

| Environment variable | Default / purpose |
| --- | --- |
| `HOOD_CORRECTION_BASELINE_RUN` | Saved baseline directory above |
| `HOOD_CORRECTION_OUTPUT_DIR` | New timestamped output directory |
| `HOOD_CORRECTION_RUN_NAME` | `mesh_geometry_correction_1704_<job_id>` |
| `HOOD_CORRECTION_EPOCHS` | `300` |
| `HOOD_CORRECTION_BATCH_SIZE` | `64` |
| `HOOD_CORRECTION_LR` | `1e-3` |
| `HOOD_CORRECTION_WEIGHT_DECAY` | `0` |
| `HOOD_CORRECTION_SEED` | `42` |
| `HOOD_CORRECTION_ANCHOR_SPACING_MM` | `10` |
| `HOOD_CORRECTION_PROFILE_K` | `8` |
| `HOOD_CORRECTION_GEOMETRY_NOISE_FLOOR_MM` | `0.1` |
| `HOOD_CORRECTION_WIDTH` | `128` |
| `HOOD_CORRECTION_LAYERS` | `3` |
| `HOOD_CORRECTION_BASELINE_CHUNK_SIZE` | `128`; baseline inference workspace size, preserving its neighbors and weights |
| `HOOD_CORRECTION_NO_PLOTS` | `0`; set to `1` for numerical exports only |
| `HOOD_CORRECTION_FOURIER_NUM_FREQUENCIES` | `6` |
| `HOOD_CORRECTION_FOURIER_MIN_FREQUENCY_HZ` | `20` |
| `HOOD_CORRECTION_FOURIER_MAX_FREQUENCY_HZ` | `640` |

`HOOD_MESH_DATA_ROOT`, `HOOD_MESH_PYTORCH_MODULE`, `HOOD_MESH_VENV`,
`HOOD_MESH_OMP_THREADS`, `HOOD_MESH_MKL_THREADS`, `HOOD_MESH_NEIGHBOR_CACHE_DIR`,
`WANDB_MODE`, `WANDB_PROJECT`
and `WANDB_ENTITY` retain their usual meanings. The W&B project defaults to
`hood-impact-geometry-correction`; online tracking is the default.

For local or manually allocated training:

```bash
python train_mesh_geometry_correction.py \
    --baseline-run runs/mesh_impact_history/20260930_204155_3283115_1704 \
    --data-root Data/HoodImpact_1704_EuroNCAP \
    --output-dir runs/mesh_geometry_correction/manual \
    --device cuda --epochs 300 --batch-size 64 --lr 1e-3 \
    --wandb-mode offline
```

To resume an interrupted correction run, use a new output directory. The source
run's basis, baseline snapshot, optimizer, scheduler, split and total epoch budget
are retained. This resumes the original experiment rather than changing its
training settings:

```bash
python train_mesh_geometry_correction.py \
    --resume-from runs/mesh_geometry_correction/<interrupted_run> \
    --output-dir runs/mesh_geometry_correction/<new_resume_run> \
    --data-root Data/HoodImpact_1704_EuroNCAP --device cuda \
    --wandb-mode offline
```

To regenerate test exports from the saved best checkpoint without training:

```bash
python train_mesh_geometry_correction.py \
    --evaluate-run runs/mesh_geometry_correction/<completed_run> \
    --data-root Data/HoodImpact_1704_EuroNCAP --device cuda
```

Numerical-only evaluation is available with `--no-plots`. The submission wrapper
starts a fresh run; the resume/evaluation commands above can be run inside an
existing GPU allocation.

Portable inference loads both branches and the frozen geometry basis from the
new run. For example, to query location 9 on design 0:

```python
import pandas as pd
from mesh_geometry_correction import GeometryCorrectionPredictor

predictor = GeometryCorrectionPredictor.from_run(
    "runs/mesh_geometry_correction/<completed_run>", device="cuda"
)
coords = pd.read_csv("Data/HoodImpact_1704_EuroNCAP/ImpactCoords_1704.csv")
xy = coords.loc[8, ["X1", "X2"]].to_numpy(dtype="float32")
result = predictor.predict_inp(
    "Data/HoodImpact_1704_EuroNCAP/inp_files/HoodImpact_9.inp",
    impact_xy=xy, return_components=True,
)
# result contains time_seconds, baseline_g, correction_g, corrected_g,
# and geometry_diagnostics. Acceleration arrays have 63 values for this baseline.
```

The correction is converted to g using the acceleration scaler's scale only;
its mean is not added a second time. The default time grid is the saved baseline
grid, and the predictor also accepts explicit `sampled_time_points`.

## Results and interpretation

The new run snapshots the required frozen baseline artifacts in `baseline/`
and saves `geometry_basis.npz`, `correction_best.pt` and `correction_last.pt`.
The best checkpoint is selected by validation acceleration MSE, not test
sensitivity. Reusable inference restores both branches from the new run.

`test_metrics.json` compares baseline and corrected absolute acceleration/HIC15
accuracy and design sensitivity on the same cases; `metrics.json` exposes the
primary test metrics. The run also writes corrected and baseline acceleration
history CSVs, design-sensitivity JSON/CSV/PNG reports, HIC parity results and
`test_acceleration_plots/` with ground truth, baseline and corrected histories.
The corresponding train/validation reports are exported separately. Geometry
reconstruction diagnostics are saved in `train_geometry_diagnostics.json`,
`validation_geometry_diagnostics.json` and `test_geometry_diagnostics.json`.

For the sensitivity reports, evaluate these together:

| Metric | Interpretation |
| --- | --- |
| `design_difference_skill` | `1 - squared difference error / squared ground-truth difference`; perfect is 1, identical predictions across designs score 0, wrong differences can score below 0 |
| `difference_amplitude_ratio` | Predicted difference norm / actual difference norm; ideal is 1 |
| `difference_alignment_cosine` | Direction agreement between predicted and actual differences; ideal is 1 |
| `difference_gain` | Signed projection onto the actual differences; ideal is 1 |
| Acceleration and HIC15 errors | Check that improved sensitivity also improves prediction accuracy |

HIC15 here is derived from the same sampled 63-point acceleration grid used for
the comparison. Do not compare it directly with an original full-resolution HIC
report without accounting for that difference. Locations with zero actual design
variation cannot define norm-based difference ratios; consult the reported counts
and per-location data rather than interpreting undefined ratios as success.

The coefficient correction is linear in geometry coefficients while nonlinear
in impact and time. Contact or buckling may require nonlinear geometry
interactions. This first experiment isolates whether a direct, inspectable
geometry-to-output path improves the observed design collapse.

## Verification

```bash
python -m pytest tests/test_geometry_correction_basis.py \
    tests/test_mesh_geometry_correction.py \
    tests/test_mesh_geometry_correction_reporting.py \
    tests/test_mesh_geometry_correction_submission.py -q
```

The 72 tests cover training-only geometry fitting, signed coefficient preservation,
zero initial correction, successful synthetic response fitting, baseline freezing,
portable inference, artifact integrity, exact interrupted-run resumption, matched
metrics, and submission with mocked Slurm commands. No test submits a real job.
The existing baseline, neighbor-cache, design-sensitivity and HIC checks also pass.

A real CUDA forward reproduced one saved 63-point baseline history within
`4.47e-5 g` maximum absolute error. A synthetic gradient check verified that the
correction head receives gradients while the baseline receives none. This is
implementation verification; no full correction training has been run locally.
