Fresh mesh-impact training now constructs **Fourier features instead of the
five handcrafted time functions**. The default frequency bank is:

```text
20, 40, 80, 160, 320, 640 Hz
```

For each output time t in seconds, its feature row is:

```text
[sin(2*pi*20*t),  cos(2*pi*20*t),
 sin(2*pi*40*t),  cos(2*pi*40*t),
 sin(2*pi*80*t),  cos(2*pi*80*t),
 sin(2*pi*160*t), cos(2*pi*160*t),
 sin(2*pi*320*t), cos(2*pi*320*t),
 sin(2*pi*640*t), cos(2*pi*640*t)]
```

There are **12 features per query**, without appending t, polynomial, tanh, or
Gaussian features. The six frequencies are fixed, not trainable parameters;
the time MLP and downstream attention/decoder weights are learned.

The 20 Hz pair has a 50 ms period and distinguishes the two ends of the roughly
25 ms prediction window. Higher pairs supply progressively faster variations.
These are starting frequencies for an experiment, not inferred physical modes
of the hood. All samples on the fixed grid still receive the same initial
features; design information enters their time tokens through mesh cross-attention.

The dataset continues to standardize time as s = (t - mean) / scale. The model
reconstructs physical seconds as t = s*scale + mean before constructing Fourier
phases. Training-only time mean and scale, plus the frequency bank, are saved as
model buffers. Consequently, frequencies retain their meaning after saving,
loading, resuming, or changing a dataset's scaler.

The query path for a batch is:

```text
Normalized sampled times                          B x 63
Restore physical seconds and construct Fourier    B x 63 x 12
Linear 12 -> 128, GELU, Linear 128 -> 128           B x 63 x 128
Two existing decoder blocks                       B x 63 x 128
Existing shared acceleration head                 B x 63 x 1
```

The XYZ node encoder, neighborhood graph, FiLM, mesh pooling, latent mixing,
cross-attention, temporal self-attention, acceleration head, time subsampling,
loss and split follow their existing settings. At width 128, the first time
linear layer gains 896 weights. The k256/two-neighborhood-layer configuration
has **1,592,457 trainable parameters**.

Submit a **fresh run** after syncing this code to DeltaAI. To reproduce the
30% cohort from the audited September 30 run:

```bash
unset HOOD_MESH_RESUME_FROM
HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT=30 bash submit_mesh_impact_history_1704_fourier_k256.sh
```

The launcher pins Fourier encoding and uses the existing filtered k256 launcher:
two neighborhood layers, 256 neighbors, 256 mesh tokens, FiLM, 20 mm positional
scaling, 286 headform nodes held out of local attention, test designs 4/5 and
validation design 11. Without the threshold override it inherits the existing
10% filter. Existing Slurm options can still be supplied at the end.

Frequency settings can be overridden through the launcher:

```bash
HOOD_MESH_FOURIER_NUM_FREQUENCIES=8 \
HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ=10 \
HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ=800 \
HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT=30 \
bash submit_mesh_impact_history_1704_fourier_k256.sh
```

Frequencies are logarithmically spaced between the specified minimum and maximum;
the feature count is twice the number of frequencies. The equivalent Python
options are `--time-encoding fourier`, `--fourier-num-frequencies`,
`--fourier-min-frequency-hz`, and `--fourier-max-frequency-hz`. Both Slurm
preflight and training receive those options.

Choose the maximum below the sampled grid's Nyquist frequency. The 63-point
1704 grid has roughly 0.4004 ms spacing, giving approximately 1,249 Hz Nyquist;
640 Hz is below that limit. Training records the frequency bank, feature order,
physical-time reconstruction, scaler, and minimum sampled Nyquist in
`config.json` under `prediction_grid.fourier`. It logs a warning if a configured
bank reaches that limit on a more coarsely sampled grid.

New runs default to `time_encoding: fourier`. Existing configs that omit that
field reconstruct `legacy_five` so their original checkpoints replay strictly.
Resuming one of those runs continues the original encoding; changing the time
encoding during resume is rejected because the first linear layer has a
different shape. Start a fresh run for the Fourier comparison. The separate
change-attention experiment retains its historical time-feature default.

For the time-resolution comparison against the all-location October 4 run
`20261004_050747_3303364_1704`, use:

```bash
unset HOOD_MESH_RESUME_FROM
bash submit_mesh_impact_history_1704_fourier_k256_t250.sh
```

This fresh launcher selects every fourth source time and acceleration value:
1000 source points become **250 query/target points**, about **0.1001 ms** apart
instead of 63 points at 0.4004 ms. It keeps all 142 locations (HIC threshold 0),
the cluster-B test split, k256/two-layer neighborhood encoder, FiLM, and the
20–640 Hz Fourier bank. Defaults remain 100 epochs, batch size 8, learning rate
0.0003, weight decay 0.00001 and seed 42. The time limit is 24 hours because the
63-point reference run took about 11.5 hours; extra Slurm options can override it.

Job 3317059 failed before dataset loading because online W&B initialization
timed out while contacting `api.wandb.ai`. This experiment launcher now defaults
to **offline W&B tracking** when `WANDB_MODE` is unset. To ensure an old exported
online setting cannot affect the retry, submit a fresh job explicitly:

```bash
unset HOOD_MESH_RESUME_FROM
WANDB_MODE=offline bash submit_mesh_impact_history_1704_fourier_k256_t250.sh
```

Checkpoints, local training reports, and W&B records are still saved in the new
run directory. Offline records do not appear on the W&B website until synced.
From a machine with working W&B access, activate the training environment and run:

```bash
python -m wandb sync /path/to/new_run/wandb/offline-run-*
```

For live tracking after service access is restored, explicitly set
`WANDB_MODE=online`; other experiment launchers retain their existing default.
The failed directory contains no checkpoint, so it cannot be resumed.

The general option is `--time-subsample-stride N`, or
`HOOD_MESH_TIME_SUBSAMPLE_STRIDE=N` with the ordinary launchers. Strides 16, 4,
2 and 1 give 63, 250, 500 and 1000 points for these histories. Preflight checks
the same stride/cutoff as training, saved configs record the effective stride,
and resume restores it and rejects conflicting explicit overrides. Shared
preprocessing defaults remain stride 16. A denser grid adds supervision at more
source times; querying an existing checkpoint more densely does not add that
training supervision. Keep frequencies fixed for this first comparison so
sampling is the experimental variable. Compare both runs on the same physical
time grid, using full-source acceleration and HIC targets, as well as the saved
design-difference diagnostics.

The design-sensitivity JSON, per-location CSV, plot and W&B values are exported
after training as before. Compare `design_difference_skill`,
`difference_amplitude_ratio` and HIC15 sensitivity on the same held-out designs,
locations and sampled grid, alongside ordinary acceleration/HIC accuracy.

Tests cover physical phases on 63 times, invariance to time normalization,
feature derivatives, CUDA/CPU backward passes, saved buffers, training/export
reloads, old checkpoint/resume behavior, and launch-option propagation.
