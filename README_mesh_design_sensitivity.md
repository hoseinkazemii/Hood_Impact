The mesh-impact training pipeline now measures whether predictions reproduce
differences between designs at the **same impact location**. Ordinary acceleration
R2 and HIC accuracy remain useful, but can hide a model that mostly learns the
location-dependent average response.

After the best validation checkpoint is evaluated, each new
`train_mesh_impact_history.py` run writes:

- `test_design_sensitivity.json`: aggregate acceleration and HIC15 diagnostics.
- `test_design_sensitivity_by_location.csv`: the same diagnostics at each matched location.
- `test_design_sensitivity.png`: true/predicted spread and skill by location.

The values are also logged to W&B under `test/design_sensitivity/`, with HIC15
under `test/design_sensitivity/hic15/`. Existing `metrics.json` retains its
ordinary acceleration metrics. The separate change-attention pipeline exports
the same files during its explicit frozen test evaluation.

**Read `design_difference_skill` first, alongside `difference_amplitude_ratio`.**

For each impact location, take every unordered pair of designs, and subtract
their acceleration histories at corresponding times. Write these true and
predicted differences as `delta_true` and `delta_pred`. Across all such pairs
and sampled times:

```text
design_difference_skill = 1 - sum((delta_pred - delta_true)^2) / sum(delta_true^2)
difference_amplitude_ratio = sqrt(sum(delta_pred^2) / sum(delta_true^2))
difference_alignment_cosine = sum(delta_pred * delta_true) / (norm(delta_pred) * norm(delta_true))
difference_gain = sum(delta_pred * delta_true) / sum(delta_true^2)
```

| Metric | Interpretation |
|---|---|
| Skill = 1 | All design differences are correct. |
| Skill = 0 | No improvement over predicting zero differences between designs. An exactly collapsed predictor scores zero. |
| Skill < 0 | Predicted differences are worse than predicting zero differences. |
| Amplitude ratio = 0 | Predictions are identical across designs. |
| Amplitude ratio = 1 | The overall magnitude of differences matches; their direction/time dependence can still be wrong. |
| Alignment = 1 / 0 / -1 | Correct direction / orthogonal differences / reversed direction, considering the complete difference vectors. |

Skill zero does not uniquely identify collapse, so always inspect amplitude too.
For example, `delta_pred = 0.1 * delta_true` gives amplitude 0.1, alignment 1,
and skill 0.19. Reversing the true differences gives amplitude 1 but skill -3.
The identity `skill = 2 * gain - amplitude_ratio^2` is another interpretation.

A shared prediction error at a given location/time cancels from acceleration
differences. Consequently, skill 1 does **not** imply accurate absolute curves;
retain ordinary acceleration/HIC metrics too. The HIC15 version first integrates
each individual history and then takes HIC differences; HIC is nonlinear, so
a shared acceleration bias need not cancel from HIC differences.

The legacy spread fields are RMS population standard deviations across designs,
averaged over matched locations/times. For exactly two designs, spread is half
the RMS design-pair difference. Pairwise amplitude and spread ratios coincide
when all locations have the same number of designs; otherwise their weightings
differ. Scores weight pairs/time samples equally. With common counts and grids,
the aggregate skill emphasizes locations with larger true differences through
its denominator, rather than averaging unstable per-location ratios.

Locations with identical ground truth remain in the report: any invented
differences still increase aggregate error. Normalized metrics at a location
with zero true difference are `null`; alignment is also `null` when predicted
differences are zero. Single-design locations cannot measure design sensitivity
and are counted as unmatched. For the current validation split, design 11 alone
cannot supply a validation design-difference score.

HIC15 uses the saved **sampled** acceleration histories in g, times in seconds,
and a maximum 15 ms window, using `hic15.batched_hic`. It does not interpolate or
read full-resolution simulation targets. Thus its reference agrees with the
"prediction vs sampled simulation" comparison from `evaluate_mesh_impact_hic.py`.
Full-resolution engineering HIC remains available from that separate evaluator.

**Backfill an existing run without loading its model or raw simulation files:**

```powershell
.\.venv\Scripts\python.exe evaluate_design_sensitivity.py runs/mesh_impact_history/20260930_204155_3283115_1704
```

Pass multiple run directories to evaluate several runs. The script reads
`config.json` and `test_acceleration_histories.csv`. It validates finite values,
unique design/location IDs and strictly increasing, exactly matched time grids.
CSV row order does not matter. Training also verifies matching physical impact
XY; older CSVs omit XY, so their reports explicitly set `impact_xy_checked: false`
and rely on the dataset's run-ID-to-location mapping.

For fair comparisons keep design IDs, location IDs and output time grid fixed.
The 0/10/20/30/40% HIC filters use different location cohorts. To inspect a shared
subset explicitly, use a distinct output prefix to preserve the full report:

```powershell
.\.venv\Scripts\python.exe evaluate_design_sensitivity.py RUN_A RUN_B --locations 9 115 137 142 --prefix common_probe
```

The script rejects requested locations absent from a run. Backfilled reports
include the source CSV SHA256 and evaluated run/location IDs. Backfilling writes
local artifacts; it does not modify historical online W&B runs.

**Trace a frozen FiLM checkpoint through its geometry pathway:**

```powershell
.\.venv\Scripts\python.exe analyze_mesh_design_pathways.py runs/mesh_impact_history/20260930_204155_3283115_1704
```

This optional, more expensive audit loads the checkpoint and raw inputs, and
probes four selected locations across eight designs by default. It saves curves,
layer activation differences, attention mass, and inference interventions in
`design_sensitivity_audit/`. Use `--locations`, `--designs`, `--data-root`,
`--device` or `--output-dir` to customize it. Default device is CUDA. Structural
neighbor processing uses smaller inference chunks to fit local GPU memory.
The checkpoint hash and saved-prediction/decoder-reconstruction tolerances are
recorded. Interventions hold weights fixed and can move activations outside the
training distribution; they are not substitutes for retraining ablations.

The October 1 audit and comparison of the five supplied FiLM runs are saved in
[the audit report](runs/mesh_impact_history/20260930_204155_3283115_1704/design_sensitivity_audit/report.md).
