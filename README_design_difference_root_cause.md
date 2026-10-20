# Design-difference root cause

Branch `feature/design-difference-root-cause`. The question: why does no
surrogate predict the response difference between near-clone hoods at the same
impact location (designs 4 vs 5, and the other near-clone pairs)? Every model
so far has a design-difference R2 of about 0, even on its own training designs.

Three approaches, run in this order. Each one decides whether the next is worth
running.

| # | Approach | Answers | Where it runs |
|---|---|---|---|
| 1 | Solver audit (Abaqus) | Is the 4-5 difference geometry, solver set-up, or chaos? | Delta CPU, ~19 h |
| 2 | Oracle ladder | Where does the signal die: data, decoder, features, or encoder? | DeltaAI GPU (or locally), ~1 h |
| 3 | Edit operator | Can a difference-first model learn what is learnable? | DeltaAI GPU (or locally), ~2-3 h |

## Shared conventions (`design_difference_data.py`)

- **Pairs.** Pairs are `(a, b)` with `a < b`, and the difference is `a_b(t) - a_a(t)`.
  - Train: the 12 near-clone pairs inside {0,1,2,3} and {6,7,8,9}.
  - Validation: (10, 11).
  - Test: (4, 5).
- **Metric.** The approved difference R2, `1 - sum (p - d)^2 / sum (d - mean d)^2`, pooled over time, locations and pairs on the 63-point model grid. A collapsed predictor scores 0.
- **Edit.** Inside a mesh family, node IDs and connectivity are identical. All 12 decks share sections, materials, contact and mass scaling; within a family only the node coordinates differ (audited 2026-10-19). So the edit is the exact field `dx = x_b - x_a`, about 50-220 moved nodes.
- **Distance bands.** These use the XY distance from the impact point to the nearest node moved by more than 0.5 mm: near < 100 mm <= mid < 250 mm <= far.
- **Cache.** Histories and per-design geometry are cached in `Data/HoodImpact_1704_EuroNCAP/design_difference_cache/` on first use.

## 1. Solver audit

Two non-geometric mechanisms are suspected:

- **Time step.** Variable mass scaling with automatic incrementation lets an edit shrink the global time step. Design 5 takes 40.3k increments, design 4 takes 26.6k.
- **Start state.** Most runs start with the headform already in contact.

`abaqus_scripts/make_solver_factorial_decks.py` writes 105 decks for designs 4
and 5 at 15 locations: 8 near, 3 mid and 4 far from the edit, one per mirror
pair. Only the `*DYNAMIC` card and/or the headform node lines differ from the
delivered deck.

| cell | change |
|---|---|
| `fixed_dt_gap_start` | `*DYNAMIC, EXPLICIT, DIRECT USER CONTROL` at 4.5e-7 s for both designs + headform 1.0 mm clear of contact. The "clean" difference |
| `fixed_dt_gap_start_jitter` | design 4 only: clean + 1e-3 mm hood jitter. The chaos floor under the clean settings |
| `fixed_dt` | fixed time step only |
| `gap_start` | gap start only |

The native/native cell is the delivered dataset. The existing noise-floor decks
(`make_noise_floor_decks.py`, 30 runs) give the chaos floor under native settings.

Run:

```bash
# Windows (decks already generated in Data/HoodImpact_SolverFactorial, ~330 MB)
python abaqus_scripts/make_solver_factorial_decks.py --dry-run   # placement report
# Delta, from the Code directory (generates decks if absent; needs Data/HoodImpact_1704_EuroNCAP)
sbatch abaqus_scripts/delta/hood_design_difference_audit.sbatch    # factorial, then noise floor; 12 lanes
# back on Windows, with both output_history_acc folders copied back
python abaqus_scripts/analyze_solver_factorial.py
python abaqus_scripts/analyze_solver_factorial.py --stride 16
python abaqus_scripts/analyze_noise_floor.py
```

Cost: ~222 core-hours, ~19 h wall at 12 lanes. Both jobs share one FlexNet pool
of 13 lanes, so do not run `hood_noise_floor.sbatch` alongside it. The job is
resume-safe.

Reading `analyze_solver_factorial.py`:

- **Artifact.** A large share of the native energy removed by `fixed_dt_gap_start` in the far band means the far-field 4-5 difference is a solver artifact.
- **Genuine signal.** `clean_over_clean_jitter` well above 1 in the near band means there is real geometry signal near the edit.
- **Chaos.** `clean_over_clean_jitter` near 1 everywhere means the near-clone difference is chaos, and the modelling goal has to change.
- **Sanity check.** `dt_effect_d4/d5` shows how much the time step alone moves one design. Runs with non-finite or |a| > 1000 g are flagged as unstable.

## 2. Oracle ladder

| script | what it does | how to read it |
|---|---|---|
| `oracle_interpolation_ceiling.py` | Interpolates a pair's own true difference curves from the 71 even checkerboard locations to the 71 odd ones (GP, IDW, nearest), plus a correlogram and a magnitude-only ceiling. | This is the ceiling for any model that has to interpolate across impact locations, even one that knows the design. |
| `oracle_design_code.py` | The surrogate's decoder (HistoryDecoderBlock + Fourier time) with an impact token and a learned code: `none`, `cluster` (A/B/C/D) or `design` (12-way). All designs train on the even locations. | Design code at seen locations is decoder capacity; at unseen locations it is the learned interpolation ceiling. |
| `oracle_delta_features.py` | Handcrafted edit descriptors around the impact point (sums of \|dx\|, dz, normal displacement, panel-gap change at 50-400 mm), distance to the edit, the time-step ratio and the baseline curve. Ridge and extra-trees models, antisymmetrized. | Shows whether the edit carries transferable signal. "Unseen edit" leaves one training design out. |
| `probe_frozen_latents.py` | Hooks the frozen no-shortcut run (`20261016_151923`) at pooling, at memory and at the decoder output. A ridge probe maps `z_b - z_a` to `a_b - a_a`. | Probe >> model means the decoder ignores what it has. Probe ~ 0 at pooling means pooling destroyed it. |

```bash
mkdir -p runs/slurm && sbatch run_design_difference_ladder.sbatch     # DeltaAI, all four
# or locally, one at a time (wandb needs protobuf<6 here; or pass --wandb-mode disabled)
python oracle_interpolation_ceiling.py
python oracle_design_code.py --wandb-mode disabled
python oracle_delta_features.py
python probe_frozen_latents.py
```

## 3. Difference-first edit operator

`edit_operator.py` and `train_edit_operator.py` predict `a_b - a_a` directly
from the edit. There is no 38k-node pooling: only the edited nodes enter, as
tokens with position relative to the impact point, panel, normal, gap and the
edit `(dx, dgap)`.

- **`linear`.** A learned sensitivity kernel is contracted with the edit and expanded on a temporal basis initialized from PCA of the training differences. It is linear in the edit, so a zero edit predicts exactly zero, swapping a and b flips the sign, and differences compose to first order.
- **`attention`.** The impact query cross-attends to the edit tokens. It is antisymmetrized as `(F(e) - F(-e)) / 2`.
- **`nuisance=dt`.** Adds `log(dt_b/dt_a) * N(context)`, so the time-step artifact has its own term. Metrics are reported for the full prediction and for the geometric term alone.

Training uses the 12 train pairs in both directions at all 142 locations, with
early stopping on (10, 11). Each run writes:

- `difference_r2.csv`: validation-selected weights, i.e. transfer;
- `difference_r2_last_epoch.csv`: last-epoch weights, i.e. capacity on seen edits;
- `predictions.npz`.

```bash
mkdir -p runs/slurm && sbatch run_edit_operator.sbatch    # 6 variants x 3 seeds, W&B online
python train_edit_operator.py --wandb-mode disabled                      # local, main variant
python train_edit_operator.py --operator none --nuisance dt --wandb-mode disabled   # time step only
```

## First local results (2026-10-20, native data, before the solver audit)

Outputs are in `runs/design_difference/local_20261020/`. All numbers are difference R2: 0 is the collapsed predictor, below 0 is worse than predicting no difference.

| probe of the problem | seen (fit) | unseen | 4 vs 5 |
|---|---|---|---|
| Existing no-shortcut surrogate (`20261016`), its own output | 0.002 | -- | -0.001 |
| GP interpolation of the pair's own true differences, unseen locations | -- | 0.02-0.04 | 0.02 |
| ... same for the difference magnitude (log RMS) only | -- | 0.77 | 0.70 |
| Design-code decoder, last epoch, seen / unseen locations | 0.92 | -0.70 | 0.92 / -0.63 |
| Edit features + extra trees, in-sample / unseen edit | 0.52 | 0.04 | 0.05 |
| ... with the baseline design's curve added | 0.60 | 0.09 | 0.09 |
| Linear probe of `z_b - z_a` after pooling / after the latent blocks | 0.20 / 0.02 | 0.02 / 0.00 | 0.01 / 0.00 |
| Edit operator (linear + dt), last epoch | 0.72 | -1.14 (val) | -0.50 |
| Edit operator, checkpoint selected on (10, 11) | 0.02 | -0.02 (val) | -0.003 |
| Edit operator (attention + dt), last epoch | 0.72 | -0.07 (val) | -0.02 |
| Time-step ratio only (edit operator `none + dt`), last epoch | 0.10 | 0.01 (val) | -0.09 |

What this says:

1. **Capacity is not the problem.**
   - The surrogate's decoder reproduces near-clone differences at R2 0.92 once a design code makes the designs distinguishable.
   - A difference-first model fits seen edits at 0.72.
   - The existing surrogate's ~0 in-sample comes from a design signal that is too weak in its representation: near-clone tokens differ by 0.3-0.6% after pooling, against 20% across families. The latent blocks then erase most of the little that survives pooling (probe 0.20 -> 0.02).
2. **Nothing transfers.**
   - The waveform of a near-clone difference is essentially uncorrelated between locations 100 mm apart: the curve correlation is 0.08, and even a GP on the true differences only reaches 0.02.
   - Its magnitude is smooth across locations (0.70-0.89).
   - No model fitted to some edits predicts another edit.
   - This is the signature of a phase-sensitive, chaotic-like divergence, or of the time-step artifact. On this data it is not a deterministic function that a geometry encoder could learn.
3. **The solver audit decides the next step.**
   - If the clean runs (fixed dt + gap start) leave a near-edit difference well above the clean jitter floor, regenerate the near-clone data under those settings and retrain approach 3 there.
   - Otherwise, restate the target: predict the difference envelope or HIC distribution (where the magnitude is coherent), not the waveform.

## Tests

```bash
python -m pytest tests/test_design_difference.py -q
```

The tests cover:

- the metric;
- the deck rewrites and gap-start placement;
- the delta features' direction handling;
- the edit operator's exact antisymmetry, zero-edit and padding invariance.
