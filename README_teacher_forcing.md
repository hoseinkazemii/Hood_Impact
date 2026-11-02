# Chunked teacher forcing

Branch `feature/teacher-forced-chunks`, from `feature/design-change-node-injection`. The option is
`teacher_forcing_chunks` on `MeshImpactHistoryNet` (`mesh_impact_history.py`), so `DesignChangeHistoryNet`
has it too. With 0, the default, the network, its weights and old checkpoints are unchanged.

## Why

On held-out cluster B, predictions track the first half of a history well and the second half poorly. The
idea tested here: predict the history in time chunks, each conditioned on the earlier chunks, and train with
the true earlier chunks (teacher forcing).

Measured before this run, on run `20261027_055436_3458374_1704`: the absolute test error does not grow over
time. After 5 ms it stays at 12-13 g, and in every window it equals the mean training curve at the same
impact location, which ignores the geometry. Given the **true** first 5 or 10 ms, simple predictors (the
nearest training design at that location, ridge regression on the early curve) did not beat that location
mean on the rest (13.9-14.9 g against 13.4-13.6 g); the model, with no history, reached 13.0 and 12.8 g.
This run asks whether a network trained with the history does better.

## What the model does

With `teacher_forcing_chunks = C`, a history of T times is split into C contiguous chunks of
ceil(T / C) times (the last can be shorter). For the 63-point grid and C = 5: 13, 13, 13, 13 and 11
points, 0-4.8, 5.2-10.0, 10.4-15.2, 15.6-20.4 and 20.8-24.8 ms.

- Every time query adds an embedding of the acceleration one chunk earlier, at the same place in that chunk
  (0 in the first chunk, which has no earlier chunk).
- The temporal self-attention is block-causal: a time reads only the times of its own chunk and earlier
  chunks. The mesh and change-node cross-attentions are per time, as before.

So a chunk's output depends only on the accelerations of earlier chunks:

| | input to chunk k | passes |
|---|---|---|
| training (`forward(..., history=target)`) | true accelerations of chunks < k | 1 |
| validation, test, `HistoryPredictor` (no `history`) | the model's own predictions of chunks < k (rollout) | C decoder passes, one mesh encoding |

The decoder needs `decoder="temporal"`, and every sample in a batch needs the same number of times.

## Training and outputs

- Training loss: acceleration MSE with teacher forcing.
- Checkpoint selection: the validation loss of the **rollout**, since that is how the model is used. W&B
  also logs `val/teacher_forced_loss` and `val/teacher_forced_mae_g`: the same model fed the true earlier
  chunks. The gap between the two is the error that rollout feeds back.
- Test, as usual (`metrics.json`, `test_acceleration_histories.csv`, HIC and difference reports): rollout.
- `teacher_forced/metrics.json` and `teacher_forced/test_acceleration_histories.csv`: the test set fed the
  true earlier chunks. This is the best case, as if the first chunks were measured.
- `time_windows.json` (every run now writes it, in 5 windows without teacher forcing): test RMSE per chunk
  for the rollout, the teacher-forced pass and the location mean, plus the test signal spread. W&B keys
  `time_windows/<name>/window_<k>`.
- The final epoch is scored the same way in `last_checkpoint/`.

## The run

```bash
bash submit_mesh_impact_history_1704_design_change_teacher_forcing.sh
```

5 chunks, 100 epochs, and otherwise the no-gate design-change model of the long runs: inner panel, XYZ change
tokens, no change self-attention layers, k256 FiLM neighborhoods, Fourier time 20-640 Hz, stride 16 (63
points), test 4,5 | validation 11, batch 8, lr 3e-4, dropout 0.1, weight decay 1e-5, seed 42. Time limit
7 h. On a laptop RTX 5050, a full-size teacher-forced training step took the same time as a plain one (5.56
vs 5.54 s) and a rollout evaluation step 0.8% longer; the extra validation pass adds about 4% per epoch, so
about 5.5 h on one GH200.

## How to read it

Baseline, run 3458374 (same split and grid, no history), RMSE in g per chunk:

| | 0-4.8 ms | 5.2-10.0 | 10.4-15.2 | 15.6-20.4 | 20.8-24.8 |
|---|---|---|---|---|---|
| model | 22.4 | 13.7 | 13.3 | 13.1 | 11.8 |
| location mean of the training designs | 22.9 | 14.5 | 13.3 | 13.7 | 12.9 |
| test signal std | 47.6 | 26.6 | 21.5 | 26.4 | 24.4 |

- `teacher_forced_rmse_g` in chunks 2-5 not below the model row: even the true history does not help, and
  the idea is ruled out.
- Below it, but `model_rmse_g` (rollout) not below: the history helps, but rollout errors cancel the gain.
- Both below: teacher forcing helps in use. Then check HIC and the design-difference reports as usual.

3458374 had the displacement gate; the closest gate-free reference is the 800-epoch run
(`submit_mesh_impact_history_1704_design_change_long.sh`), whose cosine schedule differs. A plain
100-epoch, gate-free run is the clean A/B if the result is close.
