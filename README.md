# Hood Impact: mesh-change attention model

A surrogate for pedestrian head impact on a car hood. Given the hood geometry
(the inner panel as an unordered set of mesh nodes) and the impact location, the
network predicts the headform acceleration history. It is trained to learn how
**local design edits** change the response: every hood in the dataset is design
6 with a local edit (notches, deepenings, creases), and the edited nodes are
given to the network directly.

## Layout

| Path | What |
|---|---|
| `mesh_impact_history.py` | Backbone `MeshImpactHistoryNet`: node embedding, kNN neighborhood attention, FiLM impact-conditioned pooling into 256 latent tokens, latent self-attention, and a decoder of Fourier time queries with temporal self-attention. |
| `mesh_design_change.py` | `DesignChangeHistoryNet` (the model): the backbone plus a cross-attention readout of the edited ("change") nodes after every decoder block, and the change-node table. |
| `mesh_neighborhood.py` | kNN attention block and the persistent kNN graph cache. |
| `local_edits_data.py` | The local-edit dataset: manifest, meshes (design-6 inner panel + each edit's dz), sample numbering, change nodes, design-6 reference curves. |
| `local_edit_metrics.py` | Edit-effect metrics against design 6 (`local_edit_metrics.json`), baselines, plots, time-window RMSE. |
| `train_mesh_impact_history.py` | Training, evaluation, resume, and `HistoryPredictor` for inference from a saved run. |
| `preflight_local_edits.py` | Checks a training job (files, change tables, meshes vs. decks, split, CUDA probe) before the long data load. |
| `utils/utils.py` | Config, preprocessing/scalers, dataset and batching, `Trainer`, `Evaluator`. |
| `run_mesh_change_local_edits_delta.sbatch` | The training launcher on NCSA Delta (A40). `setup_mesh_impact_history_env_delta.sh` builds its venv. |
| `evaluate_local_edits_hic.py`, `analyze_local_edits_run.py`, `compare_local_edit_runs.py` | Post-training analysis of a run: HIC15, edit-effect diagnostics and figures, run-vs-run comparison. |
| `detect_change_nodes.py` (+ `verify_*`, `plot_change_node_detection*`) | Finds the changed nodes of an edited design without node IDs (remeshed designs), and its checks on the hood and B1-B4. |
| `abaqus_scripts/` | Local-edit dataset generation: DOE (`local_edit_doe.py`), decks (`make_local_edit_decks.py`), batch solve/extract/postprocess on Delta (`delta/hood_local_edits.sbatch`), QA. See `abaqus_scripts/delta/README_local_edits.md`. |
| `benchmarks/local_edit/` | Benchmark families B1-B4 (plate, elbow, heat sink, cantilever) for testing the method beyond the hood. |
| `hic15.py` | HIC15 of acceleration histories. |
| `tests/` | `python -m pytest tests` (the toy local-edit campaign is in `tests/toy_local_edits.py`). |

## Data

`Data/HoodImpact_LocalEdits/` (not in git): `manifest_local_edits.csv`,
`output_history_acc/` (SAE1000 curves), `design_changes/<design>_changed_nodes.csv`,
`inp_files/` (one design-6 deck is enough for training), `local_edit_designs.csv`,
`local_edit_doe.csv`. Each of the 72 edits has 20 simulated impacts. The loader
excludes the two rows labeled `far` for each edit. The 16 near locations and
two mirror twins give 1,296 runs across the three splits. Design 6 supplies
reference curves at all 142 locations.

A run of design Nn at location L has sample number `142 * n + L`; `--test-designs 54`
means N54. Design 6 is never an input: its curves are the reference the edit effects
are measured from. By default it is not trained on either; `--train-on-base-design true`
adds its 142 runs as ordinary training samples (design ID 0, no change nodes). It is
never held out, and the edit metrics and baselines still score edited designs only.

## Train

On Delta, from the repository root:

```bash
bash -l setup_mesh_impact_history_env_delta.sh       # once
sbatch run_mesh_change_local_edits_delta.sbatch      # test N54, validation N55
sbatch --export=ALL,HOOD_LOCAL_EDITS_CHANGE_FEATURES=xyz run_mesh_change_local_edits_delta.sbatch
sbatch --export=ALL,HOOD_MESH_RESUME_FROM=runs/mesh_impact_history/<run> run_mesh_change_local_edits_delta.sbatch
# Options of 2026-11-28 (off by default; the default is still run 24537698):
sbatch --export=ALL,HOOD_LOCAL_EDITS_TRAIN_ON_BASE=true run_mesh_change_local_edits_delta.sbatch
sbatch --export=ALL,HOOD_LOCAL_EDITS_CHANGE_LOCALITY=true run_mesh_change_local_edits_delta.sbatch
sbatch --export=ALL,HOOD_LOCAL_EDITS_TRAIN_ON_BASE=true,HOOD_LOCAL_EDITS_CHANGE_LOCALITY=true,HOOD_LOCAL_EDITS_SEED=43 run_mesh_change_local_edits_delta.sbatch
```

`HOOD_LOCAL_EDITS_CHANGE_LOCALITY=true` (`--change-locality true`): each change-readout
head lowers its attention to a moved node by `w * r^2`, `r` = the node's distance from
the impact point in normalized mesh units, `w` learned per head; an edit far from the
impact then adds nothing. It needs a load location input, so problems with one fixed
load (the B1-B4 benchmarks) keep it off, which is the plain readout. Training on the
base design carries over to B1-B4 unchanged: each family's `base` design is stored
with zero change nodes (see `tests/test_base_design_and_locality.py`).

Locally (the trainer's defaults are the launcher's model):

```bash
python preflight_local_edits.py --device cuda
python train_mesh_impact_history.py --device cuda --test-designs 54 --val-designs 55
```

A run directory holds `config.json`, `splits.json`, `scalers.joblib`,
`prediction_times.npy`, `design_change_nodes.npz`, the best and last checkpoints,
train/validation/test histories, `metrics.json`, `local_edit_metrics.json` (+ plots,
`local_edit_runs.csv`), `time_windows.json`, and the same for the last epoch in
`last_checkpoint/`.

## Analyze

```bash
python evaluate_local_edits_hic.py runs/mesh_impact_history/<run>
python analyze_local_edits_run.py runs/mesh_impact_history/<run>
python compare_local_edit_runs.py <run_a> <run_b>
```

Earlier models (Temporal DeepONet, PointNet++, the 1704-run Euro-NCAP set, design
difference oracles, ...) and their scripts were removed on 2026-11-27; they are in git
history at commit `65a938f`.
