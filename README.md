# Hood Impact Code

Code for hood-impact acceleration-history modeling and Abaqus sample generation.

For the new irregular-mesh attention network mapping hood geometry and impact
location to acceleration history, see [Mesh impact history](README_mesh_impact_history.md).

Main areas:

- `abaqus_scripts/`: generate Abaqus impact decks, run Abaqus batches, extract and postprocess acceleration histories.
- `utils/`: visualization, metrics, and analysis helpers.
- top-level model scripts: DeepONet, temporal, PointNet++, TCN, and related training/inference code.
- `Data/`: lightweight metadata/scripts only; large generated datasets and solver outputs are ignored.

Large artifacts such as Abaqus `.odb` files, generated `.inp` decks, model checkpoints, W&B runs, mesh displacement CSVs, and raw/generated dataset outputs are excluded from Git.
