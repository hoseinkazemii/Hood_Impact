# Hood Impact Code

Code for hood-impact acceleration-history modeling and Abaqus sample generation.

For the new irregular-mesh attention network mapping hood geometry and impact
location to acceleration history, see [Mesh impact history](README_mesh_impact_history.md).

The neighborhood experiment now tests **cluster B (designs 4 and 5)**, validates
on design **11**, and trains on the remaining nine designs. Start a fresh run with
`bash submit_mesh_impact_history_1704_local_sensitivity.sh`.

For attention centered on regions that change across **training designs**, see
[Change-region attention](README_mesh_change_attention.md). It adds dedicated
geometry-change tokens and within-cluster response-difference supervision, with
test loading deferred to a separate frozen-model evaluation command.

For the previous best acceleration Temporal DeepONet trained from scratch on
the same 1704-sample cluster-B split, see
[Temporal DeepONet baseline](README_temporal_deeponet_1704.md).
Submit with `bash submit_temporal_deeponet_1704.sh`.

Main areas:

- `abaqus_scripts/`: generate Abaqus impact decks, run Abaqus batches, extract and postprocess acceleration histories.
- `utils/`: visualization, metrics, and analysis helpers.
- top-level model scripts: DeepONet, temporal, PointNet++, TCN, and related training/inference code.
- `Data/`: lightweight metadata/scripts only; large generated datasets and solver outputs are ignored.

Large artifacts such as Abaqus `.odb` files, generated `.inp` decks, model checkpoints, W&B runs, mesh displacement CSVs, and raw/generated dataset outputs are excluded from Git.
