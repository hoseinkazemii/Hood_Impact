The neighborhood encoder updates **every structural node** before mesh-to-token
pooling. The first 286 rigid-headform rows bypass these blocks and retain their
node embeddings for pooling. The current 1704 meshes contain:

| Design IDs | Total nodes | Structural nodes | k256 neighbor table |
|---|---:|---:|---|
| 0–5 | 38,977 | 38,691 | 38,691 × 256 |
| 6–11 | 40,057 | 39,771 | 39,771 × 256 |

`MeshImpactHistoryNet._encode_neighborhoods` in `mesh_impact_history.py` obtains
the table, applies both neighborhood blocks, then reattaches the headform rows.
`geometric_neighbors` in `mesh_neighborhood.py` uses SciPy `cKDTree.query` for
exact Euclidean XYZ neighbors. It does not construct an N × N distance matrix.
The count is 256 entries total; a center with distinct XYZ includes itself,
leaving 255 other nodes. Coincident-node ties follow the existing tree ordering.

For each center i, the attention block reads Q_i and K_j/V_j for its selected
neighbors j. It forms 256 scores per head for that center, rather than a
256 × 256 attention matrix for each neighborhood. At width 128 and four heads:

```text
Structural node embeddings                   N × 128
Q, K, V after the shared linear projection    N × 4 × 32 each
Neighbor indices                             N × 256
Gathered K/V for a chunk of C centers         C × 256 × 4 × 32
Dot-product scores + relative XYZ bias       C × 256 × 4
Softmax across 256 neighbors                 C × 256 × 4
Weighted value/position sum                  C × 4 × 32 → C × 128
Output projection, residual, feedforward     C × 128
Concatenate all chunks                       N × 128
```

The relative-position bias and value use `(XYZ_j - XYZ_i) / 20 mm`. The 20 mm
constant scales those features; it is not a radius for selecting neighbors.
Changing it to 1 mm changes attention but does not require a new neighbor table.
The second block reads updated features on the same graph, allowing information
to travel two neighbor hops. Both blocks execute on every forward pass and train
normally. No features, attention weights, or predictions are saved in this cache.

Previously, `NeighborGraphCache` kept each graph only in memory, normally
building it once per structural geometry in a training process, not once per
batch or epoch. It now supports shared disk files. The training pipeline prepares
the distinct raw geometries before training and reuses them in subsequent runs.
Preflight and `HistoryPredictor` use the same files.

For the 1704 dataset, the default directory is:

```text
Data/HoodImpact_1704_EuroNCAP/neighbor_graphs/
```

There are 12 k256 `.npz` files, totaling about **482 MB / 460 MiB** locally, plus
`manifest_k256.json`, which maps zero-based design IDs to files. Stored indices
are int32; attention receives torch.int64 on its device. Indices refer to rows
of the structural slice. Add 286 to obtain rows of the complete input mesh;
these are row indices, not Abaqus node IDs. GPU in-memory index tables use about
twice the disk-array space, as before.

The persistent key contains exact raw physical XYZ bytes, dtype, shape, row
order, requested k, and an algorithm version. Training maps that raw graph to
the coordinates produced by its current scaler. Thus changing a train/validation
split, time encoding, FiLM, network weights, or normalization statistics does
not rebuild an unchanged geometry. Changing geometry, row order, or k selects
a different file. Changing the excluded headform count also changes the raw
structural slice. Changed/deformed point clouds cannot reuse a fixed reference
graph automatically. Shapes, bounds, metadata and checksums are validated on
load; incomplete or corrupted files are rebuilt. Writes replace files atomically.

Precompute on the cluster after syncing the code, using the training environment:

```bash
module load python/miniforge3_pytorch/2.10.0
source .venv-mesh-history/bin/activate
python precompute_mesh_neighbors_1704.py
```

The command checks that the structural slice agrees at the first and last
impact of each design. Training additionally keys every loaded sample by its
actual raw structural geometry, so it never assumes that a design label alone
guarantees an identical graph. Repeating the command loads the saved files.
Manual precomputation is optional: training builds and saves any missing graphs.
The existing submission scripts need no new option for the default directory.

For another k or cache directory:

```bash
.venv-mesh-history/bin/python precompute_mesh_neighbors_1704.py --k 128 --cache-dir /path/to/shared/graphs
HOOD_MESH_NEIGHBOR_CACHE_DIR=/path/to/shared/graphs bash submit_mesh_impact_history_1704_fourier_k256.sh
```

The k256 launcher still uses k256; the first command only precomputes a separate
k128 graph bank. Use matching training settings when testing another k.
`--neighbor-cache-dir` is accepted by Python training and preflight. The Slurm
environment option reaches both fresh and resumed jobs. Inference can override
the directory with `HistoryPredictor.from_run(run_dir, neighbor_cache_dir=...)`.
`config.json.neighbor_graph_cache` records the directory, unique geometry count,
new searches, disk loads and recovered invalid files. Graphs remain outside
model checkpoints, so previous checkpoints still load with strict state matching.

These files are ignored by Git. Generate them once on each machine, or copy the
directory alongside the dataset. This saves graph construction at startup;
the existing cache already avoided repeated searches within a run, and learned
neighborhood attention still accounts for its usual training cost.

To visualize the exact cached lists at 100 scattered centers:

```bash
python visualize_mesh_neighborhoods_1704.py --design 0 --samples 100 --k 256
```

Results are written to `figures/mesh_neighborhoods/design_00_k256/` by default.
`neighborhood_viewer.html` contains two rotatable 3D views, a sample selector,
panel background controls, a 100-image gallery and the selected node IDs/XYZ.
It runs offline without external JavaScript libraries. Keep the output folder
together for the static image gallery and downloads; the interactive geometry
and neighbor tables are embedded in the HTML itself.

The exporter also saves a common-scale 100-neighborhood overview, individual
four-view PNGs, a 100-page PDF atlas, `neighborhood_metrics.csv`, and
`selected_neighbors.csv` containing all 25,600 entries including each center.
Centers are chosen by deterministic farthest-point sampling in physical XYZ,
starting nearest the structural centroid. They are mesh nodes, not impact
locations. Red marks the center, blue marks selected outer-panel nodes, orange
marks selected inner-panel nodes, and purple marks other selected structure.
Panel labels come from the original Abaqus element sets; graph selection still
uses the existing XYZ kNN rule. All static local views have equal physical XYZ
scale and a common extent. The dashed circle is the projection of the bounding
sphere whose radius is the farthest selected distance; a small 20 mm reference
circle is included separately. Colors identify candidates, not attention weights.

On design 0, the saved 100-center export has radii from 56.3 to 116.0 mm,
median 77.6 mm; 96 neighborhoods contain both outer- and inner-panel nodes.
These are statistics of this spatially spread sample, not all nodes or all
designs. The export was checked against all cached IDs/coordinates/distances,
and the browser controls and both WebGL views were checked in a headless browser.
