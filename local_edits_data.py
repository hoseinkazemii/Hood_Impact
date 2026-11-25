"""Local-edit dataset (Data/HoodImpact_LocalEdits): design 6 and 72 local edits of it.

Every edited hood is design 6 (HoodImpact_31.inp of the 60-design set) with some
inner-panel nodes moved in z. design_changes/<design>_changed_nodes.csv lists
them: label, x, y, z_design6, dz (mm). Each edit N1..N72 was solved at 20
impact locations (16 near the edit, 2 mirror twins, 2 far), design 6 ("D6") at
all 142. Every run used the same Abaqus time step, a 1.0 mm headform start gap
and 8 cores.

Meshes. Training reads one design-6 deck only: the mesh of design Nn is the
inner panel of that deck with dz added at the listed nodes, which equals the
design's own decks (preflight_local_edits.py checks this). Only the inner panel
is supported, as the full node block holds the headform at a location-dependent
position.

Sample numbers. The trainer identifies a run's design as (number - 1) // 142,
as on the 1704 set, so a run of design Nn at location L gets the sample number
142 * n + L, with design 6 as design 0. A slot then means the same impact
location in every design, and ``--test-designs 54`` means N54. The manifest run
number (the Abaqus job and curve file number) is kept alongside it.

Design 6 is left out of training, validation, test and the scalers. Its true
curves are read only as the reference the edit effects are measured from
(local_edit_metrics.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DESIGN = "D6"
LOCATIONS = 142
MANIFEST_FILE = "manifest_local_edits.csv"
CHANGES_DIR = "design_changes"
# The base deck's z must match the change tables' z_design6 to this tolerance.
BASE_MATCH_TOLERANCE_MM = 1e-3


def design_number(name: str) -> int:
    """'N54' -> 54; the base 'D6' -> 0."""
    if name == BASE_DESIGN:
        return 0
    if not (isinstance(name, str) and name.startswith("N") and name[1:].isdigit() and int(name[1:]) > 0):
        raise ValueError(f"Unknown local-edit design name {name!r}")
    return int(name[1:])


def design_name(number: int) -> str:
    return BASE_DESIGN if int(number) == 0 else f"N{int(number)}"


def sample_number(design: int, location: int) -> int:
    if not 1 <= int(location) <= LOCATIONS:
        raise ValueError(f"Impact location must be in 1..{LOCATIONS}; got {location}")
    return LOCATIONS * int(design) + int(location)


def sample_design(sample: int) -> int:
    return (int(sample) - 1) // LOCATIONS


def sample_location(sample: int) -> int:
    return (int(sample) - 1) % LOCATIONS + 1


def load_manifest(path) -> pd.DataFrame:
    """The run manifest with ``design_id`` and ``sample`` added, sorted by sample."""
    manifest = pd.read_csv(path)
    missing = {"run", "design", "location", "role", "x_mm", "y_mm"} - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    manifest["design_id"] = manifest["design"].map(design_number)
    manifest["sample"] = [sample_number(d, loc) for d, loc in zip(manifest["design_id"], manifest["location"])]
    if manifest["sample"].duplicated().any() or manifest["run"].duplicated().any():
        raise ValueError(f"{path} repeats a (design, location) pair or a run number")
    return manifest.sort_values("sample").reset_index(drop=True)


def base_deck_path(inp_dir, manifest: pd.DataFrame) -> Path:
    """The deck of the first design-6 run; every design-6 deck has the same hood."""
    base_runs = manifest.loc[manifest["design"] == BASE_DESIGN, "run"]
    if base_runs.empty:
        raise ValueError("The manifest has no design-6 run to take the base hood from")
    return Path(inp_dir) / f"HoodImpact_{int(base_runs.min())}.inp"


def base_inner_panel(deck) -> tuple[np.ndarray, np.ndarray]:
    """(node IDs, float32 XYZ) of the inner panel, in the row order training uses."""
    from utils.utils import inner_panel_node_ids, read_inp_nodes

    ids, coords = read_inp_nodes(deck)
    keep = np.isin(ids, inner_panel_node_ids(deck))
    return ids[keep], coords[keep]


def read_changes(changes_dir, design: str) -> pd.DataFrame:
    path = Path(changes_dir) / f"{design}_changed_nodes.csv"
    table = pd.read_csv(path)
    if list(table.columns[:5]) != ["label", "x", "y", "z_design6", "dz"]:
        raise ValueError(f"{path}: expected columns label, x, y, z_design6, dz")
    return table


class LocalEditGeometry:
    """Design-6 inner panel plus each design's changed nodes; one mesh per design."""

    def __init__(self, base_ids, base_xyz, changes_dir):
        self.base_ids = np.asarray(base_ids, dtype=np.int64)
        self.base_xyz = np.asarray(base_xyz, dtype=np.float32)
        self.changes_dir = Path(changes_dir)
        self.row_of = {int(node): row for row, node in enumerate(self.base_ids)}
        self._meshes, self._changes = {}, {}

    @classmethod
    def from_manifest(cls, manifest_path, inp_dir):
        manifest = load_manifest(manifest_path)
        deck = base_deck_path(inp_dir, manifest)
        return cls(*base_inner_panel(deck), Path(manifest_path).parent / CHANGES_DIR), manifest, deck

    def changes(self, design: int) -> dict:
        """Rows, node IDs and STATIC_CHANNELS rows of design ``design`` (an int)."""
        if design not in self._changes:
            if design == 0:
                rows, dz = np.zeros(0, np.int64), np.zeros(0, np.float32)
            else:
                table = read_changes(self.changes_dir, design_name(design))
                outside = [int(n) for n in table["label"] if int(n) not in self.row_of]
                if outside:
                    raise ValueError(f"{design_name(design)}: changed nodes {outside[:5]} are not in the inner panel")
                rows = np.array([self.row_of[int(n)] for n in table["label"]], dtype=np.int64)
                listed = table[["x", "y", "z_design6"]].to_numpy(np.float64)
                error = float(np.abs(self.base_xyz[rows].astype(np.float64) - listed).max()) if len(rows) else 0.0
                if error > BASE_MATCH_TOLERANCE_MM:
                    raise ValueError(f"{design_name(design)}: the base deck differs from z_design6 by {error:.3g} mm; "
                                     "it is not design 6")
                dz = table["dz"].to_numpy(np.float32)
                if np.abs(dz).min(initial=1.0) <= 0:
                    raise ValueError(f"{design_name(design)}: a listed node has dz = 0")
            base = self.base_xyz[rows]
            # base_x, base_y, base_z, dx, dy, dz, base_gap, gap_change, outer
            # (design_change_nodes.STATIC_CHANNELS). The edits move z only and lie on
            # the inner panel; gap channels are not computed, so only the "xyz"
            # change-token features are valid on this dataset.
            static = np.zeros((len(rows), 9), np.float32)
            static[:, :3], static[:, 5] = base, dz
            self._changes[design] = {"cluster": [0, design], "base": 0, "node_ids": self.base_ids[rows],
                                     "static": static, "rows": rows}
        return self._changes[design]

    def mesh(self, design: int) -> np.ndarray:
        """(N, 3) float32 inner-panel mesh of design ``design``, shared by all its runs."""
        if design not in self._meshes:
            change = self.changes(design)
            mesh = self.base_xyz.copy()
            mesh[change["rows"], 2] += change["static"][:, 5]
            mesh.setflags(write=False)
            self._meshes[design] = mesh
        return self._meshes[design]


def load_design_changes(manifest_path, inp_dir, designs, mesh_nodes="inner_panel"):
    """Change nodes of ``designs`` (ints) as design_change_nodes.find_design_changes returns them."""
    if mesh_nodes != "inner_panel":
        raise ValueError("The local-edit dataset supports only --mesh-nodes inner_panel")
    geometry, _, deck = LocalEditGeometry.from_manifest(manifest_path, inp_dir)
    changes = {int(design): geometry.changes(int(design)) for design in designs}
    report = {
        "rule": {"change_nodes": "the inner-panel nodes listed in design_changes/<design>_changed_nodes.csv "
                                 "(every node the edit moved, z only)",
                 "displacement_from": f"design 6 ({BASE_DESIGN}), from {deck.name}"},
        "static_channels": ["base_x", "base_y", "base_z", "dx", "dy", "dz", "base_gap", "gap_change", "outer"],
        "clusters": [{"designs": [design_name(d)], "base": BASE_DESIGN, "change_nodes": len(c["rows"]),
                      "change_nodes_in_mesh": len(c["rows"])} for d, c in sorted(changes.items())],
        "designs": {design_name(d): {"change_nodes": len(c["rows"]),
                                     "max_displacement_mm": float(np.abs(c["static"][:, 5]).max(initial=0.0))}
                    for d, c in sorted(changes.items())},
        "mesh_nodes": mesh_nodes,
        "base_deck": str(deck),
    }
    return changes, report


def load_local_edit_data(preprocessor, include_base=False):
    """Training records of every edited run, numbered by sample; design 6 only if ``include_base``.

    Reads ``config.impact_coords_path`` (the manifest), one design-6 deck in
    ``config.inp_dir`` and the curves in ``config.acceleration_dir`` with the
    configured time cutoff and stride. A missing curve is an error.
    """
    config = preprocessor.config
    if getattr(config, "mesh_nodes", "inner_panel") != "inner_panel":
        raise ValueError("The local-edit dataset supports only --mesh-nodes inner_panel")
    if config.prediction_target != "acceleration":
        raise ValueError("The local-edit dataset is set up for acceleration histories only")
    geometry, manifest, _ = LocalEditGeometry.from_manifest(config.impact_coords_path, config.inp_dir)
    rows = manifest if include_base else manifest[manifest["design"] != BASE_DESIGN]
    data = {"mesh_geometries": [], "indentor_positions": [], "time_arrays": [], "accelerations": [],
            "run_numbers": [], "source_runs": []}
    for row in rows.itertuples():
        times, accel = preprocessor.load_acceleration_history(int(row.run))
        data["mesh_geometries"].append(geometry.mesh(int(row.design_id)))
        data["indentor_positions"].append([row.x_mm, row.y_mm])
        data["time_arrays"].append(times)
        data["accelerations"].append(accel)
        data["run_numbers"].append(int(row.sample))
        data["source_runs"].append(int(row.run))
    data["indentor_positions"] = np.array(data["indentor_positions"], dtype=np.float32)
    return data


def base_reference_curves(preprocessor):
    """{location: acceleration} of design 6, sampled like the training curves, and their times."""
    locations, _, times, curves = base_reference_runs(preprocessor)
    return dict(zip(locations.tolist(), curves)), times


def base_reference_runs(preprocessor):
    """Design 6 at every location: (locations (L,), impact XY (L, 2) mm, times (T,), curves (L, T) in g).

    Sampled like the training curves. --reference-response builds the model's
    reference table from these (reference_response.py).
    """
    manifest = load_manifest(preprocessor.config.impact_coords_path)
    base = manifest[manifest["design"] == BASE_DESIGN].sort_values("location")
    if base.empty:
        raise ValueError("The manifest has no design-6 runs to use as the reference")
    curves, times = [], None
    for row in base.itertuples():
        run_times, accel = preprocessor.load_acceleration_history(int(row.run))
        if times is None:
            times = run_times
        elif not np.array_equal(times, run_times):
            raise ValueError("Design-6 curves do not share one time grid")
        curves.append(np.asarray(accel, dtype=np.float64))
    return (base["location"].to_numpy(np.int64), base[["x_mm", "y_mm"]].to_numpy(np.float64),
            times, np.stack(curves))
