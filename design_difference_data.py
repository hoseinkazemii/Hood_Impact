"""Shared data for the design-difference root-cause study on the 1704 Euro-NCAP set.

The study asks why no surrogate predicts the response difference between
near-clone hoods at a fixed impact location. Near-clones are designs of the same
shape family whose meshes share node IDs and connectivity and differ only by a
local edit of 50-220 nodes, so the edit is an exact displacement field
``dx = x_b - x_a`` on corresponding nodes.

Everything here is computed from the dataset itself (decks, SAE1000 histories,
manifest and generation metrics), never from a trained model, and cached under
``<data>/design_difference_cache`` because parsing 12 decks and 1704 CSVs is the
slow part of every script.

Conventions shared by every script of the study:

* Pairs are ordered ``(a, b)`` with ``a < b`` and the difference is ``b - a``,
  exactly as in the approved metric
  (runs/mesh_impact_history/design_difference_r2/design_diff_r2.py).
* Train pairs are the 12 near-clone pairs inside {0,1,2,3} and {6,7,8,9};
  validation is (10, 11) (design 11 is the surrogates' validation design); test
  is (4, 5), the held-out cluster.
* The metric is R2 of the difference curve pooled over time, locations and
  pairs: ``1 - sum (p - d)^2 / sum (d - mean d)^2``. Predicting identical curves
  for both designs scores ~0.
* Time grid: the surrogates' 63-point grid, every 16th of 1000 samples.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "abaqus_scripts"))
from inp_geom import Deck, HOOD_OUTER_ELSET  # noqa: E402

DEFAULT_DATA_DIR = CODE_ROOT / "Data" / "HoodImpact_1704_EuroNCAP"
CHECKERBOARD_SPLIT = CODE_ROOT / "location_splits" / "euroncap1704_checkerboard.json"
CACHE_NAME = "design_difference_cache"
NUM_DESIGNS = 12
SAMPLES_PER_DESIGN = 142
MODEL_TIME_STRIDE = 16
HEADFORM_RADIUS_MM = 82.5
# Half shell thicknesses: hood outer 1.5 mm and headform 1.0 mm. A run starts in
# contact when the sphere penetration exceeds -1.25 mm.
CONTACT_OFFSET_MM = 1.25

FAMILIES = {"A": (0, 1, 2, 3), "B": (4, 5), "C": (6, 7, 8, 9), "D": (10, 11)}
TRAIN_PAIRS = tuple(combinations(FAMILIES["A"], 2)) + tuple(combinations(FAMILIES["C"], 2))
VALIDATION_PAIRS = ((10, 11),)
TEST_PAIRS = ((4, 5),)
PAIR_GROUPS = {"train": TRAIN_PAIRS, "validation": VALIDATION_PAIRS, "test": TEST_PAIRS}
TRAIN_DESIGNS = tuple(sorted({d for pair in TRAIN_PAIRS for d in pair}))

# Edit = nodes moved by more than EDIT_TOLERANCE_MM. Distance-to-edit uses the
# 0.5 mm threshold of the earlier divergence analyses.
EDIT_TOLERANCE_MM = 1e-4
MOVED_THRESHOLD_MM = 0.5
DISTANCE_BANDS = (("near", 0.0, 100.0), ("mid", 100.0, 250.0), ("far", 250.0, math.inf))
PANELS = ("outer", "inner", "hinge", "other")
PANEL_ELSETS = {"outer": HOOD_OUTER_ELSET, "inner": "Hood_Inner-1-2", "hinge": "Hinges-1-2"}


def run_number(design, location):
    """1-indexed 1704 run id of a design (0-11) and location (1-142)."""
    return design * SAMPLES_PER_DESIGN + location


def family_of(design):
    for name, members in FAMILIES.items():
        if design in members:
            return name
    raise ValueError(f"unknown design {design}")


def cache_dir(data_dir=DEFAULT_DATA_DIR):
    path = Path(data_dir) / CACHE_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- histories
def load_histories(data_dir=DEFAULT_DATA_DIR):
    """(times (1000,) s, acceleration (12, 142, 1000) g) of all 1704 runs."""
    data_dir = Path(data_dir)
    cached = cache_dir(data_dir) / "histories_12x142.npz"
    if cached.is_file():
        stored = np.load(cached)
        return stored["times"], stored["acceleration"]
    times, rows = None, []
    for design in range(NUM_DESIGNS):
        for location in range(1, SAMPLES_PER_DESIGN + 1):
            frame = pd.read_csv(data_dir / "output_history_acc"
                                / f"HoodImpact_{run_number(design, location)}_SAE1000_interp1000.csv")
            if times is None:
                times = frame["Time"].to_numpy(np.float64)
            elif not np.allclose(frame["Time"].to_numpy(np.float64), times, rtol=0, atol=1e-12):
                raise ValueError(f"design {design} location {location}: different time grid")
            rows.append(frame["A(in g)"].to_numpy(np.float64))
    acceleration = np.asarray(rows).reshape(NUM_DESIGNS, SAMPLES_PER_DESIGN, -1)
    np.savez_compressed(cached, times=times, acceleration=acceleration)
    return times, acceleration


def model_grid(times, acceleration, stride=MODEL_TIME_STRIDE):
    """The surrogates' grid: every ``stride``-th sample, as DataPreprocessor does."""
    return times[::stride], acceleration[..., ::stride]


# --------------------------------------------------------------------------- locations and runs
def load_run_table(data_dir=DEFAULT_DATA_DIR):
    """One row per run: design, loc, X1-X3 (headform centre), surf_z, penetration, increments."""
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / "manifest_1704.csv")
    metrics = pd.read_csv(data_dir / "generation_metrics.csv")[["run", "n_raw_points", "status"]]
    table = manifest.merge(metrics, on="run", how="left", validate="one_to_one")
    if len(table) != NUM_DESIGNS * SAMPLES_PER_DESIGN or table["n_raw_points"].isna().any():
        raise ValueError("manifest and generation metrics must cover all 1704 runs")
    if not (table["run"] == table["design"] * SAMPLES_PER_DESIGN + table["loc"]).all():
        raise ValueError("run numbering differs from design * 142 + loc")
    table["in_contact"] = table["sphere_pen_mm"] > -CONTACT_OFFSET_MM
    return table.sort_values("run").reset_index(drop=True)


def per_run_array(table, column):
    """(12, 142) array of one run-table column."""
    return table.sort_values("run")[column].to_numpy().reshape(NUM_DESIGNS, SAMPLES_PER_DESIGN)


def design_increments(table):
    """Median explicit increment count per design: a proxy for 1 / stable time step."""
    return table.groupby("design")["n_raw_points"].median().reindex(range(NUM_DESIGNS)).to_numpy(float)


def log_dt_ratio(increments, a, b):
    """log(dt_b / dt_a) = log(n_a / n_b); antisymmetric in (a, b)."""
    return float(np.log(increments[a] / increments[b]))


def load_locations(data_dir=DEFAULT_DATA_DIR):
    """Lattice table indexed by loc (1-142): X1, X2, wadh_mm, col_k, mirror twin."""
    grid = pd.read_csv(Path(data_dir) / "impact_locations_142.csv").set_index("loc").sort_index()
    if list(grid.index) != list(range(1, SAMPLES_PER_DESIGN + 1)):
        raise ValueError("impact_locations_142.csv must list locations 1-142")
    key = {(round(row.wadh_mm), int(row.col_k)): loc for loc, row in grid.iterrows()}
    grid["mirror_twin"] = [key.get((round(row.wadh_mm), -int(row.col_k)), loc)
                           for loc, row in grid.iterrows()]
    return grid


def checkerboard_locations(path=CHECKERBOARD_SPLIT):
    """(even-parity locations, odd-parity locations) of the mirror-closed checkerboard."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(report["train_validation_locations"]), np.asarray(report["test_locations"])


# --------------------------------------------------------------------------- geometry
def _element_normals(deck, elements):
    """Area-weighted node normals and lumped node areas over the given shell elements."""
    normal_sum, area_sum = {}, {}
    for eid in elements:
        conn = deck.elems[eid]
        points = np.array([deck.nodes[n] for n in conn], float)
        if len(conn) == 4:
            doubled = np.cross(points[2] - points[0], points[3] - points[1])
        else:
            doubled = np.cross(points[1] - points[0], points[2] - points[0])
        area = 0.5 * np.linalg.norm(doubled)
        for node in conn:
            normal_sum[node] = normal_sum.get(node, 0.0) + 0.5 * doubled
            area_sum[node] = area_sum.get(node, 0.0) + area / len(conn)
    return normal_sum, area_sum


def load_design_geometry(design, data_dir=DEFAULT_DATA_DIR):
    """Hood structure of one design, nodes in ascending ID (headform removed).

    Returns a dict of arrays: ``ids``, ``xyz`` (mm), ``panel`` (index into
    PANELS), ``normal`` (unit, area-weighted, zero off the shells), ``area``
    (lumped mm^2) and ``gap``: the distance from an outer node to the nearest
    inner node, and from every other node to the nearest outer node.
    """
    from scipy.spatial import cKDTree

    path = cache_dir(data_dir) / f"geometry_design{design}.npz"
    if path.is_file():
        return dict(np.load(path))
    deck = Deck(str(Path(data_dir) / "inp_files" / f"HoodImpact_{run_number(design, 1)}.inp"))
    impactor = deck.impactor_node_ids()
    ids = np.array(sorted(set(deck.nodes) - impactor))
    index = {node: i for i, node in enumerate(ids)}
    xyz = np.array([deck.nodes[n] for n in ids], float)
    panel = np.full(len(ids), PANELS.index("other"))
    elements = []
    for name in ("hinge", "inner", "outer"):  # outer last: it wins a shared node
        members = deck.elsets[PANEL_ELSETS[name]]
        elements.extend(members)
        for node in deck.elset_nodes(PANEL_ELSETS[name]):
            if node in index:
                panel[index[node]] = PANELS.index(name)
    normal_sum, area_sum = _element_normals(deck, elements)
    normal = np.zeros_like(xyz)
    area = np.zeros(len(ids))
    for node, vector in normal_sum.items():
        if node in index:
            length = np.linalg.norm(vector)
            normal[index[node]] = vector / length if length > 0 else 0.0
            area[index[node]] = area_sum[node]
    outer = panel == PANELS.index("outer")
    inner = panel == PANELS.index("inner")
    gap = np.empty(len(ids))
    gap[outer] = cKDTree(xyz[inner]).query(xyz[outer])[0]
    gap[~outer] = cKDTree(xyz[outer]).query(xyz[~outer])[0]
    geometry = {"ids": ids, "xyz": xyz, "panel": panel, "normal": normal, "area": area, "gap": gap}
    np.savez_compressed(path, **geometry)
    return geometry


def pair_geometry(geometry_a, geometry_b, tolerance=EDIT_TOLERANCE_MM):
    """Exact edit field between two designs of one mesh family (b - a).

    ``edit`` marks nodes that moved or whose panel gap changed; only those can
    carry a first-order response difference.
    """
    if not np.array_equal(geometry_a["ids"], geometry_b["ids"]):
        raise ValueError("designs are from different mesh families; node IDs do not correspond")
    dx = geometry_b["xyz"] - geometry_a["xyz"]
    dgap = geometry_b["gap"] - geometry_a["gap"]
    moved = np.linalg.norm(dx, axis=1) > tolerance
    edit = moved | (np.abs(dgap) > tolerance)
    normal_mid = geometry_a["normal"] + geometry_b["normal"]
    length = np.linalg.norm(normal_mid, axis=1, keepdims=True)
    normal_mid = np.divide(normal_mid, length, out=np.zeros_like(normal_mid), where=length > 0)
    return {
        "x_mid": 0.5 * (geometry_a["xyz"] + geometry_b["xyz"]), "dx": dx, "dgap": dgap,
        "gap_mid": 0.5 * (geometry_a["gap"] + geometry_b["gap"]),
        "gap_a": geometry_a["gap"], "gap_b": geometry_b["gap"],
        "normal_mid": normal_mid, "area_mid": 0.5 * (geometry_a["area"] + geometry_b["area"]),
        "panel": geometry_a["panel"], "moved": moved, "edit": edit,
    }


def edit_distance(pair, impact_xyz, threshold=MOVED_THRESHOLD_MM):
    """XY and 3D distance (mm) from each impact point to the nearest node moved > threshold.

    Both end positions of every moved node count, as in the earlier divergence
    analysis (within_family_dt_artifact/divergence_onset.py).
    """
    from scipy.spatial import cKDTree

    moved = np.linalg.norm(pair["dx"], axis=1) > threshold
    if not moved.any():
        infinite = np.full(len(impact_xyz), np.inf)
        return infinite, infinite.copy()
    half = 0.5 * pair["dx"][moved]
    points = np.vstack((pair["x_mid"][moved] - half, pair["x_mid"][moved] + half))
    xy = cKDTree(points[:, :2]).query(np.asarray(impact_xyz)[:, :2])[0]
    xyz = cKDTree(points).query(np.asarray(impact_xyz))[0]
    return xy, xyz


def distance_band(distance_mm):
    """Band label per distance: near < 100 mm <= mid < 250 mm <= far."""
    distance_mm = np.asarray(distance_mm, float)
    labels = np.empty(distance_mm.shape, dtype=object)
    for name, low, high in DISTANCE_BANDS:
        labels[(distance_mm >= low) & (distance_mm < high)] = name
    return labels


# --------------------------------------------------------------------------- metric
def difference_r2(prediction, truth):
    """R2 of difference curves pooled over every axis; 0 for a collapsed predictor."""
    prediction = np.asarray(prediction, float).ravel()
    truth = np.asarray(truth, float).ravel()
    denominator = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - np.sum((prediction - truth) ** 2) / denominator) if denominator > 0 else float("nan")


def pair_differences(acceleration, pairs):
    """(len(pairs), 142, T) true differences b - a."""
    return np.stack([acceleration[b] - acceleration[a] for a, b in pairs])


def grouped_difference_r2(predicted, truth, locations=None):
    """{group: R2} for predicted/true dicts keyed by pair, optionally on a location subset."""
    report = {}
    for group, pairs in PAIR_GROUPS.items():
        if not all(pair in predicted for pair in pairs):
            continue
        take = slice(None) if locations is None else np.asarray(locations) - 1
        report[group] = difference_r2(np.stack([predicted[p][take] for p in pairs]),
                                      np.stack([truth[p][take] for p in pairs]))
    return report


# --------------------------------------------------------------------------- tracking
def start_wandb(mode, project, entity, run_name, config, output_dir, logger):
    """Start W&B with the main trainer's retry logic; None when disabled.

    wandb is imported only when tracking is on, so local runs with
    ``--wandb-mode disabled`` work without it.
    """
    if mode == "disabled":
        return None
    from types import SimpleNamespace

    from train_mesh_impact_history import initialize_wandb
    args = SimpleNamespace(wandb_mode=mode, wandb_project=project, wandb_entity=entity, run_name=run_name)
    return initialize_wandb(args, config, Path(output_dir), logger)


def write_json(path, payload):
    """Strict JSON (non-finite floats become null), like the training scripts."""
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        return value
    Path(path).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
