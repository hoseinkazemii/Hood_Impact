"""Direct HIC15 surrogate for the 600-run EuroNCAP hood-impact corpus.

This experiment is intentionally independent from ``temporal_deeponet.py`` and
``utils/utils.py`` so older checkpoints and training paths remain reproducible.

The data form a complete 12-design x 50-location factorial grid.  A large
temporal neural operator is a poor fit for only twelve independent geometries,
and its pointwise acceleration loss is not the metric of interest.  This model
instead:

1. recomputes canonical HIC15 from every full 1,000-point acceleration history;
2. strips the 286 moving headform nodes from one representative deck/design;
3. builds impact-centred, multi-scale hood/rib descriptors and uses X3 plus the
   EuroNCAP manifest metadata;
4. transfers the response profile of the closest *training* geometry; and
5. fits a random-forest model to the log-HIC residual between geometry pairs.

Model selection is design-wise.  By default, grouped leave-one-design-out CV is
performed only on the development designs, the residual shrinkage is selected
from those out-of-fold predictions, and the model is refit on every non-test
design before an explicitly requested held-out reference evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor

try:  # The production environment already uses numba for HIC reporting.
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # A vectorised NumPy implementation remains available.
    NUMBA_AVAILABLE = False


MODEL_VERSION = "hic-direct-v5.0"
FEATURE_VERSION = "impact-local-v2"
EXPECTED_DESIGNS = 12
EXPECTED_LOCATIONS = 50
EXPECTED_RUNS = EXPECTED_DESIGNS * EXPECTED_LOCATIONS
DEFAULT_IMPACTOR_NODES = 286
RING_EDGES_MM = (0.0, 60.0, 120.0, 200.0, 300.0, 450.0)
N_SECTORS = 8


@dataclass(frozen=True)
class MetricSet:
    n: int
    r2: float
    rmse: float
    mae: float
    bias: float


def setup_logger(output_dir: Path) -> logging.Logger:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output directory was not reserved: {output_dir}")
    logger = logging.getLogger("hic_direct_v5")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def file_content_fingerprint(paths: Iterable[Path], extra: str = "") -> str:
    """Hash input bytes and the exact implementation, not mutable mtimes alone."""

    digest = hashlib.sha256(extra.encode("utf-8"))
    implementation_path = Path(__file__).resolve()
    digest.update(implementation_path.read_bytes())
    for path in sorted((Path(p).resolve() for p in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_joblib_dump(payload: object, path: Path, compress: int = 3) -> None:
    """Publish a cache/model only after serialization completes."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(payload, temporary, compress=compress)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_provenance() -> Dict[str, object]:
    result: Dict[str, object] = {}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["commit"] = None
        result["branch"] = None
        result["dirty"] = None
    return result


def hic15_numpy(
    time: np.ndarray,
    acceleration_g: np.ndarray,
    max_window_s: float = 0.015,
) -> float:
    """Compute HIC over *all* positive-duration windows no longer than 15 ms.

    Integration is trapezoidal, matching the project's evaluation utility.  In
    contrast, the older Abaqus postprocessor checked only one (the longest)
    endpoint per start time and can substantially understate HIC.
    """

    time = np.asarray(time, dtype=np.float64)
    acceleration_g = np.asarray(acceleration_g, dtype=np.float64)
    if time.ndim != 1 or acceleration_g.ndim != 1 or len(time) != len(acceleration_g):
        raise ValueError("time and acceleration must be same-length 1-D arrays")
    if len(time) < 2:
        raise ValueError("HIC requires at least two time points")
    if not np.isfinite(time).all() or not np.isfinite(acceleration_g).all():
        raise ValueError("time/acceleration contains NaN or Inf")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not math.isfinite(max_window_s) or max_window_s <= 0.0:
        raise ValueError("max_window_s must be finite and positive")

    integral = np.zeros(len(time), dtype=np.float64)
    integral[1:] = np.cumsum(
        0.5 * (acceleration_g[1:] + acceleration_g[:-1]) * np.diff(time)
    )
    best = 0.0
    for start in range(len(time) - 1):
        stop = int(np.searchsorted(time, time[start] + max_window_s, side="right"))
        if stop <= start + 1:
            continue
        ends = np.arange(start + 1, stop)
        duration = time[ends] - time[start]
        average = (integral[ends] - integral[start]) / duration
        # A(in g) is a resultant and should be non-negative.  Clipping protects
        # against negligible floating-point undershoot without redefining HIC.
        score = duration * np.maximum(average, 0.0) ** 2.5
        best = max(best, float(np.max(score)))
    return best


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _hic15_batch_numba(times, accelerations, max_window_s):
        n_samples, n_steps = accelerations.shape
        output = np.zeros(n_samples, dtype=np.float64)
        for sample in prange(n_samples):
            integral = np.zeros(n_steps, dtype=np.float64)
            for idx in range(1, n_steps):
                dt = times[sample, idx] - times[sample, idx - 1]
                integral[idx] = integral[idx - 1] + 0.5 * (
                    accelerations[sample, idx] + accelerations[sample, idx - 1]
                ) * dt

            best = 0.0
            for start in range(n_steps - 1):
                for end in range(start + 1, n_steps):
                    duration = times[sample, end] - times[sample, start]
                    if duration > max_window_s:
                        break
                    if duration <= 0.0:
                        continue
                    average = (integral[end] - integral[start]) / duration
                    if average > 0.0:
                        score = duration * average ** 2.5
                        if score > best:
                            best = score
            output[sample] = best
        return output


def hic15_batch(
    times: Sequence[np.ndarray],
    accelerations: Sequence[np.ndarray],
    max_window_s: float,
) -> np.ndarray:
    if not math.isfinite(max_window_s) or max_window_s <= 0.0:
        raise ValueError("max_window_s must be finite and positive")
    if len(times) != len(accelerations):
        raise ValueError("times and accelerations have different sample counts")
    times = [np.asarray(item, dtype=np.float64) for item in times]
    accelerations = [np.asarray(item, dtype=np.float64) for item in accelerations]
    for time, acceleration in zip(times, accelerations):
        if time.ndim != 1 or acceleration.ndim != 1 or len(time) != len(acceleration):
            raise ValueError("time and acceleration must be same-length 1-D arrays")
        if len(time) < 2:
            raise ValueError("HIC requires at least two time points")
        if not np.isfinite(time).all() or not np.isfinite(acceleration).all():
            raise ValueError("time/acceleration contains NaN or Inf")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")
    if NUMBA_AVAILABLE and len({len(item) for item in times}) == 1:
        return _hic15_batch_numba(
            np.stack(times).astype(np.float64),
            np.stack(accelerations).astype(np.float64),
            float(max_window_s),
        )
    return np.asarray(
        [hic15_numpy(t, a, max_window_s) for t, a in zip(times, accelerations)],
        dtype=np.float64,
    )


def load_manifest(
    manifest_path: Path,
    impact_grid_path: Path,
    expected_designs: int = EXPECTED_DESIGNS,
    expected_locations: int = EXPECTED_LOCATIONS,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest.columns = [column.strip() for column in manifest.columns]
    required = {
        "run",
        "design",
        "loc",
        "X1",
        "X2",
        "X3",
        "dz_mm",
        "center_gap_mm",
        "sphere_pen_mm",
        "lift_mm",
        "surf_z_mm",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns {sorted(missing)}")
    numeric_columns = sorted(required)
    numeric_manifest = manifest[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_manifest.to_numpy(np.float64)).all():
        raise ValueError(f"{manifest_path} contains non-finite required numeric values")
    manifest[numeric_columns] = numeric_manifest
    for column in ("run", "design", "loc"):
        values = manifest[column].to_numpy(np.float64)
        if not np.array_equal(values, np.rint(values)):
            raise ValueError(f"manifest {column} IDs must be integers")
        manifest[column] = values.astype(np.int64)

    manifest = manifest.sort_values("run").reset_index(drop=True)
    expected_runs = expected_designs * expected_locations
    if len(manifest) != expected_runs:
        raise ValueError(f"expected {expected_runs} manifest rows, found {len(manifest)}")
    if not np.array_equal(manifest["run"].to_numpy(), np.arange(1, expected_runs + 1)):
        raise ValueError("manifest run IDs must be exactly 1..N")

    expected_design_ids = set(range(expected_designs))
    if set(manifest["design"].astype(int)) != expected_design_ids:
        raise ValueError("manifest design IDs are not the expected contiguous range")
    for design, group in manifest.groupby("design"):
        locations = group["loc"].astype(int).to_numpy()
        if not np.array_equal(np.sort(locations), np.arange(1, expected_locations + 1)):
            raise ValueError(f"design {design} does not contain locations 1..{expected_locations}")

    grid = pd.read_csv(impact_grid_path)
    grid.columns = [column.strip() for column in grid.columns]
    grid_required = {"loc", "X1", "X2", "wadh_mm", "col_k"}
    missing_grid = grid_required.difference(grid.columns)
    if missing_grid:
        raise ValueError(f"{impact_grid_path} is missing columns {sorted(missing_grid)}")
    numeric_grid = grid[sorted(grid_required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_grid.to_numpy(np.float64)).all():
        raise ValueError(f"{impact_grid_path} contains non-finite required numeric values")
    grid[sorted(grid_required)] = numeric_grid
    grid_locations = grid["loc"].to_numpy(np.float64)
    if not np.array_equal(grid_locations, np.rint(grid_locations)):
        raise ValueError("impact-grid location IDs must be integers")
    grid["loc"] = grid_locations.astype(np.int64)
    expected_location_ids = np.arange(1, expected_locations + 1)
    if len(grid) != expected_locations or not np.array_equal(
        np.sort(grid["loc"].to_numpy()), expected_location_ids
    ):
        raise ValueError(
            f"impact-grid location IDs must be exactly 1..{expected_locations}"
        )

    grid = grid.rename(columns={"X1": "grid_X1", "X2": "grid_X2"})
    manifest = manifest.merge(
        grid[["loc", "grid_X1", "grid_X2", "wadh_mm", "col_k"]],
        on="loc",
        how="left",
        validate="many_to_one",
    )
    xy_error = np.maximum(
        np.abs(manifest["X1"] - manifest["grid_X1"]),
        np.abs(manifest["X2"] - manifest["grid_X2"]),
    )
    merged_required = ["grid_X1", "grid_X2", "wadh_mm", "col_k"]
    if not np.isfinite(manifest[merged_required].to_numpy(np.float64)).all():
        raise ValueError("impact-grid merge produced missing or non-finite values")
    if float(xy_error.max()) > 1.0:
        raise ValueError("manifest XY coordinates do not match impact_locations_50.csv")
    return manifest.drop(columns=["grid_X1", "grid_X2"])


def history_path(history_dir: Path, run_number: int) -> Path:
    return history_dir / f"HoodImpact_{run_number}_SAE1000_interp1000.csv"


def representative_deck_paths(manifest: pd.DataFrame, inp_dir: Path) -> Dict[int, Path]:
    paths: Dict[int, Path] = {}
    for design, group in manifest.groupby("design", sort=True):
        run_number = int(group["run"].min())
        paths[int(design)] = inp_dir / f"HoodImpact_{run_number}.inp"
    return paths


def preflight(
    manifest: pd.DataFrame,
    inp_dir: Path,
    history_dir: Path,
    n_impactor_nodes: int,
) -> Dict[str, object]:
    expected_runs = len(manifest)
    decks = sorted(inp_dir.glob("HoodImpact_*.inp"))
    histories = sorted(history_dir.glob("HoodImpact_*_SAE1000_interp1000.csv"))
    if len(decks) != expected_runs:
        raise FileNotFoundError(f"expected {expected_runs} decks in {inp_dir}, found {len(decks)}")
    if len(histories) != expected_runs:
        raise FileNotFoundError(
            f"expected {expected_runs} histories in {history_dir}, found {len(histories)}"
        )

    representatives = representative_deck_paths(manifest, inp_dir)
    node_counts = {}
    for design, path in representatives.items():
        row = manifest.loc[manifest["design"] == design].sort_values("run").iloc[0]
        _, coordinates = parse_hood_mesh(
            path,
            impact_xy=np.asarray([row["X1"], row["X2"]], dtype=np.float64),
            n_impactor_nodes=n_impactor_nodes,
        )
        node_counts[design] = int(len(coordinates))

    for run_number in (1, expected_runs):
        frame = pd.read_csv(history_path(history_dir, run_number), nrows=3)
        if not {"Time", "A(in g)"}.issubset(frame.columns):
            raise ValueError(f"history {run_number} lacks Time/A(in g) columns")
    return {
        "runs": expected_runs,
        "designs": int(manifest["design"].nunique()),
        "locations_per_design": int(manifest.groupby("design").size().iloc[0]),
        "hood_node_counts": node_counts,
    }


def load_or_compute_targets(
    manifest: pd.DataFrame,
    manifest_path: Path,
    history_dir: Path,
    cache_dir: Path,
    max_window_s: float,
    rebuild_cache: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str]:
    paths = [manifest_path] + [history_path(history_dir, int(run)) for run in manifest["run"]]
    fingerprint = file_content_fingerprint(
        paths,
        extra=f"{MODEL_VERSION}|all-window-trapezoid|max={max_window_s:.9g}",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"hic15_{fingerprint[:16]}.joblib"
    if cache_path.exists() and not rebuild_cache:
        cached = joblib.load(cache_path)
        logger.info("Loaded canonical HIC targets from %s", cache_path)
        return cached["targets"].copy(), fingerprint

    logger.info("Reading %d full-resolution histories", len(manifest))
    times: List[np.ndarray] = []
    accelerations: List[np.ndarray] = []
    point_counts: List[int] = []
    for run_number in manifest["run"].astype(int):
        path = history_path(history_dir, run_number)
        frame = pd.read_csv(path, usecols=["Time", "A(in g)"])
        time = frame["Time"].to_numpy(np.float64)
        acceleration = frame["A(in g)"].to_numpy(np.float64)
        if not np.isfinite(time).all() or not np.isfinite(acceleration).all():
            raise ValueError(f"{path} contains NaN/Inf")
        if len(time) < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError(f"{path} has an invalid time grid")
        if float(acceleration.min()) < -1e-3:
            raise ValueError(f"{path} has negative resultant acceleration")
        times.append(time)
        accelerations.append(acceleration)
        point_counts.append(len(time))

    logger.info(
        "Computing all-window HIC15 (%s path)",
        "numba-parallel" if NUMBA_AVAILABLE else "NumPy",
    )
    hic = hic15_batch(times, accelerations, max_window_s)
    if not np.isfinite(hic).all() or np.any(hic <= 0.0):
        raise ValueError("computed HIC targets are not all finite and positive")

    targets = manifest[["run", "design", "loc", "X1", "X2", "X3"]].copy()
    targets["hic15"] = hic
    targets["n_time_points"] = point_counts
    targets["max_window_s"] = max_window_s
    atomic_joblib_dump({"fingerprint": fingerprint, "targets": targets}, cache_path)
    logger.info("Cached canonical HIC targets at %s", cache_path)
    return targets, fingerprint


def parse_hood_mesh(
    inp_path: Path,
    impact_xy: np.ndarray,
    n_impactor_nodes: int = DEFAULT_IMPACTOR_NODES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the first Abaqus node block and remove the highest-ID headform nodes."""

    node_ids: List[int] = []
    coordinates: List[Tuple[float, float, float]] = []
    recording = False
    with inp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not recording:
                if stripped.upper() == "*NODE":
                    recording = True
                continue
            if stripped.startswith("*"):
                break
            parts = stripped.split(",")
            if len(parts) >= 4:
                node_ids.append(int(parts[0]))
                coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))

    ids = np.asarray(node_ids, dtype=np.int64)
    xyz = np.asarray(coordinates, dtype=np.float64)
    if len(ids) <= n_impactor_nodes:
        raise ValueError(f"{inp_path}: found only {len(ids)} nodes")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{inp_path}: node IDs are not unique")

    order = np.argsort(ids)
    ids = ids[order]
    xyz = xyz[order]
    reference_xy = xyz[-1, :2]
    if float(np.max(np.abs(reference_xy - impact_xy))) > 1.0:
        raise ValueError(
            f"{inp_path}: highest-ID node {reference_xy} is not at impact XY {impact_xy}"
        )
    hood_ids = ids[:-n_impactor_nodes]
    hood_xyz = xyz[:-n_impactor_nodes]
    impactor_xyz = xyz[-n_impactor_nodes:]
    if float(impactor_xyz[:, 2].max()) <= float(hood_xyz[:, 2].max()):
        raise ValueError(f"{inp_path}: highest-ID block does not look like the headform")
    return hood_ids, hood_xyz


def local_feature_names() -> List[str]:
    names = [
        "node_count_scaled",
        "global_z_q00",
        "global_z_q10",
        "global_z_q50",
        "global_z_q90",
        "global_z_q100",
        "nearest_xy_mm",
        "nearest_z_from_surface_mm",
    ]
    quantile_names = ("q00", "q10", "q25", "q50", "q75", "q90", "q100")
    for lower, upper in zip(RING_EDGES_MM[:-1], RING_EDGES_MM[1:]):
        prefix = f"ring_{int(lower)}_{int(upper)}"
        names.append(f"{prefix}_count_scaled")
        names.extend(f"{prefix}_z_{name}" for name in quantile_names)
    for sector in range(N_SECTORS):
        names.extend(
            [
                f"sector_{sector}_count_scaled",
                f"sector_{sector}_z_mean",
                f"sector_{sector}_z_min",
                f"sector_{sector}_z_max",
            ]
        )
    return names


def describe_local_mesh(mesh_xyz: np.ndarray, sample: Mapping[str, float]) -> np.ndarray:
    x = float(sample["X1"])
    y = float(sample["X2"])
    surface_z = float(sample["surf_z_mm"])
    dx = mesh_xyz[:, 0] - x
    dy = mesh_xyz[:, 1] - y
    radius = np.sqrt(dx * dx + dy * dy)
    z = mesh_xyz[:, 2] - surface_z

    nearest = int(np.argmin(radius))
    features: List[float] = [
        len(mesh_xyz) / 40000.0,
        *np.quantile(z, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        float(radius[nearest]),
        float(z[nearest]),
    ]
    for lower, upper in zip(RING_EDGES_MM[:-1], RING_EDGES_MM[1:]):
        values = z[(radius >= lower) & (radius < upper)]
        if len(values):
            features.extend(
                [
                    len(values) / 1000.0,
                    *np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist(),
                ]
            )
        else:
            features.extend([0.0] * 8)

    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    sector_width = 2.0 * np.pi / N_SECTORS
    for sector in range(N_SECTORS):
        values = z[
            (radius < 350.0)
            & (angle >= sector * sector_width)
            & (angle < (sector + 1) * sector_width)
        ]
        if len(values):
            features.extend(
                [
                    len(values) / 1000.0,
                    float(np.mean(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                ]
            )
        else:
            features.extend([0.0] * 4)
    result = np.asarray(features, dtype=np.float64)
    if len(result) != len(local_feature_names()):
        raise AssertionError("local feature/name count mismatch")
    return result


def pair_geometry_feature_names() -> List[str]:
    return [
        "geometry_distance",
        "common_node_fraction",
        "missing_node_fraction",
        "node_count_delta_scaled",
        "displacement_rms",
        "displacement_mean",
        "displacement_q90",
        "displacement_q99",
        "displacement_max",
        "changed_fraction_0p01mm",
        "changed_fraction_0p5mm",
        "changed_fraction_2mm",
        "mean_delta_x",
        "mean_delta_y",
        "mean_delta_z",
        "mean_abs_delta_x",
        "mean_abs_delta_y",
        "mean_abs_delta_z",
    ]


def geometry_pair_features(
    query_ids: np.ndarray,
    query_xyz: np.ndarray,
    source_ids: np.ndarray,
    source_xyz: np.ndarray,
) -> np.ndarray:
    common = np.intersect1d(query_ids, source_ids, assume_unique=True)
    if not len(common):
        raise ValueError("two designs have no common node IDs")
    query_common = query_xyz[np.searchsorted(query_ids, common)]
    source_common = source_xyz[np.searchsorted(source_ids, common)]
    delta = query_common - source_common
    displacement = np.linalg.norm(delta, axis=1)
    max_nodes = max(len(query_ids), len(source_ids))
    missing_fraction = abs(len(query_ids) - len(source_ids)) / max_nodes
    common_fraction = len(common) / max_nodes
    rms = float(np.sqrt(np.mean(displacement**2)))
    distance = rms + 1000.0 * missing_fraction
    return np.asarray(
        [
            distance,
            common_fraction,
            missing_fraction,
            (len(query_ids) - len(source_ids)) / 1000.0,
            rms,
            float(np.mean(displacement)),
            float(np.quantile(displacement, 0.90)),
            float(np.quantile(displacement, 0.99)),
            float(np.max(displacement)),
            float(np.mean(displacement > 0.01)),
            float(np.mean(displacement > 0.5)),
            float(np.mean(displacement > 2.0)),
            *np.mean(delta, axis=0).tolist(),
            *np.mean(np.abs(delta), axis=0).tolist(),
        ],
        dtype=np.float64,
    )


def location_feature_names() -> List[str]:
    names = [
        "x_scaled",
        "y_scaled",
        "x3_scaled",
        "surface_z_scaled",
        "lift_scaled",
        "deck_dz_scaled",
        "center_gap_scaled",
        "sphere_pen_scaled",
        "wadh_scaled",
        "grid_column_scaled",
        "x_squared",
        "y_squared",
        "xy_interaction",
    ]
    for frequency in (1, 2, 4, 8):
        names.extend(
            [
                f"sin_x_{frequency}",
                f"cos_x_{frequency}",
                f"sin_y_{frequency}",
                f"cos_y_{frequency}",
            ]
        )
    return names


def describe_location(sample: Mapping[str, float]) -> np.ndarray:
    x = (float(sample["X1"]) - 240.0) / 800.0
    y = (float(sample["X2"]) - 155.3432) / 800.0
    values = [
        x,
        y,
        float(sample["X3"]) / 800.0,
        float(sample["surf_z_mm"]) / 800.0,
        float(sample["lift_mm"]) / 100.0,
        float(sample["dz_mm"]) / 100.0,
        float(sample["center_gap_mm"]) / 10.0,
        float(sample["sphere_pen_mm"]) / 10.0,
        float(sample["wadh_mm"]) / 1000.0,
        float(sample["col_k"]) / 10.0,
        x * x,
        y * y,
        x * y,
    ]
    for frequency in (1, 2, 4, 8):
        values.extend(
            [
                math.sin(frequency * math.pi * x),
                math.cos(frequency * math.pi * x),
                math.sin(frequency * math.pi * y),
                math.cos(frequency * math.pi * y),
            ]
        )
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(location_feature_names()):
        raise AssertionError("location feature/name count mismatch")
    return result


def load_or_compute_geometry_features(
    manifest: pd.DataFrame,
    manifest_path: Path,
    impact_grid_path: Path,
    inp_dir: Path,
    cache_dir: Path,
    n_impactor_nodes: int,
    rebuild_cache: bool,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    representatives = representative_deck_paths(manifest, inp_dir)
    fingerprint = file_content_fingerprint(
        [manifest_path, impact_grid_path, *representatives.values()],
        extra=f"{MODEL_VERSION}|{FEATURE_VERSION}|headform={n_impactor_nodes}",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"geometry_{fingerprint[:16]}.joblib"
    if cache_path.exists() and not rebuild_cache:
        cached = joblib.load(cache_path)
        logger.info("Loaded impact-centred geometry features from %s", cache_path)
        return (
            cached["local_features"],
            cached["location_features"],
            cached["pair_features"],
            cached["model_feature_names"],
            fingerprint,
        )

    n_designs = int(manifest["design"].nunique())
    n_locations = int(manifest.groupby("design").size().iloc[0])
    meshes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for design, path in representatives.items():
        row = manifest.loc[manifest["design"] == design].sort_values("run").iloc[0]
        meshes[design] = parse_hood_mesh(
            path,
            np.asarray([row["X1"], row["X2"]], dtype=np.float64),
            n_impactor_nodes=n_impactor_nodes,
        )
        logger.info("Parsed design %d hood: %d nodes", design, len(meshes[design][0]))

    local_names = local_feature_names()
    location_names = location_feature_names()
    local_features = np.empty((n_designs, n_locations, len(local_names)), dtype=np.float64)
    location_features = np.empty(
        (n_designs, n_locations, len(location_names)), dtype=np.float64
    )
    for design in range(n_designs):
        rows = manifest.loc[manifest["design"] == design].sort_values("loc")
        for location_index, (_, row) in enumerate(rows.iterrows()):
            local_features[design, location_index] = describe_local_mesh(
                meshes[design][1], row
            )
            location_features[design, location_index] = describe_location(row)

    pair_names = pair_geometry_feature_names()
    pair_features = np.zeros((n_designs, n_designs, len(pair_names)), dtype=np.float64)
    for query in range(n_designs):
        for source in range(n_designs):
            if query == source:
                continue
            pair_features[query, source] = geometry_pair_features(
                meshes[query][0],
                meshes[query][1],
                meshes[source][0],
                meshes[source][1],
            )

    model_feature_names = (
        ["log_source_hic"]
        + pair_names
        + [f"location__{name}" for name in location_names]
        + [f"query_local__{name}" for name in local_names]
        + [f"query_minus_source__{name}" for name in local_names]
    )
    atomic_joblib_dump(
        {
            "fingerprint": fingerprint,
            "local_features": local_features,
            "location_features": location_features,
            "pair_features": pair_features,
            "model_feature_names": model_feature_names,
        },
        cache_path,
    )
    logger.info("Cached geometry features at %s", cache_path)
    return local_features, location_features, pair_features, model_feature_names, fingerprint


def target_matrix(
    target_table: pd.DataFrame,
    n_designs: Optional[int] = None,
    n_locations: Optional[int] = None,
    required_designs: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if n_designs is None:
        n_designs = int(target_table["design"].max()) + 1
    if n_locations is None:
        n_locations = int(target_table["loc"].max())
    matrix = np.full((n_designs, n_locations), np.nan, dtype=np.float64)
    for row in target_table.itertuples(index=False):
        matrix[int(row.design), int(row.loc) - 1] = float(row.hic15)
    required = (
        np.arange(n_designs, dtype=np.int64)
        if required_designs is None
        else np.asarray(required_designs, dtype=np.int64)
    )
    if not np.isfinite(matrix[required]).all():
        raise ValueError("target matrix is incomplete for the required designs")
    return matrix


def validate_split_ids(n_designs: int, validation_design: int, test_design: int) -> None:
    for name, design in (
        ("validation", validation_design),
        ("test", test_design),
    ):
        if not 0 <= design < n_designs:
            raise ValueError(
                f"{name} design must be in [0, {n_designs - 1}], got {design}"
            )
    if validation_design == test_design:
        raise ValueError("validation and test design must differ")
    if n_designs < 4:
        raise ValueError(
            "at least four designs are required so pair training retains two designs"
        )


def pair_model_feature_vector(
    query: int,
    source: int,
    location: int,
    targets: np.ndarray,
    local_features: np.ndarray,
    location_features: np.ndarray,
    pair_features: np.ndarray,
) -> np.ndarray:
    query_local = local_features[query, location]
    source_local = local_features[source, location]
    return np.concatenate(
        [
            np.asarray([math.log(float(targets[source, location]))]),
            pair_features[query, source],
            location_features[query, location],
            query_local,
            query_local - source_local,
        ]
    )


def build_pair_training_data(
    designs: Sequence[int],
    targets: np.ndarray,
    local_features: np.ndarray,
    location_features: np.ndarray,
    pair_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    designs = [int(design) for design in designs]
    rows: List[np.ndarray] = []
    response: List[float] = []
    for query in designs:
        for source in designs:
            if query == source:
                continue
            for location in range(targets.shape[1]):
                rows.append(
                    pair_model_feature_vector(
                        query,
                        source,
                        location,
                        targets,
                        local_features,
                        location_features,
                        pair_features,
                    )
                )
                response.append(math.log(targets[query, location] / targets[source, location]))
    if not rows:
        raise ValueError("at least two training designs are required")
    return np.stack(rows), np.asarray(response, dtype=np.float64)


def fit_residual_model(
    designs: Sequence[int],
    targets: np.ndarray,
    local_features: np.ndarray,
    location_features: np.ndarray,
    pair_features: np.ndarray,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    random_state: int,
    n_jobs: int,
) -> Tuple[RandomForestRegressor, int]:
    features, response = build_pair_training_data(
        designs, targets, local_features, location_features, pair_features
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(features, response)
    return model, len(response)


def nearest_designs(
    query: int,
    allowed_designs: Sequence[int],
    pair_features: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    allowed = np.asarray([int(design) for design in allowed_designs], dtype=np.int64)
    if query in allowed:
        raise ValueError("query design must not appear in its source-design set")
    distance = pair_features[query, allowed, 0]
    order = np.lexsort((allowed, distance))
    return allowed[order[: min(n_neighbors, len(allowed))]]


def predict_design(
    model: RandomForestRegressor,
    query: int,
    allowed_designs: Sequence[int],
    targets: np.ndarray,
    local_features: np.ndarray,
    location_features: np.ndarray,
    pair_features: np.ndarray,
    n_neighbors: int,
    distance_power: float,
) -> Dict[str, object]:
    neighbors = nearest_designs(query, allowed_designs, pair_features, n_neighbors)
    retrieval_candidates = []
    residual_candidates = []
    raw_weights = []
    for source in neighbors:
        feature_rows = np.stack(
            [
                pair_model_feature_vector(
                    query,
                    int(source),
                    location,
                    targets,
                    local_features,
                    location_features,
                    pair_features,
                )
                for location in range(targets.shape[1])
            ]
        )
        log_ratio = np.clip(model.predict(feature_rows), -1.0, 1.0)
        source_profile = targets[int(source)].copy()
        retrieval_candidates.append(source_profile)
        residual_candidates.append(source_profile * np.exp(log_ratio))
        distance = max(float(pair_features[query, int(source), 0]), 1e-6)
        raw_weights.append(distance ** (-distance_power))

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights /= weights.sum()
    retrieval = np.sum(np.stack(retrieval_candidates) * weights[:, None], axis=0)
    residual = np.sum(np.stack(residual_candidates) * weights[:, None], axis=0)
    return {
        "neighbors": neighbors,
        "distances": pair_features[query, neighbors, 0].copy(),
        "weights": weights,
        "retrieval": retrieval,
        "residual": residual,
    }


def geometric_blend(retrieval: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    return retrieval * np.exp(alpha * np.log(np.maximum(residual, 1e-12) / retrieval))


def metrics(target: np.ndarray, prediction: np.ndarray) -> MetricSet:
    target = np.asarray(target, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(error**2)) / denominator
    return MetricSet(
        n=len(target),
        r2=r2,
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
    )


def select_blend_alpha(
    target: np.ndarray,
    retrieval: np.ndarray,
    residual: np.ndarray,
    grid_size: int = 21,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for alpha in np.linspace(0.0, 1.0, grid_size):
        score = metrics(target, geometric_blend(retrieval, residual, float(alpha)))
        rows.append({"alpha": float(alpha), **asdict(score)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["rmse", "alpha"], ascending=[True, True]).iloc[0]
    return float(best["alpha"]), table


def grouped_cross_validation(
    development_designs: Sequence[int],
    targets: np.ndarray,
    local_features: np.ndarray,
    location_features: np.ndarray,
    pair_features: np.ndarray,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    seed: int,
    n_jobs: int,
    n_neighbors: int,
    distance_power: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    development_designs = [int(design) for design in development_designs]
    for fold, held_out in enumerate(development_designs):
        fold_train = [design for design in development_designs if design != held_out]
        model, n_pairs = fit_residual_model(
            fold_train,
            targets,
            local_features,
            location_features,
            pair_features,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed + fold,
            n_jobs=n_jobs,
        )
        prediction = predict_design(
            model,
            held_out,
            fold_train,
            targets,
            local_features,
            location_features,
            pair_features,
            n_neighbors=n_neighbors,
            distance_power=distance_power,
        )
        logger.info(
            "CV design %d | %d pair examples | neighbors %s | retrieval R2 %.4f | residual R2 %.4f",
            held_out,
            n_pairs,
            prediction["neighbors"].tolist(),
            metrics(targets[held_out], prediction["retrieval"]).r2,
            metrics(targets[held_out], prediction["residual"]).r2,
        )
        for location in range(targets.shape[1]):
            rows.append(
                {
                    "design": held_out,
                    "loc": location + 1,
                    "hic_true": targets[held_out, location],
                    "hic_retrieval": prediction["retrieval"][location],
                    "hic_residual": prediction["residual"][location],
                    "nearest_design": int(prediction["neighbors"][0]),
                    "nearest_geometry_distance": float(prediction["distances"][0]),
                }
            )
    return pd.DataFrame(rows)


def prediction_frame(
    manifest: pd.DataFrame,
    design: int,
    target: np.ndarray,
    prediction: Mapping[str, object],
    alpha: float,
) -> pd.DataFrame:
    rows = manifest.loc[manifest["design"] == design].sort_values("loc").copy()
    rows["hic_true"] = target
    rows["hic_retrieval"] = prediction["retrieval"]
    rows["hic_residual"] = prediction["residual"]
    rows["hic_prediction"] = geometric_blend(
        prediction["retrieval"], prediction["residual"], alpha
    )
    rows["error"] = rows["hic_prediction"] - rows["hic_true"]
    rows["abs_error"] = np.abs(rows["error"])
    rows["nearest_design"] = int(prediction["neighbors"][0])
    rows["nearest_geometry_distance"] = float(prediction["distances"][0])
    rows["blend_alpha"] = alpha
    return rows


def save_prediction_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = frame["hic_true"].to_numpy()
    prediction = frame["hic_prediction"].to_numpy()
    score = metrics(target, prediction)
    low = float(min(target.min(), prediction.min()))
    high = float(max(target.max(), prediction.max()))
    padding = 0.04 * (high - low)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(target, prediction, alpha=0.8)
    axis.plot([low - padding, high + padding], [low - padding, high + padding], "--")
    axis.set_xlim(low - padding, high + padding)
    axis.set_ylim(low - padding, high + padding)
    axis.set_xlabel("Ground-truth full-resolution HIC15")
    axis.set_ylabel("Predicted HIC15")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.text(
        0.04,
        0.96,
        f"R2 = {score.r2:.4f}\nRMSE = {score.rmse:.2f}\nMAE = {score.mae:.2f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Data/HoodImpact_600_EuroNCAP/manifest_600.csv"),
    )
    parser.add_argument(
        "--impact-grid",
        type=Path,
        default=Path("Data/HoodImpact_600_EuroNCAP/impact_locations_50.csv"),
    )
    parser.add_argument(
        "--inp-dir", type=Path, default=Path("Data/HoodImpact_600_EuroNCAP/inp_files")
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("Data/HoodImpact_600_EuroNCAP/output_history_acc"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/_hic_direct_v5_cache"))
    parser.add_argument("--validation-design", type=int, default=10)
    parser.add_argument("--test-design", type=int, default=11)
    parser.add_argument("--max-window", type=float, default=0.015)
    parser.add_argument("--impactor-nodes", type=int, default=DEFAULT_IMPACTOR_NODES)
    parser.add_argument("--n-estimators", type=int, default=768)
    parser.add_argument("--cv-estimators", type=int, default=192)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "-1")),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("group-cv", "validation"),
        default="group-cv",
        help="Select residual shrinkage by grouped CV or the explicit validation design.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Explicitly evaluate/write the held-out reference test design. "
            "Omit during development or hyperparameter iteration."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.neighbors < 1:
        raise ValueError("neighbors must be positive")
    if args.impactor_nodes < 1:
        raise ValueError("impactor-nodes must be positive")

    manifest = load_manifest(args.manifest, args.impact_grid)
    validate_split_ids(
        int(manifest["design"].nunique()),
        args.validation_design,
        args.test_design,
    )
    preflight_summary = preflight(
        manifest, args.inp_dir, args.history_dir, args.impactor_nodes
    )
    if args.preflight_only:
        print(json.dumps(preflight_summary, indent=2, sort_keys=True))
        return Path(".")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    identity = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:8])
    output_dir = args.output_dir or Path("runs") / f"{stamp}_{identity}_hic_direct_v5"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite or interleave an existing output directory: {output_dir}"
        ) from error
    logger = setup_logger(output_dir)
    logger.info("Starting %s", MODEL_VERSION)
    logger.info("Output directory: %s", output_dir)
    logger.info("Preflight: %s", preflight_summary)

    config = vars(args).copy()
    config.update(
        {
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "numba_available": NUMBA_AVAILABLE,
            "git": git_provenance(),
        }
    )
    write_json(output_dir / "config.json", config)

    all_designs = list(range(int(manifest["design"].nunique())))
    train_designs = [
        design
        for design in all_designs
        if design not in (args.validation_design, args.test_design)
    ]
    development_designs = train_designs + [args.validation_design]
    logger.info(
        "Strict split | train %s | validation %d | held-out reference test %d",
        train_designs,
        args.validation_design,
        args.test_design,
    )

    development_manifest = manifest.loc[
        manifest["design"] != args.test_design
    ].copy()
    target_table, target_fingerprint = load_or_compute_targets(
        development_manifest,
        args.manifest,
        args.history_dir,
        args.cache_dir,
        args.max_window,
        args.rebuild_cache,
        logger,
    )
    n_locations = int(manifest.groupby("design").size().iloc[0])
    targets = target_matrix(
        target_table,
        n_designs=len(all_designs),
        n_locations=n_locations,
        required_designs=development_designs,
    )
    target_table["split"] = "train"
    target_table.loc[target_table["design"] == args.validation_design, "split"] = "validation"
    target_table.to_csv(output_dir / "canonical_hic15_targets.csv", index=False)
    logger.info(
        "Development target range %.2f..%.2f (mean %.2f); test targets not computed",
        float(target_table["hic15"].min()),
        float(target_table["hic15"].max()),
        float(target_table["hic15"].mean()),
    )

    (
        local_features,
        location_features,
        pair_features,
        model_feature_names,
        geometry_fingerprint,
    ) = load_or_compute_geometry_features(
        manifest,
        args.manifest,
        args.impact_grid,
        args.inp_dir,
        args.cache_dir,
        args.impactor_nodes,
        args.rebuild_cache,
        logger,
    )
    pd.DataFrame(pair_features[:, :, 0]).to_csv(
        output_dir / "geometry_distance_matrix.csv", index_label="query_design"
    )

    validation_model, validation_pairs = fit_residual_model(
        train_designs,
        targets,
        local_features,
        location_features,
        pair_features,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    validation_prediction = predict_design(
        validation_model,
        args.validation_design,
        train_designs,
        targets,
        local_features,
        location_features,
        pair_features,
        n_neighbors=args.neighbors,
        distance_power=args.distance_power,
    )
    validation_alpha, validation_alpha_table = select_blend_alpha(
        targets[args.validation_design],
        validation_prediction["retrieval"],
        validation_prediction["residual"],
    )
    validation_alpha_table.to_csv(output_dir / "validation_blend_search.csv", index=False)
    validation_frame = prediction_frame(
        manifest,
        args.validation_design,
        targets[args.validation_design],
        validation_prediction,
        validation_alpha,
    )
    validation_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    cv_frame = None
    cv_alpha = None
    cv_alpha_table = None
    if args.selection_mode == "group-cv":
        logger.info(
            "Running grouped leave-one-design-out model selection on %s",
            development_designs,
        )
        cv_frame = grouped_cross_validation(
            development_designs,
            targets,
            local_features,
            location_features,
            pair_features,
            n_estimators=args.cv_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            seed=args.seed + 1000,
            n_jobs=args.n_jobs,
            n_neighbors=args.neighbors,
            distance_power=args.distance_power,
            logger=logger,
        )
        cv_alpha, cv_alpha_table = select_blend_alpha(
            cv_frame["hic_true"].to_numpy(),
            cv_frame["hic_retrieval"].to_numpy(),
            cv_frame["hic_residual"].to_numpy(),
        )
        cv_frame["hic_prediction"] = geometric_blend(
            cv_frame["hic_retrieval"].to_numpy(),
            cv_frame["hic_residual"].to_numpy(),
            cv_alpha,
        )
        cv_frame["error"] = cv_frame["hic_prediction"] - cv_frame["hic_true"]
        cv_frame.to_csv(output_dir / "group_cv_predictions.csv", index=False)
        cv_alpha_table.to_csv(output_dir / "group_cv_blend_search.csv", index=False)
        cv_design_rows = []
        for design, group in cv_frame.groupby("design", sort=True):
            for variant, column in (
                ("retrieval", "hic_retrieval"),
                ("residual", "hic_residual"),
                ("blended", "hic_prediction"),
            ):
                cv_design_rows.append(
                    {
                        "design": int(design),
                        "variant": variant,
                        **asdict(metrics(group["hic_true"], group[column])),
                    }
                )
        pd.DataFrame(cv_design_rows).to_csv(
            output_dir / "group_cv_design_metrics.csv", index=False
        )
        selected_alpha = cv_alpha
    else:
        selected_alpha = validation_alpha

    validation_metrics = {
        "pair_examples": validation_pairs,
        "neighbors": validation_prediction["neighbors"],
        "neighbor_distances": validation_prediction["distances"],
        "retrieval": asdict(
            metrics(targets[args.validation_design], validation_prediction["retrieval"])
        ),
        "residual": asdict(
            metrics(targets[args.validation_design], validation_prediction["residual"])
        ),
        "selected_alpha": validation_alpha,
        "blended": asdict(
            metrics(
                targets[args.validation_design],
                geometric_blend(
                    validation_prediction["retrieval"],
                    validation_prediction["residual"],
                    validation_alpha,
                ),
            )
        ),
    }
    model_selection = {
        "mode": args.selection_mode,
        "validation": validation_metrics,
        "group_cv_alpha": cv_alpha,
        "group_cv_metrics": None
        if cv_frame is None
        else asdict(metrics(cv_frame["hic_true"], cv_frame["hic_prediction"])),
        "final_alpha": selected_alpha,
    }
    write_json(output_dir / "model_selection.json", model_selection)
    logger.info(
        "Validation | retrieval R2 %.4f | residual R2 %.4f | alpha %.2f | blended R2 %.4f",
        validation_metrics["retrieval"]["r2"],
        validation_metrics["residual"]["r2"],
        validation_alpha,
        validation_metrics["blended"]["r2"],
    )
    if cv_frame is not None:
        logger.info(
            "Grouped CV | alpha %.2f | pooled R2 %.4f | RMSE %.2f",
            cv_alpha,
            model_selection["group_cv_metrics"]["r2"],
            model_selection["group_cv_metrics"]["rmse"],
        )

    logger.info("Refitting on all non-test designs %s", development_designs)
    final_model, final_pairs = fit_residual_model(
        development_designs,
        targets,
        local_features,
        location_features,
        pair_features,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed + 1,
        n_jobs=args.n_jobs,
    )
    importance = pd.DataFrame(
        {
            "feature": model_feature_names,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    atomic_joblib_dump(
        {
            "model_version": MODEL_VERSION,
            "model": final_model,
            "blend_alpha": selected_alpha,
            "feature_names": model_feature_names,
            "source_designs": development_designs,
            "source_hic_profiles": targets[development_designs],
            "target_fingerprint": target_fingerprint,
            "geometry_fingerprint": geometry_fingerprint,
            "config": config,
        },
        output_dir / "hic_direct_v5_model.joblib",
    )

    if not args.evaluate_test:
        logger.info(
            "Final model frozen without computing test targets. "
            "Pass --evaluate-test only after model selection to produce reference-test metrics."
        )
        logger.info("Finished. Outputs: %s", output_dir)
        close_logger(logger)
        return output_dir

    logger.info(
        "Model selection and final fit are complete; reading held-out reference targets now"
    )
    test_manifest = manifest.loc[manifest["design"] == args.test_design].copy()
    test_target_table, test_target_fingerprint = load_or_compute_targets(
        test_manifest,
        args.manifest,
        args.history_dir,
        args.cache_dir,
        args.max_window,
        args.rebuild_cache,
        logger,
    )
    test_matrix = target_matrix(
        test_target_table,
        n_designs=len(all_designs),
        n_locations=n_locations,
        required_designs=[args.test_design],
    )
    targets[args.test_design] = test_matrix[args.test_design]
    test_target_table["split"] = "test"
    pd.concat([target_table, test_target_table], ignore_index=True).sort_values(
        ["run"]
    ).to_csv(output_dir / "canonical_hic15_targets.csv", index=False)

    test_prediction = predict_design(
        final_model,
        args.test_design,
        development_designs,
        targets,
        local_features,
        location_features,
        pair_features,
        n_neighbors=args.neighbors,
        distance_power=args.distance_power,
    )
    test_frame = prediction_frame(
        manifest,
        args.test_design,
        targets[args.test_design],
        test_prediction,
        selected_alpha,
    )
    test_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    test_metric_sets = {
        "retrieval": asdict(metrics(test_frame["hic_true"], test_frame["hic_retrieval"])),
        "residual": asdict(metrics(test_frame["hic_true"], test_frame["hic_residual"])),
        "blended": asdict(metrics(test_frame["hic_true"], test_frame["hic_prediction"])),
    }
    test_summary = {
        "test_design": args.test_design,
        "source_designs": development_designs,
        "pair_examples": final_pairs,
        "neighbors": test_prediction["neighbors"],
        "neighbor_distances": test_prediction["distances"],
        "neighbor_weights": test_prediction["weights"],
        "blend_alpha": selected_alpha,
        "metrics": test_metric_sets,
        "target_definition": {
            "name": "HIC15",
            "history_points": "full resolution (no temporal subsampling)",
            "integration": "trapezoidal",
            "windows": "all positive-duration windows <= max_window_s",
            "max_window_s": args.max_window,
        },
        "development_target_fingerprint": target_fingerprint,
        "test_target_fingerprint": test_target_fingerprint,
        "geometry_fingerprint": geometry_fingerprint,
    }
    write_json(output_dir / "test_metrics.json", test_summary)

    save_prediction_plot(
        test_frame,
        output_dir / "test_predicted_vs_true_hic.png",
        f"Direct HIC surrogate | held-out design {args.test_design}",
    )

    blended = test_metric_sets["blended"]
    logger.info(
        "HELD-OUT REFERENCE TEST | design %d | neighbor(s) %s | alpha %.2f | R2 %.4f | RMSE %.2f | MAE %.2f",
        args.test_design,
        test_prediction["neighbors"].tolist(),
        selected_alpha,
        blended["r2"],
        blended["rmse"],
        blended["mae"],
    )
    logger.info("Finished. Outputs: %s", output_dir)
    close_logger(logger)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
