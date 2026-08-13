"""GPU multi-scale geometry CNN for the 660-sample hood-impact corpus.

This experiment is intentionally self-contained and does not import the legacy
``temporal_deeponet.py`` or ``utils/utils.py`` training paths.  It predicts
canonical full-resolution HIC15 directly while using the complete acceleration
history as an auxiliary task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from abaqus_scripts.inp_geom import Deck, HOOD_OUTER_ELSET


MODEL_VERSION = "gpu-multiscale-hic-v6.0"
PREPROCESS_VERSION = "impact-raster-v5-inner-connectors"
EXPECTED_DESIGNS = 12
INDUSTRY_LOCATIONS = 5
EURONCAP_LOCATIONS = 50
EXPECTED_SAMPLES = EXPECTED_DESIGNS * (INDUSTRY_LOCATIONS + EURONCAP_LOCATIONS)
EXPECTED_TIME_POINTS = 1000
HEADFORM_RADIUS_MM = 82.5
HOOD_INNER_ELSET = "Hood_Inner-1-2"
GRID_SIZE = 48
HALF_WIDTHS_MM = (175.0, 400.0, 1800.0)
MAP_CHANNELS = (
    "occupancy",
    "log_node_count",
    "z_min_relative",
    "z_max_relative",
    "z_mean_relative",
    "outer_z_mean_relative",
    "outer_occupancy",
    "inner_occupancy",
    "inner_log_node_count",
    "inner_z_min_relative",
    "inner_z_max_relative",
    "inner_z_mean_relative",
    "connector_occupancy",
    "connector_log_node_count",
    "connector_z_mean_relative",
)
SCALAR_FEATURES = (
    "impact_x",
    "impact_y",
    "impact_z",
    "surface_z",
    "headform_surface_gap",
    "surface_slope_x",
    "surface_slope_y",
    "surface_fit_rmse",
    "nearest_outer_xy_distance",
    "bbox_edge_distance",
    "hood_z_span",
)


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    r2: float
    rmse: float
    mae: float
    bias: float


@dataclass
class PreparedData:
    metadata: pd.DataFrame
    maps: np.ndarray
    scalars: np.ndarray
    times: np.ndarray
    accelerations: np.ndarray
    hic15: np.ndarray
    fingerprint: str


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def reserve_output_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        identity = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:8])
        output_dir = Path("runs") / f"{stamp}_{identity}_gpu_multiscale_hic_v6"
    else:
        output_dir = requested
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}") from error
    return output_dir


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("gpu_multiscale_hic_v6")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
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
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result.update(commit=None, branch=None, dirty=None)
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> RegressionMetrics:
    target = np.asarray(target, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    if len(target) != len(prediction) or len(target) == 0:
        raise ValueError("target and prediction must be non-empty and have equal length")
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator <= 0.0 else 1.0 - float(np.sum(error**2)) / denominator
    return RegressionMetrics(
        n=len(target),
        r2=r2,
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
    )


def hic15_numpy(
    time_s: np.ndarray,
    acceleration_g: np.ndarray,
    max_window_s: float = 0.015,
) -> float:
    """Exhaustive trapezoidal HIC search over every admissible time window."""

    time_s = np.asarray(time_s, dtype=np.float64)
    acceleration_g = np.asarray(acceleration_g, dtype=np.float64)
    if time_s.ndim != 1 or acceleration_g.ndim != 1 or len(time_s) != len(acceleration_g):
        raise ValueError("time and acceleration must be same-length 1-D arrays")
    if len(time_s) < 2 or not np.isfinite(time_s).all() or not np.isfinite(acceleration_g).all():
        raise ValueError("HIC inputs must contain at least two finite points")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not math.isfinite(max_window_s) or max_window_s <= 0.0:
        raise ValueError("max_window_s must be finite and positive")

    integral = np.zeros(len(time_s), dtype=np.float64)
    integral[1:] = np.cumsum(
        0.5 * (acceleration_g[1:] + acceleration_g[:-1]) * np.diff(time_s)
    )
    best = 0.0
    for start in range(len(time_s) - 1):
        stop = int(np.searchsorted(time_s, time_s[start] + max_window_s, side="right"))
        if stop <= start + 1:
            continue
        ends = np.arange(start + 1, stop)
        duration = time_s[ends] - time_s[start]
        average = (integral[ends] - integral[start]) / duration
        score = duration * np.maximum(average, 0.0) ** 2.5
        best = max(best, float(score.max()))
    return best


def hic15_batch(times: np.ndarray, accelerations: np.ndarray, max_window_s: float) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    accelerations = np.asarray(accelerations, dtype=np.float64)
    if times.shape != accelerations.shape or times.ndim != 2:
        raise ValueError("batched HIC inputs must be same-shape 2-D arrays")
    return np.asarray(
        [hic15_numpy(t, a, max_window_s) for t, a in zip(times, accelerations)],
        dtype=np.float64,
    )


def content_fingerprint(paths: Iterable[Path], extra: str) -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Mapping, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def history_path(root: Path, source: str, source_run: int) -> Path:
    folder = (
        root / "HoodImpact_60_IndustryLike" / "output_history_acc"
        if source == "industry5"
        else root / "HoodImpact_600_EuroNCAP" / "output_history_acc"
    )
    return folder / f"HoodImpact_{source_run}_SAE1000_interp1000.csv"


def build_metadata(data_root: Path) -> pd.DataFrame:
    industry_coords_path = data_root / "HoodImpact_60_IndustryLike" / "ImpactCoords_60.csv"
    manifest_path = data_root / "HoodImpact_600_EuroNCAP" / "manifest_600.csv"
    industry = pd.read_csv(industry_coords_path)
    industry.columns = [column.strip() for column in industry.columns]
    if len(industry) != EXPECTED_DESIGNS * INDUSTRY_LOCATIONS:
        raise ValueError(f"expected 60 IndustryLike coordinates, found {len(industry)}")
    for column in ("X1", "X2", "X3"):
        industry[column] = pd.to_numeric(industry[column], errors="coerce")
    if not np.isfinite(industry[["X1", "X2", "X3"]].to_numpy(np.float64)).all():
        raise ValueError("IndustryLike coordinates contain non-finite values")
    industry["source"] = "industry5"
    industry["source_run"] = np.arange(1, len(industry) + 1)
    industry["global_run"] = industry["source_run"]
    industry["design"] = (industry["source_run"] - 1) // INDUSTRY_LOCATIONS
    industry["location"] = (industry["source_run"] - 1) % INDUSTRY_LOCATIONS + 1

    euro = pd.read_csv(manifest_path)
    euro.columns = [column.strip() for column in euro.columns]
    required = {"run", "design", "loc", "X1", "X2", "X3"}
    missing = required.difference(euro.columns)
    if missing:
        raise ValueError(f"manifest_600.csv is missing columns {sorted(missing)}")
    numeric = euro[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise ValueError("manifest_600.csv contains non-finite required values")
    euro[list(required)] = numeric
    euro = euro.sort_values("run").reset_index(drop=True)
    if len(euro) != EXPECTED_DESIGNS * EURONCAP_LOCATIONS:
        raise ValueError(f"expected 600 EuroNCAP rows, found {len(euro)}")
    if not np.array_equal(euro["run"].to_numpy(), np.arange(1, len(euro) + 1)):
        raise ValueError("EuroNCAP manifest run IDs must be exactly 1..600")
    for design, group in euro.groupby("design"):
        if int(design) not in range(EXPECTED_DESIGNS) or not np.array_equal(
            np.sort(group["loc"].astype(int).to_numpy()), np.arange(1, 51)
        ):
            raise ValueError(f"invalid EuroNCAP location mapping for design {design}")
    euro = euro.rename(columns={"run": "source_run", "loc": "location"})
    euro["source"] = "euroncap50"
    euro["global_run"] = euro["source_run"].astype(int) + 60
    euro["design"] = euro["design"].astype(int)
    euro["location"] = euro["location"].astype(int)

    columns = ["global_run", "source", "source_run", "design", "location", "X1", "X2", "X3"]
    metadata = pd.concat([industry[columns], euro[columns]], ignore_index=True)
    metadata.insert(0, "sample_index", np.arange(len(metadata), dtype=np.int64))
    metadata["sample_key"] = metadata["source"] + ":" + metadata["source_run"].astype(str)
    if len(metadata) != EXPECTED_SAMPLES or metadata["sample_key"].duplicated().any():
        raise ValueError("combined metadata is incomplete or contains duplicate sample keys")
    return metadata


def connector_node_ids(deck_path: Path) -> np.ndarray:
    """Return node IDs referenced by Abaqus ``CONN3D2`` fastener elements."""

    node_ids = set()
    in_connector_block = False
    with deck_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("**"):
                continue
            if stripped.startswith("*"):
                upper = stripped.upper().replace(" ", "")
                in_connector_block = upper.startswith("*ELEMENT") and "TYPE=CONN3D2" in upper
                continue
            if in_connector_block:
                values = [int(value) for value in stripped.split(",") if value.strip()]
                if len(values) != 3:
                    raise ValueError(f"invalid CONN3D2 record in {deck_path}: {stripped}")
                node_ids.update(values[1:])
    return np.asarray(sorted(node_ids), dtype=np.int64)


def shell_geometry(
    deck_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return non-headform shell, outer/inner panels, and connector geometry."""

    deck = Deck(str(deck_path))
    if (
        deck.rigid_ref is None
        or HOOD_OUTER_ELSET not in deck.elsets
        or HOOD_INNER_ELSET not in deck.elsets
    ):
        raise ValueError(f"{deck_path} lacks rigid-headform or hood-panel metadata")
    impactor_ids = deck.impactor_node_ids()
    if len(impactor_ids) != 286:
        raise ValueError(f"expected 286 rigid-headform nodes in {deck_path}, found {len(impactor_ids)}")
    shell_ids = set()
    for connectivity in deck.elems.values():
        shell_ids.update(connectivity)
    hood_ids = np.asarray(sorted(shell_ids.difference(impactor_ids)), dtype=np.int64)
    outer_ids = np.asarray(
        sorted(deck.elset_nodes(HOOD_OUTER_ELSET).difference(impactor_ids)), dtype=np.int64
    )
    inner_ids = np.asarray(
        sorted(deck.elset_nodes(HOOD_INNER_ELSET).difference(impactor_ids)), dtype=np.int64
    )
    hood_xyz = np.asarray([deck.nodes[int(node)] for node in hood_ids], dtype=np.float64)
    outer_xyz = np.asarray([deck.nodes[int(node)] for node in outer_ids], dtype=np.float64)
    inner_xyz = np.asarray([deck.nodes[int(node)] for node in inner_ids], dtype=np.float64)
    connector_ids = connector_node_ids(deck_path)
    connector_ids = np.asarray(
        sorted(set(connector_ids).difference(impactor_ids)), dtype=np.int64
    )
    missing_connectors = set(connector_ids).difference(deck.nodes)
    if missing_connectors:
        raise ValueError(
            f"connector nodes are absent from the node block in {deck_path}: "
            f"{sorted(missing_connectors)[:5]}"
        )
    connector_xyz = np.asarray(
        [deck.nodes[int(node)] for node in connector_ids], dtype=np.float64
    )
    if (
        len(hood_xyz) < 30000
        or len(outer_xyz) < 10000
        or len(inner_xyz) < 10000
        or len(connector_xyz) < 500
    ):
        raise ValueError(f"unexpectedly small hood geometry in {deck_path}")
    return hood_ids, hood_xyz, outer_xyz, inner_xyz, connector_xyz


def representative_deck(data_root: Path, design: int, source: str = "industry5") -> Path:
    if source == "industry5":
        run = design * INDUSTRY_LOCATIONS + 1
        folder = data_root / "HoodImpact_60_IndustryLike" / "inp_files"
    else:
        run = design * EURONCAP_LOCATIONS + 1
        folder = data_root / "HoodImpact_600_EuroNCAP" / "inp_files"
    return folder / f"HoodImpact_{run}.inp"


def preflight(data_root: Path) -> Dict[str, object]:
    metadata = build_metadata(data_root)
    expected_counts = {
        "industry_inp": (data_root / "HoodImpact_60_IndustryLike" / "inp_files", 60, "*.inp"),
        "industry_history": (
            data_root / "HoodImpact_60_IndustryLike" / "output_history_acc",
            60,
            "*.csv",
        ),
        "euroncap_inp": (data_root / "HoodImpact_600_EuroNCAP" / "inp_files", 600, "*.inp"),
        "euroncap_history": (
            data_root / "HoodImpact_600_EuroNCAP" / "output_history_acc",
            600,
            "*.csv",
        ),
    }
    counts = {}
    for name, (folder, expected, pattern) in expected_counts.items():
        actual = len(list(folder.glob(pattern)))
        counts[name] = actual
        if actual != expected:
            raise FileNotFoundError(f"expected {expected} files in {folder}, found {actual}")

    hood_counts: Dict[int, int] = {}
    outer_counts: Dict[int, int] = {}
    inner_counts: Dict[int, int] = {}
    connector_counts: Dict[int, int] = {}
    for design in range(EXPECTED_DESIGNS):
        ids_60, xyz_60, outer_60, inner_60, connector_60 = shell_geometry(
            representative_deck(data_root, design)
        )
        ids_600, xyz_600, _, inner_600, connector_600 = shell_geometry(
            representative_deck(data_root, design, source="euroncap50")
        )
        if (
            not np.array_equal(ids_60, ids_600)
            or not np.array_equal(xyz_60, xyz_600)
            or not np.array_equal(inner_60, inner_600)
            or not np.array_equal(connector_60, connector_600)
        ):
            raise ValueError(f"source geometries disagree for design {design}")
        hood_counts[design] = len(xyz_60)
        outer_counts[design] = len(outer_60)
        inner_counts[design] = len(inner_60)
        connector_counts[design] = len(connector_60)

    for row in (metadata.iloc[0], metadata.iloc[-1]):
        frame = pd.read_csv(
            history_path(data_root, str(row["source"]), int(row["source_run"])), nrows=3
        )
        if not {"Time", "A(in g)"}.issubset(frame.columns):
            raise ValueError(f"history {row['sample_key']} lacks Time/A(in g) columns")
    return {
        "samples": len(metadata),
        "designs": int(metadata["design"].nunique()),
        "samples_per_design": metadata.groupby("design").size().astype(int).to_dict(),
        "file_counts": counts,
        "hood_shell_node_counts": hood_counts,
        "outer_skin_node_counts": outer_counts,
        "inner_panel_node_counts": inner_counts,
        "connector_node_counts": connector_counts,
    }


def fit_local_surface(outer_xyz: np.ndarray, impact_xy: np.ndarray, neighbors: int = 24):
    distances = np.linalg.norm(outer_xyz[:, :2] - impact_xy[None, :], axis=1)
    count = min(neighbors, len(outer_xyz))
    indices = np.argpartition(distances, count - 1)[:count]
    delta = outer_xyz[indices, :2] - impact_xy[None, :]
    design = np.column_stack([delta[:, 0], delta[:, 1], np.ones(count)])
    weights = 1.0 / (distances[indices] + 2.0)
    coefficient = np.linalg.lstsq(
        design * weights[:, None], outer_xyz[indices, 2] * weights, rcond=None
    )[0]
    fitted = design @ coefficient
    rmse = float(np.sqrt(np.mean((fitted - outer_xyz[indices, 2]) ** 2)))
    return float(coefficient[2]), float(coefficient[0]), float(coefficient[1]), rmse, float(distances.min())


def _aggregate_bins(
    x_index: np.ndarray,
    y_index: np.ndarray,
    values: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = y_index * grid_size + x_index
    size = grid_size * grid_size
    count = np.bincount(flat, minlength=size).astype(np.float64)
    total = np.bincount(flat, weights=values, minlength=size)
    minimum = np.full(size, np.inf, dtype=np.float64)
    maximum = np.full(size, -np.inf, dtype=np.float64)
    np.minimum.at(minimum, flat, values)
    np.maximum.at(maximum, flat, values)
    occupied = count > 0
    mean = np.zeros(size, dtype=np.float64)
    mean[occupied] = total[occupied] / count[occupied]
    minimum[~occupied] = 0.0
    maximum[~occupied] = 0.0
    shape = (grid_size, grid_size)
    return count.reshape(shape), minimum.reshape(shape), maximum.reshape(shape), mean.reshape(shape)


def rasterize_geometry(
    hood_xyz: np.ndarray,
    outer_xyz: np.ndarray,
    inner_xyz: np.ndarray,
    connector_xyz: np.ndarray,
    impact_xyz: np.ndarray,
    grid_size: int = GRID_SIZE,
    half_widths_mm: Sequence[float] = HALF_WIDTHS_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    impact_xyz = np.asarray(impact_xyz, dtype=np.float64)
    surface_z, slope_x, slope_y, fit_rmse, nearest = fit_local_surface(
        outer_xyz, impact_xyz[:2]
    )
    relative_xy = hood_xyz[:, :2] - impact_xyz[None, :2]
    relative_z = np.clip(hood_xyz[:, 2] - surface_z, -600.0, 300.0)
    outer_xy = outer_xyz[:, :2] - impact_xyz[None, :2]
    outer_z = np.clip(outer_xyz[:, 2] - surface_z, -200.0, 200.0)
    inner_xy = inner_xyz[:, :2] - impact_xyz[None, :2]
    inner_z = np.clip(inner_xyz[:, 2] - surface_z, -600.0, 300.0)
    connector_xy = connector_xyz[:, :2] - impact_xyz[None, :2]
    connector_z = np.clip(connector_xyz[:, 2] - surface_z, -600.0, 300.0)
    maps = np.zeros(
        (len(half_widths_mm), len(MAP_CHANNELS), grid_size, grid_size), dtype=np.float32
    )

    for scale_index, half_width in enumerate(half_widths_mm):
        inside = np.max(np.abs(relative_xy), axis=1) < half_width
        normalized = (relative_xy[inside] + half_width) / (2.0 * half_width)
        indices = np.clip((normalized * grid_size).astype(np.int64), 0, grid_size - 1)
        count, z_min, z_max, z_mean = _aggregate_bins(
            indices[:, 0], indices[:, 1], relative_z[inside], grid_size
        )

        outer_inside = np.max(np.abs(outer_xy), axis=1) < half_width
        outer_normalized = (outer_xy[outer_inside] + half_width) / (2.0 * half_width)
        outer_indices = np.clip(
            (outer_normalized * grid_size).astype(np.int64), 0, grid_size - 1
        )
        outer_count, _, _, outer_mean = _aggregate_bins(
            outer_indices[:, 0], outer_indices[:, 1], outer_z[outer_inside], grid_size
        )

        inner_inside = np.max(np.abs(inner_xy), axis=1) < half_width
        inner_normalized = (inner_xy[inner_inside] + half_width) / (2.0 * half_width)
        inner_indices = np.clip(
            (inner_normalized * grid_size).astype(np.int64), 0, grid_size - 1
        )
        inner_count, inner_min, inner_max, inner_mean = _aggregate_bins(
            inner_indices[:, 0], inner_indices[:, 1], inner_z[inner_inside], grid_size
        )

        connector_inside = np.max(np.abs(connector_xy), axis=1) < half_width
        connector_normalized = (
            connector_xy[connector_inside] + half_width
        ) / (2.0 * half_width)
        connector_indices = np.clip(
            (connector_normalized * grid_size).astype(np.int64), 0, grid_size - 1
        )
        connector_count, _, _, connector_mean = _aggregate_bins(
            connector_indices[:, 0],
            connector_indices[:, 1],
            connector_z[connector_inside],
            grid_size,
        )
        maps[scale_index] = np.stack(
            [
                count > 0,
                np.log1p(count),
                z_min,
                z_max,
                z_mean,
                outer_mean,
                outer_count > 0,
                inner_count > 0,
                np.log1p(inner_count),
                inner_min,
                inner_max,
                inner_mean,
                connector_count > 0,
                np.log1p(connector_count),
                connector_mean,
            ]
        ).astype(np.float32)

    minimum = hood_xyz.min(axis=0)
    maximum = hood_xyz.max(axis=0)
    edge_distance = min(
        impact_xyz[0] - minimum[0],
        maximum[0] - impact_xyz[0],
        impact_xyz[1] - minimum[1],
        maximum[1] - impact_xyz[1],
    )
    scalars = np.asarray(
        [
            impact_xyz[0],
            impact_xyz[1],
            impact_xyz[2],
            surface_z,
            impact_xyz[2] - HEADFORM_RADIUS_MM - surface_z,
            slope_x,
            slope_y,
            fit_rmse,
            nearest,
            edge_distance,
            maximum[2] - minimum[2],
        ],
        dtype=np.float32,
    )
    return maps, scalars


def dataset_fingerprint(data_root: Path, metadata: pd.DataFrame, max_window_s: float) -> str:
    paths: List[Path] = [
        data_root / "HoodImpact_60_IndustryLike" / "ImpactCoords_60.csv",
        data_root / "HoodImpact_600_EuroNCAP" / "manifest_600.csv",
    ]
    paths.extend(representative_deck(data_root, design) for design in range(EXPECTED_DESIGNS))
    paths.extend(
        history_path(data_root, str(row.source), int(row.source_run))
        for row in metadata.itertuples(index=False)
    )
    extra = (
        f"{MODEL_VERSION}|{PREPROCESS_VERSION}|grid={GRID_SIZE}|"
        f"widths={HALF_WIDTHS_MM}|hic={max_window_s:.9g}"
    )
    return content_fingerprint(paths, extra)


def load_or_prepare_data(
    data_root: Path,
    cache_dir: Path,
    max_window_s: float,
    rebuild_cache: bool,
    logger: logging.Logger,
) -> PreparedData:
    metadata = build_metadata(data_root)
    fingerprint = dataset_fingerprint(data_root, metadata, max_window_s)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dataset_{fingerprint[:16]}.npz"
    if cache_path.exists() and not rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        logger.info("Loaded preprocessed raster/history cache from %s", cache_path)
        return PreparedData(
            metadata=metadata,
            maps=cached["maps"],
            scalars=cached["scalars"],
            times=cached["times"],
            accelerations=cached["accelerations"],
            hic15=cached["hic15"],
            fingerprint=fingerprint,
        )

    logger.info("Parsing one validated static hood geometry per design")
    geometries = {}
    for design in range(EXPECTED_DESIGNS):
        _, hood_xyz, outer_xyz, inner_xyz, connector_xyz = shell_geometry(
            representative_deck(data_root, design)
        )
        geometries[design] = (hood_xyz, outer_xyz, inner_xyz, connector_xyz)
        logger.info(
            "Design %d geometry: %d shell nodes, %d outer-skin nodes, "
            "%d inner-panel nodes, "
            "%d connector nodes",
            design,
            len(hood_xyz),
            len(outer_xyz),
            len(inner_xyz),
            len(connector_xyz),
        )

    maps = np.empty(
        (len(metadata), len(HALF_WIDTHS_MM), len(MAP_CHANNELS), GRID_SIZE, GRID_SIZE),
        dtype=np.float32,
    )
    scalars = np.empty((len(metadata), len(SCALAR_FEATURES)), dtype=np.float32)
    times = np.empty((len(metadata), EXPECTED_TIME_POINTS), dtype=np.float32)
    accelerations = np.empty_like(times)
    logger.info("Rasterizing %d design/location samples and reading full histories", len(metadata))
    for sample_index, row in enumerate(metadata.itertuples(index=False)):
        hood_xyz, outer_xyz, inner_xyz, connector_xyz = geometries[int(row.design)]
        maps[sample_index], scalars[sample_index] = rasterize_geometry(
            hood_xyz,
            outer_xyz,
            inner_xyz,
            connector_xyz,
            np.asarray([row.X1, row.X2, row.X3], dtype=np.float64),
        )
        history = pd.read_csv(
            history_path(data_root, str(row.source), int(row.source_run)),
            usecols=["Time", "A(in g)"],
        )
        if len(history) != EXPECTED_TIME_POINTS:
            raise ValueError(f"history {row.sample_key} has {len(history)} points, expected 1000")
        times[sample_index] = history["Time"].to_numpy(np.float32)
        accelerations[sample_index] = history["A(in g)"].to_numpy(np.float32)
        if sample_index % 100 == 99:
            logger.info("Prepared %d/%d samples", sample_index + 1, len(metadata))

    if not all(np.isfinite(array).all() for array in (maps, scalars, times, accelerations)):
        raise ValueError("preprocessed arrays contain NaN or Inf")
    if np.any(accelerations < -1e-3) or np.any(np.diff(times, axis=1) <= 0.0):
        raise ValueError("histories contain negative resultant acceleration or invalid time grids")
    accelerations = np.maximum(accelerations, 0.0)
    logger.info("Computing canonical full-resolution all-window HIC15 targets")
    hic15 = hic15_batch(times, accelerations, max_window_s).astype(np.float32)
    if not np.isfinite(hic15).all() or np.any(hic15 <= 0.0):
        raise ValueError("canonical HIC15 targets are not finite and positive")
    atomic_npz(
        cache_path,
        maps=maps,
        scalars=scalars,
        times=times,
        accelerations=accelerations,
        hic15=hic15,
    )
    logger.info("Cached prepared dataset at %s", cache_path)
    return PreparedData(metadata, maps, scalars, times, accelerations, hic15, fingerprint)


class TrainingNormalizer:
    """Fold-local normalization fitted only on training-design samples."""

    def __init__(self):
        self.map_mean: Optional[np.ndarray] = None
        self.map_std: Optional[np.ndarray] = None
        self.scalar_mean: Optional[np.ndarray] = None
        self.scalar_std: Optional[np.ndarray] = None
        self.industry_hic_mean: Optional[np.ndarray] = None
        self.euroncap_hic_mean: Optional[np.ndarray] = None
        self.hic_residual_std: Optional[float] = None
        self.accel_log_mean: Optional[float] = None
        self.accel_log_std: Optional[float] = None

    def fit(self, data: PreparedData, train_indices: np.ndarray) -> "TrainingNormalizer":
        train_maps = data.maps[train_indices].astype(np.float64)
        self.map_mean = train_maps.mean(axis=(0, 3, 4), keepdims=True).astype(np.float32)
        self.map_std = train_maps.std(axis=(0, 3, 4), keepdims=True).astype(np.float32)
        self.map_std = np.maximum(self.map_std, 1e-5)

        train_scalars = data.scalars[train_indices].astype(np.float64)
        self.scalar_mean = train_scalars.mean(axis=0, keepdims=True).astype(np.float32)
        self.scalar_std = train_scalars.std(axis=0, keepdims=True).astype(np.float32)
        self.scalar_std = np.maximum(self.scalar_std, 1e-5)

        train_metadata = data.metadata.iloc[train_indices].copy()
        train_metadata["hic15"] = data.hic15[train_indices]
        location_means = train_metadata.groupby(["source", "location"])["hic15"].mean()
        self.industry_hic_mean = np.asarray(
            [location_means.loc[("industry5", location)] for location in range(1, 6)],
            dtype=np.float32,
        )
        self.euroncap_hic_mean = np.asarray(
            [location_means.loc[("euroncap50", location)] for location in range(1, 51)],
            dtype=np.float32,
        )
        baseline = self.hic_baseline(train_metadata)
        residual = data.hic15[train_indices].astype(np.float64) - baseline
        self.hic_residual_std = max(float(residual.std()), 1e-5)

        log_acceleration = np.log1p(data.accelerations[train_indices].astype(np.float64))
        self.accel_log_mean = float(log_acceleration.mean())
        self.accel_log_std = max(float(log_acceleration.std()), 1e-5)
        return self

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.map_mean,
                self.map_std,
                self.scalar_mean,
                self.scalar_std,
                self.industry_hic_mean,
                self.euroncap_hic_mean,
                self.hic_residual_std,
                self.accel_log_mean,
                self.accel_log_std,
            )
        ):
            raise RuntimeError("normalizer is not fitted")

    def transform_maps(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return ((values - self.map_mean) / self.map_std).astype(np.float32)

    def transform_scalars(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return ((values - self.scalar_mean) / self.scalar_std).astype(np.float32)

    def hic_baseline(self, metadata: pd.DataFrame) -> np.ndarray:
        if self.industry_hic_mean is None or self.euroncap_hic_mean is None:
            raise RuntimeError("normalizer HIC baseline is not fitted")
        baseline = np.empty(len(metadata), dtype=np.float64)
        for index, row in enumerate(metadata.itertuples(index=False)):
            if row.source == "industry5":
                table = self.industry_hic_mean
            elif row.source == "euroncap50":
                table = self.euroncap_hic_mean
            else:
                raise ValueError(f"unknown HIC-baseline source: {row.source!r}")
            location = int(row.location)
            if location < 1 or location > len(table):
                raise ValueError(
                    f"location {location} is invalid for HIC-baseline source {row.source!r}"
                )
            baseline[index] = table[location - 1]
        return baseline

    def transform_hic(self, values: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        return (
            (np.asarray(values) - self.hic_baseline(metadata)) / self.hic_residual_std
        ).astype(np.float32)

    def inverse_hic(self, values: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        return (
            np.asarray(values, dtype=np.float64) * self.hic_residual_std
            + self.hic_baseline(metadata)
        )

    def transform_acceleration(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return (
            (np.log1p(np.maximum(values, 0.0)) - self.accel_log_mean) / self.accel_log_std
        ).astype(np.float32)

    def inverse_acceleration(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        log_acceleration = np.asarray(values, dtype=np.float64) * self.accel_log_std
        log_acceleration += self.accel_log_mean
        # The clamp is far above the observed range and only prevents numerical
        # explosions from an untrained auxiliary decoder.
        return np.maximum(np.expm1(np.clip(log_acceleration, 0.0, 12.0)), 0.0)

    def to_dict(self) -> Dict[str, object]:
        self._require_fitted()
        return {
            "map_mean": self.map_mean,
            "map_std": self.map_std,
            "scalar_mean": self.scalar_mean,
            "scalar_std": self.scalar_std,
            "industry_hic_mean": self.industry_hic_mean,
            "euroncap_hic_mean": self.euroncap_hic_mean,
            "hic_residual_std": self.hic_residual_std,
            "accel_log_mean": self.accel_log_mean,
            "accel_log_std": self.accel_log_std,
        }

    def save(self, path: Path) -> None:
        payload = self.to_dict()
        atomic_npz(path, **payload)


class HoodImpactDataset(Dataset):
    def __init__(
        self,
        data: PreparedData,
        indices: Sequence[int],
        normalizer: TrainingNormalizer,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.maps = normalizer.transform_maps(data.maps[self.indices])
        self.scalars = normalizer.transform_scalars(data.scalars[self.indices])
        self.hic = normalizer.transform_hic(
            data.hic15[self.indices], data.metadata.iloc[self.indices]
        )
        self.waveform = normalizer.transform_acceleration(data.accelerations[self.indices])
        raw = data.accelerations[self.indices]
        peak = np.maximum(raw.max(axis=1, keepdims=True), 1e-6)
        self.peak_weight = (1.0 + 4.0 * (raw / peak) ** 1.5).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "maps": torch.from_numpy(self.maps[index]),
            "scalars": torch.from_numpy(self.scalars[index]),
            "hic": torch.tensor(self.hic[index], dtype=torch.float32),
            "waveform": torch.from_numpy(self.waveform[index]),
            "peak_weight": torch.from_numpy(self.peak_weight[index]),
            "sample_index": torch.tensor(self.indices[index], dtype=torch.long),
        }


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            group_norm(channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values + self.block(values))


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, bias=False),
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=2, bias=False),
            group_norm(channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values + self.block(values))


class SharedScaleEncoder(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            group_norm(32),
            nn.SiLU(inplace=True),
            ResidualBlock2d(32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            group_norm(64),
            nn.SiLU(inplace=True),
            ResidualBlock2d(64),
            ResidualBlock2d(64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            group_norm(96),
            nn.SiLU(inplace=True),
            ResidualBlock2d(96),
            ResidualBlock2d(96),
        )
        self.projection = nn.Sequential(
            nn.Linear(192, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stage3(self.stage2(self.stem(values)))
        average = F.adaptive_avg_pool2d(values, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(values, 1).flatten(1)
        return self.projection(torch.cat([average, maximum], dim=1))


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim: int, seed_channels: int = 32, seed_length: int = 125):
        super().__init__()
        self.seed_channels = seed_channels
        self.seed_length = seed_length
        self.seed = nn.Sequential(
            nn.Linear(latent_dim, seed_channels * seed_length),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock1d(seed_channels) for _ in range(4)])
        self.output = nn.Conv1d(seed_channels, 1, 5, padding=2)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.seed(latent).view(-1, self.seed_channels, self.seed_length)
        values = self.blocks[0](values)
        for block in self.blocks[1:]:
            values = F.interpolate(values, scale_factor=2.0, mode="linear", align_corners=False)
            values = block(values)
        return self.output(values).squeeze(1)


class MultiScaleHoodImpactNet(nn.Module):
    """Impact-centered raster CNN with direct-HIC and waveform heads."""

    def __init__(
        self,
        map_channels: int = len(MAP_CHANNELS),
        scalar_dim: int = len(SCALAR_FEATURES),
        num_scales: int = len(HALF_WIDTHS_MM),
        scale_embedding_dim: int = 128,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.scale_encoder = SharedScaleEncoder(map_channels, scale_embedding_dim)
        self.scale_identity = nn.Parameter(torch.zeros(num_scales, scale_embedding_dim))
        nn.init.normal_(self.scale_identity, std=0.02)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        fusion_input = num_scales * scale_embedding_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.hic_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.waveform_decoder = WaveformDecoder(latent_dim)

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if maps.ndim != 5 or maps.shape[1] != self.num_scales:
            raise ValueError("maps must have shape (batch, scales, channels, height, width)")
        batch_size, scales, channels, height, width = maps.shape
        encoded = self.scale_encoder(maps.reshape(batch_size * scales, channels, height, width))
        encoded = encoded.view(batch_size, scales, -1)
        encoded = encoded + self.scale_identity.unsqueeze(0)
        scalar_features = self.scalar_encoder(scalars)
        latent = self.fusion(torch.cat([encoded.flatten(1), scalar_features], dim=1))
        hic = self.hic_head(latent).squeeze(1)
        waveform = self.waveform_decoder(latent)
        return hic, waveform


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_indices(
    metadata: pd.DataFrame,
    validation_design: int,
    test_design: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if validation_design == test_design:
        raise ValueError("validation and test design must differ")
    for name, design in (("validation", validation_design), ("test", test_design)):
        if design not in range(EXPECTED_DESIGNS):
            raise ValueError(f"{name} design must be in [0, 11], got {design}")
    design = metadata["design"].to_numpy(np.int64)
    train = np.flatnonzero((design != validation_design) & (design != test_design))
    validation = np.flatnonzero(design == validation_design)
    test = np.flatnonzero(design == test_design)
    expected = (550, 55, 55)
    if (len(train), len(validation), len(test)) != expected:
        raise ValueError(
            f"expected design split sizes {expected}, found {(len(train), len(validation), len(test))}"
        )
    return train, validation, test


def make_loader(
    data: PreparedData,
    indices: np.ndarray,
    normalizer: TrainingNormalizer,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(
        HoodImpactDataset(data, indices, normalizer),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    wave_loss_weight: float,
    peak_loss_weight: float,
    map_noise_std: float,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "hic_loss": 0.0, "wave_loss": 0.0, "peak_loss": 0.0}
    samples = 0
    dtype = autocast_dtype(device)
    for batch in loader:
        maps = batch["maps"].to(device, non_blocking=True)
        scalars = batch["scalars"].to(device, non_blocking=True)
        target_hic = batch["hic"].to(device, non_blocking=True)
        target_wave = batch["waveform"].to(device, non_blocking=True)
        peak_weight = batch["peak_weight"].to(device, non_blocking=True)
        if map_noise_std > 0.0:
            maps = maps + torch.randn_like(maps) * map_noise_std

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            predicted_hic, predicted_wave = model(maps, scalars)
            hic_loss = F.mse_loss(predicted_hic, target_hic)
            wave_loss = F.smooth_l1_loss(predicted_wave, target_wave, beta=0.5)
            peak_loss = torch.mean(peak_weight * (predicted_wave - target_wave) ** 2)
            loss = hic_loss + wave_loss_weight * wave_loss + peak_loss_weight * peak_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = maps.shape[0]
        samples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["hic_loss"] += float(hic_loss.detach()) * batch_size
        totals["wave_loss"] += float(wave_loss.detach()) * batch_size
        totals["peak_loss"] += float(peak_loss.detach()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    data: PreparedData,
    normalizer: TrainingNormalizer,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, np.ndarray]:
    model.eval()
    sample_indices: List[np.ndarray] = []
    predicted_hic_z: List[np.ndarray] = []
    predicted_wave_z: List[np.ndarray] = []
    dtype = autocast_dtype(device)
    for batch in loader:
        maps = batch["maps"].to(device, non_blocking=True)
        scalars = batch["scalars"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            hic, waveform = model(maps, scalars)
        sample_indices.append(batch["sample_index"].numpy())
        predicted_hic_z.append(hic.float().cpu().numpy())
        predicted_wave_z.append(waveform.float().cpu().numpy())

    indices = np.concatenate(sample_indices)
    hic = normalizer.inverse_hic(
        np.concatenate(predicted_hic_z), data.metadata.iloc[indices]
    )
    waveform = normalizer.inverse_acceleration(np.concatenate(predicted_wave_z))
    order = np.argsort(indices)
    return {
        "sample_indices": indices[order],
        "hic_direct": hic[order],
        "acceleration": waveform[order],
    }


def basic_validation_metrics(data: PreparedData, prediction: Mapping[str, np.ndarray]) -> Dict[str, object]:
    indices = prediction["sample_indices"]
    hic = asdict(regression_metrics(data.hic15[indices], prediction["hic_direct"]))
    acceleration = asdict(
        regression_metrics(
            data.accelerations[indices].reshape(-1),
            prediction["acceleration"].reshape(-1),
        )
    )
    return {"hic_direct": hic, "acceleration": acceleration}


def detailed_metrics(
    data: PreparedData,
    prediction: Mapping[str, np.ndarray],
    max_window_s: float,
) -> Tuple[Dict[str, object], np.ndarray]:
    indices = prediction["sample_indices"]
    waveform_hic = hic15_batch(
        data.times[indices], prediction["acceleration"], max_window_s
    )
    frame = data.metadata.iloc[indices].reset_index(drop=True)
    metrics: Dict[str, object] = {
        "combined": {
            "hic_direct": asdict(
                regression_metrics(data.hic15[indices], prediction["hic_direct"])
            ),
            "hic_from_waveform": asdict(
                regression_metrics(data.hic15[indices], waveform_hic)
            ),
            "acceleration": asdict(
                regression_metrics(
                    data.accelerations[indices].reshape(-1),
                    prediction["acceleration"].reshape(-1),
                )
            ),
        }
    }
    for source in ("industry5", "euroncap50"):
        mask = frame["source"].to_numpy() == source
        metrics[source] = {
            "hic_direct": asdict(
                regression_metrics(data.hic15[indices][mask], prediction["hic_direct"][mask])
            ),
            "hic_from_waveform": asdict(
                regression_metrics(data.hic15[indices][mask], waveform_hic[mask])
            ),
            "acceleration": asdict(
                regression_metrics(
                    data.accelerations[indices][mask].reshape(-1),
                    prediction["acceleration"][mask].reshape(-1),
                )
            ),
        }
    return metrics, waveform_hic


def location_mean_baseline(
    data: PreparedData,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
) -> np.ndarray:
    train = data.metadata.iloc[train_indices].copy()
    train["hic15"] = data.hic15[train_indices]
    lookup = train.groupby(["source", "location"])["hic15"].mean()
    query = data.metadata.iloc[query_indices]
    return np.asarray(
        [lookup.loc[(row.source, int(row.location))] for row in query.itertuples()],
        dtype=np.float64,
    )


def export_predictions(
    output_dir: Path,
    split_name: str,
    data: PreparedData,
    prediction: Mapping[str, np.ndarray],
    waveform_hic: np.ndarray,
    baseline: np.ndarray,
) -> None:
    indices = prediction["sample_indices"]
    frame = data.metadata.iloc[indices].reset_index(drop=True).copy()
    frame["hic15_true"] = data.hic15[indices]
    frame["hic15_direct_prediction"] = prediction["hic_direct"]
    frame["hic15_waveform_prediction"] = waveform_hic
    frame["hic15_location_mean_baseline"] = baseline
    frame["direct_error"] = frame["hic15_direct_prediction"] - frame["hic15_true"]
    frame["direct_abs_error"] = np.abs(frame["direct_error"])
    frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    atomic_npz(
        output_dir / f"{split_name}_waveforms.npz",
        sample_indices=indices,
        times=data.times[indices],
        acceleration_true=data.accelerations[indices],
        acceleration_prediction=prediction["acceleration"].astype(np.float32),
    )


def save_prediction_plot(
    path: Path,
    target: np.ndarray,
    direct: np.ndarray,
    waveform_hic: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = np.asarray(target)
    lower = float(min(target.min(), direct.min(), waveform_hic.min()))
    upper = float(max(target.max(), direct.max(), waveform_hic.max()))
    padding = 0.04 * max(upper - lower, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for axis, prediction, label in (
        (axes[0], direct, "Direct HIC head"),
        (axes[1], waveform_hic, "HIC from predicted history"),
    ):
        score = regression_metrics(target, prediction)
        axis.scatter(target, prediction, alpha=0.8, edgecolor="white", linewidth=0.4)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "--")
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Ground-truth canonical HIC15")
        axis.set_ylabel("Predicted HIC15")
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.text(
            0.04,
            0.96,
            f"R² = {score.r2:.4f}\nRMSE = {score.rmse:.2f}\nMAE = {score.mae:.2f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_training_plot(path: Path, history: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train composite")
    axes[0].plot(history["epoch"], history["train_hic_loss"], label="train HIC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], history["val_hic_rmse"], label="validation HIC RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HIC")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def cuda_smoke_check(device: torch.device) -> Dict[str, object]:
    model = MultiScaleHoodImpactNet().to(device)
    model.train()
    maps = torch.randn(
        2,
        len(HALF_WIDTHS_MM),
        len(MAP_CHANNELS),
        GRID_SIZE,
        GRID_SIZE,
        device=device,
    )
    scalars = torch.randn(2, len(SCALAR_FEATURES), device=device)
    hic, waveform = model(maps, scalars)
    loss = hic.square().mean() + waveform.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(item).all() for item in gradients):
        raise RuntimeError("CUDA forward/backward smoke check produced non-finite values")
    nonzero = sum(int(torch.count_nonzero(item)) for item in gradients)
    if nonzero == 0:
        raise RuntimeError("CUDA forward/backward smoke check produced no nonzero gradients")
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": model_parameter_count(model),
        "hic_shape": list(hic.shape),
        "waveform_shape": list(waveform.shape),
        "nonzero_gradient_values": nonzero,
    }


def add_baseline_metrics(
    metric_sets: Dict[str, object],
    data: PreparedData,
    indices: np.ndarray,
    baseline: np.ndarray,
) -> None:
    metric_sets["combined"]["location_mean_baseline"] = asdict(
        regression_metrics(data.hic15[indices], baseline)
    )
    frame = data.metadata.iloc[indices].reset_index(drop=True)
    for source in ("industry5", "euroncap50"):
        mask = frame["source"].to_numpy() == source
        metric_sets[source]["location_mean_baseline"] = asdict(
            regression_metrics(data.hic15[indices][mask], baseline[mask])
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("runs/_gpu_multiscale_hic_v6_cache")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validation-design", type=int, default=10)
    parser.add_argument("--test-design", type=int, default=11)
    parser.add_argument("--max-window", type=float, default=0.015)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--wave-loss-weight", type=float, default=0.20)
    parser.add_argument("--peak-loss-weight", type=float, default=0.05)
    parser.add_argument("--map-noise-std", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--refit-development",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after validation selection, fit a fresh model on train+validation "
            "for the selected epoch count"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args(argv)


def validate_training_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        raise ValueError("epochs, min-epochs, and patience must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.wave_loss_weight < 0.0 or args.peak_loss_weight < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if args.map_noise_std < 0.0:
        raise ValueError("map-noise-std must be non-negative")
    if not math.isfinite(args.max_window) or args.max_window <= 0.0:
        raise ValueError("max-window must be finite and positive")


def scheduler_multiplier(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def fit_fixed_epochs(
    data: PreparedData,
    train_indices: np.ndarray,
    normalizer: TrainingNormalizer,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    epochs: int,
    logger: logging.Logger,
) -> Tuple[nn.Module, pd.DataFrame]:
    """Fit a fresh final model without consulting validation or test targets."""

    if epochs < 1:
        raise ValueError("fixed refit epoch count must be positive")
    seed_everything(args.seed)
    loader = make_loader(
        data,
        train_indices,
        normalizer,
        args.batch_size,
        args.workers,
        True,
        device,
    )
    model = MultiScaleHoodImpactNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Preserve the selection run's schedule horizon. Shortening the cosine
    # horizon to ``epochs`` would silently change the chosen training recipe.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: scheduler_multiplier(
            epoch, args.warmup_epochs, args.epochs
        ),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and autocast_dtype(device) == torch.float16,
    )
    rows: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.wave_loss_weight,
            args.peak_loss_weight,
            args.map_noise_std,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "train_hic_loss": metrics["hic_loss"],
            "train_wave_loss": metrics["wave_loss"],
            "train_peak_loss": metrics["peak_loss"],
            "learning_rate": learning_rate,
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        logger.info(
            "Development refit epoch %03d/%03d | loss %.4f (HIC %.4f)",
            epoch,
            epochs,
            metrics["loss"],
            metrics["hic_loss"],
        )
    return model, pd.DataFrame(rows)


def initialize_wandb(args: argparse.Namespace, output_dir: Path, config: Mapping):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("--wandb-project was requested but wandb is not installed") from error
    return wandb.init(
        project=args.wandb_project,
        name=output_dir.name,
        dir=str(output_dir),
        config=dict(config),
    )


def flatten_numeric_metrics(
    payload: Mapping, prefix: str = ""
) -> Dict[str, float]:
    """Flatten nested metric dictionaries into W&B-friendly scalar keys."""

    flattened: Dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_numeric_metrics(value, name))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flattened[name] = float(value)
    return flattened


def safe_wandb_log(
    wandb_run,
    payload: Mapping,
    logger: logging.Logger,
    step: Optional[int] = None,
) -> None:
    """Keep a transient tracking failure from killing an expensive GPU run."""

    if wandb_run is None:
        return
    try:
        wandb_run.log(dict(payload), step=step)
    except Exception as error:  # W&B is observability, not the training result.
        logger.warning("W&B logging failed; continuing training: %s", error)


def safe_wandb_summary_update(
    wandb_run,
    payload: Mapping[str, float],
    logger: logging.Logger,
) -> None:
    if wandb_run is None:
        return
    try:
        for key, value in payload.items():
            wandb_run.summary[key] = value
    except Exception as error:
        logger.warning("W&B summary update failed; continuing: %s", error)


def log_training_diagnostics_to_wandb(
    wandb_run,
    output_dir: Path,
    history: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb

        safe_wandb_log(
            wandb_run,
            {
                "diagnostics/training_curves": wandb.Image(
                    str(output_dir / "training_history.png")
                ),
                "diagnostics/training_history": wandb.Table(dataframe=history),
            },
            logger,
        )
    except Exception as error:
        logger.warning("W&B training-diagnostic upload failed; continuing: %s", error)


def log_evaluation_to_wandb(
    wandb_run,
    output_dir: Path,
    split_name: str,
    metric_sets: Mapping,
    logger: logging.Logger,
) -> None:
    """Upload scalar metrics, the HIC plot, and the prediction table."""

    if wandb_run is None:
        return
    try:
        import wandb

        prefix = f"evaluation/{split_name}"
        metrics = flatten_numeric_metrics(metric_sets, prefix)
        plot_path = output_dir / f"{split_name}_hic_predictions.png"
        predictions_path = output_dir / f"{split_name}_predictions.csv"
        payload: Dict[str, object] = dict(metrics)
        if plot_path.is_file():
            payload[f"{prefix}/hic_plot"] = wandb.Image(str(plot_path))
        if predictions_path.is_file():
            payload[f"{prefix}/predictions"] = wandb.Table(
                dataframe=pd.read_csv(predictions_path)
            )
        safe_wandb_log(wandb_run, payload, logger)
        safe_wandb_summary_update(wandb_run, metrics, logger)
    except Exception as error:
        logger.warning("W&B %s upload failed; continuing: %s", split_name, error)


def log_results_artifact_to_wandb(
    wandb_run,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Version checkpoints, normalizers, reports, tables, and plots together."""

    if wandb_run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(
            name="gpu-multiscale-hic-v6-results",
            type="model",
            metadata={
                "model_version": MODEL_VERSION,
                "run_directory": output_dir.name,
            },
        )
        patterns = (
            "*.json",
            "*.csv",
            "*.png",
            "*.pt",
            "*.npz",
            "model_architecture.txt",
            "train.log",
        )
        files = sorted(
            {path for pattern in patterns for path in output_dir.glob(pattern)}
        )
        for path in files:
            artifact.add_file(str(path), name=path.name)
        wandb_run.log_artifact(artifact, aliases=["latest"])
        logger.info("Queued %d result files as a W&B model artifact", len(files))
    except Exception as error:
        logger.warning("W&B artifact upload failed; local outputs are intact: %s", error)


def run(args: argparse.Namespace) -> Path:
    validate_training_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this GPU experiment but is unavailable")
    split_indices(build_metadata(args.data_root), args.validation_design, args.test_design)
    preflight_summary = preflight(args.data_root)
    device_smoke = None
    if args.require_cuda:
        device_smoke = cuda_smoke_check(device)
    if args.preflight_only:
        print(
            json.dumps(
                {"data": preflight_summary, "device_smoke": device_smoke},
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return Path(".")

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.min_epochs = 1
        args.patience = 2
        args.workers = 0

    output_dir = reserve_output_dir(args.output_dir)
    logger = setup_logger(output_dir)
    wandb_run = None
    try:
        logger.info("Starting %s on %s", MODEL_VERSION, device)
        logger.info("Output directory: %s", output_dir)
        logger.info("Preflight: %s", preflight_summary)
        if device.type == "cuda":
            logger.info(
                "GPU: %s | capability %s | CUDA %s",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_capability(0),
                torch.version.cuda,
            )

        config = vars(args).copy()
        config.update(
            {
                "model_version": MODEL_VERSION,
                "preprocess_version": PREPROCESS_VERSION,
                "grid_size": GRID_SIZE,
                "half_widths_mm": HALF_WIDTHS_MM,
                "map_channels": MAP_CHANNELS,
                "scalar_features": SCALAR_FEATURES,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "device": str(device),
                "git": git_provenance(),
            }
        )
        write_json(output_dir / "config.json", config)
        wandb_run = initialize_wandb(args, output_dir, config)

        data = load_or_prepare_data(
            args.data_root,
            args.cache_dir,
            args.max_window,
            args.rebuild_cache,
            logger,
        )
        train_indices, validation_indices, test_indices = split_indices(
            data.metadata, args.validation_design, args.test_design
        )
        logger.info(
            "Design split | train designs %s (%d) | validation design %d (%d) | "
            "historical reference design %d (%d)",
            sorted(data.metadata.iloc[train_indices]["design"].unique().tolist()),
            len(train_indices),
            args.validation_design,
            len(validation_indices),
            args.test_design,
            len(test_indices),
        )
        normalizer = TrainingNormalizer().fit(data, train_indices)
        normalizer.save(output_dir / "normalization.npz")

        manifest_export = data.metadata.copy()
        manifest_export["split"] = "train"
        manifest_export.loc[validation_indices, "split"] = "validation"
        manifest_export.loc[test_indices, "split"] = "reference_test"
        manifest_export["canonical_hic15"] = data.hic15
        if not args.evaluate_test:
            manifest_export.loc[test_indices, "canonical_hic15"] = np.nan
        manifest_export.to_csv(output_dir / "dataset_manifest.csv", index=False)

        train_loader = make_loader(
            data,
            train_indices,
            normalizer,
            args.batch_size,
            args.workers,
            True,
            device,
        )
        validation_loader = make_loader(
            data,
            validation_indices,
            normalizer,
            args.batch_size,
            args.workers,
            False,
            device,
        )

        # CUDA/preflight checks consume random numbers. Reset immediately before
        # construction so the configured seed fully defines the selection run.
        seed_everything(args.seed)
        model = MultiScaleHoodImpactNet().to(device)
        parameters = model_parameter_count(model)
        logger.info("Trainable parameters: %s", f"{parameters:,}")
        (output_dir / "model_architecture.txt").write_text(
            f"{model}\n\nTrainable parameters: {parameters:,}\n", encoding="utf-8"
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: scheduler_multiplier(
                epoch, args.warmup_epochs, args.epochs
            ),
        )
        amp_enabled = bool(args.amp and device.type == "cuda")
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=amp_enabled and autocast_dtype(device) == torch.float16,
        )
        best_path = output_dir / "best_model.pt"
        best_rmse = float("inf")
        best_epoch = 0
        stale_epochs = 0
        history_rows: List[Dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp_enabled,
                args.wave_loss_weight,
                args.peak_loss_weight,
                args.map_noise_std,
            )
            validation_prediction = predict_loader(
                model, validation_loader, data, normalizer, device, amp_enabled
            )
            validation_metrics = basic_validation_metrics(data, validation_prediction)
            validation_rmse = float(validation_metrics["hic_direct"]["rmse"])
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            elapsed = time.perf_counter() - started
            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_hic_loss": train_metrics["hic_loss"],
                "train_wave_loss": train_metrics["wave_loss"],
                "train_peak_loss": train_metrics["peak_loss"],
                "val_hic_r2": validation_metrics["hic_direct"]["r2"],
                "val_hic_rmse": validation_rmse,
                "val_hic_mae": validation_metrics["hic_direct"]["mae"],
                "val_acceleration_r2": validation_metrics["acceleration"]["r2"],
                "learning_rate": learning_rate,
                "seconds": elapsed,
            }
            history_rows.append(history_row)
            logger.info(
                "Epoch %03d | loss %.4f (HIC %.4f, wave %.4f, peak %.4f) | "
                "val HIC R2 %.4f RMSE %.2f | val accel R2 %.4f | %.1fs",
                epoch,
                train_metrics["loss"],
                train_metrics["hic_loss"],
                train_metrics["wave_loss"],
                train_metrics["peak_loss"],
                validation_metrics["hic_direct"]["r2"],
                validation_rmse,
                validation_metrics["acceleration"]["r2"],
                elapsed,
            )
            safe_wandb_log(
                wandb_run,
                {f"selection/{key}": value for key, value in history_row.items()},
                logger,
                step=epoch,
            )

            if validation_rmse < best_rmse - 1e-6:
                best_rmse = validation_rmse
                best_epoch = epoch
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_version": MODEL_VERSION,
                        "epoch": epoch,
                        "validation_hic_rmse": validation_rmse,
                        "data_fingerprint": data.fingerprint,
                        "training_designs": sorted(
                            data.metadata.iloc[train_indices]["design"].unique().tolist()
                        ),
                        "target": "standardized residual from training-only source/location mean",
                        "normalizer_file": "normalization.npz",
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                logger.info(
                    "Early stopping at epoch %d after %d epochs without HIC-RMSE improvement",
                    epoch,
                    stale_epochs,
                )
                break

        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restored epoch %d selected by validation HIC RMSE %.2f", best_epoch, best_rmse)
        history = pd.DataFrame(history_rows)
        save_training_plot(output_dir / "training_history.png", history)
        log_training_diagnostics_to_wandb(wandb_run, output_dir, history, logger)

        validation_prediction = predict_loader(
            model, validation_loader, data, normalizer, device, amp_enabled
        )
        validation_metric_sets, validation_wave_hic = detailed_metrics(
            data, validation_prediction, args.max_window
        )
        validation_baseline = location_mean_baseline(
            data, train_indices, validation_indices
        )
        add_baseline_metrics(
            validation_metric_sets, data, validation_indices, validation_baseline
        )
        validation_summary = {
            "design": args.validation_design,
            "selected_epoch": best_epoch,
            "metrics": validation_metric_sets,
            "target_definition": {
                "name": "HIC15",
                "history_points": EXPECTED_TIME_POINTS,
                "integration": "trapezoidal",
                "windows": "all positive-duration windows <= max_window_s",
                "max_window_s": args.max_window,
            },
            "data_fingerprint": data.fingerprint,
        }
        write_json(output_dir / "validation_metrics.json", validation_summary)
        export_predictions(
            output_dir,
            "validation",
            data,
            validation_prediction,
            validation_wave_hic,
            validation_baseline,
        )
        save_prediction_plot(
            output_dir / "validation_hic_predictions.png",
            data.hic15[validation_indices],
            validation_prediction["hic_direct"],
            validation_wave_hic,
            f"Validation design {args.validation_design}",
        )
        log_evaluation_to_wandb(
            wandb_run,
            output_dir,
            "validation",
            validation_metric_sets,
            logger,
        )
        logger.info(
            "VALIDATION | direct HIC R2 %.4f RMSE %.2f | waveform HIC R2 %.4f",
            validation_metric_sets["combined"]["hic_direct"]["r2"],
            validation_metric_sets["combined"]["hic_direct"]["rmse"],
            validation_metric_sets["combined"]["hic_from_waveform"]["r2"],
        )

        selection_model = model
        final_model = selection_model
        final_normalizer = normalizer
        final_training_indices = train_indices
        if args.refit_development:
            development_indices = np.sort(
                np.concatenate([train_indices, validation_indices])
            )
            final_training_indices = development_indices
            final_normalizer = TrainingNormalizer().fit(data, development_indices)
            final_normalizer.save(output_dir / "final_normalization.npz")
            logger.info(
                "Refitting a fresh model on development designs %s for %d selected epochs",
                sorted(
                    data.metadata.iloc[development_indices]["design"].unique().tolist()
                ),
                best_epoch,
            )
            final_model, refit_history = fit_fixed_epochs(
                data,
                development_indices,
                final_normalizer,
                args,
                device,
                amp_enabled,
                best_epoch,
                logger,
            )
            refit_history.to_csv(output_dir / "refit_history.csv", index=False)
            atomic_torch_save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "model_version": MODEL_VERSION,
                    "data_fingerprint": data.fingerprint,
                    "epochs": best_epoch,
                    "selection_validation_design": args.validation_design,
                    "training_designs": sorted(
                        data.metadata.iloc[development_indices]["design"].unique().tolist()
                    ),
                    "target": "standardized residual from development-only source/location mean",
                    "normalizer_file": "final_normalization.npz",
                    "seed": args.seed,
                    "scheduler_horizon_epochs": args.epochs,
                },
                output_dir / "final_model.pt",
            )
            logger.info("Development refit complete; saved final_model.pt")

        if args.evaluate_test:
            # Report the selection model as well as the standard train+validation
            # refit. This exposes how much the historically close design 10 adds
            # to the design-11 reference result; neither score is used for tuning.
            selection_test_loader = make_loader(
                data,
                test_indices,
                normalizer,
                args.batch_size,
                args.workers,
                False,
                device,
            )
            selection_test_prediction = predict_loader(
                selection_model,
                selection_test_loader,
                data,
                normalizer,
                device,
                amp_enabled,
            )
            selection_test_metric_sets, selection_test_wave_hic = detailed_metrics(
                data, selection_test_prediction, args.max_window
            )
            selection_test_baseline = location_mean_baseline(
                data, train_indices, test_indices
            )
            add_baseline_metrics(
                selection_test_metric_sets,
                data,
                test_indices,
                selection_test_baseline,
            )
            write_json(
                output_dir / "test_selection_metrics.json",
                {
                    "design": args.test_design,
                    "status": "pre-refit historical reference benchmark",
                    "training_designs": sorted(
                        data.metadata.iloc[train_indices]["design"].unique().tolist()
                    ),
                    "selected_epoch": best_epoch,
                    "metrics": selection_test_metric_sets,
                    "data_fingerprint": data.fingerprint,
                },
            )
            export_predictions(
                output_dir,
                "test_selection",
                data,
                selection_test_prediction,
                selection_test_wave_hic,
                selection_test_baseline,
            )
            save_prediction_plot(
                output_dir / "test_selection_hic_predictions.png",
                data.hic15[test_indices],
                selection_test_prediction["hic_direct"],
                selection_test_wave_hic,
                f"Pre-refit historical reference design {args.test_design}",
            )
            log_evaluation_to_wandb(
                wandb_run,
                output_dir,
                "test_selection",
                selection_test_metric_sets,
                logger,
            )

            test_loader = make_loader(
                data,
                test_indices,
                final_normalizer,
                args.batch_size,
                args.workers,
                False,
                device,
            )
            test_prediction = predict_loader(
                final_model,
                test_loader,
                data,
                final_normalizer,
                device,
                amp_enabled,
            )
            test_metric_sets, test_wave_hic = detailed_metrics(
                data, test_prediction, args.max_window
            )
            test_baseline = location_mean_baseline(
                data, final_training_indices, test_indices
            )
            add_baseline_metrics(test_metric_sets, data, test_indices, test_baseline)
            test_summary = {
                "design": args.test_design,
                "status": "historical reference benchmark; not a statistically untouched test",
                "selected_epoch": best_epoch,
                "refit_development": bool(args.refit_development),
                "training_designs": sorted(
                    data.metadata.iloc[final_training_indices]["design"].unique().tolist()
                ),
                "metrics": test_metric_sets,
                "data_fingerprint": data.fingerprint,
            }
            write_json(output_dir / "test_metrics.json", test_summary)
            export_predictions(
                output_dir,
                "test",
                data,
                test_prediction,
                test_wave_hic,
                test_baseline,
            )
            save_prediction_plot(
                output_dir / "test_hic_predictions.png",
                data.hic15[test_indices],
                test_prediction["hic_direct"],
                test_wave_hic,
                f"Historical reference design {args.test_design}",
            )
            log_evaluation_to_wandb(
                wandb_run,
                output_dir,
                "test",
                test_metric_sets,
                logger,
            )
            logger.info(
                "REFERENCE TEST | direct HIC R2 %.4f RMSE %.2f MAE %.2f | "
                "waveform HIC R2 %.4f | acceleration R2 %.4f",
                test_metric_sets["combined"]["hic_direct"]["r2"],
                test_metric_sets["combined"]["hic_direct"]["rmse"],
                test_metric_sets["combined"]["hic_direct"]["mae"],
                test_metric_sets["combined"]["hic_from_waveform"]["r2"],
                test_metric_sets["combined"]["acceleration"]["r2"],
            )
            logger.info(
                "PRE-REFIT REFERENCE | direct HIC R2 %.4f | final refit used: %s",
                selection_test_metric_sets["combined"]["hic_direct"]["r2"],
                args.refit_development,
            )
        else:
            logger.info("Reference-test evaluation skipped; pass --evaluate-test after model selection")

        if device.type == "cuda":
            logger.info(
                "Peak allocated GPU memory: %.2f GiB",
                torch.cuda.max_memory_allocated(device) / 1024**3,
            )
        safe_wandb_summary_update(
            wandb_run,
            {
                "selection/best_epoch": float(best_epoch),
                "selection/best_validation_hic_rmse": best_rmse,
                "runtime/peak_gpu_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                ),
            },
            logger,
        )
        log_results_artifact_to_wandb(wandb_run, output_dir, logger)
        logger.info("Finished. Outputs: %s", output_dir)
        return output_dir
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as error:
                logger.warning("W&B finish failed; local outputs are intact: %s", error)
        close_logger(logger)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
