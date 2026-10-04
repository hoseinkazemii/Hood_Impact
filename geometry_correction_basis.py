"""Training-only signed geometry measurements for a direct output correction.

This module describes unordered physical XYZ clouds; it does not construct a
graph or use response labels. Anchors are fixed by the supplied training clouds.
At each anchor, the descriptor contains a signed nearest-point offset and a
sorted local distance profile, separately for each supplied structural part.
Exact duplicate points are removed. Different node counts and row orders are
therefore supported, but different mesh densities can still change profiles.
Nearest-point offsets are not material-point correspondence or surface normals.

The descriptor is centered and scaled using training geometries only. Thin SVD
retains every numerically resolved training mode, without an explained-variance
cutoff. A physical noise floor prevents near-constant descriptor dimensions
from being amplified. Coefficient normalization also has a floor, so arbitrarily
small singular modes are not whitened to unit amplitude. Held-out geometries
only project onto the saved basis; reconstruction diagnostics expose geometry
variation that the coefficient branch cannot represent.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


SCHEMA_VERSION = 1


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _cloud(geometry: np.ndarray) -> np.ndarray:
    cloud = np.asarray(geometry, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud):
        raise ValueError("geometry must have nonempty shape (N, 3)")
    if not np.isfinite(cloud).all():
        raise ValueError("geometry contains nonfinite XYZ values")
    return cloud


def _labels(values: np.ndarray | None, count: int) -> np.ndarray:
    if values is None:
        return np.full(count, "all", dtype="U3")
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != count:
        raise ValueError("part_labels must be a one-dimensional label per node")
    if array.dtype.kind not in "iuUS":
        raise ValueError("part_labels must contain integer or string labels")
    labels = array.astype(str)
    if np.any(np.char.str_len(labels) == 0):
        raise ValueError("part_labels cannot contain empty labels")
    return labels


def _parts(geometry: np.ndarray, labels: np.ndarray | None) -> dict[str, np.ndarray]:
    cloud = _cloud(geometry)
    label_array = _labels(labels, len(cloud))
    return {label: np.unique(cloud[label_array == label], axis=0)
            for label in np.unique(label_array)}


def _voxel_anchors(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """One deterministic point per physical voxel, ordered by voxel coordinate."""
    points = np.unique(points, axis=0)
    voxel_float = np.floor(points / spacing_mm)
    if np.any(np.abs(voxel_float) > np.iinfo(np.int64).max // 2):
        raise ValueError("coordinates/anchor_spacing_mm exceed supported voxel range")
    voxels = voxel_float.astype(np.int64)
    centers = (voxels.astype(np.float64) + .5) * spacing_mm
    squared_distance = np.sum((points - centers) ** 2, axis=1)
    # Voxel first, then distance to its center, then XYZ to break exact ties.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], squared_distance,
                        voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    ordered_voxels = voxels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered_voxels[1:] != ordered_voxels[:-1], axis=1)
    return points[order[first]].copy()


def _nearest_features(points: np.ndarray, anchors: np.ndarray, profile_k: int) -> np.ndarray:
    """Resolve nearest-point ties lexicographically, and pad small-cloud profiles."""
    tree = cKDTree(points)
    query_k = min(len(points), max(2, profile_k))
    distances, indices = tree.query(anchors, k=query_k, workers=1)
    distances = np.asarray(distances).reshape(len(anchors), query_k)
    indices = np.asarray(indices).reshape(len(anchors), query_k)
    nearest = indices[:, 0].copy()
    if query_k > 1:
        ties = np.isclose(distances[:, 0], distances[:, 1], rtol=1e-12, atol=1e-12)
        for row in np.flatnonzero(ties):
            radius = distances[row, 0] * (1 + 1e-12) + 1e-12
            candidates = np.asarray(tree.query_ball_point(anchors[row], radius), dtype=np.int64)
            candidate_distances = np.linalg.norm(points[candidates] - anchors[row], axis=1)
            minimum = candidate_distances.min()
            candidates = candidates[np.isclose(candidate_distances, minimum, rtol=1e-12, atol=1e-12)]
            # np.unique sorted points lexicographically, so smallest index wins.
            nearest[row] = candidates.min()
    profile = distances[:, :profile_k]
    if query_k < profile_k:
        profile = np.pad(profile, ((0, 0), (0, profile_k - query_k)), mode="edge")
    return np.concatenate((points[nearest] - anchors, profile), axis=1)


@dataclass
class GeometryCorrectionBasis:
    """A frozen measurement reference and orthogonal training geometry basis."""

    anchors_mm: np.ndarray
    anchor_labels: np.ndarray
    feature_mean_mm: np.ndarray
    feature_scale_mm: np.ndarray
    components: np.ndarray
    coefficient_scale: np.ndarray
    singular_values: np.ndarray
    training_coefficients: np.ndarray
    training_design_ids: list[int]
    config: dict[str, Any]

    @property
    def coefficient_width(self) -> int:
        return int(self.components.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(len(self.anchors_mm))

    @property
    def num_features(self) -> int:
        return int(len(self.feature_mean_mm))

    @classmethod
    def fit(cls, geometries: Sequence[np.ndarray], design_ids: Sequence[int],
            part_labels: Sequence[np.ndarray] | None = None, *,
            anchor_spacing_mm: float = 10., profile_k: int = 8,
            noise_floor_mm: float = .1, rank_rtol: float = 1e-10,
            coefficient_scale_floor: float = 1e-6) -> "GeometryCorrectionBasis":
        """Fit on explicitly supplied training geometries, never held-out clouds.

        Inputs must already exclude headform nodes and use a shared physical
        coordinate frame in millimeters. When part labels are supplied, each
        training and subsequently projected geometry must contain the same
        groups. Input design order is canonicalized by ascending design ID.
        """
        geometries, design_ids = list(geometries), list(design_ids)
        if len(geometries) < 2 or len(geometries) != len(design_ids):
            raise ValueError("fit requires at least two geometries and one design ID per geometry")
        if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 0
               for d in design_ids) or len(set(design_ids)) != len(design_ids):
            raise ValueError("design IDs must be unique nonnegative integers")
        anchor_spacing_mm = _positive(anchor_spacing_mm, "anchor_spacing_mm")
        noise_floor_mm = _positive(noise_floor_mm, "noise_floor_mm")
        rank_rtol = _positive(rank_rtol, "rank_rtol")
        coefficient_scale_floor = _positive(coefficient_scale_floor, "coefficient_scale_floor")
        if rank_rtol >= 1:
            raise ValueError("rank_rtol must be smaller than one")
        if isinstance(profile_k, (bool, np.bool_)) or not isinstance(profile_k, (int, np.integer)) or profile_k < 1:
            raise ValueError("profile_k must be a positive integer")
        profile_k = int(profile_k)
        if part_labels is not None:
            part_labels = list(part_labels)
            if len(part_labels) != len(geometries):
                raise ValueError("part_labels must have one array per training geometry")
        else:
            part_labels = [None] * len(geometries)
        order = np.argsort(design_ids)
        training_ids = [int(design_ids[i]) for i in order]
        prepared = [_parts(geometries[i], part_labels[i]) for i in order]
        groups = sorted(prepared[0])
        if any(sorted(parts) != groups for parts in prepared):
            raise ValueError("all geometries must contain the same part label groups")
        anchors, anchor_groups = [], []
        for group in groups:
            group_anchors = _voxel_anchors(np.concatenate([parts[group] for parts in prepared]), anchor_spacing_mm)
            anchors.append(group_anchors)
            anchor_groups.extend([group] * len(group_anchors))
        anchors_array = np.concatenate(anchors)
        anchor_labels_array = np.asarray(anchor_groups, dtype=str)
        descriptors = np.stack([
            cls._descriptor(parts, anchors_array, anchor_labels_array, profile_k)
            for parts in prepared])
        mean = descriptors.mean(axis=0)
        centered = descriptors - mean
        feature_scale = np.maximum(np.sqrt(np.mean(centered ** 2, axis=0)), noise_floor_mm)
        standardized = centered / feature_scale
        _, singular, components = np.linalg.svd(standardized, full_matrices=False)
        arithmetic_floor = (64 * np.finfo(np.float64).eps
                            * max(1., float(np.max(np.abs(descriptors) / feature_scale)))
                            * np.sqrt(standardized.size))
        threshold = max(float(singular[0]) * rank_rtol,
                        float(singular[0]) * np.finfo(np.float64).eps * max(standardized.shape),
                        arithmetic_floor)
        keep = singular > threshold
        # Centering implies at most N-1 modes. Explicitly exclude arithmetic
        # centering residue if a very small user rtol would otherwise retain it.
        keep[len(geometries) - 1:] = False
        if not keep.any():
            raise ValueError("training geometries have no numerically resolved geometry variation")
        components, singular = components[keep], singular[keep]
        # Resolve SVD's arbitrary sign with each mode's largest loading.
        pivots = np.argmax(np.abs(components), axis=1)
        signs = np.where(components[np.arange(len(components)), pivots] < 0., -1., 1.)
        components *= signs[:, None]
        raw_coefficients = standardized @ components.T
        physical_mode_norm = np.linalg.norm(components * feature_scale[None, :], axis=1)
        mode_noise_floor = noise_floor_mm / physical_mode_norm
        coefficient_scale = np.maximum(np.sqrt(np.mean(raw_coefficients ** 2, axis=0)),
                                       np.maximum(mode_noise_floor, coefficient_scale_floor))
        coefficients = raw_coefficients / coefficient_scale
        config = dict(schema_version=SCHEMA_VERSION, fit_split="train", units="mm",
                      descriptor="part_aware_signed_nearest_xyz_and_sorted_distance_profile",
                      anchor_policy="training_union_one_point_per_physical_voxel",
                      anchor_spacing_mm=anchor_spacing_mm, profile_k=profile_k,
                      noise_floor_mm=noise_floor_mm, rank_rtol=rank_rtol,
                      coefficient_scale_floor=coefficient_scale_floor,
                      coefficient_scaling="training_rms_with_physical_mode_noise_floor",
                      physical_mode_norm_mm=physical_mode_norm.tolist(),
                      mode_coefficient_noise_floor=mode_noise_floor.tolist(),
                      part_labels=groups, part_aware=groups != ["all"],
                      rank_threshold=float(threshold), training_design_ids=training_ids,
                      num_anchors=len(anchors_array), num_features=descriptors.shape[1],
                      coefficient_width=len(components),
                      limitations=["unordered node-cloud descriptors depend on mesh sampling",
                                   "nearest point is not material correspondence",
                                   "out-of-span geometry variation has no correction coefficient"])
        return cls(anchors_array, anchor_labels_array, mean, feature_scale, components,
                   coefficient_scale, singular, coefficients.astype(np.float32), training_ids, config)

    @staticmethod
    def _descriptor(parts: dict[str, np.ndarray], anchors: np.ndarray,
                    anchor_labels: np.ndarray, profile_k: int) -> np.ndarray:
        expected = sorted(np.unique(anchor_labels).tolist())
        if sorted(parts) != expected:
            raise ValueError(f"geometry part groups {sorted(parts)} do not match fitted groups {expected}")
        values = np.empty((len(anchors), 3 + profile_k), dtype=np.float64)
        for group in expected:
            selection = anchor_labels == group
            values[selection] = _nearest_features(parts[group], anchors[selection], profile_k)
        return values.ravel()

    def transform(self, geometry: np.ndarray, part_labels: np.ndarray | None = None
                  ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Project one cloud without changing anchors, scales, modes, or reference."""
        descriptor = self._descriptor(_parts(geometry, part_labels), self.anchors_mm,
                                      self.anchor_labels, self.config["profile_k"])
        delta = descriptor - self.feature_mean_mm
        standardized = delta / self.feature_scale_mm
        raw_coefficients = standardized @ self.components.T
        coefficients = raw_coefficients / self.coefficient_scale
        reconstructed = raw_coefficients @ self.components
        residual = standardized - reconstructed
        delta_norm, residual_norm = np.linalg.norm(standardized), np.linalg.norm(residual)
        diagnostics = dict(
            coefficient_width=self.coefficient_width, num_anchors=self.num_anchors,
            descriptor_delta_rms_mm=float(np.sqrt(np.mean(delta ** 2))),
            descriptor_delta_max_mm=float(np.max(np.abs(delta))),
            reconstruction_rms_mm=float(np.sqrt(np.mean((residual * self.feature_scale_mm) ** 2))),
            reconstruction_max_mm=float(np.max(np.abs(residual * self.feature_scale_mm))),
            standardized_delta_norm=float(delta_norm),
            projection_norm=float(np.linalg.norm(raw_coefficients)),
            out_of_span_norm=float(residual_norm),
            relative_reconstruction_error=float(residual_norm / delta_norm) if delta_norm > 0 else 0.,
            out_of_span_energy_fraction=float((residual_norm / delta_norm) ** 2) if delta_norm > 0 else 0.,
            normalized_coefficient_norm=float(np.linalg.norm(coefficients)))
        return coefficients.astype(np.float32), diagnostics

    def save(self, path: str | Path) -> None:
        """Write a single atomic NPZ artifact readable with allow_pickle=False."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(config=self.config, training_design_ids=self.training_design_ids)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(handle, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                                    anchors_mm=self.anchors_mm, anchor_labels=self.anchor_labels,
                                    feature_mean_mm=self.feature_mean_mm,
                                    feature_scale_mm=self.feature_scale_mm, components=self.components,
                                    coefficient_scale=self.coefficient_scale, singular_values=self.singular_values,
                                    training_coefficients=self.training_coefficients)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "GeometryCorrectionBasis":
        """Load a frozen artifact; reject unsupported versions and invalid arrays."""
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            config = metadata["config"]
            if config.get("schema_version") != SCHEMA_VERSION or config.get("fit_split") != "train":
                raise ValueError("unsupported geometry correction basis schema or fit split")
            arrays = {key: artifact[key].copy() for key in (
                "anchors_mm", "anchor_labels", "feature_mean_mm", "feature_scale_mm",
                "components", "coefficient_scale", "singular_values", "training_coefficients")}
        basis = cls(**arrays, training_design_ids=metadata["training_design_ids"], config=config)
        rank, features, anchors = config["coefficient_width"], config["num_features"], config["num_anchors"]
        expected_shapes = dict(anchors_mm=(anchors, 3), anchor_labels=(anchors,), feature_mean_mm=(features,),
                               feature_scale_mm=(features,), components=(rank, features), coefficient_scale=(rank,),
                               singular_values=(rank,), training_coefficients=(len(basis.training_design_ids), rank))
        for key, shape in expected_shapes.items():
            array = arrays[key]
            if array.shape != shape or (key != "anchor_labels" and not np.isfinite(array).all()):
                raise ValueError(f"invalid saved geometry basis array: {key}")
        if (rank < 1 or anchors < 1 or features != anchors * (3 + config["profile_k"])
                or np.any(basis.feature_scale_mm <= 0) or np.any(basis.coefficient_scale <= 0)
                or basis.training_design_ids != config["training_design_ids"]
                or sorted(np.unique(basis.anchor_labels).tolist()) != config["part_labels"]):
            raise ValueError("invalid saved geometry correction basis metadata or scales")
        return basis
