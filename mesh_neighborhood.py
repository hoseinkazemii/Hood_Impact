"""Sparse geometric point attention before mesh-to-latent pooling."""

import hashlib
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def default_neighbor_cache_directory(data_format, inp_dir=None):
    if inp_dir is not None and data_format in ("euroncap1704", "industrylike"):
        return Path(inp_dir).parent / "neighbor_graphs"
    datasets = {"euroncap1704": "HoodImpact_1704_EuroNCAP", "industrylike": "HoodImpact_60_IndustryLike"}
    if data_format in datasets:
        return Path("Data") / datasets[data_format] / "neighbor_graphs"
    return Path("runs") / "mesh_neighbor_graphs"


def geometric_neighbors(coordinates, count):
    """Exact Euclidean kNN in physical XYZ; never allocate an N-by-N matrix.

    Selection is discrete, but relative coordinates in the attention block
    remain differentiable. Canonical tree order makes distance ties independent
    of input node order (coincident nodes have identical input features).
    """
    points = coordinates.detach().to(device="cpu", dtype=torch.float64).numpy()
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    _, indices = cKDTree(points[order]).query(points, k=min(count, len(points)), workers=1)
    indices = np.asarray(indices).reshape(len(points), -1)
    return torch.as_tensor(order[indices], device=coordinates.device, dtype=torch.long)


class NeighborGraphCache:
    """Reuse exact geometric kNN graphs in memory and, optionally, on disk.

    Raw physical XYZ and its input row order identify persistent graphs.
    Registering a geometry associates that graph with the current scaler/device
    representation without including the scaler in the persistent key.
    """

    ALGORITHM = "physical_xyz_ckdtree_lexsort_v1"

    def __init__(self, capacity=16, directory=None):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.directory = Path(directory).expanduser() if directory is not None else None
        self.entries = {}
        self.references = {}
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.builds = 0
        self.writes = 0
        self.invalid_files = 0

    @staticmethod
    def points(coordinates):
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        points = np.ascontiguousarray(coordinates)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("Neighbor coordinates must have shape (N, 3), N > 0")
        if points.dtype.kind != "f" or not np.isfinite(points).all():
            raise ValueError("Neighbor coordinates must contain finite floating-point XYZ")
        return points

    @staticmethod
    def geometry_digest(points):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(points.shape).encode("ascii"))
        digest.update(points.dtype.str.encode("ascii"))
        digest.update(memoryview(points).cast("B"))
        return digest.hexdigest()

    def path_for(self, coordinates, count):
        if self.directory is None:
            return None
        points = self.points(coordinates)
        return self.directory / f"{self.ALGORITHM}_{self.geometry_digest(points)}_k{count}.npz"

    def register_geometry(self, raw_coordinates, runtime_coordinates, count):
        """Bind raw XYZ to the exact coordinates used by this model forward."""
        raw = self.points(raw_coordinates)
        runtime = self.points(runtime_coordinates)
        if raw.shape != runtime.shape:
            raise ValueError("Raw and runtime neighbor coordinates must have the same shape")
        identity = (self.geometry_digest(runtime), count)
        existing = self.references.get(identity)
        if existing is not None and self.geometry_digest(existing) != self.geometry_digest(raw):
            raise ValueError("Different raw geometries map to the same runtime coordinates")
        if existing is None:
            self.references[identity] = raw.copy()
            # A previously unregistered call may have built from rounded,
            # centered coordinates. Bind the canonical raw graph instead.
            for key in list(self.entries):
                if key[:2] == identity:
                    self.entries.pop(key)
        return self.neighbors(runtime_coordinates, count)

    def _load(self, path, points, count):
        with np.load(path, allow_pickle=False) as saved:
            indices = saved["neighbors"]
            digest = self.geometry_digest(points)
            if (str(saved["algorithm"]) != self.ALGORITHM or str(saved["geometry_digest"]) != digest
                    or int(saved["requested_k"]) != count or int(saved["node_count"]) != len(points)
                    or indices.dtype != np.int32 or indices.shape != (len(points), min(count, len(points)))
                    or indices.min() < 0 or indices.max() >= len(points)
                    or str(saved["indices_digest"]) != hashlib.blake2b(indices, digest_size=16).hexdigest()):
                raise ValueError("Neighbor graph metadata, shape, indices, or checksum do not match")
            return indices

    def _save(self, path, points, count, indices):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "_", suffix=".tmp", delete=False) as file:
                temporary_path = Path(file.name)
                np.savez(file, neighbors=indices, algorithm=self.ALGORITHM,
                         geometry_digest=self.geometry_digest(points), requested_k=count, node_count=len(points),
                         indices_digest=hashlib.blake2b(indices, digest_size=16).hexdigest())
            os.replace(temporary_path, path)
            self.writes += 1
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _graph(self, points, count, device):
        path = self.path_for(points, count)
        if path is not None and path.is_file():
            try:
                indices = self._load(path, points, count)
            except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
                self.invalid_files += 1
            else:
                self.disk_hits += 1
                return torch.as_tensor(indices, device=device, dtype=torch.long)
        graph = geometric_neighbors(torch.from_numpy(points), count)
        self.builds += 1
        if path is not None:
            self._save(path, points, count, graph.numpy().astype(np.int32))
        return graph.to(device=device)

    def neighbors(self, coordinates, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Neighbor count must be a positive integer")
        points = self.points(coordinates)
        identity = (self.geometry_digest(points), count)
        key = (*identity, str(coordinates.device))
        cached = self.entries.get(key)
        if cached is None:
            self.misses += 1
            reference = self.references.get(identity, points)
            cached = self.entries[key] = self._graph(reference, count, coordinates.device)
            while len(self.entries) > self.capacity:
                self.entries.pop(next(iter(self.entries)))
        else:
            self.hits += 1
        return cached


def prepare_model_neighbor_graphs(model, preprocessor, mesh_geometries, directory, progress=None):
    """Load/build each distinct raw structural geometry once before training."""
    if not model.neighborhood_blocks:
        return None
    model.neighbor_cache = NeighborGraphCache(directory=directory)
    cache = model.neighbor_cache
    seen = set()
    for mesh in mesh_geometries:
        raw = cache.points(mesh[model.impactor_nodes:])
        digest = cache.geometry_digest(raw)
        if digest in seen:
            continue
        normalized = torch.as_tensor(preprocessor.transform_mesh(raw),
                                     dtype=model.mesh_scale.dtype, device=model.mesh_scale.device)
        cache.register_geometry(raw, normalized * model.mesh_scale, model.neighborhood_k)
        seen.add(digest)
        if progress is not None:
            progress(len(seen), cache)
    return {
        "directory": str(cache.directory.resolve()) if cache.directory is not None else None,
        "algorithm": cache.ALGORITHM, "requested_k": model.neighborhood_k,
        "impactor_nodes_excluded": model.impactor_nodes,
        "unique_structural_geometries": len(seen), "built": cache.builds,
        "loaded_from_disk": cache.disk_hits, "files_written": cache.writes,
        "invalid_files_rebuilt": cache.invalid_files,
        "key_uses": "raw physical XYZ in input row order, dtype, shape, k, algorithm version",
    }


class NeighborhoodAttentionBlock(nn.Module):
    """Multihead kNN attention with learned relative-position scores/messages.

    Chunking and activation recomputation bound edge activation memory during
    training. Every node receives an update, including nodes far from impact.
    """

    def __init__(self, width, num_heads, dropout, position_scale, chunk_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.position_scale = position_scale
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.position_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, num_heads))
        self.position_value = nn.Linear(3, width, bias=False)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )

    def _attend(self, query, key, value, centers, coordinates, neighbors):
        # (centers, neighbors, heads, head_width); no attention across meshes.
        relative = (coordinates[neighbors] - centers[:, None]) / self.position_scale
        scores = (query[:, None] * key[neighbors]).sum(-1) / math.sqrt(self.head_width)
        scores = scores + self.position_bias(relative)
        weights = self.dropout(scores.float().softmax(dim=1).to(value.dtype))
        position = self.position_value(relative).reshape(*neighbors.shape, self.num_heads, self.head_width)
        message = (weights[..., None] * (value[neighbors] + position)).sum(dim=1)
        return message.flatten(1)

    def forward(self, nodes, coordinates, neighbors):
        query, key, value = self.qkv(self.norm(nodes)).reshape(
            len(nodes), 3, self.num_heads, self.head_width
        ).unbind(dim=1)
        chunks = []
        for start in range(0, len(nodes), self.chunk_size):
            stop = start + self.chunk_size
            arguments = (query[start:stop], key, value, coordinates[start:stop], coordinates, neighbors[start:stop])
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._attend, *arguments, use_reentrant=False))
            else:
                chunks.append(self._attend(*arguments))
        nodes = nodes + self.dropout(self.output(torch.cat(chunks)))
        return nodes + self.dropout(self.feedforward(self.ff_norm(nodes)))
