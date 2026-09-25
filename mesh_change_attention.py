"""Training-atlas change patches alongside global and impact-local mesh memory.

The atlas is fitted outside the network from training geometry only. Inference
queries the frozen anchors against one supplied mesh; it never refits the atlas
or compares held-out designs with one another.
"""

import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from mesh_impact_history import MeshImpactHistoryNet
from mesh_neighborhood import NeighborhoodAttentionBlock


class MeshChangeAttentionNet(MeshImpactHistoryNet):
    """Preserve small, possibly remote geometry changes as separate mesh tokens.

    Each training-derived anchor selects K nearest STRUCTURAL nodes, independently
    of impact position. Two shared point-attention blocks exchange information
    within each patch; impact-conditioned attention compresses each patch to one
    token. The inherited mixer/temporal decoder reads both these tokens and the
    ordinary full-mesh global/local tokens.
    """

    def __init__(self, change_anchors=32, change_k=256, change_layers=2,
                 change_scale_mm=20.0, change_chunk_size=256, **kwargs):
        for name, value in (("change_anchors", change_anchors), ("change_k", change_k),
                            ("change_layers", change_layers), ("change_chunk_size", change_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(change_scale_mm) or change_scale_mm <= 0:
            raise ValueError("change_scale_mm must be finite and positive")
        if kwargs.get("neighborhood_layers", 0):
            raise ValueError("Use change_layers for change patches; neighborhood_layers must be zero")
        super().__init__(**kwargs)
        self.change_anchors, self.change_k = change_anchors, change_k
        self.change_scale_mm = change_scale_mm
        self.register_buffer("change_anchor_xyz", torch.zeros(change_anchors, 3))
        self.register_buffer("change_reference_offsets", torch.zeros(change_anchors, 3))
        self.register_buffer("change_reference_distances", torch.zeros(change_anchors, change_k))
        self.register_buffer("change_anchor_scores", torch.zeros(change_anchors))
        self.register_buffer("change_anchor_count", torch.zeros((), dtype=torch.long))
        # Existing six features plus anchor-relative XYZ, radial deviation from
        # training mean, and nearest-surface displacement from training mean.
        self.change_embedding = nn.Sequential(
            nn.Linear(13, self.width), nn.GELU(), nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
        )
        self.change_blocks = nn.ModuleList([
            NeighborhoodAttentionBlock(self.width, self.num_heads, self.attention_dropout,
                                       change_scale_mm, change_chunk_size)
            for _ in range(change_layers)
        ])
        self.change_query = nn.Sequential(nn.Linear(5, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        self.change_pool = nn.MultiheadAttention(self.width, self.num_heads,
                                                dropout=self.attention_dropout, batch_first=True)
        self.change_pool_norm = nn.LayerNorm(self.width)
        self.change_type_embedding = nn.Parameter(torch.empty(self.width))
        nn.init.normal_(self.change_type_embedding, std=.02)
        # Derived query indices only, bounded by designs; not checkpoint data.
        self._patch_cache = {}
        self._patch_graphs = {}

    @torch.no_grad()
    def set_change_atlas(self, atlas):
        if atlas.get("fit_split", "train") != "train":
            raise ValueError("Change atlas must be fitted on training geometry only")
        if atlas.get("impactor_nodes", self.impactor_nodes) != self.impactor_nodes:
            raise ValueError("Atlas and model impactor_nodes must match")
        if atlas.get("neighborhood_k", self.change_k) != self.change_k:
            raise ValueError("Atlas neighborhood_k must match model change_k")
        arrays = {
            "change_anchor_xyz": np.asarray(atlas["anchors_mm"]),
            "change_reference_offsets": np.asarray(atlas["reference_offsets_mm"]),
            "change_reference_distances": np.asarray(atlas["reference_distance_profiles_mm"]),
            "change_anchor_scores": np.asarray(atlas["change_scores_mm"]),
        }
        count = len(arrays["change_anchor_xyz"])
        if not 1 <= count <= self.change_anchors:
            raise ValueError("Atlas must contain between one and change_anchors anchors")
        for name, array in arrays.items():
            buffer = getattr(self, name)
            if array.shape != (count, *buffer.shape[1:]) or not np.isfinite(array).all():
                raise ValueError(f"Invalid atlas shape or nonfinite values for {name}")
        distances = arrays["change_reference_distances"]
        if np.any(distances < 0) or np.any(np.diff(distances, axis=1) < 0):
            raise ValueError("Reference distance profiles must be nonnegative and sorted")
        if np.any(arrays["change_anchor_scores"] <= 0):
            raise ValueError("Change scores must be positive")
        for name, array in arrays.items():
            buffer = getattr(self, name)
            buffer.zero_()
            buffer[:count].copy_(torch.as_tensor(array, dtype=buffer.dtype, device=buffer.device))
        self.change_anchor_count.fill_(count)
        self._patch_cache.clear()
        self._patch_graphs.clear()

    def load_state_dict(self, *args, **kwargs):
        self._patch_cache.clear()
        self._patch_graphs.clear()
        return super().load_state_dict(*args, **kwargs)

    def _change_indices(self, coordinates, anchors):
        points = np.ascontiguousarray(coordinates.detach().cpu().numpy())
        key = (hashlib.blake2b(points, digest_size=16).digest(), str(coordinates.device))
        if key not in self._patch_cache:
            # The atlas uses distinct XYZ positions. Match that convention so
            # coincident mesh rows cannot alter the radial reference features.
            unique, order = np.unique(points, axis=0, return_index=True)
            count = min(self.change_k, len(unique))
            tree = cKDTree(unique)
            queries = anchors.detach().cpu().numpy()
            _, indices = tree.query(queries, k=count, workers=1)
            indices = np.asarray(indices).reshape(len(anchors), count)
            # k=1 and k=K may order equidistant neighbors differently. The atlas
            # defines its nearest displacement using k=1; preserve that exact
            # convention while retaining the sorted radial distance profile.
            nearest = np.asarray(tree.query(queries, k=1, workers=1)[1])
            for row, first in zip(indices, nearest):
                match = np.flatnonzero(row == first)
                if len(match):
                    row[0], row[match[0]] = row[match[0]], row[0]
                else:
                    row[0] = first
            self._patch_cache[key] = torch.as_tensor(order[indices], device=coordinates.device, dtype=torch.long)
            while len(self._patch_cache) > 16:
                self._patch_cache.pop(next(iter(self._patch_cache)))
        return self._patch_cache[key]

    def _encode_changes(self, mesh, impact, condition):
        count = int(self.change_anchor_count.item())
        if not 1 <= count <= self.change_anchors:
            raise RuntimeError("Fit and set a training-only change atlas before using this model")
        structural = mesh[self.impactor_nodes:]
        if len(structural) < 2:
            raise ValueError("Change attention needs at least two structural nodes after impactor exclusion")
        coordinates = structural * self.mesh_scale + self.mesh_mean
        anchors = self.change_anchor_xyz[:count]
        indices = self._change_indices(coordinates, anchors)
        patch_xyz = coordinates[indices]
        offset = patch_xyz - anchors[:, None, :]
        radius = torch.linalg.vector_norm(offset, dim=-1)
        k = indices.shape[1]
        radial_delta = radius - self.change_reference_distances[:count, :k]
        nearest_delta = offset[:, 0] - self.change_reference_offsets[:count]
        features = torch.cat((
            self.node_features(structural, impact)[indices],
            offset / self.change_scale_mm,
            radial_delta[..., None] / self.change_scale_mm,
            nearest_delta[:, None, :].expand(-1, k, -1) / self.change_scale_mm,
        ), dim=-1)
        nodes = self.change_embedding(features.flatten(0, 1))
        graph_key = (count, k, str(mesh.device))
        if graph_key not in self._patch_graphs:
            members = torch.arange(count * k, device=mesh.device).reshape(count, k)
            self._patch_graphs[graph_key] = members[:, None, :].expand(-1, k, -1).reshape(count * k, k)
        graph = self._patch_graphs[graph_key]
        for block in self.change_blocks:
            nodes = block(nodes, patch_xyz.flatten(0, 1), graph)
        anchor_mesh = (anchors - self.mesh_mean) / self.mesh_scale
        impact_mm = impact * self.impact_scale + self.impact_mean
        relative_impact = (anchors[:, :2] - impact_mm) / self.mesh_scale[:2]
        query = self.change_query(torch.cat((anchor_mesh, relative_impact), dim=-1))
        query = query + condition + self.change_type_embedding
        normalized = self.change_pool_norm(nodes.reshape(count, k, self.width))
        update = self.change_pool(query[:, None], normalized, normalized, need_weights=False)[0][:, 0]
        return query + self.mesh_dropout(update)

    def _encode_mesh(self, mesh, impact, condition):
        ordinary = super()._encode_mesh(mesh, impact, condition)
        changes = self._encode_changes(mesh, impact, condition)
        return torch.cat((ordinary, changes), dim=0)
