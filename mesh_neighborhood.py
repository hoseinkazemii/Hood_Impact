"""Sparse geometric point attention before mesh-to-latent pooling."""

import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


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
