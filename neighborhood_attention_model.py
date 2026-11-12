"""Neighborhood attention network: hood mesh + impact location -> acceleration history.

    model = MeshImpactHistoryNet()
    acceleration = model(mesh, indentor, time)

    mesh      (B, N, 3)  node XYZ
    indentor  (B, 2)     impact XY
    time      (B, T)     output times
    returns   (B, T)     acceleration

All inputs and the output are normalized with the training scalers
(standardization of mesh XYZ, impact XY, time and acceleration). The scaler
statistics of mesh, impact and time are also stored in the model buffers, so
the network can recover millimetres and seconds internally.

Per sample, the network
  1. embeds every node's XYZ,
  2. runs kNN neighborhood attention between nodes in physical XYZ,
  3. pools all nodes into latent tokens with impact-conditioned (FiLM)
     attention: half of the tokens are global, half have learned radial
     priors around the impact point,
  4. mixes the latent tokens with self-attention,
  5. decodes Fourier time queries with cross-attention to the tokens and
     self-attention across times into one acceleration per time.

Checkpoints of ``mesh_impact_history.MeshImpactHistoryNet`` trained with the
same settings (temporal decoder, FiLM conditioning, Fourier time encoding)
load into this class directly; pass the run's architecture values, e.g.

    model = MeshImpactHistoryNet(neighborhood_k=256, impactor_nodes=286)
    state = torch.load("hood_impact_best_model.pt", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
"""

import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def feedforward(width: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))


def geometric_neighbors(coordinates: Tensor, k: int) -> Tensor:
    """Exact kNN (each node included) in physical XYZ, shape (N, k).

    The tree is built on lexicographically sorted points, so distance ties are
    broken independently of the input node order.
    """
    points = coordinates.detach().to(device="cpu", dtype=torch.float64).numpy()
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    _, indices = cKDTree(points[order]).query(points, k=min(k, len(points)), workers=1)
    indices = np.asarray(indices).reshape(len(points), -1)
    return torch.as_tensor(order[indices], device=coordinates.device, dtype=torch.long)


class FiLM(nn.Module):
    """Condition-dependent scale and shift of every feature, starting at identity."""

    def __init__(self, width: int):
        super().__init__()
        self.affine = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        scale_delta, shift = self.affine(condition).chunk(2, dim=-1)
        return tokens * (1 + scale_delta) + shift


class NeighborhoodAttentionBlock(nn.Module):
    """Multihead kNN attention with learned relative-position scores and messages.

    Nodes are processed in chunks (with activation recomputation in training)
    to bound the memory of the (N, k) edge tensors.
    """

    def __init__(self, width: int, num_heads: int, dropout: float, position_scale: float, chunk_size: int):
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
        self.feedforward = feedforward(width, 2 * width, dropout)

    def _attend(self, query, key, value, centers, coordinates, neighbors):
        # (centers, k, heads, head_width)
        relative = (coordinates[neighbors] - centers[:, None]) / self.position_scale
        scores = (query[:, None] * key[neighbors]).sum(-1) / math.sqrt(self.head_width)
        scores = scores + self.position_bias(relative)
        weights = self.dropout(scores.float().softmax(dim=1).to(value.dtype))
        position = self.position_value(relative).reshape(*neighbors.shape, self.num_heads, self.head_width)
        message = (weights[..., None] * (value[neighbors] + position)).sum(dim=1)
        return message.flatten(1)

    def forward(self, nodes: Tensor, coordinates: Tensor, neighbors: Tensor) -> Tensor:
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


class LatentMixingBlock(nn.Module):
    """Self-attention among the latent mesh tokens."""

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = feedforward(width, 4 * width, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        update = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.dropout(update)
        return tokens + self.dropout(self.feedforward(self.feedforward_norm(tokens)))


class HistoryDecoderBlock(nn.Module):
    """Time queries read the mesh memory, then attend to each other across times."""

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.time_attention = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = feedforward(width, 4 * width, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update = self.cross_attention(self.query_norm(queries), normalized_memory, normalized_memory,
                                      need_weights=False)[0]
        queries = queries + self.dropout(update)
        normalized_time = self.time_norm(queries)
        update = self.time_attention(normalized_time, normalized_time, normalized_time, need_weights=False)[0]
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.feedforward(self.feedforward_norm(queries)))


class MeshImpactHistoryNet(nn.Module):
    """Neighborhood attention encoder, impact-conditioned pooling, temporal decoder.

    ``impactor_nodes`` > 0 marks the leading mesh rows as the rigid headform:
    they are left out of the kNN graph and get zero pooling weight. Use 0 when
    the mesh contains hood nodes only.
    """

    def __init__(
        self,
        width: int = 128,
        num_heads: int = 4,
        num_latents: int = 256,
        latent_layers: int = 3,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        neighborhood_layers: int = 2,
        neighborhood_k: int = 256,
        neighborhood_scale_mm: float = 20.0,
        neighborhood_chunk_size: int = 1024,
        impactor_nodes: int = 0,
        fourier_num_frequencies: int = 6,
        fourier_min_frequency_hz: float = 20.0,
        fourier_max_frequency_hz: float = 640.0,
    ):
        super().__init__()
        assert width % num_heads == 0, "width must be divisible by num_heads"
        self.width = width
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.num_global_latents = num_latents // 2
        self.attention_dropout = dropout
        self.neighborhood_k = neighborhood_k
        self.impactor_nodes = impactor_nodes

        # Module order fixes the state-dict layout and the seeded initialization.
        # Training scaler statistics (mesh and impact in their own scalers).
        self.register_buffer("mesh_mean", torch.zeros(3))
        self.register_buffer("mesh_scale", torch.ones(3))
        self.register_buffer("impact_mean", torch.zeros(2))
        self.register_buffer("impact_scale", torch.ones(2))

        self.node_embedding = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.impact_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))

        # Mesh -> latent pooling.
        self.latent_queries = nn.Parameter(torch.empty(num_latents, width))
        nn.init.normal_(self.latent_queries, std=0.02)
        self.query_norm = nn.LayerNorm(width)
        self.mesh_query = nn.Linear(width, width)
        self.mesh_key_value = nn.Linear(width, 2 * width)
        self.mesh_output = nn.Linear(width, width)
        self.mesh_dropout = nn.Dropout(dropout)
        # Radial prior precision of the local tokens, from 0.5 to 32 (softplus-parameterized).
        initial_precision = torch.logspace(math.log10(0.5), math.log10(32.0), num_latents - self.num_global_latents)
        self.local_log_precision = nn.Parameter(torch.log(torch.expm1(initial_precision)))

        self.latent_blocks = nn.ModuleList([LatentMixingBlock(width, num_heads, dropout) for _ in range(latent_layers)])
        self.memory_norm = nn.LayerNorm(width)

        # Time queries: sin/cos at log-spaced frequencies, in seconds.
        frequencies = [fourier_min_frequency_hz * (fourier_max_frequency_hz / fourier_min_frequency_hz)
                       ** (i / max(fourier_num_frequencies - 1, 1)) for i in range(fourier_num_frequencies)]
        self.register_buffer("time_frequencies_hz", torch.tensor(frequencies, dtype=torch.float32))
        self.register_buffer("time_mean", torch.zeros(()))
        self.register_buffer("time_scale", torch.ones(()))
        self.time_embedding = nn.Sequential(
            nn.Linear(2 * fourier_num_frequencies, width), nn.GELU(), nn.Linear(width, width),
        )
        self.decoder_blocks = nn.ModuleList([HistoryDecoderBlock(width, num_heads, dropout)
                                             for _ in range(temporal_layers)])
        self.acceleration_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )

        self.neighborhood_blocks = nn.ModuleList([
            NeighborhoodAttentionBlock(width, num_heads, dropout, neighborhood_scale_mm, neighborhood_chunk_size)
            for _ in range(neighborhood_layers)
        ])
        self.impact_film = FiLM(width)

        # kNN graphs depend only on the mesh; keep the last few (not saved in the state dict).
        self.neighbor_cache = {}
        self.neighbor_cache_size = 16

    @torch.no_grad()
    def set_coordinate_scalers(self, mesh_mean, mesh_scale, impact_mean, impact_scale) -> None:
        """Store the training mesh/impact scaler statistics (mean and standard deviation)."""
        for name, value in (("mesh_mean", mesh_mean), ("mesh_scale", mesh_scale),
                            ("impact_mean", impact_mean), ("impact_scale", impact_scale)):
            buffer = getattr(self, name)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))

    @torch.no_grad()
    def set_time_scaler(self, mean: float, scale: float) -> None:
        """Store the training time scaler so the Fourier frequencies stay in Hz."""
        self.time_mean.fill_(float(mean))
        self.time_scale.fill_(float(scale))

    def forward(self, mesh: Tensor, indentor: Tensor, time: Tensor) -> Tensor:
        """(B, N, 3) mesh, (B, 2) impact XY, (B, T) times -> (B, T) acceleration."""
        condition = self.impact_embedding(indentor)
        # Each mesh is encoded on its own; only the compact latent memory is batched.
        memory = torch.stack([self.encode_mesh(mesh[b], indentor[b], condition[b]) for b in range(len(mesh))])
        for block in self.latent_blocks:
            memory = block(memory)
        memory = self.memory_norm(memory)

        queries = self.time_embedding(self.time_features(time))
        for block in self.decoder_blocks:
            queries = block(queries, memory)
        return self.acceleration_head(queries).squeeze(-1)

    def time_features(self, time: Tensor) -> Tensor:
        """(..., T) normalized time -> (..., T, 2F) sin/cos features in physical seconds."""
        seconds = time * self.time_scale + self.time_mean
        phases = 2 * math.pi * seconds.unsqueeze(-1) * self.time_frequencies_hz
        return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2)

    def impact_relative_xy(self, mesh: Tensor, indentor: Tensor) -> Tensor:
        """Node XY relative to the impact point, in mesh-scaler units, shape (N, 2)."""
        impact_xy = (indentor * self.impact_scale + self.impact_mean - self.mesh_mean[:2]) / self.mesh_scale[:2]
        return mesh[:, :2] - impact_xy

    def encode_mesh(self, mesh: Tensor, indentor: Tensor, condition: Tensor) -> Tensor:
        """One mesh (N, 3) -> latent memory tokens (num_latents, width)."""
        nodes = self.node_embedding(mesh)

        # Local attention among hood nodes; the headform rows keep their embedding.
        start = self.impactor_nodes
        if len(self.neighborhood_blocks):
            coordinates = mesh[start:] * self.mesh_scale  # millimetres, up to a translation
            neighbors = self.neighbors(coordinates)
            hood = nodes[start:]
            for block in self.neighborhood_blocks:
                hood = block(hood, coordinates, neighbors)
            nodes = torch.cat((nodes[:start], hood))

        # Impact-conditioned latent queries attend to every node.
        tokens = self.impact_film(self.query_norm(self.latent_queries), condition)
        head_width = self.width // self.num_heads
        query = self.mesh_query(tokens).reshape(self.num_latents, self.num_heads, head_width).transpose(0, 1)
        key, value = self.mesh_key_value(nodes).chunk(2, dim=-1)
        key = key.reshape(-1, self.num_heads, head_width).transpose(0, 1)
        value = value.reshape(-1, self.num_heads, head_width).transpose(0, 1)

        # Global tokens: no spatial bias. Local tokens: -precision * r^2 around the impact.
        precision = torch.cat((self.local_log_precision.new_zeros(self.num_global_latents),
                               F.softplus(self.local_log_precision)))
        radius_squared = self.impact_relative_xy(mesh, indentor).square().sum(dim=-1)
        spatial_bias = -precision[:, None] * radius_squared[None, :]
        spatial_bias[:, :start] = float("-inf")  # headform rows get zero weight

        update = F.scaled_dot_product_attention(
            query[None], key[None], value[None], attn_mask=spatial_bias.to(query.dtype)[None, None],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        update = update[0].transpose(0, 1).reshape(self.num_latents, self.width)
        return self.mesh_dropout(self.mesh_output(update))

    def neighbors(self, coordinates: Tensor) -> Tensor:
        """Cached kNN graph of the hood nodes, shape (N_hood, k)."""
        points = coordinates.detach().cpu().numpy()
        key = (hashlib.blake2b(points.tobytes(), digest_size=16).hexdigest(), points.shape, str(coordinates.device))
        if key not in self.neighbor_cache:
            self.neighbor_cache[key] = geometric_neighbors(coordinates, self.neighborhood_k)
            while len(self.neighbor_cache) > self.neighbor_cache_size:
                self.neighbor_cache.pop(next(iter(self.neighbor_cache)))
        return self.neighbor_cache[key]
