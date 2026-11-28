"""Backbone attention network: unstructured hood mesh + impact location -> acceleration history.

The mesh is an unordered set of XYZ nodes; no grid or element connectivity is
required. Nodes are embedded, optionally refined by sparse kNN attention
(mesh_neighborhood.py), and pooled into a fixed set of latent tokens by
impact-conditioned (FiLM) cross-attention. Latent self-attention mixes the
tokens, and a decoder of time queries (Fourier features of physical time)
reads them with cross-attention and couples the requested times with temporal
self-attention.

The model used for training is :class:`mesh_design_change.DesignChangeHistoryNet`,
which adds a readout of the edited nodes after every decoder block.

``forward`` accepts the flattened batches built by ``utils.utils.collate_fn``.
Time contains output-grid coordinates, never measured acceleration. For the
two-input physical-units API, use ``HistoryPredictor`` in
``train_mesh_impact_history.py``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from mesh_neighborhood import NeighborGraphCache, NeighborhoodAttentionBlock

# What configs saved before these keys existed were trained with.
HISTORICAL_DEFAULTS = {
    "impact_conditioning": "legacy_additive",
    "time_encoding": "legacy_five",
    "pooling_query_residual": True,
    "pool_impactor_nodes": True,
    "neighborhood_scale_mm": 20.0,
    "local_latents": None,  # half the pooling tokens local, every run before 2026-11-22
}
# Keyword arguments of earlier saved runs that now have only one supported value.
# saved_model_kwargs accepts a saved config only when each holds that value.
RETIRED_KWARGS = {
    "decoder": "temporal",
    "impact_conditioning": "film",
    "time_encoding": "fourier",
    "pooling_query_residual": False,
    "pool_impactor_nodes": False,
    "impactor_nodes": 0,
    "geometry_film": "none",
    "teacher_forcing_chunks": 0,
    "node_extra_features": 0,
}
# Saved but no longer meaningful once the options above are fixed.
IGNORED_KWARGS = ("geometry_fourier_octaves",)


def saved_model_kwargs(kwargs: dict, *, format_version: int) -> dict:
    """Constructor kwargs of a saved run (config.json), rejecting runs that used a retired option.

    Configs of ``format_version`` 1 (before 2026-11-27) list every option of
    their time; keys a version-1 config lacks take the value runs were trained
    with before the key existed (HISTORICAL_DEFAULTS), so runs before
    2026-11-22, with half the pooling tokens local and 20 mm kNN offsets,
    still load. Version 2 lists exactly the current constructor options.
    """
    if format_version not in (1, 2):
        raise ValueError(f"Unknown config format_version {format_version!r}")
    kwargs = {**HISTORICAL_DEFAULTS, **kwargs} if format_version == 1 else dict(kwargs)
    if kwargs.pop("reference_response", False):
        raise ValueError("This run fed the base design's simulated response to the network "
                         "(reference_response, removed on 2026-11-26); it cannot be rebuilt")
    for key, supported in RETIRED_KWARGS.items():
        value = kwargs.pop(key, supported)
        if value != supported:
            raise ValueError(f"This run used {key}={value!r}, an option removed from the code; "
                             f"only {supported!r} is supported")
    for key in IGNORED_KWARGS:
        kwargs.pop(key, None)
    return kwargs


class LatentMixingBlock(nn.Module):
    """Exchange information among the compact mesh tokens."""

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        update = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.dropout(update)
        return tokens + self.dropout(self.feedforward(self.feedforward_norm(tokens)))


class HistoryDecoderBlock(nn.Module):
    """Read mesh memory and couple all requested times within each history."""

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.time_norm = nn.LayerNorm(width)
        self.time_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, padding_mask: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update = self.cross_attention(
            self.query_norm(queries), normalized_memory, normalized_memory,
            need_weights=False,
        )[0]
        queries = queries + self.dropout(update)
        normalized_time = self.time_norm(queries)
        update = self.time_attention(
            normalized_time, normalized_time, normalized_time,
            key_padding_mask=padding_mask, need_weights=False,
        )[0]
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.feedforward(self.feedforward_norm(queries)))


class FiLM(nn.Module):
    """Condition-dependent scale and shift of every feature, starting at identity."""

    def __init__(self, width: int):
        super().__init__()
        self.affine = nn.Linear(width, 2 * width)
        # Start with identity modulation; learn deviations from scale=1, shift=0.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        scale_delta, shift = self.affine(condition).chunk(2, dim=-1)
        return tokens * (1 + scale_delta) + shift


class MeshImpactHistoryNet(nn.Module):
    """Impact-conditioned full-mesh attention, then a mesh-reading decoder.

    Every node is embedded from its normalized XYZ. ``neighborhood_layers > 0``
    inserts sparse kNN attention in physical XYZ between the node embedding and
    pooling. ``num_latents`` learned queries, FiLM-modulated by the impact
    location, pool all nodes into memory tokens. ``local_latents`` of them carry
    a learned radial prior around the impact point (0 since 2026-11-22: every
    query reads the whole hood without spatial bias). Attention costs O(N * L)
    for N nodes and L latents; no N-by-N matrix is built.

    The impact location only chooses *where* the queries read: each memory
    token is an attention-weighted sum of hood-node values, with no
    location-only residual.

    Time queries embed sine/cosine pairs of physical time (seconds, recovered
    with the training time scaler) at ``fourier_num_frequencies`` log-spaced
    frequencies. ``temporal_layers`` decoder blocks each cross-attend to the
    memory and run self-attention across the requested times.
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
        neighborhood_scale_mm: float = 0.0,
        neighborhood_chunk_size: int = 1024,
        local_latents: int | None = 0,
        fourier_num_frequencies: int = 6,
        fourier_min_frequency_hz: float = 20.0,
        fourier_max_frequency_hz: float = 640.0,
    ):
        super().__init__()
        integer_options = {
            "width": width, "num_heads": num_heads, "num_latents": num_latents,
            "latent_layers": latent_layers, "temporal_layers": temporal_layers,
            "neighborhood_k": neighborhood_k, "neighborhood_chunk_size": neighborhood_chunk_size,
            "fourier_num_frequencies": fourier_num_frequencies,
        }
        for name, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if num_latents < 2:
            raise ValueError("num_latents must be at least 2")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name, frequency in (("fourier_min_frequency_hz", fourier_min_frequency_hz),
                                ("fourier_max_frequency_hz", fourier_max_frequency_hz)):
            if isinstance(frequency, bool) or not math.isfinite(frequency) or frequency <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if fourier_max_frequency_hz < fourier_min_frequency_hz:
            raise ValueError("fourier_max_frequency_hz must be at least fourier_min_frequency_hz")
        if fourier_num_frequencies == 1 and fourier_min_frequency_hz != fourier_max_frequency_hz:
            raise ValueError("A single Fourier frequency requires equal minimum and maximum frequencies")
        if isinstance(neighborhood_layers, bool) or not isinstance(neighborhood_layers, int) or neighborhood_layers < 0:
            raise ValueError("neighborhood_layers must be a nonnegative integer")
        # 0: offsets in units of the mesh's own median distance to the k-th neighbour.
        if not math.isfinite(neighborhood_scale_mm) or neighborhood_scale_mm < 0:
            raise ValueError("neighborhood_scale_mm must be finite and nonnegative (0 = from the mesh)")
        if local_latents is None:
            local_latents = num_latents // 2  # every run before 2026-11-22
        if isinstance(local_latents, bool) or not isinstance(local_latents, int) or not 0 <= local_latents < num_latents:
            raise ValueError("local_latents must be an integer in [0, num_latents)")

        self.width = width
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.local_latents = local_latents
        self.num_global_latents = num_latents - local_latents
        self.neighborhood_scale_mm = neighborhood_scale_mm
        self.attention_dropout = dropout
        self.neighborhood_k = neighborhood_k

        # The preprocessor fits mesh and impact scalers independently. Store
        # their affine maps in the checkpoint so relative coordinates always
        # use one common coordinate system, including after reload.
        self.register_buffer("mesh_mean", torch.zeros(3))
        self.register_buffer("mesh_scale", torch.ones(3))
        self.register_buffer("impact_mean", torch.zeros(2))
        self.register_buffer("impact_scale", torch.ones(2))

        # Module creation order fixes the seeded initial weights; keep it.
        self.node_embedding = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.impact_embedding = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, width),
        )
        self.latent_queries = nn.Parameter(torch.empty(num_latents, width))
        nn.init.normal_(self.latent_queries, std=0.02)
        self.query_norm = nn.LayerNorm(width)
        self.mesh_query = nn.Linear(width, width)
        self.mesh_key_value = nn.Linear(width, 2 * width)
        self.mesh_output = nn.Linear(width, width)
        self.mesh_dropout = nn.Dropout(dropout)

        self.local_log_precision = None
        if local_latents:
            initial_precision = torch.logspace(math.log10(0.5), math.log10(32.0), local_latents)
            self.local_log_precision = nn.Parameter(torch.log(torch.expm1(initial_precision)))
        self.latent_blocks = nn.ModuleList([
            LatentMixingBlock(width, num_heads, dropout) for _ in range(latent_layers)
        ])
        self.memory_norm = nn.LayerNorm(width)

        # Fourier phases use physical seconds, reconstructed from the training time scaler.
        self.time_feature_count = 2 * fourier_num_frequencies
        frequencies = [fourier_min_frequency_hz * (fourier_max_frequency_hz / fourier_min_frequency_hz)
                       ** (i / max(fourier_num_frequencies - 1, 1)) for i in range(fourier_num_frequencies)]
        frequency_tensor = torch.tensor(frequencies, dtype=torch.float32)
        if not torch.isfinite(frequency_tensor).all() or not (frequency_tensor > 0).all():
            raise ValueError("Fourier frequencies must be finite and positive in float32")
        self.register_buffer("time_frequencies_hz", frequency_tensor)
        self.register_buffer("time_mean", torch.zeros(()))
        self.register_buffer("time_scale", torch.ones(()))
        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_feature_count, width), nn.GELU(), nn.Linear(width, width),
        )
        # temporal_layers is the decoder depth: how many times the time queries re-read the memory.
        self.decoder_blocks = nn.ModuleList([
            HistoryDecoderBlock(width, num_heads, dropout) for _ in range(temporal_layers)
        ])
        self.acceleration_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )
        self.neighborhood_blocks = nn.ModuleList([
            NeighborhoodAttentionBlock(width, num_heads, dropout, neighborhood_scale_mm or 1.0,
                                       neighborhood_chunk_size)
            for _ in range(neighborhood_layers)
        ])
        # Plain attribute, never a buffer: the graph is derived from the inputs,
        # so it must not enter the checkpoint or a state dict.
        self.neighbor_cache = NeighborGraphCache() if neighborhood_layers else None
        self.impact_film = FiLM(width)

    @torch.no_grad()
    def set_time_scaler(self, mean: float, scale: float) -> None:
        """Save the training time scaler so Fourier frequencies remain in Hz."""
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Time scaler must have a finite mean and finite positive scale")
        self.time_mean.fill_(float(mean))
        self.time_scale.fill_(float(scale))

    def time_features(self, normalized_time: Tensor) -> Tensor:
        """Map normalized times (...,) to [sin, cos](2*pi*f*t_seconds) per frequency, (..., C)."""
        physical_time = normalized_time * self.time_scale + self.time_mean
        phases = 2 * math.pi * physical_time.unsqueeze(-1) * self.time_frequencies_hz
        return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2)

    @torch.no_grad()
    def set_coordinate_scalers(
        self,
        mesh_mean: Sequence[float],
        mesh_scale: Sequence[float],
        impact_mean: Sequence[float],
        impact_scale: Sequence[float],
    ) -> None:
        """Copy training-only scaler statistics into persistent model buffers."""
        converted = {}
        for name, values in (
            ("mesh_mean", mesh_mean), ("mesh_scale", mesh_scale),
            ("impact_mean", impact_mean), ("impact_scale", impact_scale),
        ):
            reference = getattr(self, name)
            value = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
            if value.shape != reference.shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain {reference.numel()} finite values")
            if name.endswith("scale") and not (value > 0).all():
                raise ValueError(f"{name} must be strictly positive")
            converted[name] = value
        for name, value in converted.items():
            getattr(self, name).copy_(value)

    def impact_relative_xy(self, mesh: Tensor, impact: Tensor) -> Tensor:
        """Relative XY in common mesh-scaler units, used by the pooling prior."""
        impact_mesh_xy = (
            impact * self.impact_scale + self.impact_mean - self.mesh_mean[:2]
        ) / self.mesh_scale[:2]
        return mesh[:, :2] - impact_mesh_xy

    def _encode_neighborhoods(self, mesh: Tensor, nodes: Tensor) -> Tensor:
        """Local kNN attention over all nodes, in input order.

        The graph depends only on the hood, so it is built once per design and
        reused for every impact. Training registers raw-XYZ graphs for
        persistent reuse across scalers and runs (prepare_model_neighbor_graphs).
        """
        # Undo anisotropic standardization; the omitted mean is a common
        # translation and cannot change distances or relative positions.
        coordinates = mesh * self.mesh_scale
        if len(coordinates) < 2:
            raise ValueError(f"Neighborhood attention needs at least two nodes; got {len(coordinates)}")
        neighbors = self.neighbor_cache.neighbors(coordinates, self.neighborhood_k)
        if not self.neighborhood_scale_mm:
            # No hand-set length: offsets in units of this mesh's median distance to its k-th neighbour.
            radius = (coordinates[neighbors[:, -1]] - coordinates).norm(dim=-1).median().detach()
            coordinates = coordinates / radius.clamp_min(1e-6)
        for block in self.neighborhood_blocks:
            nodes = block(nodes, coordinates, neighbors)
        return nodes

    def _encode_mesh(self, mesh: Tensor, impact: Tensor, condition: Tensor) -> Tensor:
        """(num_latents, width) memory tokens of one mesh at one impact location."""
        nodes = self.node_embedding(mesh)
        if self.neighborhood_blocks:
            nodes = self._encode_neighborhoods(mesh, nodes)
        # Normalize before modulation so no immediate LayerNorm removes the FiLM scale/shift.
        query = self.mesh_query(self.impact_film(self.query_norm(self.latent_queries), condition))
        head_width = self.width // self.num_heads
        query = query.reshape(self.num_latents, self.num_heads, head_width).transpose(0, 1)
        key, value = self.mesh_key_value(nodes).chunk(2, dim=-1)
        key = key.reshape(-1, self.num_heads, head_width).transpose(0, 1)
        value = value.reshape(-1, self.num_heads, head_width).transpose(0, 1)

        # Global tokens have zero spatial bias. Local tokens (if any) have soft
        # learned radial priors, never a hard node cutoff or node sampling.
        precision = mesh.new_zeros(self.num_global_latents)
        if self.local_latents:
            precision = torch.cat((precision, F.softplus(self.local_log_precision)))
        radius_squared = self.impact_relative_xy(mesh, impact).square().sum(dim=-1)
        spatial_bias = -precision[:, None] * radius_squared.unsqueeze(0)
        update = F.scaled_dot_product_attention(
            query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
            attn_mask=spatial_bias.to(query.dtype)[None, None, :, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        update = update.squeeze(0).transpose(0, 1).reshape(self.num_latents, self.width)
        return self.mesh_dropout(self.mesh_output(update))

    def _decoder_context(self, mesh: Tensor, mesh_batch: Tensor, indentor: Tensor, batch_size: int):
        """A second memory the decoder reads after each block; none in the backbone.

        DesignChangeHistoryNet (mesh_design_change.py) returns its change-node
        tokens here and reads them in ``_read_decoder_context``.
        """
        return None

    def _read_decoder_context(self, index: int, queries: Tensor, context) -> Tensor:
        raise NotImplementedError("_decoder_context returned a context without a reader")

    def _validate_inputs(
        self, mesh: Tensor, mesh_batch: Tensor, indentor: Tensor,
        time: Tensor, time_batch: Tensor, batch_size: int,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if mesh.ndim != 2 or mesh.shape[1] != 3 or mesh.shape[0] == 0:
            raise ValueError("mesh must have shape (N, 3) with at least one node")
        if indentor.shape != (batch_size, 2):
            raise ValueError("indentor must have shape (batch_size, 2)")
        if time.ndim != 1 or time.numel() == 0:
            raise ValueError("time must be a nonempty flat tensor")
        for name, value in (("mesh", mesh), ("indentor", indentor), ("time", time)):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
            if value.device != mesh.device or value.dtype != mesh.dtype:
                raise ValueError("mesh, indentor and time must share a device and dtype")
        for name, membership, count in (
            ("mesh_batch", mesh_batch, mesh.shape[0]),
            ("time_batch", time_batch, time.shape[0]),
        ):
            if membership.shape != (count,) or membership.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor of shape ({count},)")
            if membership.device != mesh.device:
                raise ValueError(f"{name} must be on the mesh device")
            if (membership < 0).any() or (membership >= batch_size).any():
                raise ValueError(f"{name} contains an invalid sample index")
            if (torch.bincount(membership.long(), minlength=batch_size) == 0).any():
                raise ValueError(f"{name} must include at least one entry for every sample")

    def forward(
        self,
        mesh: Tensor,
        mesh_batch: Tensor,
        indentor: Tensor,
        time: Tensor,
        time_batch: Tensor,
        batch_size: int | None = None,
    ) -> Tensor:
        """Return one normalized acceleration per input time, in the same order.

        Args:
            mesh: Normalized XYZ nodes, shape (sum_nodes, 3).
            mesh_batch: Mesh sample membership, shape (sum_nodes,).
            indentor: Normalized impact XY for each sample, shape (B, 2).
            time: Normalized, already-subsampled times, shape (sum_times,).
            time_batch: Time sample membership, shape (sum_times,).
            batch_size: Optional B; inferred from indentor when omitted.
        """
        if batch_size is None:
            if indentor.ndim != 2:
                raise ValueError("indentor must have shape (batch_size, 2)")
            batch_size = indentor.shape[0]
        self._validate_inputs(mesh, mesh_batch, indentor, time, time_batch, batch_size)
        condition = self.impact_embedding(indentor)

        # Encoding each mesh separately avoids padding tens of thousands of
        # nodes. Only the compact memory and short time sequences are batched.
        memory = torch.stack([
            self._encode_mesh(mesh[mesh_batch == b], indentor[b], condition[b])
            for b in range(batch_size)
        ])
        for block in self.latent_blocks:
            memory = block(memory)
        memory = self.memory_norm(memory)

        time_indices = [torch.where(time_batch == b)[0] for b in range(batch_size)]
        sequences = [time[index] for index in time_indices]
        padded_time = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([seq.numel() for seq in sequences], device=time.device)
        padding_mask = torch.arange(padded_time.shape[1], device=time.device)[None, :] >= lengths[:, None]
        queries = self.time_embedding(self.time_features(padded_time))
        decoder_context = self._decoder_context(mesh, mesh_batch, indentor, batch_size)
        for index, block in enumerate(self.decoder_blocks):
            queries = block(queries, memory, padding_mask)
            if decoder_context is not None:
                queries = self._read_decoder_context(index, queries, decoder_context)
        padded_acceleration = self.acceleration_head(queries).squeeze(-1)

        # Restore even interleaved input time batches without cross-sample
        # attention or predictions for padded positions.
        output = padded_acceleration.new_zeros(time.shape)
        for b, index in enumerate(time_indices):
            output = output.index_copy(0, index, padded_acceleration[b, :index.numel()])
        return output
