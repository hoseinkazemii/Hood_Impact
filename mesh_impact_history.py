"""A new attention network for unstructured hood geometry -> acceleration history.

The mesh is an unordered set of XYZ nodes; no grid or element connectivity is
required. Every node participates in impact-conditioned attention. A compact
latent mesh representation drives a decoder with optional temporal attention.

``forward`` accepts the flattened batches used by the shared training utilities.
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


class MeshOnlyDecoderBlock(nn.Module):
    """Read mesh memory without letting the requested times talk to each other.

    The ablation of :class:`HistoryDecoderBlock`: identical except that the
    temporal self-attention stage is gone. Every output uses the mesh memory,
    impact location and its own time, without mixing across requested times.
    It isolates what the impact-conditioned global+local mesh attention carries
    on its own.

    Retained submodules keep their original names, so the ablated state-dict
    keys are a strict subset of the full model's keys.
    """

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
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
        # Keep the common decoder interface. All operations are independent
        # across queries, so padded times cannot affect real predictions.
        del padding_mask
        normalized_memory = self.memory_norm(memory)
        update = self.cross_attention(
            self.query_norm(queries), normalized_memory, normalized_memory,
            need_weights=False,
        )[0]
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.feedforward(self.feedforward_norm(queries)))


# The decoder arms, selected by the ``decoder`` keyword. "temporal" is the
# original network; "mesh_only" is the temporal-attention ablation.
DECODER_BLOCKS = {"temporal": HistoryDecoderBlock, "mesh_only": MeshOnlyDecoderBlock}


IMPACT_CONDITIONING = ("film", "legacy_additive")
TIME_ENCODINGS = ("fourier", "legacy_five")


def saved_model_kwargs(kwargs: dict) -> dict:
    """Reconstruct the original conditioning/time features of older configs."""
    return {"impact_conditioning": "legacy_additive", "time_encoding": "legacy_five", **kwargs}


class ImpactFiLM(nn.Module):
    """Apply impact-dependent scale and shift to every pooling token feature."""

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

    Half of the learned mesh queries have unrestricted global attention. The
    remaining queries learn different radial preferences around the impact XY
    location. Both banks see every node, preserving the original irregular mesh.
    Attention costs O(N * L) for N nodes and L latent tokens; an N-by-N mesh
    attention matrix is never constructed.

    ``decoder`` selects the arm:

    * ``"temporal"`` (default) -- the original network. Each decoder block
      cross-attends to the mesh memory and then runs self-attention across the
      requested times, at O(T * T).
    * ``"mesh_only"`` -- the temporal-attention ablation. The blocks keep the
      cross-attention to mesh memory but never mix along time, so each time
      point is decoded from the geometry, impact location and its own time.
      Use it to ask what the global+local mesh attention carries without
      temporal coupling.

    All trainable weights are new. Only linear layers, attention, normalization,
    activations and dropout are used; there are no convolution layers.

    ``neighborhood_layers > 0`` inserts sparse, physical-XYZ kNN attention
    between the node embedding and global/local pooling. See
    README_mesh_local_sensitivity.md for the
    two-layer experiment with ordinary acceleration MSE.

    New models embed XYZ only and FiLM-condition normalized pooling tokens.
    Time queries use sine/cosine pairs in physical seconds by default.
    ``legacy_five`` reconstructs older time embeddings without adding buffers.
    ``legacy_additive`` reconstructs historical
    checkpoints with six node features and additive impact conditioning.

    ``impactor_nodes`` holds the leading rigid-headform nodes out of that local
    attention. They still reach the pooled memory unchanged; only the local
    graph skips them. See ``_encode_neighborhoods``.
    """

    def __init__(
        self,
        width: int = 128,
        num_heads: int = 4,
        num_latents: int = 256,
        latent_layers: int = 3,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        decoder: str = "temporal",
        neighborhood_layers: int = 0,
        neighborhood_k: int = 16,
        neighborhood_scale_mm: float = 20.0,
        neighborhood_chunk_size: int = 1024,
        impactor_nodes: int = 0,
        impact_conditioning: str = "film",
        time_encoding: str = "fourier",
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
            raise ValueError("num_latents must be at least 2 for global and local queries")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        # isinstance first: a bare membership test on an unhashable value would
        # raise TypeError instead of the intended ValueError.
        if not isinstance(decoder, str) or decoder not in DECODER_BLOCKS:
            raise ValueError(f"decoder must be one of {sorted(DECODER_BLOCKS)}")
        if not isinstance(impact_conditioning, str) or impact_conditioning not in IMPACT_CONDITIONING:
            raise ValueError(f"impact_conditioning must be one of {IMPACT_CONDITIONING}")
        if not isinstance(time_encoding, str) or time_encoding not in TIME_ENCODINGS:
            raise ValueError(f"time_encoding must be one of {TIME_ENCODINGS}")
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
        if not math.isfinite(neighborhood_scale_mm) or neighborhood_scale_mm <= 0:
            raise ValueError("neighborhood_scale_mm must be finite and positive")
        if isinstance(impactor_nodes, bool) or not isinstance(impactor_nodes, int) or impactor_nodes < 0:
            raise ValueError("impactor_nodes must be a nonnegative integer")

        self.decoder = decoder
        self.impact_conditioning = impact_conditioning
        self.time_encoding = time_encoding
        self.width = width
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.num_global_latents = num_latents // 2
        self.attention_dropout = dropout
        self.neighborhood_k = neighborhood_k
        self.impactor_nodes = impactor_nodes

        # The shared preprocessor fits mesh and impact scalers independently.
        # Store their affine maps in the checkpoint so relative coordinates
        # always use one common coordinate system, including after reload.
        self.register_buffer("mesh_mean", torch.zeros(3))
        self.register_buffer("mesh_scale", torch.ones(3))
        self.register_buffer("impact_mean", torch.zeros(2))
        self.register_buffer("impact_scale", torch.ones(2))

        self.node_embedding = nn.Sequential(
            nn.Linear(3 if impact_conditioning == "film" else 6, width),
            nn.GELU(), nn.Linear(width, width),
            nn.LayerNorm(width),
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

        local_count = num_latents - self.num_global_latents
        initial_precision = torch.logspace(math.log10(0.5), math.log10(32.0), local_count)
        self.local_log_precision = nn.Parameter(torch.log(torch.expm1(initial_precision)))
        self.latent_blocks = nn.ModuleList([
            LatentMixingBlock(width, num_heads, dropout) for _ in range(latent_layers)
        ])
        self.memory_norm = nn.LayerNorm(width)

        # Fourier phases use physical seconds, reconstructed from the same
        # training-only scaler used by the dataset. No old features are appended.
        # Old modes retain exactly their parameter/buffer keys for strict replay.
        self.time_feature_count = 2 * fourier_num_frequencies if time_encoding == "fourier" else 5
        if time_encoding == "fourier":
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
        # temporal_layers is the decoder depth: how many times the time queries
        # re-read the mesh memory. Under decoder="mesh_only" that is all it is,
        # since those blocks carry no temporal attention. The name is kept so
        # saved configs and existing checkpoints keep replaying by keyword.
        self.decoder_blocks = nn.ModuleList([
            DECODER_BLOCKS[decoder](width, num_heads, dropout)
            for _ in range(temporal_layers)
        ])
        self.acceleration_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )
        # Add after the original modules: baseline initialization and old
        # checkpoint keys stay identical when neighborhood_layers=0.
        self.neighborhood_blocks = nn.ModuleList()
        self.neighbor_cache = None
        if neighborhood_layers:
            from mesh_neighborhood import NeighborGraphCache, NeighborhoodAttentionBlock
            self.neighborhood_blocks.extend([
                NeighborhoodAttentionBlock(width, num_heads, dropout, neighborhood_scale_mm, neighborhood_chunk_size)
                for _ in range(neighborhood_layers)
            ])
            # Plain attribute, never a buffer: the graph is derived from the
            # inputs, so it must not enter the checkpoint or a state dict.
            self.neighbor_cache = NeighborGraphCache()

        # Append so legacy construction keeps the original parameter order/RNG.
        if impact_conditioning == "film":
            self.impact_film = ImpactFiLM(width)

    @torch.no_grad()
    def set_time_scaler(self, mean: float, scale: float) -> None:
        """Save the training time scaler so Fourier frequencies remain in Hz."""
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Time scaler must have a finite mean and finite positive scale")
        if self.time_encoding == "fourier":
            self.time_mean.fill_(float(mean))
            self.time_scale.fill_(float(scale))

    def time_features(self, normalized_time: Tensor) -> Tensor:
        """Map normalized output coordinates (...,) to features (..., C).

        New runs use [sin(2*pi*f*t_seconds), cos(2*pi*f*t_seconds)] for each
        fixed frequency f, interleaved in ascending frequency order. The
        historical mapping is used only for explicitly selected/saved old runs.
        """
        if self.time_encoding == "legacy_five":
            return torch.stack((normalized_time, normalized_time.square(), normalized_time.pow(3),
                                normalized_time.tanh(), torch.exp(-normalized_time.square())), dim=-1)
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

    def node_features(self, mesh: Tensor, impact: Tensor) -> Tensor:
        """Use XYZ only for FiLM; retain six features for historical runs."""
        if self.impact_conditioning == "film":
            return mesh
        relative_xy = self.impact_relative_xy(mesh, impact)
        radius_squared = relative_xy.square().sum(dim=-1, keepdim=True)
        return torch.cat((mesh, relative_xy, radius_squared), dim=-1)

    def _encode_neighborhoods(self, mesh: Tensor, nodes: Tensor) -> Tensor:
        """Run local attention over the structural nodes, in input order.

        The leading ``impactor_nodes`` rows are the rigid headform. It travels
        with the impact location, which ``indentor`` already states exactly, and
        it is not hood structure, so it takes no part in the local graph. Every
        remaining node still attends, and the held-out rows reach pooling with
        their embedding intact. Excluding them also makes the graph identical
        for every impact on a design, so it is built once and reused.
        """
        # Undo anisotropic standardization; the omitted mean is a common
        # translation and cannot change distances or relative positions.
        coordinates = mesh[self.impactor_nodes:] * self.mesh_scale
        if len(coordinates) < 2:
            raise ValueError(
                f"Neighborhood attention needs at least two structural nodes; "
                f"got {len(coordinates)} after holding out {self.impactor_nodes} impactor nodes"
            )
        neighbors = self.neighbor_cache.neighbors(coordinates, self.neighborhood_k)
        updated = nodes[self.impactor_nodes:]
        for block in self.neighborhood_blocks:
            updated = block(updated, coordinates, neighbors)
        if not self.impactor_nodes:
            return updated
        return torch.cat((nodes[:self.impactor_nodes], updated))

    def _mesh_pooling_inputs(self, mesh: Tensor, impact: Tensor, condition: Tensor):
        features = self.node_features(mesh, impact)
        nodes = self.node_embedding(features)
        if self.neighborhood_blocks:
            nodes = self._encode_neighborhoods(mesh, nodes)
        if self.impact_conditioning == "film":
            # Normalize before modulation so no immediate LayerNorm removes
            # the FiLM scale/shift. Both query and residual use these tokens.
            tokens = self.impact_film(self.query_norm(self.latent_queries), condition)
            query = self.mesh_query(tokens)
        else:
            tokens = self.latent_queries + condition.unsqueeze(0)
            query = self.mesh_query(self.query_norm(tokens))
        head_width = self.width // self.num_heads
        query = query.reshape(self.num_latents, self.num_heads, head_width).transpose(0, 1)
        key, value = self.mesh_key_value(nodes).chunk(2, dim=-1)
        key = key.reshape(-1, self.num_heads, head_width).transpose(0, 1)
        value = value.reshape(-1, self.num_heads, head_width).transpose(0, 1)

        # Global tokens have zero spatial bias. Local tokens have soft learned
        # radial priors, never a hard node cutoff or arbitrary node sampling.
        precision = torch.cat((
            self.local_log_precision.new_zeros(self.num_global_latents),
            F.softplus(self.local_log_precision),
        ))
        radius_squared = self.impact_relative_xy(mesh, impact).square().sum(dim=-1)
        spatial_bias = -precision[:, None] * radius_squared.unsqueeze(0)
        return tokens, query, key, value, spatial_bias

    @torch.no_grad()
    def iter_pooling_attention(self, mesh: Tensor, impact: Tensor, token_chunk_size: int = 16):
        """Yield (first_token, weights[heads, chunk_tokens, nodes]) in eval mode.

        Inputs use the saved training scalers, as in forward(). The weights
        include the learned local radial bias and softmax over ALL mesh nodes.
        No dropout is applied. No weights are retained in the model/checkpoint.
        """
        if self.training:
            raise ValueError("Attention export requires model.eval()")
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        # Generator execution must itself be inside no_grad.
        with torch.no_grad():
            condition = self.impact_embedding(impact)
            _, query, key, _, spatial_bias = self._mesh_pooling_inputs(mesh, impact, condition)
            for start in range(0, self.num_latents, token_chunk_size):
                stop = min(start + token_chunk_size, self.num_latents)
                scores = query[:, start:stop].float() @ key.float().transpose(-1, -2)
                scores = scores / math.sqrt(self.width // self.num_heads)
                yield start, torch.softmax(scores + spatial_bias[None, start:stop].float(), dim=-1)

    def _encode_mesh(self, mesh: Tensor, impact: Tensor, condition: Tensor) -> Tensor:
        tokens, query, key, value, spatial_bias = self._mesh_pooling_inputs(mesh, impact, condition)
        update = F.scaled_dot_product_attention(
            query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
            attn_mask=spatial_bias.to(query.dtype)[None, None, :, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        update = update.squeeze(0).transpose(0, 1).reshape(self.num_latents, self.width)
        return tokens + self.mesh_dropout(self.mesh_output(update))

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
        time_features = self.time_features(padded_time)
        queries = self.time_embedding(time_features)
        if self.impact_conditioning == "legacy_additive":
            queries = queries + condition[:, None, :]
        for block in self.decoder_blocks:
            queries = block(queries, memory, padding_mask)
        padded_acceleration = self.acceleration_head(queries).squeeze(-1)

        # Restore even interleaved input time batches without cross-sample
        # attention or predictions for padded positions.
        output = padded_acceleration.new_zeros(time.shape)
        for b, index in enumerate(time_indices):
            output = output.index_copy(0, index, padded_acceleration[b, :index.numel()])
        return output
