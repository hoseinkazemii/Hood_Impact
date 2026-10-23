"""MeshImpactHistoryNet plus a decoder path that reads the design-change nodes.

Pooling compresses the ~38k hood nodes into 256 tokens, and near-clone designs
differ at only 110-333 of them, so the pooled memory hardly changes between
them (0.5% relative after pooling against 0.8% at the node embedding; probe of
run 20261016_151923_3394500_1704). Here the change nodes found by
design_change_nodes.py skip pooling. Each one becomes its own token, built from:

* its position relative to the impact point: XY offset and distance
  (/ 250 mm), and sin/cos of the offset at wavelengths 800 / 2^k mm;
* its displacement from the cluster base (/ 10 mm) and the displacement length;
* the base inner-outer panel gap and its change (/ 10 mm), and an outer-panel flag.

The impact-relative position is the part meant to transfer to held-out
designs. It describes "a patch raised 10 mm, 40 mm from the head", not where
on this particular hood the patch sits.

``design_change_layers`` self-attention blocks run over each design's change
tokens plus one learned null token. After every decoder block the time queries
read these tokens with a second cross-attention (:class:`ChangeReadout`). Its
values are scaled by each node's displacement, and its output projection starts
at zero. So:

* a new model starts as exactly the backbone, and
* a design with no displacement (each cluster's base: 0, 4, 6, 10) always gets
  exactly the backbone prediction. The path predicts only the effect of an edit.

Training uses the backbone's plain acceleration MSE.

The model recognizes a hood by its normalized coordinates, as the neighbor-graph
cache does. ``design_changes`` (a :class:`DesignChangeTable`) maps the digest
of each known hood's structural rows to its change nodes. The trainer fills it
and saves it beside the checkpoint (design_change_nodes.npz); it is never part
of the state dict.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from design_change_nodes import STATIC_CHANNELS
from mesh_impact_history import LatentMixingBlock, MeshImpactHistoryNet
from mesh_neighborhood import NeighborGraphCache

CHANGE_POSITION_SCALE_MM = 250.0
CHANGE_DISPLACEMENT_SCALE_MM = 10.0
CHANGE_LONGEST_WAVELENGTH_MM = 800.0
TABLE_FILE = "design_change_nodes.npz"


class DesignChangeTable:
    """Change-node features of every known hood, found from the hood's coordinates.

    ``static`` is (designs, max nodes, len(STATIC_CHANNELS)) in mm, zero-padded
    past ``counts``. ``register`` binds the normalized structural rows (headform
    removed) of a design's mesh to its row of the table.
    """

    def __init__(self, designs, static, counts, index_of=None):
        self.designs = np.asarray(designs, dtype=np.int64)
        self.static = np.asarray(static, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.int64)
        if self.static.ndim != 3 or self.static.shape[0] != len(self.designs) or self.static.shape[2] != len(STATIC_CHANNELS):
            raise ValueError(f"static must have shape (designs, nodes, {len(STATIC_CHANNELS)})")
        if self.counts.shape != self.designs.shape or (self.counts > self.static.shape[1]).any():
            raise ValueError("counts must give one node count per design, within the padded width")
        self.index_of = dict(index_of or {})
        self._tensors = {}

    @classmethod
    def from_changes(cls, changes):
        """Build from design_change_nodes.find_design_changes output."""
        designs = sorted(changes)
        counts = [len(changes[design]["static"]) for design in designs]
        static = np.zeros((len(designs), max(counts + [1]), len(STATIC_CHANNELS)), np.float32)
        for index, design in enumerate(designs):
            static[index, :counts[index]] = changes[design]["static"]
        return cls(designs, static, counts)

    @staticmethod
    def digest(structural):
        return NeighborGraphCache.geometry_digest(NeighborGraphCache.points(structural))

    def register(self, structural, design):
        matches = np.flatnonzero(self.designs == design)
        if not len(matches):
            raise KeyError(f"Design {design} is not in the design-change table")
        key, index = self.digest(structural), int(matches[0])
        if self.index_of.setdefault(key, index) != index:
            raise ValueError(f"Design {design} has the same hood coordinates as design "
                             f"{self.designs[self.index_of[key]]}")
        return key

    def lookup(self, structural):
        index = self.index_of.get(self.digest(structural))
        if index is None:
            raise KeyError("This hood is not in the design-change table. The table holds the decks the "
                           "model was trained with; register a new hood with DesignChangeTable.register.")
        return index

    def tensors(self, device):
        """(static, valid) on ``device``: (designs, nodes, channels) and (designs, nodes)."""
        key = str(device)
        if key not in self._tensors:
            static = torch.as_tensor(self.static, device=device)
            counts = torch.as_tensor(self.counts, device=device)
            valid = torch.arange(static.shape[1], device=device)[None, :] < counts[:, None]
            self._tensors[key] = (static, valid)
        return self._tensors[key]

    def save(self, path):
        keys = sorted(self.index_of)
        np.savez(path, designs=self.designs, static=self.static, counts=self.counts,
                 channels=np.array(STATIC_CHANNELS), digests=np.array(keys, dtype=str),
                 digest_index=np.array([self.index_of[key] for key in keys], dtype=np.int64))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            if tuple(saved["channels"].tolist()) != STATIC_CHANNELS:
                raise ValueError(f"{path} stores channels {saved['channels'].tolist()}, expected {STATIC_CHANNELS}")
            index_of = dict(zip(saved["digests"].tolist(), saved["digest_index"].tolist()))
            return cls(saved["designs"], saved["static"], saved["counts"], index_of)

    def same_as(self, other):
        return (np.array_equal(self.designs, other.designs) and np.array_equal(self.counts, other.counts)
                and np.array_equal(self.static, other.static) and self.index_of == other.index_of)


class ChangeReadout(nn.Module):
    """Time queries read the change-node tokens, with displacement-scaled values.

    A node's value is multiplied by its displacement length / 10 mm, and the
    null token's by zero. A query can therefore read nothing, and a design with
    no displacement adds nothing. The output projection starts at zero.
    """

    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.dropout_p = dropout
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        # No bias: zero values must give a zero update.
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, tokens: Tensor, gate: Tensor, valid: Tensor) -> Tensor:
        """queries (B, T, W); tokens (B, M, W); gate and valid (B, M)."""
        batch, steps, width = queries.shape

        def heads(values):
            return values.reshape(batch, -1, self.num_heads, self.head_width).transpose(1, 2)

        key, value = self.key_value(tokens).chunk(2, dim=-1)
        update = F.scaled_dot_product_attention(
            heads(self.query(self.query_norm(queries))), heads(key), heads(value * gate.unsqueeze(-1)),
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return queries + self.dropout(self.output(update.transpose(1, 2).reshape(batch, steps, width)))


class DesignChangeHistoryNet(MeshImpactHistoryNet):
    """MeshImpactHistoryNet whose decoder also reads the design-change nodes directly."""

    def __init__(self, design_change_layers: int = 2, design_change_fourier_octaves: int = 5, **kwargs):
        if isinstance(design_change_layers, bool) or not isinstance(design_change_layers, int) or design_change_layers < 0:
            raise ValueError("design_change_layers must be a nonnegative integer")
        if (isinstance(design_change_fourier_octaves, bool) or not isinstance(design_change_fourier_octaves, int)
                or design_change_fourier_octaves < 1):
            raise ValueError("design_change_fourier_octaves must be a positive integer")
        super().__init__(**kwargs)
        if self.node_extra_features:
            raise ValueError("DesignChangeHistoryNet reads XYZ meshes; node_extra_features must be 0")
        self.design_change_layers = design_change_layers
        # Added after every backbone module, so one seed gives the backbone the
        # same initial weights as MeshImpactHistoryNet.
        frequencies = 2 * math.pi * 2.0 ** torch.arange(design_change_fourier_octaves) / CHANGE_LONGEST_WAVELENGTH_MM
        self.register_buffer("change_frequencies", frequencies.float(), persistent=False)
        width = self.width
        self.change_embedding = nn.Sequential(
            nn.Linear(10 + 4 * design_change_fourier_octaves, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.change_null_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.change_null_token, std=0.02)
        self.change_blocks = nn.ModuleList([
            LatentMixingBlock(width, self.num_heads, self.attention_dropout) for _ in range(design_change_layers)
        ])
        self.change_norm = nn.LayerNorm(width)
        self.change_readouts = nn.ModuleList([
            ChangeReadout(width, self.num_heads, self.attention_dropout) for _ in self.decoder_blocks
        ])
        # Plain attribute, never a buffer: it is derived from the decks.
        self.design_changes = None

    def change_features(self, static: Tensor, impact_mm: Tensor) -> Tensor:
        """(B, M, 10 + 4 * octaves) token inputs from static rows (B, M, S) and impact XY (B, 2) in mm."""
        base, displacement = static[..., :3], static[..., 3:6]
        relative = base[..., :2] - impact_mm[:, None, :]
        phases = (relative.unsqueeze(-1) * self.change_frequencies).flatten(2)
        return torch.cat((
            relative / CHANGE_POSITION_SCALE_MM,
            relative.norm(dim=-1, keepdim=True) / CHANGE_POSITION_SCALE_MM,
            phases.sin(), phases.cos(),
            displacement / CHANGE_DISPLACEMENT_SCALE_MM,
            displacement.norm(dim=-1, keepdim=True) / CHANGE_DISPLACEMENT_SCALE_MM,
            static[..., 6:8] / CHANGE_DISPLACEMENT_SCALE_MM,
            static[..., 8:9],
        ), dim=-1)

    def _decoder_context(self, mesh: Tensor, mesh_batch: Tensor, indentor: Tensor, batch_size: int):
        if self.design_changes is None:
            raise RuntimeError("DesignChangeHistoryNet has no design_changes table; the trainer builds it "
                               f"from the decks, and saved runs keep it in {TABLE_FILE}")
        index = torch.tensor([self.design_changes.lookup(mesh[mesh_batch == b][self.impactor_nodes:])
                              for b in range(batch_size)], device=mesh.device)
        static, valid = self.design_changes.tensors(mesh.device)
        static, valid = static[index].to(mesh.dtype), valid[index]
        impact_mm = indentor * self.impact_scale + self.impact_mean
        tokens = self.change_embedding(self.change_features(static, impact_mm))
        tokens = torch.cat((self.change_null_token.expand(batch_size, -1, -1), tokens), dim=1)
        valid = torch.cat((valid.new_ones(batch_size, 1), valid), dim=1)
        for block in self.change_blocks:
            tokens = block(tokens, padding_mask=~valid)
        gate = static[..., 3:6].norm(dim=-1) / CHANGE_DISPLACEMENT_SCALE_MM
        gate = torch.cat((gate.new_zeros(batch_size, 1), gate), dim=1)
        return self.change_norm(tokens), gate, valid

    def _read_decoder_context(self, index: int, queries: Tensor, context) -> Tensor:
        return self.change_readouts[index](queries, *context)


def register_design_changes(model, changes, meshes, designs, transform_mesh):
    """Fill ``model.design_changes`` for the loaded runs and return a summary.

    meshes: raw (N, 3) meshes of the runs; designs: each run's design;
    transform_mesh: the training scaler. Every run's change rows are checked
    against its own mesh, so a table built from the wrong decks cannot load.
    """
    table = DesignChangeTable.from_changes(changes)
    start = model.impactor_nodes
    checked, worst = set(), 0.0
    for mesh, design in zip(meshes, designs):
        change = changes[design]
        expected = change["static"][:, :3] + change["static"][:, 3:6]
        error = float(np.abs(np.asarray(mesh)[change["rows"]] - expected).max()) if len(expected) else 0.0
        if error > 1e-2:
            raise ValueError(f"Design {design}: mesh rows differ from the deck change nodes by {error:.3g} mm")
        worst = max(worst, error)
        normalized = np.asarray(transform_mesh(np.asarray(mesh)), dtype=np.float32)
        checked.add(table.register(normalized[start:], design))
    model.design_changes = table
    return {"registered_hoods": len(checked), "registered_designs": sorted({int(d) for d in designs}),
            "max_row_check_error_mm": worst, "table_file": TABLE_FILE}
