"""DesignChangeHistoryNet: the backbone plus a decoder readout of the edited nodes.

Pooling compresses the ~18k inner-panel nodes into 256 tokens, and a local edit
moves only 50-377 of them, so the pooled memory hardly changes between a design
and its edits. Here the edited ("change") nodes skip pooling: each becomes its
own token, and after every decoder block the time queries read these tokens
with a second cross-attention (:class:`ChangeReadout`) whose output projection
starts at zero, so a new model starts as exactly the backbone.

Token inputs (``design_change_features``):

* ``"edit"`` (default, since 2026-11-24): the node's position in the reference
  design, normalized as the backbone's mesh input, and its 3D displacement
  from there divided by one isotropic length, ``change_displacement_scale``
  (the trainer sets it to the RMS displacement length of the training
  designs' change nodes). Nothing in it is tied to an axis or to the hood.
  The null token's value is zero, so a design without change nodes gets no
  update from the readouts; its key still lets attention look away from the
  edit, which sets how strongly the edit acts at each time.
* ``"xyz"``: the node's coordinates in this design, normalized as the mesh
  (runs before 2026-11-24).

A design without change nodes (the base design itself) has only the null
token. With "edit" tokens its value is zero, so every readout adds exactly
nothing and the prediction is the backbone's. Training on the base design
(``--train-on-base-design``) therefore teaches the backbone the unedited
response and leaves the readouts to carry the edits.

Change locality (``change_locality=True``, since 2026-11-28): each readout head
lowers its attention score for a change node by ``w * r**2``, where ``r`` is the
node's distance from the load point (the impact point) and ``w`` a learned
per-head rate (stored as log w). The null token is not penalized, so an edit far
from the impact sends the attention to the null token and adds nothing. Distances
are taken in the coordinates the load point is given in (x, y for the hood's
vertical impact), in the backbone's normalized mesh units, so no length is set
by hand. Starting rates are spread over the heads (0.5 to 32, ranges of about
1.4 to 0.18 normalized units) and every rate is learned; a head that needs a
long reach drives its rate to zero. Use it only when the load location is an
input that varies between samples. A problem with one fixed load (the B1-B4
benchmarks) has nothing to measure distances from; leave it off there, which
is exactly the readout without locality.

The model recognizes a hood by its normalized coordinates, as the
neighbor-graph cache does. ``design_changes`` (a :class:`DesignChangeTable`)
maps the digest of each known hood to its change nodes. The trainer fills it
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

from mesh_impact_history import MeshImpactHistoryNet, saved_model_kwargs
from mesh_neighborhood import NeighborGraphCache

TABLE_FILE = "design_change_nodes.npz"
# Columns of a design change's "static" rows: reference-hood position and the
# design's displacement from it, in mm.
STATIC_CHANNELS = ("base_x", "base_y", "base_z", "dx", "dy", "dz")
# Tables saved before 2026-11-27 also carried three panel-gap channels, zero on
# the local-edit set; DesignChangeTable.load drops them.
LEGACY_STATIC_CHANNELS = (*STATIC_CHANNELS, "base_gap", "gap_change", "outer")
CHANGE_FEATURES = ("edit", "xyz")
CHANGE_INPUTS = {"xyz": 3, "edit": 6}
# Keyword arguments of earlier saved runs that now have only one supported value.
RETIRED_KWARGS = {"design_change_layers": 0, "design_change_displacement_gate": False}
IGNORED_KWARGS = ("design_change_fourier_octaves",)
# Starting locality rates w of the readout heads (normalized mesh units^-2),
# log-spaced over the heads, like the radial pooling priors of earlier runs.
CHANGE_LOCALITY_INITIAL_RATES = (0.5, 32.0)


def saved_design_change_kwargs(kwargs: dict, *, format_version: int) -> dict:
    """Constructor kwargs of a saved DesignChangeHistoryNet run (see saved_model_kwargs)."""
    kwargs = saved_model_kwargs(kwargs, format_version=format_version)
    if format_version == 1:
        # Configs saved before these keys existed used impact-relative tokens and the gate.
        kwargs = {"design_change_features": "impact_relative", "design_change_displacement_gate": True, **kwargs}
    if kwargs.get("design_change_features") not in CHANGE_FEATURES:
        raise ValueError("This run used impact-relative change tokens (before 2026-10-26), "
                         "which were removed from the code")
    for key, supported in RETIRED_KWARGS.items():
        value = kwargs.pop(key, supported)
        if value != supported:
            raise ValueError(f"This run used {key}={value!r}, an option removed from the code")
    for key in IGNORED_KWARGS:
        kwargs.pop(key, None)
    return kwargs


class DesignChangeTable:
    """Change-node rows of every known hood, found from the hood's coordinates.

    ``static`` is (designs, max nodes, len(STATIC_CHANNELS)) in mm, zero-padded
    past ``counts``. ``register`` binds the normalized mesh of a design to its
    row of the table.
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
        """Build from local_edits_data.load_design_changes output."""
        designs = sorted(changes)
        counts = [len(changes[design]["static"]) for design in designs]
        static = np.zeros((len(designs), max(counts + [1]), len(STATIC_CHANNELS)), np.float32)
        for index, design in enumerate(designs):
            static[index, :counts[index]] = changes[design]["static"]
        return cls(designs, static, counts)

    @staticmethod
    def digest(mesh):
        return NeighborGraphCache.geometry_digest(NeighborGraphCache.points(mesh))

    def register(self, mesh, design):
        matches = np.flatnonzero(self.designs == design)
        if not len(matches):
            raise KeyError(f"Design {design} is not in the design-change table")
        key, index = self.digest(mesh), int(matches[0])
        if self.index_of.setdefault(key, index) != index:
            raise ValueError(f"Design {design} has the same hood coordinates as design "
                             f"{self.designs[self.index_of[key]]}")
        return key

    def lookup(self, mesh):
        index = self.index_of.get(self.digest(mesh))
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
            channels, static = tuple(saved["channels"].tolist()), saved["static"]
            if channels == LEGACY_STATIC_CHANNELS and not static[..., len(STATIC_CHANNELS):].any():
                static, channels = static[..., :len(STATIC_CHANNELS)], STATIC_CHANNELS
            if channels != STATIC_CHANNELS:
                raise ValueError(f"{path} stores channels {list(channels)}, expected {list(STATIC_CHANNELS)}")
            index_of = dict(zip(saved["digests"].tolist(), saved["digest_index"].tolist()))
            return cls(saved["designs"], static, saved["counts"], index_of)

    def same_as(self, other):
        return (np.array_equal(self.designs, other.designs) and np.array_equal(self.counts, other.counts)
                and np.array_equal(self.static, other.static) and self.index_of == other.index_of)


class ChangeReadout(nn.Module):
    """Time queries read the change-node tokens. The output projection starts at zero.

    With a ``gate`` each value is multiplied by it; "edit" tokens gate only the
    null token, by zero, so a design without an edit gets no update.
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

    def forward(self, queries: Tensor, tokens: Tensor, gate: Tensor | None, valid: Tensor,
                bias: Tensor | None = None) -> Tensor:
        """queries (B, T, W); tokens (B, M, W); gate (B, M) or None; valid (B, M);
        bias (B, heads, M) or None, added to every query's attention scores."""
        batch, steps, width = queries.shape

        def heads(values):
            return values.reshape(batch, -1, self.num_heads, self.head_width).transpose(1, 2)

        key, value = self.key_value(tokens).chunk(2, dim=-1)
        if gate is not None:
            value = value * gate.unsqueeze(-1)
        if bias is None:
            mask = valid[:, None, None, :]
        else:
            mask = bias.masked_fill(~valid[:, None, :], float("-inf"))[:, :, None, :].to(queries.dtype)
        update = F.scaled_dot_product_attention(
            heads(self.query(self.query_norm(queries))), heads(key), heads(value),
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return queries + self.dropout(self.output(update.transpose(1, 2).reshape(batch, steps, width)))


class DesignChangeHistoryNet(MeshImpactHistoryNet):
    """MeshImpactHistoryNet whose decoder also reads the design-change nodes directly."""

    def __init__(self, design_change_features: str = "edit", change_locality: bool = False, **kwargs):
        if design_change_features not in CHANGE_FEATURES:
            raise ValueError(f"design_change_features must be one of {CHANGE_FEATURES}; got {design_change_features!r}")
        if not isinstance(change_locality, bool):
            raise ValueError(f"change_locality must be True or False; got {change_locality!r}")
        super().__init__(**kwargs)
        self.design_change_features = design_change_features
        self.change_locality = change_locality
        # Added after every backbone module, so one seed gives the backbone the
        # same initial weights as MeshImpactHistoryNet.
        width = self.width
        self.change_embedding = nn.Sequential(
            nn.Linear(CHANGE_INPUTS[design_change_features], width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.change_null_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.change_null_token, std=0.02)
        self.change_norm = nn.LayerNorm(width)
        self.change_readouts = nn.ModuleList([
            ChangeReadout(width, self.num_heads, self.attention_dropout) for _ in self.decoder_blocks
        ])
        # Plain attribute, never a buffer: it is derived from the decks.
        self.design_changes = None
        if design_change_features == "edit":
            # Set from the training designs before use; NaN until then.
            self.register_buffer("change_displacement_scale", torch.full((), float("nan")))
        self.change_log_rate = None
        if change_locality:
            # (readouts, heads) log of the rate w in the score penalty w * r^2. Drawn
            # from no random generator, so seeded weights are unchanged.
            low, high = CHANGE_LOCALITY_INITIAL_RATES
            rates = torch.logspace(math.log10(low), math.log10(high), self.num_heads)
            self.change_log_rate = nn.Parameter(rates.log().repeat(len(self.change_readouts), 1))

    @torch.no_grad()
    def set_change_displacement_scale(self, scale_mm: float) -> None:
        """The one length (mm) every displacement of an "edit" token is divided by."""
        if self.design_change_features != "edit":
            raise ValueError("Only design_change_features='edit' tokens carry displacements")
        if not math.isfinite(scale_mm) or scale_mm <= 0:
            raise ValueError("The change displacement scale must be finite and positive")
        self.change_displacement_scale.fill_(float(scale_mm))

    def change_features(self, static: Tensor) -> Tensor:
        """Token inputs from static rows (B, M, 6) in mm.

        "edit": (B, M, 6), the node's reference position normalized as the mesh,
        then its displacement / change_displacement_scale. "xyz": (B, M, 3), the
        node's coordinates in this design, normalized as the mesh.
        """
        base, displacement = static[..., :3], static[..., 3:6]
        if self.design_change_features == "xyz":
            return (base + displacement - self.mesh_mean) / self.mesh_scale
        if not torch.isfinite(self.change_displacement_scale):
            raise RuntimeError("Set the edit tokens' displacement scale (set_change_displacement_scale) first")
        return torch.cat(((base - self.mesh_mean) / self.mesh_scale,
                          displacement / self.change_displacement_scale), dim=-1)

    def _decoder_context(self, mesh: Tensor, mesh_batch: Tensor, indentor: Tensor, batch_size: int):
        if self.design_changes is None:
            raise RuntimeError("DesignChangeHistoryNet has no design_changes table; the trainer builds it "
                               f"from the decks, and saved runs keep it in {TABLE_FILE}")
        index = torch.tensor([self.design_changes.lookup(mesh[mesh_batch == b]) for b in range(batch_size)],
                             device=mesh.device)
        static, valid = self.design_changes.tensors(mesh.device)
        static, valid = static[index].to(mesh.dtype), valid[index]
        tokens = self.change_embedding(self.change_features(static))
        tokens = torch.cat((self.change_null_token.expand(batch_size, -1, -1), tokens), dim=1)
        valid = torch.cat((valid.new_ones(batch_size, 1), valid), dim=1)
        gate = None
        if self.design_change_features == "edit":
            # Zero value for the null token only: no edit, no update.
            gate = torch.cat((static.new_zeros(batch_size, 1), static.new_ones(batch_size, static.shape[1])), dim=1)
        distance_squared = self.change_distance_squared(static, indentor) if self.change_locality else None
        return self.change_norm(tokens), gate, valid, distance_squared

    def change_distance_squared(self, static: Tensor, indentor: Tensor) -> Tensor:
        """(B, M) squared distance of each change node from the load point, in normalized mesh units.

        The load point (``indentor``, normalized by the impact scaler) has D
        coordinates; they are the first D mesh axes (x, y for the hood), and the
        distance uses those axes. Both points are mapped to the mesh scaler's
        units, the coordinates the backbone itself reads.
        """
        dims = indentor.shape[1]
        load = (indentor * self.impact_scale + self.impact_mean - self.mesh_mean[:dims]) / self.mesh_scale[:dims]
        position = (static[..., :dims] - self.mesh_mean[:dims]) / self.mesh_scale[:dims]
        return (position - load[:, None, :]).square().sum(dim=-1)

    def _read_decoder_context(self, index: int, queries: Tensor, context) -> Tensor:
        tokens, gate, valid, distance_squared = context
        bias = None
        if distance_squared is not None:
            # The null token (column 0) is never penalized.
            rate = self.change_log_rate[index].exp()
            penalty = -rate[None, :, None] * distance_squared[:, None, :]
            bias = torch.cat((penalty.new_zeros(penalty.shape[0], penalty.shape[1], 1), penalty), dim=2)
        return self.change_readouts[index](queries, tokens, gate, valid, bias)


def change_displacement_rms(changes, designs) -> float:
    """RMS displacement length (mm) of the moved change nodes of ``designs``: the "edit" tokens' scale."""
    lengths = [np.linalg.norm(np.asarray(changes[design]["static"][:, 3:6], dtype=np.float64), axis=1)
               for design in designs if design in changes]
    lengths = np.concatenate(lengths) if lengths else np.zeros(0)
    lengths = lengths[lengths > 0]
    if not len(lengths):
        raise ValueError("The training designs move no change node, so edit tokens have no displacement scale")
    return float(np.sqrt(np.mean(np.square(lengths))))


def register_design_changes(model, changes, meshes, designs, transform_mesh):
    """Fill ``model.design_changes`` for the loaded runs and return a summary.

    meshes: raw (N, 3) meshes of the runs; designs: each run's design;
    transform_mesh: the training scaler. Every run's change rows are checked
    against its own mesh, so a table built from the wrong decks cannot load.
    """
    table = DesignChangeTable.from_changes(changes)
    checked, worst = set(), 0.0
    for mesh, design in zip(meshes, designs):
        change = changes[design]
        expected = change["static"][:, :3] + change["static"][:, 3:6]
        error = float(np.abs(np.asarray(mesh)[change["rows"]] - expected).max()) if len(expected) else 0.0
        if error > 1e-2:
            raise ValueError(f"Design {design}: mesh rows differ from the deck change nodes by {error:.3g} mm")
        worst = max(worst, error)
        normalized = np.asarray(transform_mesh(np.asarray(mesh)), dtype=np.float32)
        checked.add(table.register(normalized, design))
    model.design_changes = table
    return {"registered_hoods": len(checked), "registered_designs": sorted({int(d) for d in designs}),
            "max_row_check_error_mm": worst, "table_file": TABLE_FILE}
