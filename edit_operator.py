"""Difference-first surrogate: predict how a local hood edit changes the response.

Given a baseline hood (design a, already simulated) and an edited hood b of the
same mesh family, predict d(t) = a_b(t) - a_a(t) at impact point p directly
from the edit, instead of encoding two nearly identical 38k-node meshes and
subtracting. Only the edited nodes enter (50-220 moved nodes plus nodes whose
panel gap changed); nothing is pooled over the unchanged hood, so the edit
cannot be diluted.

Each edited node i carries
    r_i = x_i - p            position relative to the impact point (mid-edit geometry)
    f_i                      panel (outer/inner/hinge/other), unit normal, panel gap,
                             lumped area -- all evaluated on the mid-edit geometry,
                             so they are the same for a -> b and b -> a
    e_i = (dx, dgap)         the edit itself, which flips sign for b -> a
and the impact point carries a context c (location, start penetration, in
contact; optionally the baseline's own response, see ``context_dim``).

Operators (``operator``):

``linear``  first-order perturbation operator
    d(t) = sum_m B_m(t) sum_i sum_k K_mk(r_i, f_i, c) e_ik
    A learned sensitivity kernel K, evaluated per node, contracted with the edit
    and expanded on a learned temporal basis B. Linear in the edit, so a zero
    edit gives exactly zero, swapping a and b flips the sign, and differences
    compose (d_ac = d_ab + d_bc to first order).
``attention``  nonlinear: the impact query cross-attends to edit tokens
    (Perceiver/GINOT style, no graph), antisymmetrized as
    d = (F(e) - F(-e)) / 2 so the same identities hold exactly.
``none``  no geometry term (for the time-step-only ablation).

``nuisance="dt"`` adds rho * N(c)(t) with rho = log(dt_b / dt_a): the solver
time-step artifact, separated so the geometric term need not absorb it. It is
also antisymmetric. Reported predictions with and without it show how much of
the fit is solver artifact.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

OPERATORS = ("linear", "attention", "none")
NUISANCES = ("none", "dt")
NODE_FEATURES = 9     # panel one-hot (4), normal (3), gap / 50 mm, area / mean area
EDIT_CHANNELS = 4     # dx, dy, dz, dgap (mm / 10)
EDIT_NORMALIZER = 100.0


class RelativePositionEncoding(nn.Module):
    """Multi-scale sin/cos of x - p plus radial terms, in units of ``length_scale_mm``."""

    def __init__(self, length_scale_mm: float = 100.0, octaves: int = 6):
        super().__init__()
        self.length_scale_mm = length_scale_mm
        # Wavelengths from 8 L down to L / 4 (800 mm to 25 mm at L = 100 mm).
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(-2, octaves - 2, dtype=torch.float32),
                             persistent=False)
        self.dim = 3 + 6 * octaves + 3

    def forward(self, relative_mm: Tensor) -> Tensor:
        u = relative_mm / self.length_scale_mm
        phases = (u.unsqueeze(-1) * self.frequencies).flatten(-2)
        radius = torch.linalg.vector_norm(u[..., :2], dim=-1, keepdim=True)
        radial = torch.cat((radius, torch.exp(-radius), torch.exp(-radius.square())), dim=-1)
        return torch.cat((u, phases.sin(), phases.cos(), radial), dim=-1)


def mlp(inputs: int, width: int, outputs: int, layers: int = 2) -> nn.Sequential:
    modules, size = [], inputs
    for _ in range(layers):
        modules += [nn.Linear(size, width), nn.GELU()]
        size = width
    modules.append(nn.Linear(size, outputs))
    return nn.Sequential(*modules)


class EditOperator(nn.Module):
    """d^(t) = geometric term (linear | attention | none) + optional time-step nuisance."""

    def __init__(self, num_times: int, context_dim: int, operator: str = "linear", nuisance: str = "dt",
                 width: int = 128, basis_size: int = 16, num_heads: int = 4, attention_layers: int = 2,
                 length_scale_mm: float = 100.0, dropout: float = 0.0):
        super().__init__()
        if operator not in OPERATORS:
            raise ValueError(f"operator must be one of {OPERATORS}")
        if nuisance not in NUISANCES:
            raise ValueError(f"nuisance must be one of {NUISANCES}")
        if operator == "none" and nuisance == "none":
            raise ValueError("a model needs a geometric operator, a nuisance term, or both")
        self.operator, self.nuisance, self.basis_size = operator, nuisance, basis_size
        self.position = RelativePositionEncoding(length_scale_mm)
        self.basis = nn.Parameter(0.1 * torch.randn(basis_size, num_times))
        token_inputs = self.position.dim + NODE_FEATURES + context_dim
        if operator == "linear":
            self.kernel = mlp(token_inputs, width, basis_size * EDIT_CHANNELS, layers=3)
        elif operator == "attention":
            self.token = mlp(self.position.dim + NODE_FEATURES + EDIT_CHANNELS + 1, width, width)
            self.query = mlp(context_dim, width, width)
            self.blocks = nn.ModuleList()
            for _ in range(attention_layers):
                self.blocks.append(nn.ModuleDict({
                    "query_norm": nn.LayerNorm(width), "token_norm": nn.LayerNorm(width),
                    "attention": nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True),
                    "feedforward_norm": nn.LayerNorm(width), "feedforward": mlp(width, 4 * width, width, layers=1),
                }))
            self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, basis_size))
        if nuisance == "dt":
            self.nuisance_net = mlp(context_dim, width, basis_size)

    @torch.no_grad()
    def set_temporal_basis(self, components: Tensor) -> None:
        """Start the temporal basis from principal components of training differences."""
        if components.shape != self.basis.shape:
            raise ValueError(f"basis must have shape {tuple(self.basis.shape)}")
        self.basis.copy_(components)

    def geometric_coefficients(self, relative: Tensor, features: Tensor, edit: Tensor, mask: Tensor,
                               context: Tensor) -> Tensor:
        """(B, M) basis coefficients of the geometric term."""
        position = self.position(relative)
        if self.operator == "linear":
            nodes = relative.shape[1]
            inputs = torch.cat((position, features, context[:, None].expand(-1, nodes, -1)), dim=-1)
            kernel = self.kernel(inputs).unflatten(-1, (self.basis_size, EDIT_CHANNELS))
            weighted = (kernel * edit.unsqueeze(-2)).sum(-1) * mask.unsqueeze(-1)
            return weighted.sum(1) / EDIT_NORMALIZER
        return 0.5 * (self._attend(position, features, edit, mask, context)
                      - self._attend(position, features, -edit, mask, context))

    def _attend(self, position, features, edit, mask, context):
        magnitude = torch.linalg.vector_norm(edit[..., :3], dim=-1, keepdim=True)
        tokens = self.token(torch.cat((position, features, edit, magnitude), dim=-1))
        query = self.query(context)[:, None]
        padding = ~mask
        for block in self.blocks:
            normalized = block["token_norm"](tokens)
            update = block["attention"](block["query_norm"](query), normalized, normalized,
                                        key_padding_mask=padding, need_weights=False)[0]
            query = query + update
            query = query + block["feedforward"](block["feedforward_norm"](query))
        return self.readout(query[:, 0])

    def forward(self, relative: Tensor, features: Tensor, edit: Tensor, mask: Tensor, context: Tensor,
                log_dt_ratio: Tensor) -> tuple[Tensor, Tensor]:
        """Return (geometric, nuisance) difference curves, each (B, T), normalized units.

        relative (B, N, 3) mm, features (B, N, NODE_FEATURES), edit (B, N, EDIT_CHANNELS),
        mask (B, N) bool (False = padding), context (B, C), log_dt_ratio (B,).
        """
        batch = context.shape[0]
        zero = context.new_zeros(batch, self.basis.shape[1])
        geometric = zero
        if self.operator != "none":
            geometric = self.geometric_coefficients(relative, features, edit, mask, context) @ self.basis
        nuisance = zero
        if self.nuisance == "dt":
            nuisance = (log_dt_ratio[:, None] * self.nuisance_net(context)) @ self.basis
        return geometric, nuisance
