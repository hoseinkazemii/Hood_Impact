"""Common-mode + design-residual reformulation of the hood-impact surrogate.

Implements "The crucial output reformulation" of the literature review
(Literature Review and Research Direction for Geometry-Sensitive Hood-Impact
Surrogate Modeling, pp. 12-13). For each impact location p let

    a_bar_p(t)      = mean over the TRAINING designs of a_{d,p}(t)
    delta a_{d,p}   = a_{d,p}(t) - a_bar_p(t)          (design residual)

and predict with two paths,

    a_hat_{d,p}(t) = c_hat_p(t) + r_hat_{d,p}(t),

* c (common path): impact location only -> the common impact-location response.
  The decoder of MeshImpactHistoryNet behind an impact token (the "none" arm of
  oracle_design_code.DesignCodeHistoryNet).
* r (residual path): MeshImpactHistoryNet fed the design's explicit deviation
  from a reference hood, (G_d - G_ref, p) -> delta a_{d,p}. Node rows are the
  reference hood's coordinates plus ``DEVIATION_CHANNELS`` per node, so the
  edit is visible before any pooling.

The review averages over all 12 designs; here the mean and the reference hoods
use training designs only, so no test curve or test geometry leaks into a
target or an input frame. References are per mesh family (designs 0-5 and 6-11
have different node sets): the node-wise mean of the family's training designs.
A held-out design is described by its deviation from that frame.

Loss, with every term in normalized units:

    L = l_abs   ||a_hat - a||^2 / sigma_a^2
      + l_com   ||c_hat - a_bar||^2 / sigma_a^2              (pins the split)
      + l_delta ||r_hat - delta a||^2 / s_r^2
      + l_pair  sum_p w_p ||(a_hat_i - a_hat_j) - (a_i - a_j)||^2 / s_r^2
      + l_hic   (log(1 + HIC_hat) - log(1 + HIC))^2

The pair term runs over designs i < j at the same impact location in a batch;
the common path cancels in it, so it penalizes only design-to-design error.
w_p = Var_d[HIC_{d,p}] / mean_p Var_d[HIC_{d,p}] over training designs, clipped,
so locations where geometry moves HIC weigh more without dropping the rest.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from mesh_impact_history import MeshImpactHistoryNet
from oracle_design_code import DesignCodeHistoryNet

DEVIATION_CHANNELS = ("dx", "dy", "dz", "abs_dx", "dnx", "dny", "dnz", "dgap",
                      "outer", "inner", "hinge", "other")
DEVIATION_SCALE_MM = 10.0
LOSS_TERMS = ("abs", "common", "delta", "pair", "hic")


def family_references(geometries, train_designs):
    """Node-wise mean hood per mesh family over its training designs.

    ``geometries`` maps design -> design_difference_data.load_design_geometry
    output. Returns ({design: family index}, [reference dicts]) where a reference
    has ids, xyz, normal (unit), gap, panel.
    """
    families, family_of = [], {}
    for design in sorted(geometries):
        ids = geometries[design]["ids"]
        for index, members in enumerate(families):
            if np.array_equal(geometries[members[0]]["ids"], ids):
                members.append(design)
                family_of[design] = index
                break
        else:
            families.append([design])
            family_of[design] = len(families) - 1
    references = []
    for members in families:
        trained = [d for d in members if d in set(train_designs)]
        if not trained:
            raise ValueError(f"mesh family {members} has no training design to define its reference")
        normal = np.mean([geometries[d]["normal"] for d in trained], axis=0)
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        references.append({
            "designs": trained, "ids": geometries[trained[0]]["ids"],
            "xyz": np.mean([geometries[d]["xyz"] for d in trained], axis=0),
            "normal": np.divide(normal, length, out=np.zeros_like(normal), where=length > 0),
            "gap": np.mean([geometries[d]["gap"] for d in trained], axis=0),
            "panel": geometries[trained[0]]["panel"],
        })
    return family_of, references


def deviation_features(geometry, reference):
    """(N, len(DEVIATION_CHANNELS)) explicit deviation of one design from its reference."""
    if not np.array_equal(geometry["ids"], reference["ids"]):
        raise ValueError("design and reference hoods do not share node IDs")
    dx = (geometry["xyz"] - reference["xyz"]) / DEVIATION_SCALE_MM
    dgap = (geometry["gap"] - reference["gap"]) / DEVIATION_SCALE_MM
    panel = np.eye(4)[geometry["panel"]]
    features = np.column_stack((dx, np.linalg.norm(dx, axis=1), geometry["normal"] - reference["normal"],
                                dgap, panel))
    return features.astype(np.float32)


def torch_hic(acceleration_g: Tensor, times_s: Tensor, window: float = 0.015) -> Tensor:
    """Differentiable HIC over every admissible window, as hic15.batched_hic."""
    widths = times_s[1:] - times_s[:-1]
    integral = torch.cat((acceleration_g.new_zeros(*acceleration_g.shape[:-1], 1),
                          torch.cumsum(0.5 * (acceleration_g[..., 1:] + acceleration_g[..., :-1]) * widths, -1)), -1)
    best = acceleration_g.new_zeros(acceleration_g.shape[:-1])
    for lag in range(1, len(times_s)):
        duration = times_s[lag:] - times_s[:-lag]
        valid = duration <= window
        if not bool(valid.any()):
            break
        mean = (integral[..., lag:] - integral[..., :-lag])[..., valid] / duration[valid]
        best = torch.maximum(best, (torch.relu(mean) ** 2.5 * duration[valid]).amax(-1))
    return best


class CommonResidualNet(nn.Module):
    """a_hat = mu + sigma * common(p, t) + s_r * residual(G_ref, G_d - G_ref, p, t)."""

    def __init__(self, residual_kwargs: dict, common_width: int = 128, common_dropout: float = 0.1):
        super().__init__()
        self.common = DesignCodeHistoryNet(num_codes=0, width=common_width, dropout=common_dropout)
        self.residual = MeshImpactHistoryNet(**{**residual_kwargs, "node_extra_features": len(DEVIATION_CHANNELS)})

    def forward(self, mesh: Tensor, mesh_batch: Tensor, indentor: Tensor, time_normalized: Tensor,
                time_seconds: Tensor) -> tuple[Tensor, Tensor]:
        """Return (common, residual), each (B, T) in its own normalized units.

        mesh (sum_nodes, 3 + len(DEVIATION_CHANNELS)): normalized reference XYZ +
        deviation channels; indentor (B, 2) normalized; one shared time grid.
        """
        batch, steps = indentor.shape[0], time_seconds.shape[0]
        common = self.common(indentor, torch.zeros(batch, dtype=torch.long, device=indentor.device), time_seconds)
        time_batch = torch.arange(batch, device=indentor.device).repeat_interleave(steps)
        residual = self.residual(mesh, mesh_batch, indentor, time_normalized.repeat(batch), time_batch, batch)
        return common, residual.view(batch, steps)


def reformulation_losses(common: Tensor, residual: Tensor, target_g: Tensor, mean_g: Tensor, group: Tensor,
                         pair_weight: Tensor, scales: dict, weights: dict, times_s: Tensor):
    """Total loss and its terms for one batch.

    common, residual: (B, T) model outputs; target_g: a_{d,p}; mean_g: a_bar_p;
    group: (B,) location id (pairs are formed within a group); pair_weight: (B,)
    w_p of each sample's location; scales: mu, sigma (acceleration), residual
    (s_r); weights: lambda per LOSS_TERMS.
    """
    mu, sigma, residual_scale = scales["mu"], scales["sigma"], scales["residual"]
    prediction_g = mu + sigma * common + residual_scale * residual
    true_residual = (target_g - mean_g) / residual_scale
    error = residual - true_residual
    terms = {
        "abs": torch.mean(((prediction_g - target_g) / sigma) ** 2),
        "common": torch.mean((common - (mean_g - mu) / sigma) ** 2),
        "delta": torch.mean(error ** 2),
    }
    count = group.shape[0]
    upper = torch.ones(count, count, device=group.device).triu(1).bool()
    same = (group[:, None] == group[None, :]) & upper
    if bool(same.any()):
        pair_error = ((error[:, None, :] - error[None, :, :]) ** 2).mean(-1)
        pair_weights = pair_weight[:, None].expand_as(pair_error)[same]
        terms["pair"] = (pair_error[same] * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)
    else:
        terms["pair"] = error.new_zeros(())
    if weights.get("hic", 0.0) > 0:
        terms["hic"] = torch.mean((torch.log1p(torch_hic(prediction_g, times_s))
                                   - torch.log1p(torch_hic(target_g, times_s))) ** 2)
    else:
        terms["hic"] = error.new_zeros(())
    total = sum(weights.get(name, 0.0) * value for name, value in terms.items())
    return total, {name: float(value.detach()) for name, value in terms.items()}, prediction_g


def hic_pair_weights(hic, train_designs, maximum=5.0):
    """w_p = Var over training designs of HIC at p, divided by its mean over p, clipped to maximum."""
    variance = np.var(np.asarray(hic)[list(train_designs)], axis=0)
    weights = variance / max(variance.mean(), 1e-12)
    return np.clip(weights, 0.0, maximum) if math.isfinite(maximum) else weights
