"""Direct coefficient-times-response correction for a frozen mesh predictor.

The new branch contains only Fourier features, MLPs, and a dot product. Its
geometry coefficients come from a separately fitted, training-only basis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CoefficientHistoryCorrection(nn.Module):
    """Map coefficients (B,R), impact (B,2), and seconds (T,) to delta (B,T).

    Corrections use the baseline's normalized acceleration units. There is no
    design embedding, graph, attention, pooling, or geometry-independent bias
    in the final response. A zero coefficient vector produces exactly zero.
    """

    def __init__(self, coefficient_width, width=128, layers=3,
                 fourier_num_frequencies=6, fourier_min_frequency_hz=20.,
                 fourier_max_frequency_hz=640.):
        super().__init__()
        for name, value in (("coefficient_width", coefficient_width), ("width", width),
                            ("layers", layers), ("fourier_num_frequencies", fourier_num_frequencies)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not np.isfinite([fourier_min_frequency_hz, fourier_max_frequency_hz]).all()
                or not 0 < fourier_min_frequency_hz <= fourier_max_frequency_hz):
            raise ValueError("Fourier frequencies must be finite, positive and ordered")
        self.coefficient_width = coefficient_width
        self.kwargs = dict(coefficient_width=coefficient_width, width=width, layers=layers,
                           fourier_num_frequencies=fourier_num_frequencies,
                           fourier_min_frequency_hz=fourier_min_frequency_hz,
                           fourier_max_frequency_hz=fourier_max_frequency_hz)
        frequencies = np.geomspace(fourier_min_frequency_hz, fourier_max_frequency_hz,
                                   fourier_num_frequencies).astype(np.float32)
        self.register_buffer("frequencies_hz", torch.from_numpy(frequencies))
        modules = []
        incoming = 2 + 2 * fourier_num_frequencies
        for _ in range(layers):
            modules.extend((nn.Linear(incoming, width), nn.GELU()))
            incoming = width
        self.response_functions = nn.Sequential(*modules, nn.Linear(width, coefficient_width))
        # The first prediction exactly reproduces the frozen baseline.
        nn.init.zeros_(self.response_functions[-1].weight)
        nn.init.zeros_(self.response_functions[-1].bias)

    def response_basis(self, impact, time_seconds):
        if impact.ndim != 2 or impact.shape[1] != 2:
            raise ValueError("impact must have shape (B,2)")
        if time_seconds.ndim == 1:
            times = time_seconds[None].expand(len(impact), -1)
        elif time_seconds.ndim == 2 and time_seconds.shape[0] == len(impact):
            times = time_seconds
        else:
            raise ValueError("time_seconds must have shape (T,) or (B,T)")
        if not len(impact) or not times.shape[1]:
            raise ValueError("Impact and time batches must be nonempty")
        for value in (impact, times):
            if value.device != self.frequencies_hz.device or value.dtype != self.frequencies_hz.dtype:
                raise ValueError("Inputs must match the model's floating dtype and device")
            if not torch.isfinite(value).all():
                raise ValueError("Impact and time must be finite")
        phases = 2 * torch.pi * times[..., None] * self.frequencies_hz
        fourier = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2)
        features = torch.cat((impact[:, None].expand(-1, times.shape[1], -1), fourier), dim=-1)
        return self.response_functions(features)  # B,T,R

    def forward(self, coefficients, impact, time_seconds):
        if coefficients.ndim != 2 or coefficients.shape != (len(impact), self.coefficient_width):
            raise ValueError("coefficients must have shape (B,coefficient_width)")
        if coefficients.dtype != impact.dtype or coefficients.device != impact.device:
            raise ValueError("Coefficients and impact must share dtype/device")
        if not torch.isfinite(coefficients).all():
            raise ValueError("Coefficients must be finite")
        return torch.einsum("br,btr->bt", coefficients, self.response_basis(impact, time_seconds))


def load_correction_geometry(inp_path, impactor_nodes):
    """Return full XYZ and panel labels for the structural rows, without a graph.

    Element sets identify panels only; connectivity is not given to the model.
    The full XYZ preserves the headform for the frozen baseline.
    """
    from abaqus_scripts.inp_geom import Deck
    deck = Deck(Path(inp_path))
    ids = np.asarray(list(deck.nodes), dtype=np.int64)
    mesh = np.asarray(list(deck.nodes.values()), dtype=np.float32)
    if isinstance(impactor_nodes, bool) or not isinstance(impactor_nodes, int) or not 0 <= impactor_nodes < len(mesh):
        raise ValueError("Invalid structural/headform prefix")
    if impactor_nodes and set(ids[:impactor_nodes]) != deck.impactor_node_ids():
        raise ValueError("Saved impactor prefix does not match the deck's headform node set")
    outer_name, inner_name = "Hood_Outer-1-2", "Hood_Inner-1-2"
    if outer_name not in deck.elsets or inner_name not in deck.elsets:
        raise ValueError("Geometry correction requires the hood outer/inner panel element sets")
    outer, inner = deck.elset_nodes(outer_name), deck.elset_nodes(inner_name)
    labels = np.asarray(["shared" if node in outer and node in inner else
                         "outer" if node in outer else "inner" if node in inner else "other"
                         for node in ids[impactor_nodes:]])
    return mesh, labels


class GeometryCorrectionPredictor:
    """Portable inference from a correction run containing its baseline snapshot."""

    def __init__(self, baseline, basis, correction):
        self.baseline, self.basis, self.correction = baseline, basis, correction.eval()
        self.time_points = baseline.time_points.copy()
        self.device = baseline.device
        self.impactor_nodes = baseline.model.impactor_nodes
        baseline.model.requires_grad_(False)

    @classmethod
    def from_run(cls, run_dir, device="cpu", neighbor_cache_dir=None):
        from geometry_correction_basis import GeometryCorrectionBasis
        from train_mesh_impact_history import HistoryPredictor
        run_dir = Path(run_dir)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("architecture", {}).get("name") != "CoefficientHistoryCorrection":
            raise ValueError("Not a supported coefficient correction run")
        def digest(path):
            result = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    result.update(block)
            return result.hexdigest()
        for name in ("config.json", "splits.json", "scalers.joblib", "prediction_times.npy",
                     "hood_impact_best_model.pt"):
            if digest(run_dir / "baseline" / name) != config["baseline"]["artifact_sha256"].get(name):
                raise ValueError(f"Frozen baseline artifact changed: {name}")
        if digest(run_dir / "geometry_basis.npz") != config["geometry_basis"]["sha256"]:
            raise ValueError("Saved geometry basis changed")
        baseline = HistoryPredictor.from_run(run_dir / "baseline", device=device,
                                             neighbor_cache_dir=neighbor_cache_dir)
        basis = GeometryCorrectionBasis.load(run_dir / "geometry_basis.npz")
        correction = CoefficientHistoryCorrection(**config["architecture"]["kwargs"]).to(device)
        checkpoint = torch.load(run_dir / "correction_best.pt", map_location=device, weights_only=True)
        if (checkpoint["baseline_artifact_sha256"] != config["baseline"]["artifact_sha256"]
                or checkpoint["basis_sha256"] != config["geometry_basis"]["sha256"]):
            raise ValueError("Correction checkpoint does not match baseline/basis artifacts")
        correction.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if basis.coefficient_width != correction.coefficient_width:
            raise ValueError("Geometry basis and correction checkpoint widths disagree")
        return cls(baseline, basis, correction)

    @torch.no_grad()
    def predict(self, mesh_xyz, impact_xy, part_labels=None, sampled_time_points=None,
                return_components=False):
        mesh = np.asarray(mesh_xyz, dtype=np.float32)
        times = self.time_points if sampled_time_points is None else np.asarray(sampled_time_points, dtype=np.float32)
        structural = mesh[self.impactor_nodes:]
        if part_labels is not None and len(part_labels) == len(mesh):
            part_labels = np.asarray(part_labels)[self.impactor_nodes:]
        coefficients, diagnostics = self.basis.transform(structural, part_labels=part_labels)
        baseline = self.baseline.predict(mesh, impact_xy, sampled_time_points=times)
        impact = self.baseline.preprocessor.transform_indentor(np.asarray(impact_xy, dtype=np.float32))
        delta = self.correction(torch.as_tensor(coefficients[None], device=self.device),
                                torch.as_tensor(impact[None], dtype=torch.float32, device=self.device),
                                torch.as_tensor(times, device=self.device))[0].cpu().numpy()
        # A difference is multiplied by scale ONLY, without the scaler's mean.
        delta_g = delta * float(self.baseline.preprocessor.accel_scaler.scale_[0])
        corrected = baseline + delta_g
        if return_components:
            return dict(time_seconds=times.copy(), baseline_g=baseline, correction_g=delta_g,
                        corrected_g=corrected, geometry_diagnostics=diagnostics)
        return corrected

    def predict_inp(self, inp_path, impact_xy, **kwargs):
        mesh, labels = load_correction_geometry(inp_path, self.impactor_nodes)
        return self.predict(mesh, impact_xy, part_labels=labels, **kwargs)
