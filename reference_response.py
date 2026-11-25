"""The reference design's simulated response, as an input and as the base of the output.

A local-edit study always has an unedited reference design that is simulated at
every load case. With ``reference_response=True`` the network predicts an edited
design's history as

    a(t) = a_ref(t) + s * delta(t)

where a_ref is the reference design's own simulated curve at the same load case,
delta is the network's output and s is a fixed scale: the RMS of
(a - a_ref) over the training runs, in normalized units. Every time query also
reads a_ref(t). The loss is unchanged (acceleration MSE), but since a_ref is
given exactly it is spent entirely on the edit effect a - a_ref. The output head
starts at zero, so a new model is exactly "copy the reference".

:class:`ReferenceResponseTable` holds the reference curves, in the model's
normalized time and acceleration units, keyed by the load case: here the impact
XY in mm, the only load parameter of the hood data. Like the design-change
table it is derived from the data, saved beside the checkpoint
(reference_responses.npz) and never part of the state dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

TABLE_FILE = "reference_responses.npz"
# A load case matches a reference row within this distance (mm).
MATCH_TOLERANCE_MM = 0.5
# Query times may lie this far outside the reference grid (normalized units).
TIME_TOLERANCE = 1e-5


class ReferenceResponseTable:
    """Reference curves (L, T) at normalized ``times`` (T,), one per load case ``loads`` (L, P).

    ``labels`` (L,) names each load case (the impact location number) for reports.
    """

    def __init__(self, loads, times, curves, labels=None):
        self.loads = np.asarray(loads, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float32)
        self.curves = np.asarray(curves, dtype=np.float32)
        self.labels = np.arange(len(self.loads)) if labels is None else np.asarray(labels, dtype=np.int64)
        if self.loads.ndim != 2 or not len(self.loads):
            raise ValueError("loads must have shape (load cases, load parameters)")
        if self.times.ndim != 1 or not len(self.times) or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be a nonempty, strictly increasing vector")
        if self.curves.shape != (len(self.loads), len(self.times)):
            raise ValueError(f"curves must have shape {(len(self.loads), len(self.times))}; got {self.curves.shape}")
        if self.labels.shape != (len(self.loads),):
            raise ValueError("labels must give one label per load case")
        if not all(np.isfinite(array).all() for array in (self.loads, self.times, self.curves)):
            raise ValueError("Reference loads, times and curves must be finite")
        if len(self.loads) > 1:
            gaps = np.linalg.norm(self.loads[:, None] - self.loads[None], axis=-1)
            np.fill_diagonal(gaps, np.inf)
            if gaps.min() <= 2 * MATCH_TOLERANCE_MM:
                raise ValueError("Two reference load cases are within the matching tolerance of each other")
        self._tensors = {}

    def lookup(self, loads) -> np.ndarray:
        """Row of each load case (B, P): the nearest reference, within MATCH_TOLERANCE_MM."""
        loads = np.asarray(loads, dtype=np.float64)
        if loads.ndim != 2 or loads.shape[1] != self.loads.shape[1]:
            raise ValueError(f"loads must have shape (B, {self.loads.shape[1]})")
        distance = np.linalg.norm(loads[:, None] - self.loads[None], axis=-1)
        index = distance.argmin(axis=1)
        far = distance[np.arange(len(loads)), index] > MATCH_TOLERANCE_MM
        if far.any():
            raise KeyError(f"No reference response within {MATCH_TOLERANCE_MM} mm of load case "
                           f"{loads[far][0].tolist()}; the table holds the reference runs the model was trained with")
        return index

    def curve(self, load, times) -> np.ndarray:
        """Reference curve of one load case (P,) at normalized ``times`` (T,), linearly interpolated."""
        times = np.asarray(times, dtype=np.float64)
        self._check_range(times.min(initial=self.times[0]), times.max(initial=self.times[-1]))
        row = self.lookup(np.asarray(load, dtype=np.float64)[None])[0]
        return np.interp(times, self.times.astype(np.float64), self.curves[row].astype(np.float64))

    def curves_at(self, index: Tensor, time: Tensor, valid: Tensor) -> Tensor:
        """Curves of rows ``index`` (B,) at normalized times (B, T); 0 where ``valid`` (B, T) is False."""
        times, curves = self.tensors(time.device)
        times, curves = times.to(time.dtype), curves.to(time.dtype)[index]
        if valid.any():
            inside = time[valid]
            self._check_range(float(inside.min()), float(inside.max()))
        if len(times) == 1:
            return torch.where(valid, curves.expand_as(time), torch.zeros_like(time))
        right = torch.searchsorted(times, time.contiguous()).clamp(1, len(times) - 1)
        left = right - 1
        weight = ((time - times[left]) / (times[right] - times[left])).clamp(0.0, 1.0)
        values = curves.gather(1, left) * (1 - weight) + curves.gather(1, right) * weight
        return torch.where(valid, values, torch.zeros_like(values))

    def _check_range(self, low, high):
        if low < self.times[0] - TIME_TOLERANCE or high > self.times[-1] + TIME_TOLERANCE:
            raise ValueError(f"Query times [{low:.4g}, {high:.4g}] fall outside the reference grid "
                             f"[{self.times[0]:.4g}, {self.times[-1]:.4g}] (normalized)")

    def tensors(self, device):
        """(times (T,), curves (L, T)) on ``device``."""
        key = str(device)
        if key not in self._tensors:
            self._tensors[key] = (torch.as_tensor(self.times, device=device),
                                  torch.as_tensor(self.curves, device=device))
        return self._tensors[key]

    def save(self, path):
        np.savez(path, loads=self.loads, times=self.times, curves=self.curves, labels=self.labels)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            return cls(saved["loads"], saved["times"], saved["curves"], saved["labels"])

    def same_as(self, other):
        return all(np.array_equal(getattr(self, name), getattr(other, name))
                   for name in ("loads", "times", "curves", "labels"))


def residual_rms(table, loads, time_arrays, accelerations) -> float:
    """RMS of (acceleration - reference) over runs, all in normalized units: the output scale s."""
    squares, count = 0.0, 0
    for load, times, acceleration in zip(loads, time_arrays, accelerations):
        residual = np.asarray(acceleration, dtype=np.float64) - table.curve(load, times)
        squares += float(np.square(residual).sum())
        count += residual.size
    if not count:
        raise ValueError("No runs to measure the reference residual on")
    return float(np.sqrt(squares / count))
