"""HIC15 from an acceleration history in g. NumPy only, no torch, no plotting.

Kept dependency-free on purpose: the Abaqus post-processing environment on
Delta has neither torch nor the training stack, and the solver-side tooling
must compute HIC exactly the way the model-side tooling does.

Do not use ``abaqus_scripts.postprocess_acc.compute_hic`` for analysis. For
each window start it evaluates only the longest admissible window, so a shorter
window with a higher mean is missed and the result can be far too low. The
values recorded in ``generation_metrics.csv`` carry that error. ``batched_hic``
below searches every admissible pair.
"""

import numpy as np


def batched_hic(times, acceleration, window=.015):
    """Exact trapezoidal HIC over all admissible pairs on a shared time grid."""
    times = np.asarray(times, float)
    a = np.asarray(acceleration, float)
    if a.shape[-1] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Acceleration and strictly increasing times must align")
    integral = np.concatenate((np.zeros((*a.shape[:-1], 1)), np.cumsum(
        .5 * (a[..., 1:] + a[..., :-1]) * np.diff(times), axis=-1)), axis=-1)
    best = np.zeros(a.shape[:-1])
    for lag in range(1, len(times)):
        duration = times[lag:] - times[:-lag]
        valid = duration <= window
        if not valid.any():
            break
        mean = (integral[..., lag:] - integral[..., :-lag])[..., valid] / duration[valid]
        best = np.maximum(best, (np.maximum(mean, 0)**2.5 * duration[valid]).max(axis=-1))
    return best
