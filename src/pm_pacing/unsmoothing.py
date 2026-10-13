"""Invertible MA(1) specialization of Getmansky-Lo-Makarov return smoothing."""

from collections.abc import Sequence
import math
import numpy as np
from .numerics import bisect, finite_vector
from .cma import autocorrelation

SOURCE = "https://www.mit.edu/~alo/Papers/JFE2004Pub.pdf"
SOURCE_STATUS = "verified: Section 4, equations (21) to (23), MA(1) specialization"


def _lag_weight(weight: float) -> None:
    if not math.isfinite(weight) or not 0 <= weight < 0.5:
        raise ValueError("lag weight must lie in [0, 0.5) for stable inversion")


def smooth_ma1(true_returns: Sequence[float], lag_weight: float,
               previous_true: float = 0.0) -> np.ndarray:
    """Mix current and previous true returns using weights summing to one."""
    _lag_weight(lag_weight)
    values = finite_vector(true_returns, "true returns")
    if not math.isfinite(previous_true):
        raise ValueError("previous true return must be finite")
    return (1 - lag_weight) * values + lag_weight * np.append(previous_true, values[:-1])


def estimate_ma1(reported: Sequence[float]) -> float:
    """Fit the invertible MA(1) moment equation assuming iid true returns."""
    correlation = autocorrelation(reported)
    if not 0 <= correlation < 0.5:
        raise ValueError("lag correlation is incompatible with invertible nonnegative MA(1)")
    def moment(weight: float) -> float:
        return weight * (1 - weight) / ((1 - weight)**2 + weight**2) - correlation
    return bisect(moment, 0.0, 0.5, tolerance=1e-13)


def unsmooth_ma1(reported: Sequence[float], lag_weight: float | None = None,
                 previous_true: float = 0.0) -> np.ndarray:
    """Recover true returns recursively, conditional on the initial true lag."""
    values = finite_vector(reported, "reported returns")
    weight = estimate_ma1(values) if lag_weight is None else lag_weight
    _lag_weight(weight)
    if not math.isfinite(previous_true):
        raise ValueError("previous true return must be finite")
    recovered = np.empty_like(values)
    for i, value in enumerate(values):
        previous_true = (value - weight * previous_true) / (1 - weight)
        recovered[i] = previous_true
    return recovered
