"""Small numerical routines with explicit convergence failures."""

from collections.abc import Callable, Sequence
import math
import numpy as np


def finite_vector(values: Sequence[float], name: str, minimum: int = 1) -> np.ndarray:
    """Validate a finite, nonempty vector."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) < minimum or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length at least {minimum}")
    return result


def bisect(function: Callable[[float], float], low: float, high: float,
           tolerance: float = 1e-12, max_iterations: int = 200) -> float:
    """Find a bracketed root, including either endpoint."""
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("invalid root bracket")
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid convergence controls")
    left, right = function(low), function(high)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("nonfinite function value")
    if abs(left) <= tolerance:
        return low
    if abs(right) <= tolerance:
        return high
    if (left > 0) == (right > 0):
        raise ValueError("root is not bracketed")
    for _ in range(max_iterations):
        midpoint = (low + high) / 2
        value = function(midpoint)
        if not math.isfinite(value):
            raise ValueError("nonfinite function value")
        if abs(value) <= tolerance or high - low <= tolerance:
            return midpoint
        if (value > 0) == (left > 0):
            low, left = midpoint, value
        else:
            high = midpoint
    raise RuntimeError("bisection did not converge")
