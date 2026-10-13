"""Geltner-style first-order unsmoothing and fund performance measures."""

from collections.abc import Sequence
import math
import numpy as np
import pandas as pd
from .numerics import bisect, finite_vector

UNSMOOTHING_SOURCE_STATUS = "stylized first-order Geltner-inspired filter; original equation attribution unverified"


def _coefficient(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("smoothing coefficient must lie in [0, 1)")


def smooth(true_returns: Sequence[float], alpha: float, previous: float = 0.0) -> np.ndarray:
    """Simulate recursive appraisal smoothing with a known starting observation."""
    _coefficient(alpha)
    values = finite_vector(true_returns, "returns")
    if not math.isfinite(previous):
        raise ValueError("previous return must be finite")
    result = np.empty_like(values)
    for i, value in enumerate(values):
        previous = (1 - alpha) * value + alpha * previous
        result[i] = previous
    return result


def estimate_ar1(returns: Sequence[float]) -> float:
    """Fit an AR(1) slope with an intercept, without silent clipping."""
    values = finite_vector(returns, "returns", 3)
    x, y = values[:-1], values[1:]
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        raise ValueError("cannot fit AR(1) to a constant series")
    x, y = x / scale, y / scale
    centered = x - x.mean()
    variance = float(centered @ centered)
    if variance == 0:
        raise ValueError("cannot fit AR(1) to a constant series")
    return float(centered @ (y - y.mean()) / variance)


def autocorrelation(returns: Sequence[float]) -> float:
    """Compute the sample correlation of adjacent observations."""
    values = finite_vector(returns, "returns", 3)
    if values[:-1].std() == 0 or values[1:].std() == 0:
        raise ValueError("autocorrelation is undefined for a constant series")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def unsmooth(reported: Sequence[float], alpha: float | None = None,
             previous: float | None = None) -> np.ndarray:
    """Invert first-order smoothing; omit the first result when its lag is unknown."""
    values = finite_vector(reported, "reported", 2 if previous is None else 1)
    alpha = estimate_ar1(values) if alpha is None else alpha
    _coefficient(alpha)
    if previous is None:
        return (values[1:] - alpha * values[:-1]) / (1 - alpha)
    if not math.isfinite(previous):
        raise ValueError("previous return must be finite")
    lags = np.append(previous, values[:-1])
    return (values - alpha * lags) / (1 - alpha)


def capital_market_assumptions(returns: pd.DataFrame, periods_per_year: int = 4
                               ) -> tuple[pd.Series, pd.DataFrame]:
    """Annualize arithmetic means and sample covariance under an iid convention."""
    values = returns.to_numpy(dtype=float)
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1
            or not np.all(np.isfinite(values)) or not returns.columns.is_unique
            or not isinstance(periods_per_year, int) or periods_per_year < 1):
        raise ValueError("invalid return panel or annualization frequency")
    return returns.mean() * periods_per_year, returns.cov() * periods_per_year


def ks_pme(calls: Sequence[float], distributions: Sequence[float],
           index: Sequence[float], terminal_nav: float = 0.0) -> float:
    """Compute Kaplan-Schoar PME with residual NAV realized at the final index level."""
    c, d, levels = (finite_vector(x, name) for x, name in
                    [(calls, "calls"), (distributions, "distributions"), (index, "index")])
    if len(c) != len(d) or len(c) != len(levels):
        raise ValueError("cash flows and index must have equal lengths")
    if np.any(c < 0) or np.any(d < 0) or np.any(levels <= 0):
        raise ValueError("invalid cash flows or index levels")
    if not math.isfinite(terminal_nav) or terminal_nav < 0:
        raise ValueError("terminal NAV must be nonnegative")
    denominator = float(np.sum(c / levels))
    if denominator <= 0:
        raise ValueError("PME requires positive contributions")
    return float((np.sum(d / levels) + terminal_nav / levels[-1]) / denominator)


def multiples(calls: Sequence[float], distributions: Sequence[float],
              terminal_nav: float = 0.0) -> dict[str, float]:
    """Return TVPI, DPI and RVPI on paid-in capital, with no recycling."""
    c, d = finite_vector(calls, "calls"), finite_vector(distributions, "distributions")
    if len(c) != len(d) or np.any(c < 0) or np.any(d < 0):
        raise ValueError("cash flows must be aligned and nonnegative")
    if not math.isfinite(terminal_nav) or terminal_nav < 0 or c.sum() <= 0:
        raise ValueError("invalid residual NAV or paid-in capital")
    dpi, rvpi = float(d.sum() / c.sum()), float(terminal_nav / c.sum())
    return {"tvpi": dpi + rvpi, "dpi": dpi, "rvpi": rvpi}


def irr(cash_flows: Sequence[float], times: Sequence[float] | None = None) -> float:
    """Find the unique conventional annual IRR; reject multiple sign changes."""
    flows = finite_vector(cash_flows, "cash flows", 2)
    dates = np.arange(len(flows), dtype=float) if times is None else finite_vector(times, "times", 2)
    if len(flows) != len(dates) or np.any(np.diff(dates) <= 0):
        raise ValueError("times must align and increase strictly")
    nonzero = flows[flows != 0]
    if len(nonzero) < 2 or nonzero[0] >= 0 or nonzero[-1] <= 0:
        raise ValueError("IRR requires initial outflows and later inflows")
    if np.sum(np.diff(np.sign(nonzero)) != 0) != 1:
        raise ValueError("nonconventional cash flows can have multiple IRRs")
    active = flows != 0
    signs = np.sign(flows[active])
    log_amounts = np.log(np.abs(flows[active]))
    dates = dates[active] - dates[active][0]
    def scaled_npv(log_rate: float) -> float:
        exponents = log_amounts - dates * log_rate
        exponents -= np.max(exponents)
        discounted = np.exp(exponents)
        return float((signs @ discounted) / discounted.sum())
    root = bisect(scaled_npv, -50.0, 50.0, tolerance=1e-13)
    result = math.expm1(root)
    if result <= -1:
        raise ValueError("IRR cannot be represented above negative one")
    return result
