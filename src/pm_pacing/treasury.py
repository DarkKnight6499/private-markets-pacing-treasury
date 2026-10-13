"""Quarterly subscription line, liquidity buffers and cash reconciliation."""

from dataclasses import dataclass
from collections.abc import Sequence
import math
import numpy as np
import pandas as pd
from .numerics import finite_vector


@dataclass(frozen=True)
class TreasuryPolicy:
    """Stylized facility terms and cash buffer policy, not regulatory parameters."""

    line_limit: float = 20.0
    repayment_quarters: int = 2
    annual_interest: float = 0.07
    annual_cash_yield: float = 0.03
    financed_fraction: float = 0.8
    buffer_quarters: int = 4
    buffer_fraction: float = 0.25

    def __post_init__(self) -> None:
        rates = [self.line_limit, self.annual_interest, self.annual_cash_yield, self.buffer_fraction]
        if any(not math.isfinite(x) or x < 0 for x in rates):
            raise ValueError("amounts and rates must be finite and nonnegative")
        if not math.isfinite(self.financed_fraction) or not 0 <= self.financed_fraction <= 1:
            raise ValueError("financed fraction must lie in the unit interval")
        if any(not isinstance(x, int) or x < 1 for x in
               [self.repayment_quarters, self.buffer_quarters]):
            raise ValueError("horizons must be positive integer quarters")


def quarterly_flows(annual: pd.DataFrame) -> pd.DataFrame:
    """Split annual calls and distributions equally across quarters."""
    if not {"year", "calls", "distributions"}.issubset(annual):
        raise ValueError("annual table lacks required columns")
    if annual.empty:
        raise ValueError("annual table must not be empty")
    years = finite_vector(annual["year"], "years")
    calls = finite_vector(annual["calls"], "calls")
    distributions = finite_vector(annual["distributions"], "distributions")
    if (np.any(years != np.floor(years)) or len(set(years)) != len(years)
            or np.any(calls < 0) or np.any(distributions < 0)):
        raise ValueError("annual years must be unique integers and flows nonnegative")
    ordered = annual.set_index("year").reindex(
        range(int(years.min()), int(years.max()) + 1), fill_value=0.0).reset_index()
    rows = [{"period": f"{int(row.year)}Q{quarter}", "calls": row.calls / 4,
             "distributions": row.distributions / 4}
            for row in ordered.itertuples(index=False) for quarter in range(1, 5)]
    return pd.DataFrame(rows)


def stress_flows(calls: Sequence[float], distributions: Sequence[float],
                 distribution_multiplier: float = 0.5,
                 acceleration: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Move part of each future call one quarter earlier, preserving total calls."""
    c, d = finite_vector(calls, "calls"), finite_vector(distributions, "distributions")
    if len(c) != len(d) or np.any(c < 0) or np.any(d < 0):
        raise ValueError("cash flows must be aligned and nonnegative")
    if not (math.isfinite(acceleration) and 0 <= acceleration <= 1
            and math.isfinite(distribution_multiplier) and 0 <= distribution_multiplier <= 1):
        raise ValueError("invalid stress parameters")
    accelerated = c.copy()
    shifted = c[1:] * acceleration
    accelerated[:-1] += shifted
    accelerated[1:] -= shifted
    return accelerated, d * distribution_multiplier


def liquidity_buffer(calls: Sequence[float], window: int = 4,
                     fraction: float = 0.25) -> np.ndarray:
    """Size a prospective gross call buffer at each quarter opening."""
    c = finite_vector(calls, "calls")
    if np.any(c < 0) or not isinstance(window, int) or window < 1:
        raise ValueError("invalid calls or window")
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("buffer fraction must be nonnegative")
    return np.array([fraction * c[t:t + window].sum() for t in range(len(c))])


def cash_ladder(calls: Sequence[float], distributions: Sequence[float],
                policy: TreasuryPolicy = TreasuryPolicy(), initial_cash: float = 0.0,
                periods: Sequence[str] | None = None) -> pd.DataFrame:
    """Build a funded ladder with explicit LP top-ups and a facility runoff tail."""
    c, d = finite_vector(calls, "calls"), finite_vector(distributions, "distributions")
    if len(c) != len(d) or np.any(c < 0) or np.any(d < 0):
        raise ValueError("cash flows must be aligned and nonnegative")
    if not math.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError("initial cash must be finite and nonnegative")
    labels = list(periods) if periods is not None else [f"Q{t + 1}" for t in range(len(c))]
    if len(labels) != len(c):
        raise ValueError("period labels must align with cash flows")
    labels += [f"runoff_Q{t + 1}" for t in range(policy.repayment_quarters)]
    c = np.append(c, np.zeros(policy.repayment_quarters))
    d = np.append(d, np.zeros(policy.repayment_quarters))
    buffer = liquidity_buffer(c, policy.buffer_quarters, policy.buffer_fraction)
    stressed_c, stressed_d = stress_flows(c, d)
    debt, cash = 0.0, initial_cash
    tranches: list[tuple[int, float]] = []
    rows = []
    for t in range(len(c)):
        opening_cash, opening_debt = cash, debt
        opening_tranches = tranches.copy()
        repayment = sum(amount for due, amount in tranches if due == t)
        tranches = [(due, amount) for due, amount in tranches if due > t]
        debt = sum(amount for _, amount in tranches)
        draw = min(policy.line_limit - debt, c[t] * policy.financed_fraction)
        draw = max(0.0, draw)
        if draw:
            tranches.append((t + policy.repayment_quarters, draw))
        debt = sum(amount for _, amount in tranches)
        # Opening maturities incur no interest in their repayment quarter.
        interest = debt * policy.annual_interest / 4
        cash_income = opening_cash * policy.annual_cash_yield / 4
        pre_funding_cash = cash + cash_income + d[t] + draw - c[t] - repayment - interest
        next_buffer = buffer[t + 1] if t + 1 < len(c) else 0.0
        lp_funding = max(0.0, next_buffer - pre_funding_cash)
        cash = pre_funding_cash + lp_funding
        maturity = np.zeros(min(4, len(c) - t))
        for due, amount in opening_tranches:
            if t <= due < t + len(maturity):
                maturity[due - t] += amount
        forecast_interest = np.array([
            sum(amount for due, amount in opening_tranches if due > t + ahead)
            * policy.annual_interest / 4 for ahead in range(len(maturity))])
        net_stress = stressed_c[t:t + 4] - stressed_d[t:t + 4] + maturity + forecast_interest
        stressed_need = max(0.0, float(np.cumsum(net_stress).max()))
        coverage = opening_cash / stressed_need if stressed_need > 0 else math.inf
        rows.append({"period": labels[t], "calls": c[t], "distributions": d[t],
                     "opening_cash": opening_cash, "opening_line": opening_debt,
                     "draw": draw, "repayment": repayment, "interest": interest,
                     "cash_income": cash_income, "lp_funding": lp_funding,
                     "pre_funding_cash": pre_funding_cash,
                     "funding_shortfall": max(0.0, -pre_funding_cash),
                     "closing_cash": cash, "closing_line": debt,
                     "required_buffer": next_buffer,
                     "stressed_outflows_12m": stressed_need, "liquidity_coverage_12m": coverage,
                     "coverage_forecast_quarters": len(maturity)})
    return pd.DataFrame(rows)
