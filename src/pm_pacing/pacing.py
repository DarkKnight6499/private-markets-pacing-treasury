"""Annual commitment pacing with a self-financing public portfolio."""

from dataclasses import replace
import math
import numpy as np
import pandas as pd
from .numerics import bisect
from .ta_model import FundParameters, Vintage, project_fund


def solve_pacing(initial_public_nav: float, target: float, years: int,
                 parameters: FundParameters, max_commitment: float,
                 public_growth: float = 0.05, ramp_years: int = 4,
                 start_year: int = 2026) -> pd.DataFrame:
    """Reach a ramped allocation and hold it whenever funding and caps permit."""
    if (not math.isfinite(initial_public_nav) or initial_public_nav <= 0
            or not math.isfinite(target) or not 0 <= target < 1
            or not math.isfinite(max_commitment) or max_commitment < 0
            or not math.isfinite(public_growth) or public_growth <= -1
            or not isinstance(years, int) or years < 1
            or not isinstance(ramp_years, int) or ramp_years < 1):
        raise ValueError("invalid pacing inputs")
    kernel = project_fund(Vintage("unit", start_year, 1.0, parameters))
    unit_nav = float(kernel.nav.iloc[0])
    unit_call = float(kernel.calls.iloc[0])
    if unit_nav <= 0 or unit_call <= 0:
        raise ValueError("new vintages must create positive first-year NAV")
    commitments: list[float] = []
    public = initial_public_nav
    previous_private = 0.0
    rows = []
    for t in range(years):
        totals = {name: 0.0 for name in ["nav", "calls", "distributions", "unfunded", "growth_pnl"]}
        for vintage, commitment in enumerate(commitments):
            age_index = t - vintage
            if age_index < parameters.life:
                for name in totals:
                    totals[name] += commitment * float(kernel[name].iloc[age_index])
        public_gain = public * public_growth
        available = public + public_gain + totals["distributions"] - totals["calls"]
        total = available + totals["nav"]
        if total <= 0 or available < -1e-9:
            raise ValueError("existing commitments exhaust public funding")
        desired = target * min((t + 1) / ramp_years, 1.0)
        funding_bound = max(0.0, available) / unit_call
        forecast_public, marginal_public = available, -unit_call
        for ahead in range(parameters.life):
            if ahead:
                future_net = 0.0
                for vintage, prior_commitment in enumerate(commitments):
                    age_index = t + ahead - vintage
                    if age_index < parameters.life:
                        future_net += prior_commitment * float(
                            kernel.distributions.iloc[age_index] - kernel.calls.iloc[age_index])
                forecast_public = forecast_public * (1 + public_growth) + future_net
                marginal_public = marginal_public * (1 + public_growth) + float(
                    kernel.distributions.iloc[ahead] - kernel.calls.iloc[ahead])
            if marginal_public < 0:
                funding_bound = min(funding_bound, max(0.0, forecast_public) / -marginal_public)
        bound = min(max_commitment, funding_bound)
        residual = lambda value: (totals["nav"] + unit_nav * value) / total - desired
        if residual(0.0) >= -1e-12:
            commitment = 0.0
            status = "on_target" if abs(residual(0.0)) <= 1e-10 else "above_target"
        elif bound == 0 or residual(bound) < -1e-12:
            commitment = bound
            status = "capped" if bound == max_commitment else "funding_limited"
        else:
            commitment = bisect(residual, 0.0, bound)
            status = "on_target"
        commitments.append(commitment)
        calls = totals["calls"] + commitment * unit_call
        distributions = totals["distributions"] + commitment * float(kernel.distributions.iloc[0])
        public = available - commitment * unit_call + commitment * float(kernel.distributions.iloc[0])
        private = totals["nav"] + commitment * unit_nav
        unfunded = totals["unfunded"] + commitment * float(kernel.unfunded.iloc[0])
        active_commitments = sum(commitments[max(0, t - parameters.life + 1):])
        rows.append({"year": start_year + t, "commitment": commitment,
                     "public_nav": public, "private_nav": private, "total_nav": public + private,
                     "calls": calls, "distributions": distributions, "unfunded": unfunded,
                     "public_growth_pnl": public_gain, "private_growth_pnl": totals["growth_pnl"],
                     "opening_private_nav": previous_private, "target": desired,
                     "allocation": private / (public + private),
                     "target_error": private / (public + private) - desired,
                     "overcommitment_ratio": (private + unfunded) / (public + private),
                     "active_commitment_ratio": active_commitments / (public + private),
                     "status": status})
        previous_private = private
    return pd.DataFrame(rows)


def sensitivity_table(initial_public_nav: float, target: float, years: int,
                      parameters: FundParameters, max_commitment: float) -> pd.DataFrame:
    """Re-solve pacing under one-at-a-time stylized parameter changes."""
    scenarios = {"base": parameters,
                 "faster_distributions": replace(parameters, bow=parameters.bow * 0.75),
                 "slower_distributions": replace(parameters, bow=parameters.bow * 1.25),
                 "lower_growth": replace(parameters, growth=parameters.growth - 0.04),
                 "higher_growth": replace(parameters, growth=parameters.growth + 0.04),
                 "faster_calls": replace(parameters, call_rates=tuple(min(1.0, r * 1.25)
                                                                      for r in parameters.call_rates)),
                 "slower_calls": replace(parameters, call_rates=tuple(r * 0.75 for r in parameters.call_rates))}
    rows = []
    for name, p in scenarios.items():
        path = solve_pacing(initial_public_nav, target, years, p, max_commitment)
        rows.append({"scenario": name, "bow": p.bow, "growth": p.growth,
                     "initial_call_rate": p.call_rates[0],
                     "total_commitments": path.commitment.sum(),
                     "final_commitment": path.commitment.iloc[-1],
                     "final_allocation": path.allocation.iloc[-1],
                     "max_target_error": np.abs(path.target_error).max(),
                     "constrained_years": int((path.status != "on_target").sum())})
    return pd.DataFrame(rows)
