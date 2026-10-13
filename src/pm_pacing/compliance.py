"""Data-driven checks on post-trade NAV plus unfunded economic exposure."""

from dataclasses import dataclass
from collections.abc import Sequence, Mapping
from typing import Any
import math
import pandas as pd


@dataclass(frozen=True)
class Holding:
    """A complete issuer look-through share of an exposure."""

    issuer: str
    weight: float
    sector: str
    jurisdiction: str

    def __post_init__(self) -> None:
        if not all([self.issuer, self.sector, self.jurisdiction]):
            raise ValueError("issuer, sector and jurisdiction are required")
        if not math.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("look-through weight must lie in the unit interval")


@dataclass(frozen=True)
class Exposure:
    """A fund or co-invest sleeve with no fee or carry calculation attached."""

    fund: str
    manager: str
    strategy: str
    vintage: int
    geography: str
    nav: float
    unfunded: float
    holdings: tuple[Holding, ...]

    def __post_init__(self) -> None:
        if not all([self.fund, self.manager, self.strategy, self.geography]):
            raise ValueError("exposure labels are required")
        if not isinstance(self.vintage, int):
            raise ValueError("vintage must be an integer")
        if any(not math.isfinite(x) or x < 0 for x in [self.nav, self.unfunded]):
            raise ValueError("NAV and unfunded must be finite and nonnegative")
        if not self.holdings or abs(sum(h.weight for h in self.holdings) - 1.0) > 1e-10:
            raise ValueError("complete look-through weights must sum to one")

    @property
    def economic_exposure(self) -> float:
        """Reserve both invested NAV and uncalled obligations."""
        return self.nav + self.unfunded


def check_trade(portfolio: Sequence[Exposure], trade: Exposure, total_nav: float,
                rules: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Evaluate explicit policy rules; an empty rule set is invalid."""
    if not math.isfinite(total_nav) or total_nav <= 0:
        raise ValueError("total portfolio NAV must be positive")
    if not rules or len({rule['id'] for rule in rules}) != len(rules):
        raise ValueError("provide rules with unique ids")
    post_trade = [*portfolio, trade]
    rows = []
    for rule in rules:
        kind = rule["kind"]
        if kind == "band":
            dimension = rule["dimension"]
            if dimension not in {"strategy", "vintage", "geography"}:
                raise ValueError("unsupported policy dimension")
            amount = sum(e.economic_exposure for e in post_trade
                         if getattr(e, dimension) == rule["category"])
            measured = amount / total_nav
            lower, upper = float(rule.get("lower", 0.0)), float(rule["upper"])
        elif kind == "concentration":
            dimension = rule["dimension"]
            amounts: dict[str, float] = {}
            for e in post_trade:
                if dimension == "issuer":
                    for h in e.holdings:
                        amounts[h.issuer] = amounts.get(h.issuer, 0.0) + e.economic_exposure * h.weight
                elif dimension in {"manager", "fund"}:
                    label = getattr(e, dimension)
                    amounts[label] = amounts.get(label, 0.0) + e.economic_exposure
                else:
                    raise ValueError("unsupported concentration dimension")
            measured = max(amounts.values()) / total_nav
            lower, upper = 0.0, float(rule["upper"])
        elif kind == "unfunded":
            measured = sum(e.unfunded for e in post_trade) / total_nav
            lower, upper = 0.0, float(rule["upper"])
        elif kind == "exclusion":
            dimension = rule["dimension"]
            if dimension not in {"sector", "jurisdiction"}:
                raise ValueError("unsupported exclusion dimension")
            measured = sum(e.economic_exposure * h.weight for e in post_trade
                           for h in e.holdings if getattr(h, dimension) in rule["excluded"]) / total_nav
            lower, upper = 0.0, 0.0
        else:
            raise ValueError(f"unsupported rule kind: {kind}")
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper):
            raise ValueError("invalid policy limits")
        warn_at = float(rule.get("warn_at", max(lower, upper * 0.9)))
        if not math.isfinite(warn_at) or not lower <= warn_at <= upper:
            raise ValueError("warning threshold must lie inside the policy band")
        status = "fail" if measured < lower - 1e-12 or measured > upper + 1e-12 else "pass"
        if status == "pass" and upper > 0 and measured >= warn_at - 1e-12:
            status = "warn"
        rows.append({"rule_id": rule["id"], "kind": kind, "status": status,
                     "measured": measured, "lower": lower, "upper": upper,
                     "warn_at": warn_at, "limit": f"[{lower:.6f}, {upper:.6f}]",
                     "basis": "NAV plus unfunded divided by total portfolio NAV"})
    return pd.DataFrame(rows)


def overall_status(results: pd.DataFrame) -> str:
    """Return the most severe rule outcome."""
    if results.empty:
        raise ValueError("results must not be empty")
    return max(results.status, key={"pass": 0, "warn": 1, "fail": 2}.__getitem__)
