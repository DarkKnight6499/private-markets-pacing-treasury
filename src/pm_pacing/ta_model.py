"""Annual Takahashi-Alexander cash flows with explicit terminal liquidation."""

from dataclasses import dataclass
from collections.abc import Sequence
import math
import pandas as pd


@dataclass(frozen=True)
class FundParameters:
    """Net growth and annual rates; all supplied defaults are stylized."""

    life: int = 12
    growth: float = 0.12
    call_rates: tuple[float, ...] = (0.25, 0.35, 0.50)
    bow: float = 2.5
    yield_rate: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.life, bool) or not isinstance(self.life, int) or self.life < 1:
            raise ValueError("life must be a positive integer")
        if not math.isfinite(self.growth) or self.growth <= -1:
            raise ValueError("growth must exceed negative one")
        if not math.isfinite(self.bow) or self.bow <= 0:
            raise ValueError("bow must be positive")
        if not self.call_rates or any(not math.isfinite(r) or not 0 <= r <= 1
                                      for r in self.call_rates):
            raise ValueError("call rates must lie in the unit interval")
        if not math.isfinite(self.yield_rate) or not 0 <= self.yield_rate <= 1:
            raise ValueError("yield must lie in the unit interval")

    def call_rate(self, age: int) -> float:
        """Repeat the last call rate after the supplied schedule ends."""
        if age < 1:
            raise ValueError("age must be positive")
        return self.call_rates[min(age - 1, len(self.call_rates) - 1)]


STYLIZED_PARAMETERS = {
    "buyout": FundParameters(),
    "venture": FundParameters(14, 0.15, (0.20, 0.30, 0.45), 3.0),
    "private_credit": FundParameters(8, 0.08, (0.40, 0.50), 1.5, 0.06),
    "real_assets": FundParameters(12, 0.09, (0.35, 0.40), 2.5, 0.04),
    "secondaries": FundParameters(6, 0.11, (0.70, 0.70), 1.0, 0.04),
    "co_invest": FundParameters(7, 0.12, (1.0,), 2.0),
}


@dataclass(frozen=True)
class Vintage:
    """A new commitment whose first projection year is its vintage year."""

    name: str
    year: int
    commitment: float
    parameters: FundParameters

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.year, int):
            raise ValueError("name and integer year are required")
        if not math.isfinite(self.commitment) or self.commitment < 0:
            raise ValueError("commitment must be finite and nonnegative")


def project_fund(fund: Vintage, liquidate_terminal: bool = True) -> pd.DataFrame:
    """Project annual fund states, calls, distributions and canceled commitments."""
    nav, unfunded, rows = 0.0, fund.commitment, []
    for age in range(1, fund.parameters.life + 1):
        p = fund.parameters
        opening_nav, opening_unfunded = nav, unfunded
        calls = unfunded * p.call_rate(age)
        unfunded -= calls
        gain = nav * p.growth
        rate = max(p.yield_rate, (age / p.life) ** p.bow)
        distributions = rate * (nav + gain)
        terminal_extra = calls if liquidate_terminal and age == p.life else 0.0
        distributions += terminal_extra
        nav = nav + gain + calls - distributions
        canceled = unfunded if liquidate_terminal and age == p.life else 0.0
        unfunded -= canceled
        rows.append({"fund": fund.name, "year": fund.year + age - 1, "age": age,
                     "commitment": fund.commitment, "opening_nav": opening_nav,
                     "opening_unfunded": opening_unfunded, "calls": calls,
                     "distributions": distributions, "growth_pnl": gain,
                     "nav": max(0.0, nav), "unfunded": unfunded, "canceled": canceled,
                     "distribution_rate": rate, "terminal_extra": terminal_extra,
                     "net_cash_flow": distributions - calls})
    result = pd.DataFrame(rows)
    result["cumulative_net_cash_flow"] = result.net_cash_flow.cumsum()
    return result


def project_program(vintages: Sequence[Vintage]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fund detail and a calendar-aligned, additive annual program."""
    if not vintages or len({f.name for f in vintages}) != len(vintages):
        raise ValueError("provide nonempty vintages with unique names")
    detail = pd.concat([project_fund(f) for f in vintages], ignore_index=True)
    columns = ["opening_nav", "opening_unfunded", "calls", "distributions",
               "growth_pnl", "nav", "unfunded", "canceled", "net_cash_flow"]
    annual = detail.groupby("year")[columns].sum()
    annual = annual.reindex(range(int(detail.year.min()), int(detail.year.max()) + 1),
                            fill_value=0.0).reset_index()
    annual["new_commitments"] = annual.year.map(
        {year: sum(f.commitment for f in vintages if f.year == year)
         for year in {f.year for f in vintages}}).fillna(0.0)
    annual["cumulative_net_cash_flow"] = annual.net_cash_flow.cumsum()
    return detail, annual
