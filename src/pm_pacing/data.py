"""Seeded synthetic portfolio, returns, index and policy fixtures."""

import numpy as np
import pandas as pd
from .cma import smooth
from .ta_model import Vintage, STYLIZED_PARAMETERS
from .compliance import Exposure, Holding
from .unsmoothing import smooth_ma1

SEED = 20261012
ASSET_NAMES = ["public_equity", "public_bonds", "private_equity", "private_credit", "real_assets"]


def synthetic_returns(periods: int = 400, seed: int = SEED
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return true and appraisal-smoothed quarterly panels from common factors."""
    if not isinstance(periods, int) or periods < 4:
        raise ValueError("periods must be an integer of at least four")
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((periods, 3))
    noise = rng.standard_normal((periods, 5))
    loading = np.array([[0.075, 0.0, 0.0], [-0.004, 0.023, 0.0],
                        [0.080, 0.0, 0.018], [0.012, 0.012, 0.019],
                        [0.035, 0.010, 0.030]])
    raw = factors @ loading.T + noise * np.array([0.035, 0.010, 0.060, 0.022, 0.035])
    raw += np.array([0.075, 0.035, 0.115, 0.075, 0.085]) / 4 - raw.mean(axis=0)
    truth = pd.DataFrame(raw, columns=ASSET_NAMES)
    reported = truth.copy()
    coefficients = {"private_equity": 0.65, "private_credit": 0.55, "real_assets": 0.70}
    for name, alpha in coefficients.items():
        reported[name] = smooth(truth[name], alpha, previous=0.0)
    return truth, reported, coefficients


def synthetic_index(periods: int, seed: int = SEED + 1) -> np.ndarray:
    """Create a positive synthetic annual total return index at cash-flow dates."""
    if periods < 1:
        raise ValueError("periods must be positive")
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.06, 0.12, periods)))


def synthetic_vintages() -> list[Vintage]:
    """Include stylized secondaries and a co-invest with no fee or carry assumed."""
    return [Vintage("buyout_A", 2026, 80.0, STYLIZED_PARAMETERS["buyout"]),
            Vintage("venture_B", 2027, 45.0, STYLIZED_PARAMETERS["venture"]),
            Vintage("credit_C", 2026, 60.0, STYLIZED_PARAMETERS["private_credit"]),
            Vintage("real_assets_D", 2028, 40.0, STYLIZED_PARAMETERS["real_assets"]),
            Vintage("secondaries_E", 2026, 35.0, STYLIZED_PARAMETERS["secondaries"]),
            Vintage("co_invest_F", 2027, 20.0, STYLIZED_PARAMETERS["co_invest"])]


def synthetic_exposures() -> list[Exposure]:
    """A fully looked-through synthetic current portfolio."""
    labels = [("buyout", 75.0, 45.0), ("venture", 40.0, 30.0),
              ("private_credit", 65.0, 20.0), ("real_assets", 35.0, 15.0),
              ("secondaries", 30.0, 8.0), ("co_invest", 20.0, 0.0)]
    return [Exposure(f"fund_{i}", f"manager_{i}", strategy, 2024 + i % 3,
                     "North America" if i % 2 == 0 else "Europe", nav, unfunded,
                     (Holding(f"issuer_{i}", 0.7, "technology", "US"),
                      Holding("shared_issuer", 0.3, "healthcare", "UK")))
            for i, (strategy, nav, unfunded) in enumerate(labels)]


def stylized_rules() -> list[dict]:
    """Illustrative investment policy and side letters, with no regulatory claims."""
    return [
        {"id": "strategy_buyout", "kind": "band", "dimension": "strategy", "category": "buyout", "lower": 0.0, "upper": 0.20},
        {"id": "strategy_secondaries", "kind": "band", "dimension": "strategy", "category": "secondaries", "lower": 0.0, "upper": 0.10},
        {"id": "strategy_co_invest", "kind": "band", "dimension": "strategy", "category": "co_invest", "lower": 0.0, "upper": 0.08},
        {"id": "vintage_current", "kind": "band", "dimension": "vintage", "category": 2026, "lower": 0.0, "upper": 0.20},
        {"id": "geography_na", "kind": "band", "dimension": "geography", "category": "North America", "lower": 0.0, "upper": 0.45},
        {"id": "single_manager", "kind": "concentration", "dimension": "manager", "upper": 0.18},
        {"id": "single_fund", "kind": "concentration", "dimension": "fund", "upper": 0.15},
        {"id": "single_issuer", "kind": "concentration", "dimension": "issuer", "upper": 0.16},
        {"id": "side_letter_sector", "kind": "exclusion", "dimension": "sector", "excluded": ["thermal_coal"]},
        {"id": "side_letter_jurisdiction", "kind": "exclusion", "dimension": "jurisdiction", "excluded": ["Excluded_Jurisdiction"]},
        {"id": "unfunded_limit", "kind": "unfunded", "upper": 0.20},
    ]


def synthetic_glm_returns(periods: int = 400, seed: int = SEED
                          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Apply a stylized invertible GLM MA(1) filter to seeded true returns."""
    truth, _, _ = synthetic_returns(periods, seed)
    reported = truth.copy()
    coefficients = {"private_equity": 0.30, "private_credit": 0.25, "real_assets": 0.35}
    for name, weight in coefficients.items():
        reported[name] = smooth_ma1(truth[name], weight, previous_true=0.0)
    return truth, reported, coefficients
