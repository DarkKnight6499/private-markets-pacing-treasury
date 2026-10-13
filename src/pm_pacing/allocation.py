"""Small-universe convex allocation solved by enumerating active constraints."""

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Sequence
import math
import numpy as np
import pandas as pd
from .numerics import finite_vector


@dataclass(frozen=True)
class AllocationConstraints:
    """Haircut liquid value must cover fixed and allocation-dependent stressed calls."""

    illiquid: tuple[bool, ...]
    max_illiquid: float
    liquid_haircuts: tuple[float, ...]
    stressed_outflows: tuple[float, ...]
    fixed_liquidity_need: float = 0.0

    def matrices(self, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build inequalities A w <= b in fractions of total portfolio NAV."""
        illiquid = np.asarray(self.illiquid)
        if illiquid.ndim != 1 or illiquid.dtype != np.dtype(bool):
            raise ValueError("illiquid mask must be a vector of booleans")
        liquid = finite_vector(self.liquid_haircuts, "liquid haircuts")
        stress = finite_vector(self.stressed_outflows, "stressed outflows")
        if len(illiquid) != count or len(liquid) != count or len(stress) != count:
            raise ValueError("constraint dimensions must match assets")
        if np.any(liquid < 0) or np.any(liquid > 1) or np.any(stress < 0):
            raise ValueError("invalid liquidity coefficients")
        if np.any(liquid[illiquid] != 0):
            raise ValueError("illiquid assets cannot supply liquid value")
        if not math.isfinite(self.max_illiquid) or not 0 <= self.max_illiquid <= 1:
            raise ValueError("illiquid cap must lie in the unit interval")
        if not math.isfinite(self.fixed_liquidity_need) or self.fixed_liquidity_need < 0:
            raise ValueError("fixed liquidity need must be nonnegative")
        matrix = np.vstack([-np.eye(count), illiquid.astype(float), stress - liquid])
        bounds = np.append(np.zeros(count), [self.max_illiquid, -self.fixed_liquidity_need])
        names = [f"long_only_{i}" for i in range(count)] + ["illiquid_cap", "stressed_liquidity"]
        return matrix, bounds, names


@dataclass(frozen=True)
class AllocationResult:
    """Weights and the numerical optimality certificate."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    objective: float
    active_rules: tuple[str, ...]
    stationarity_residual: float
    liquidity_surplus: float


def _inputs(expected_returns: Sequence[float], covariance: Sequence[Sequence[float]]
            ) -> tuple[np.ndarray, np.ndarray]:
    mean = finite_vector(expected_returns, "expected returns", 2)
    if len(mean) > 8:
        raise ValueError("active-set enumeration supports at most eight assets")
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (len(mean), len(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("invalid covariance matrix")
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(cov).min() <= 1e-12:
        raise ValueError("covariance must be positive definite")
    return mean, cov


def optimize(expected_returns: Sequence[float], covariance: Sequence[Sequence[float]],
             constraints: AllocationConstraints, risk_aversion: float = 4.0,
             target_return: float | None = None, minimum_variance: bool = False
             ) -> AllocationResult:
    """Solve a long-only quadratic program and verify primal and dual feasibility."""
    mean, cov = _inputs(expected_returns, covariance)
    n = len(mean)
    if not math.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk aversion must be positive")
    matrix, bounds, names = constraints.matrices(n)
    if target_return is not None:
        if not math.isfinite(target_return):
            raise ValueError("target return must be finite")
        matrix, bounds = np.vstack([matrix, -mean]), np.append(bounds, -target_return)
        names.append("target_return")
    hessian = risk_aversion * cov
    linear = np.zeros(n) if minimum_variance else -mean
    best = None
    for size in range(n):
        for active in combinations(range(len(bounds)), size):
            equality = np.vstack([np.ones(n), matrix[list(active)]])
            rhs = np.append(1.0, bounds[list(active)])
            kkt = np.block([[hessian, equality.T],
                            [equality, np.zeros((size + 1, size + 1))]])
            try:
                solution = np.linalg.solve(kkt, np.append(-linear, rhs))
            except np.linalg.LinAlgError:
                continue
            weights, dual = solution[:n], solution[n:]
            if (np.max(matrix @ weights - bounds) > 1e-9
                    or abs(weights.sum() - 1) > 1e-9 or np.any(dual[1:] < -1e-9)):
                continue
            objective = float(0.5 * weights @ hessian @ weights + linear @ weights)
            residual = float(np.max(np.abs(hessian @ weights + linear + equality.T @ dual)))
            if residual > 1e-8:
                continue
            if best is None or objective < best.objective:
                best = AllocationResult(weights.copy(), float(mean @ weights),
                                        math.sqrt(float(weights @ cov @ weights)), objective,
                                        tuple(names[i] for i in active), residual,
                                        float(bounds[n + 1] - matrix[n + 1] @ weights))
    if best is None:
        raise ValueError("allocation constraints or return target are infeasible")
    return best


def _maximum_return(mean: np.ndarray, constraints: AllocationConstraints) -> float:
    matrix, bounds, _ = constraints.matrices(len(mean))
    maximum = -math.inf
    for active in combinations(range(len(bounds)), len(mean) - 1):
        equality = np.vstack([np.ones(len(mean)), matrix[list(active)]])
        try:
            weights = np.linalg.solve(equality, np.append(1.0, bounds[list(active)]))
        except np.linalg.LinAlgError:
            continue
        if np.max(matrix @ weights - bounds) <= 1e-9:
            maximum = max(maximum, float(mean @ weights))
    if not math.isfinite(maximum):
        raise ValueError("allocation constraints are infeasible")
    return maximum


def efficient_frontier(expected_returns: Sequence[float], covariance: Sequence[Sequence[float]],
                       constraints: AllocationConstraints, points: int = 11,
                       asset_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Return the upper efficient branch from minimum variance to maximum return."""
    mean, cov = _inputs(expected_returns, covariance)
    if not isinstance(points, int) or points < 2:
        raise ValueError("frontier requires at least two points")
    labels = list(asset_names) if asset_names is not None else [f"asset_{i}" for i in range(len(mean))]
    reserved = {"target_return", "expected_return", "volatility", "liquidity_surplus"}
    if (len(labels) != len(mean) or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label or label in reserved for label in labels)):
        raise ValueError("asset names must be unique and aligned")
    minimum = optimize(mean, cov, constraints, minimum_variance=True)
    maximum = _maximum_return(mean, constraints)
    rows = []
    for target in np.linspace(minimum.expected_return, maximum, points):
        result = optimize(mean, cov, constraints, target_return=float(target), minimum_variance=True)
        rows.append({"target_return": target, "expected_return": result.expected_return,
                     "volatility": result.volatility, "liquidity_surplus": result.liquidity_surplus,
                     **dict(zip(labels, result.weights.tolist()))})
    return pd.DataFrame(rows)
