import _bootstrap
import unittest
import numpy as np
from pm_pacing.allocation import AllocationConstraints, optimize, efficient_frontier
from pm_pacing.cma import capital_market_assumptions, unsmooth, estimate_ar1
from pm_pacing.data import synthetic_returns


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.array([.08, .05])
        self.cov = np.diag([.04, .09])
        self.free = AllocationConstraints((False, False), 1, (1, 1), (0, 0))

    def test_analytic_minimum_variance(self):
        result = optimize(self.mean, self.cov, self.free, minimum_variance=True)
        expected = np.array([1/.04, 1/.09])
        np.testing.assert_allclose(result.weights, expected/expected.sum(), atol=1e-10)

    def test_analytic_mean_variance(self):
        result = optimize(self.mean, self.cov, self.free, risk_aversion=4)
        np.testing.assert_allclose(result.weights, [.75, .25], atol=1e-10)

    def test_simplex(self):
        result = optimize(self.mean, self.cov, self.free)
        self.assertAlmostEqual(result.weights.sum(), 1)
        self.assertTrue((result.weights >= -1e-10).all())

    def test_long_only_boundary(self):
        result = optimize([.8, -.5], self.cov, self.free)
        np.testing.assert_allclose(result.weights, [1, 0], atol=1e-10)

    def test_illiquid_cap_binding(self):
        limits = AllocationConstraints((False, True), .3, (1, 0), (0, 0))
        result = optimize([.01, .5], self.cov, limits)
        self.assertAlmostEqual(result.weights[1], .3)
        self.assertIn("illiquid_cap", result.active_rules)

    def test_stressed_liquidity_binding(self):
        limits = AllocationConstraints((False, True), 1, (.5, 0), (0, .5), .1)
        result = optimize([.01, .5], self.cov, limits)
        self.assertAlmostEqual(result.weights[1], .4)
        self.assertAlmostEqual(result.liquidity_surplus, 0)
        self.assertIn("stressed_liquidity", result.active_rules)

    def test_zero_illiquid_cap(self):
        limits = AllocationConstraints((False, True), 0, (1, 0), (0, .2))
        result = optimize(self.mean, self.cov, limits)
        np.testing.assert_allclose(result.weights, [1, 0], atol=1e-10)

    def test_target_return_respected(self):
        result = optimize(self.mean, self.cov, self.free, target_return=.075, minimum_variance=True)
        self.assertGreaterEqual(result.expected_return, .075 - 1e-10)
        self.assertAlmostEqual(result.expected_return, .075)

    def test_infeasible_target(self):
        with self.assertRaises(ValueError):
            optimize(self.mean, self.cov, self.free, target_return=.1)

    def test_infeasible_liquidity(self):
        limits = AllocationConstraints((False, True), .5, (.5, 0), (0, 1), 1)
        with self.assertRaises(ValueError):
            optimize(self.mean, self.cov, limits)

    def test_frontier_monotone(self):
        frontier = efficient_frontier(self.mean, self.cov, self.free)
        self.assertTrue((np.diff(frontier.expected_return) >= -1e-9).all())
        self.assertTrue((np.diff(frontier.volatility) >= -1e-9).all())
        self.assertAlmostEqual(frontier.expected_return.iloc[-1], .08)

    def test_frontier_weights_reconcile(self):
        f = efficient_frontier(self.mean, self.cov, self.free)
        np.testing.assert_allclose(f[["asset_0", "asset_1"]].sum(axis=1), 1)

    def test_stationarity_certificate(self):
        result = optimize(self.mean, self.cov, self.free)
        self.assertLess(result.stationarity_residual, 1e-10)

    def test_solution_beats_feasible_grid(self):
        result = optimize(self.mean, self.cov, self.free)
        for equity in np.linspace(0, 1, 101):
            weights = np.array([equity, 1-equity])
            objective = 2*weights @ self.cov @ weights - self.mean @ weights
            self.assertLessEqual(result.objective, objective + 1e-12)

    def test_five_asset_constraints(self):
        truth, _, _ = synthetic_returns()
        mean, cov = capital_market_assumptions(truth)
        limits = AllocationConstraints((False, False, True, True, True), .45, (.6, .9, 0, 0, 0), (0, 0, .5, .4, .45), .15)
        result = optimize(mean, cov, limits)
        self.assertLessEqual(result.weights[2:].sum(), .45 + 1e-10)
        self.assertGreaterEqual(result.liquidity_surplus, -1e-10)
        f = efficient_frontier(mean, cov, limits)
        self.assertTrue((np.diff(f.volatility) >= -1e-9).all())

    def test_asset_permutation_invariance(self):
        a = optimize(self.mean, self.cov, self.free)
        b = optimize(self.mean[::-1], self.cov[::-1, ::-1], self.free)
        np.testing.assert_allclose(a.weights, b.weights[::-1])

    def test_bad_covariance_rejected(self):
        for covariance in [[[1, 2], [0, 1]], [[1, 2], [2, 1]], [[0, 0], [0, 0]]]:
            with self.assertRaises(ValueError):
                optimize(self.mean, covariance, self.free)

    def test_bad_constraints_rejected(self):
        for limits in [AllocationConstraints((False,), .5, (1,), (0,)), AllocationConstraints((False, True), 2, (1, 0), (0, 0)), AllocationConstraints((False, True), .5, (1, 1), (0, 0))]:
            with self.assertRaises(ValueError):
                optimize(self.mean, self.cov, limits)

    def test_unsmoothing_changes_allocation_and_reveals_naive_risk(self):
        _, reported, coefficients = synthetic_returns()
        adjusted = reported.iloc[1:].copy()
        for name in coefficients:
            adjusted[name] = unsmooth(reported[name], estimate_ar1(reported[name]))
        raw_mean, raw_cov = capital_market_assumptions(reported.iloc[1:])
        mean, cov = capital_market_assumptions(adjusted)
        limits = AllocationConstraints((False, False, True, True, True), .45, (.6, .9, 0, 0, 0), (0, 0, .5, .4, .45), .15)
        raw = optimize(raw_mean, raw_cov, limits)
        recovered = optimize(mean, cov, limits)
        self.assertGreater(raw.weights[2], recovered.weights[2])
        self.assertGreater(float(raw.weights @ cov @ raw.weights), raw.volatility**2)

    def test_mask_rejects_implicit_truthiness(self):
        for mask in [("False", "True"), ((False, False), (True, True)), (0, 1)]:
            limits = AllocationConstraints(mask, .5, (1, 0), (0, 0))
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                optimize(self.mean, self.cov, limits)

    def test_frontier_rejects_colliding_asset_labels(self):
        for labels in [["expected_return", "bonds"], ["", "bonds"], ["equity", "equity"]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                efficient_frontier(self.mean, self.cov, self.free, asset_names=labels)
