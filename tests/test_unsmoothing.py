import _bootstrap
import unittest
import numpy as np
from pm_pacing.unsmoothing import smooth_ma1, unsmooth_ma1, estimate_ma1
from pm_pacing.cma import autocorrelation, estimate_ar1
from pm_pacing.data import synthetic_glm_returns


class UnsmoothingTests(unittest.TestCase):
    def test_known_lag_recovers_true_series(self):
        truth = np.random.default_rng(31).normal(.02, .08, 2000)
        for weight in [0, .1, .3, .49]:
            reported = smooth_ma1(truth, weight, previous_true=.01)
            np.testing.assert_allclose(unsmooth_ma1(reported, weight, .01), truth, atol=1e-14)

    def test_analytic_single_step(self):
        np.testing.assert_allclose(smooth_ma1([.1, -.05], .25, .02), [.08, -.0125])

    def test_zero_weight_is_identity(self):
        np.testing.assert_array_equal(unsmooth_ma1([.1, -.2], 0), [.1, -.2])

    def test_volatility_and_ar1_diagnostics(self):
        truth = np.random.default_rng(91).normal(.02, .08, 30000)
        reported = smooth_ma1(truth, .3)
        recovered = unsmooth_ma1(reported)
        self.assertGreater(recovered.std(), reported.std() * 1.2)
        self.assertGreater(estimate_ar1(reported), .3)
        self.assertLess(abs(estimate_ar1(recovered)), .02)
        self.assertLess(abs(autocorrelation(recovered)), .02)

    def test_moment_estimator_recovers_weight(self):
        truth = np.random.default_rng(91).normal(.02, .08, 30000)
        self.assertAlmostEqual(estimate_ma1(smooth_ma1(truth, .3)), .3, delta=.01)

    def test_ma1_variance_benchmark(self):
        truth = np.random.default_rng(91).normal(0, .08, 30000)
        reported = smooth_ma1(truth, .3)
        self.assertAlmostEqual(reported.var()/truth.var(), .7**2 + .3**2, delta=.01)

    def test_initial_error_decays_geometrically(self):
        truth = np.full(12, .02)
        reported = smooth_ma1(truth, .25, .1)
        recovered = unsmooth_ma1(reported, .25, 0)
        errors = recovered - truth
        np.testing.assert_allclose(errors[1:], errors[:-1] * (-.25/.75), atol=1e-14)

    def test_invalid_weights_rejected(self):
        for weight in [-.1, .5, .9, np.nan]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                unsmooth_ma1([.1, .2], weight)

    def test_incompatible_serial_correlation_rejected(self):
        for reported in [[1, -1, 1, -1], [1, 2, 3, 4], [0, 0, 0]]:
            with self.subTest(reported=reported), self.assertRaises(ValueError):
                estimate_ma1(reported)

    def test_invalid_initial_lag_rejected(self):
        with self.assertRaises(ValueError):
            unsmooth_ma1([.1], .2, np.nan)

    def test_seed_reproducible(self):
        first, second = synthetic_glm_returns(), synthetic_glm_returns()
        np.testing.assert_array_equal(first[1], second[1])

    def test_scale_invariant_estimation(self):
        truth = np.random.default_rng(91).normal(0, .08, 2000)
        reported = smooth_ma1(truth, .3)
        self.assertAlmostEqual(estimate_ma1(reported), estimate_ma1(reported * 1e-12))

    def test_glm_allocation_uses_recovered_risk(self):
        from pm_pacing.cma import capital_market_assumptions
        from pm_pacing.allocation import AllocationConstraints, optimize, efficient_frontier
        truth, reported, coefficients = synthetic_glm_returns()
        recovered = reported.iloc[1:].copy()
        for name in coefficients:
            recovered[name] = unsmooth_ma1(reported[name])[1:]
        mean, covariance = capital_market_assumptions(recovered)
        raw_mean, raw_covariance = capital_market_assumptions(reported.iloc[1:])
        limits = AllocationConstraints((False, False, True, True, True), .45,
                                       (.6, .9, 0, 0, 0), (0, 0, .5, .4, .45), .15)
        result = optimize(mean, covariance, limits)
        raw_result = optimize(raw_mean, raw_covariance, limits)
        self.assertLess(result.weights[2], raw_result.weights[2])
        self.assertGreater(float(raw_result.weights @ covariance @ raw_result.weights),
                           raw_result.volatility**2)
        self.assertAlmostEqual(result.weights.sum(), 1)
        self.assertGreaterEqual(result.liquidity_surplus, -1e-10)
        frontier = efficient_frontier(mean, covariance, limits)
        self.assertTrue((np.diff(frontier.volatility) >= -1e-9).all())

    def test_glm_exact_cma_reconciliation(self):
        from pm_pacing.cma import capital_market_assumptions
        truth, reported, coefficients = synthetic_glm_returns()
        recovered = reported.copy()
        for name, weight in coefficients.items():
            recovered[name] = unsmooth_ma1(reported[name], weight)
        mean, covariance = capital_market_assumptions(recovered)
        true_mean, true_covariance = capital_market_assumptions(truth)
        np.testing.assert_allclose(mean, true_mean, atol=1e-14)
        np.testing.assert_allclose(covariance, true_covariance, atol=1e-14)
