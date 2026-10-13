import _bootstrap
import unittest
import numpy as np
from pm_pacing.cma import smooth, unsmooth, estimate_ar1, autocorrelation, irr, ks_pme, multiples, capital_market_assumptions
from pm_pacing.data import synthetic_returns, synthetic_index


class CMATests(unittest.TestCase):
    def test_unsmoothing_recovers_truth(self):
        rng = np.random.default_rng(17)
        truth = rng.normal(.02, .08, 2000)
        for alpha in [0, .3, .7, .95]:
            reported = smooth(truth, alpha, previous=.01)
            np.testing.assert_allclose(unsmooth(reported, alpha, previous=.01), truth, atol=1e-14)

    def test_unknown_initial_lag_is_dropped(self):
        truth = np.array([.01, .1, -.1, .04])
        reported = smooth(truth, .6)
        np.testing.assert_allclose(unsmooth(reported, .6), truth[1:])

    def test_volatility_rises(self):
        truth = np.random.default_rng(6).normal(.02, .08, 10000)
        reported = smooth(truth, .7)
        recovered = unsmooth(reported, .7)
        self.assertGreater(recovered.std(), reported.std()*2)

    def test_ar1_estimation_and_autocorrelation_falls(self):
        truth = np.random.default_rng(6).normal(.02, .08, 10000)
        reported = smooth(truth, .7)
        self.assertAlmostEqual(estimate_ar1(reported), .7, delta=.02)
        recovered = unsmooth(reported)
        self.assertLess(abs(autocorrelation(recovered)), .03)
        self.assertGreater(autocorrelation(reported), .65)

    def test_invalid_unsmoothing_coefficient(self):
        for alpha in [-.1, 1, np.nan]:
            with self.assertRaises(ValueError):
                unsmooth([.1, .2], alpha)

    def test_constant_ar1_rejected(self):
        with self.assertRaises(ValueError):
            estimate_ar1([.1, .1, .1])

    def test_synthetic_seed_reproducible(self):
        first = synthetic_returns()
        second = synthetic_returns()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(synthetic_index(20), synthetic_index(20))

    def test_cma_annualization(self):
        truth, _, _ = synthetic_returns()
        mean, cov = capital_market_assumptions(truth)
        np.testing.assert_allclose(mean, truth.mean()*4)
        np.testing.assert_allclose(cov, truth.cov()*4)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)

    def test_pme_one_when_index_matched(self):
        self.assertAlmostEqual(ks_pme([100, 0, 0], [0, 0, 121], [100, 110, 121]), 1)

    def test_pme_one_with_intermediate_flows(self):
        index = np.array([100, 110, 90, 120])
        calls = np.array([100, 25, 0, 0])
        distributions = np.array([0, 0, 30, 0.0])
        shares = np.sum(calls/index) - np.sum(distributions/index)
        distributions[-1] = shares*index[-1]
        self.assertAlmostEqual(ks_pme(calls, distributions, index), 1)

    def test_pme_residual_nav(self):
        self.assertAlmostEqual(ks_pme([100, 0], [0, 0], [100, 110], terminal_nav=110), 1)

    def test_pme_index_scale_invariance(self):
        a = ks_pme([100, 0], [0, 120], [100, 110])
        b = ks_pme([100, 0], [0, 120], [1000, 1100])
        self.assertAlmostEqual(a, b)

    def test_pme_increases_with_distributions(self):
        self.assertGreater(ks_pme([100, 0], [0, 130], [100, 110]), ks_pme([100, 0], [0, 120], [100, 110]))

    def test_pme_invalid_inputs(self):
        for calls, distributions, index in [([0], [0], [1]), ([1], [0], [0]), ([1], [-1], [1]), ([1], [0, 0], [1])]:
            with self.assertRaises(ValueError):
                ks_pme(calls, distributions, index)

    def test_irr_known_single_period(self):
        self.assertAlmostEqual(irr([-100, 110]), .1, places=10)

    def test_irr_known_two_period(self):
        self.assertAlmostEqual(irr([-100, 0, 121]), .1, places=10)

    def test_irr_irregular_dates(self):
        self.assertAlmostEqual(irr([-100, 121], [0, 2]), .1, places=10)

    def test_irr_negative(self):
        self.assertAlmostEqual(irr([-100, 81], [0, 2]), -.1, places=10)

    def test_irr_zero(self):
        self.assertAlmostEqual(irr([-100, 100]), 0)

    def test_irr_scale_invariance(self):
        self.assertAlmostEqual(irr([-10, -20, 40]), irr([-100, -200, 400]))

    def test_irr_multiple_roots_rejected(self):
        with self.assertRaises(ValueError):
            irr([-100, 230, -132])

    def test_irr_no_root_rejected(self):
        for flows in [[1, 2], [-1, -2], [0, 0]]:
            with self.assertRaises(ValueError):
                irr(flows)

    def test_multiples_reconcile(self):
        result = multiples([100, 0], [0, 150], 50)
        self.assertEqual(result["tvpi"], 2)
        self.assertEqual(result["dpi"], 1.5)
        self.assertEqual(result["rvpi"], .5)
        self.assertEqual(result["tvpi"], result["dpi"] + result["rvpi"])

    def test_multiples_invalid_paid_in(self):
        with self.assertRaises(ValueError):
            multiples([0], [100])

    def test_irr_small_terminal_flow_is_not_false_endpoint(self):
        result = irr([-100, 1e-8])
        self.assertAlmostEqual(result, 1e-10 - 1, delta=1e-15)

    def test_ar1_scale_invariance(self):
        reported = smooth(np.random.default_rng(11).normal(0, .08, 300), .6)
        self.assertAlmostEqual(estimate_ar1(reported), estimate_ar1(reported * 1e-12))

    def test_irr_extreme_cash_flow_scale(self):
        self.assertAlmostEqual(irr([-1e-300, 1.1e-300]), .1, places=10)
        self.assertAlmostEqual(irr([-1e300, 1.1e300]), .1, places=10)

    def test_irr_extreme_relative_magnitudes(self):
        rate = irr([-1e300, 1e-300], [0, 100])
        self.assertAlmostEqual(rate, 1e-6 - 1, delta=1e-14)
