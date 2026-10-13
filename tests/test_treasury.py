import _bootstrap
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from pm_pacing.treasury import TreasuryPolicy, cash_ladder, stress_flows, liquidity_buffer, quarterly_flows


class TreasuryTests(unittest.TestCase):
    def setUp(self):
        self.calls = [10, 20, 15, 5, 0, 0]
        self.distributions = [0, 0, 2, 15, 30, 5]
        self.policy = TreasuryPolicy(line_limit=12)
        self.path = cash_ladder(self.calls, self.distributions, self.policy, initial_cash=10)

    def test_facility_bounds(self):
        self.assertTrue((self.path.closing_line >= 0).all())
        self.assertTrue((self.path.closing_line <= 12 + 1e-12).all())

    def test_principal_identity(self):
        np.testing.assert_allclose(self.path.closing_line, self.path.opening_line + self.path.draw - self.path.repayment)

    def test_cash_identity(self):
        p = self.path
        np.testing.assert_allclose(p.closing_cash, p.opening_cash + p.cash_income + p.distributions + p.draw + p.lp_funding - p.calls - p.repayment - p.interest)

    def test_cash_nonnegative(self):
        self.assertTrue((self.path.closing_cash >= 0).all())
        self.assertTrue((self.path.lp_funding >= 0).all())

    def test_buffer_held(self):
        self.assertTrue((self.path.closing_cash + 1e-12 >= self.path.required_buffer).all())

    def test_full_line_runoff(self):
        self.assertAlmostEqual(self.path.closing_line.iloc[-1], 0)
        self.assertAlmostEqual(self.path.draw.sum(), self.path.repayment.sum())

    def test_horizon_repayments(self):
        np.testing.assert_allclose(self.path.repayment.iloc[2:], self.path.draw.iloc[:-2])
        self.assertEqual(self.path.repayment.iloc[:2].sum(), 0)

    def test_known_interest(self):
        p = cash_ladder([10, 0, 0], [0, 0, 0], TreasuryPolicy(line_limit=10, financed_fraction=1, annual_interest=.08, annual_cash_yield=0, buffer_fraction=0))
        self.assertAlmostEqual(p.interest.iloc[0], .2)
        self.assertAlmostEqual(p.interest.iloc[1], .2)
        self.assertAlmostEqual(p.interest.iloc[2], 0)
        self.assertAlmostEqual(p.interest.sum(), .4)

    def test_cash_yield(self):
        p = cash_ladder([0, 0], [0, 0], TreasuryPolicy(annual_cash_yield=.04), initial_cash=100)
        self.assertAlmostEqual(p.cash_income.iloc[0], 1)
        self.assertAlmostEqual(p.closing_cash.iloc[0], 101)

    def test_zero_facility(self):
        p = cash_ladder(self.calls, self.distributions, replace(self.policy, line_limit=0))
        self.assertEqual(p.draw.sum(), 0)
        self.assertEqual(p.interest.sum(), 0)
        self.assertGreater(p.lp_funding.sum(), 0)

    def test_stress_conserves_calls(self):
        c, d = stress_flows(self.calls, self.distributions)
        self.assertAlmostEqual(c.sum(), sum(self.calls))
        self.assertAlmostEqual(d.sum(), sum(self.distributions)/2)
        self.assertTrue((np.cumsum(c) >= np.cumsum(self.calls) - 1e-12).all())

    def test_acceleration_moves_only_existing_calls(self):
        c, _ = stress_flows([1, 2, 3], [0, 0, 0], acceleration=1)
        np.testing.assert_array_equal(c, [3, 3, 0])

    def test_stress_increases_lp_funding(self):
        c, d = stress_flows(self.calls, self.distributions)
        stressed = cash_ladder(c, d, self.policy, initial_cash=10)
        self.assertGreater(stressed.lp_funding.sum(), self.path.lp_funding.sum())

    def test_buffer_known_window(self):
        np.testing.assert_allclose(liquidity_buffer([10, 20, 30], window=2, fraction=.5), [15, 25, 15])

    def test_quarterly_reconciliation(self):
        annual = pd.DataFrame({"year": [2026, 2027], "calls": [40, 20], "distributions": [10, 30]})
        quarterly = quarterly_flows(annual)
        self.assertEqual(quarterly.calls.sum(), 60)
        self.assertEqual(quarterly.distributions.sum(), 40)
        self.assertEqual(quarterly.period.iloc[-1], "2027Q4")

    def test_coverage_known_peak(self):
        p = cash_ladder([10, 0, 0, 0], [0, 0, 0, 0], TreasuryPolicy(line_limit=0, annual_interest=0), initial_cash=20)
        self.assertAlmostEqual(p.liquidity_coverage_12m.iloc[0], 2)

    def test_no_outflows_coverage(self):
        p = cash_ladder([0, 0], [1, 1])
        self.assertTrue(np.isinf(p.liquidity_coverage_12m.iloc[0]))

    def test_invalid_cashflows(self):
        for c, d in [([-1], [0]), ([1], [0, 0]), ([np.nan], [0])]:
            with self.subTest(c=c), self.assertRaises(ValueError):
                cash_ladder(c, d)

    def test_invalid_policy(self):
        for kwargs in [{"line_limit": -1}, {"repayment_quarters": 0}, {"financed_fraction": 2}, {"annual_interest": np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TreasuryPolicy(**kwargs)

    def test_quarterly_calendar_gaps_are_zero_flows(self):
        annual = pd.DataFrame({"year": [2026, 2028], "calls": [40, 20], "distributions": [10, 30]})
        quarterly = quarterly_flows(annual)
        self.assertEqual(len(quarterly), 12)
        self.assertEqual(quarterly.calls.iloc[4:8].sum(), 0)
        self.assertEqual(quarterly.calls.sum(), 60)

    def test_quarterly_sorts_calendar(self):
        annual = pd.DataFrame({"year": [2027, 2026], "calls": [20, 40], "distributions": [30, 10]})
        quarterly = quarterly_flows(annual)
        self.assertEqual(quarterly.period.iloc[0], "2026Q1")
        self.assertEqual(quarterly.calls.iloc[0], 10)

    def test_quarterly_invalid_calendar_or_flows(self):
        for annual in [pd.DataFrame(columns=["year", "calls", "distributions"]),
                       pd.DataFrame({"year": [2026, 2026], "calls": [1, 2], "distributions": [0, 0]}),
                       pd.DataFrame({"year": [2026.5], "calls": [1], "distributions": [0]}),
                       pd.DataFrame({"year": [2026], "calls": [-1], "distributions": [0]}),
                       pd.DataFrame({"year": [2026], "calls": [1], "distributions": [np.nan]})]:
            with self.subTest(annual=annual), self.assertRaises(ValueError):
                quarterly_flows(annual)
