import _bootstrap
import unittest
from dataclasses import replace
import numpy as np
from pm_pacing.ta_model import FundParameters, Vintage, project_fund, project_program, STYLIZED_PARAMETERS
from pm_pacing.cma import irr


class FundTests(unittest.TestCase):
    def setUp(self):
        self.p = FundParameters()
        self.f = Vintage("buyout", 2026, 100.0, self.p)
        self.path = project_fund(self.f)

    def test_nav_identity(self):
        for p in STYLIZED_PARAMETERS.values():
            frame = project_fund(Vintage("fund", 0, 100, p))
            np.testing.assert_allclose(frame.nav, frame.opening_nav * (1+p.growth) + frame.calls - frame.distributions, atol=1e-12)

    def test_unfunded_identity(self):
        np.testing.assert_allclose(self.path.unfunded, self.path.opening_unfunded - self.path.calls - self.path.canceled)

    def test_commitment_conservation(self):
        np.testing.assert_allclose(self.path.calls.cumsum() + self.path.canceled.cumsum() + self.path.unfunded, 100)

    def test_never_overcalls(self):
        self.assertTrue((self.path.calls.cumsum() <= 100).all())

    def test_terminal_liquidation(self):
        for name, p in STYLIZED_PARAMETERS.items():
            with self.subTest(name=name):
                frame = project_fund(Vintage(name, 0, 100, p))
                self.assertAlmostEqual(frame.nav.iloc[-1], 0)
                self.assertAlmostEqual(frame.unfunded.iloc[-1], 0)

    def test_j_curve(self):
        self.assertLess(self.path.cumulative_net_cash_flow.iloc[0], 0)
        self.assertGreater(self.path.cumulative_net_cash_flow.iloc[-1], 0)
        self.assertLess(self.path.cumulative_net_cash_flow.min(), self.path.cumulative_net_cash_flow.iloc[0])

    def test_published_first_period(self):
        self.assertEqual(self.path.calls.iloc[0], 25)
        self.assertEqual(self.path.distributions.iloc[0], 0)
        self.assertEqual(self.path.nav.iloc[0], 25)

    def test_published_second_period(self):
        self.assertAlmostEqual(self.path.calls.iloc[1], 75 * .35)
        self.assertAlmostEqual(self.path.distributions.iloc[1], (2/12)**2.5 * 25 * 1.12)

    def test_strict_published_terminal_difference(self):
        strict = project_fund(self.f, liquidate_terminal=False)
        self.assertAlmostEqual(strict.nav.iloc[-1], strict.calls.iloc[-1])
        self.assertAlmostEqual(self.path.distributions.iloc[-1] - strict.distributions.iloc[-1], strict.calls.iloc[-1])

    def test_irr_equals_growth(self):
        self.assertAlmostEqual(irr(self.path.net_cash_flow), self.p.growth, places=9)

    def test_zero_growth_cash_conservation(self):
        frame = project_fund(replace(self.f, parameters=replace(self.p, growth=0)))
        self.assertAlmostEqual(frame.calls.sum(), frame.distributions.sum())

    def test_program_additivity(self):
        other = Vintage("credit", 2028, 50, STYLIZED_PARAMETERS["private_credit"])
        detail, annual = project_program([self.f, other])
        self.assertAlmostEqual(annual.calls.sum(), detail.calls.sum())
        np.testing.assert_allclose(annual.nav, detail.groupby("year").nav.sum().reindex(annual.year, fill_value=0))

    def test_program_calendar_gaps(self):
        _, annual = project_program([Vintage("a", 0, 1, FundParameters(life=2)), Vintage("b", 5, 1, FundParameters(life=2))])
        self.assertEqual(annual.year.tolist(), list(range(7)))
        self.assertEqual(annual.loc[annual.year == 3, "calls"].iloc[0], 0)

    def test_linear_scaling(self):
        double = project_fund(replace(self.f, commitment=200))
        np.testing.assert_allclose(double.nav, self.path.nav*2)
        np.testing.assert_allclose(double.calls, self.path.calls*2)

    def test_distribution_rate_rises(self):
        self.assertTrue((np.diff(self.path.distribution_rate) >= 0).all())

    def test_faster_distributions(self):
        fast = project_fund(replace(self.f, parameters=replace(self.p, bow=1)))
        self.assertGreater(fast.distributions.iloc[:5].sum(), self.path.distributions.iloc[:5].sum())

    def test_secondaries_earlier_distribution(self):
        secondary = project_fund(replace(self.f, parameters=STYLIZED_PARAMETERS["secondaries"]))
        self.assertGreater(secondary.distributions.iloc[:3].sum(), self.path.distributions.iloc[:3].sum())
        self.assertLess(len(secondary), len(self.path))

    def test_co_invest_immediate_call(self):
        frame = project_fund(replace(self.f, parameters=STYLIZED_PARAMETERS["co_invest"]))
        self.assertEqual(frame.calls.iloc[0], 100)
        self.assertEqual(frame.calls.iloc[1:].sum(), 0)

    def test_zero_commitment(self):
        frame = project_fund(replace(self.f, commitment=0))
        self.assertEqual(frame.nav.sum() + frame.calls.sum() + frame.distributions.sum(), 0)

    def test_one_period_terminal(self):
        frame = project_fund(replace(self.f, parameters=FundParameters(life=1)))
        self.assertEqual(frame.nav.iloc[0], 0)
        self.assertEqual(frame.calls.iloc[0], frame.distributions.iloc[0])

    def test_negative_growth(self):
        frame = project_fund(replace(self.f, parameters=replace(self.p, growth=-.2)))
        self.assertTrue((frame.nav >= 0).all())
        self.assertLess(frame.distributions.sum(), frame.calls.sum())

    def test_invalid_parameters(self):
        for kwargs in [{"life": 0}, {"life": 2.5}, {"growth": -1}, {"bow": 0}, {"call_rates": ()}, {"call_rates": (1.1,)}, {"yield_rate": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FundParameters(**kwargs)

    def test_invalid_program(self):
        for funds in [[], [self.f, self.f]]:
            with self.assertRaises(ValueError):
                project_program(funds)

    def test_program_unfunded_calendar_identity(self):
        _, annual = project_program([self.f, Vintage("other", 2028, 75, self.p)])
        opening = np.append(0, annual.unfunded.to_numpy()[:-1])
        np.testing.assert_allclose(annual.unfunded, opening + annual.new_commitments - annual.calls - annual.canceled, atol=1e-12)
