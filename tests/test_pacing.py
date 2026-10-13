import _bootstrap
import unittest
import numpy as np
from pm_pacing.pacing import solve_pacing, sensitivity_table
from pm_pacing.ta_model import FundParameters


class PacingTests(unittest.TestCase):
    def path(self, **kwargs):
        inputs = dict(initial_public_nav=1000, target=.25, years=25, parameters=FundParameters(), max_commitment=400)
        inputs.update(kwargs)
        return solve_pacing(**inputs)

    def test_hits_and_holds_target(self):
        path = self.path(parameters=FundParameters(6, .1, (.5,), 1), max_commitment=5000)
        self.assertLess(np.abs(path.target_error).max(), 1e-9)
        np.testing.assert_allclose(path.allocation.iloc[3:], .25, atol=1e-9)

    def test_cap_respected(self):
        path = self.path(max_commitment=20)
        self.assertLessEqual(path.commitment.max(), 20)
        self.assertIn("capped", path.status.tolist())
        self.assertLess(path.allocation.iloc[3], .25)

    def test_public_cash_identity(self):
        path = self.path()
        opening = np.append(1000, path.public_nav.to_numpy()[:-1])
        np.testing.assert_allclose(path.public_nav, opening + path.public_growth_pnl + path.distributions - path.calls)

    def test_total_wealth_conserved(self):
        path = self.path()
        opening = np.append(1000, path.total_nav.to_numpy()[:-1])
        np.testing.assert_allclose(path.total_nav, opening + path.public_growth_pnl + path.private_growth_pnl)

    def test_overcommitment_definition(self):
        path = self.path()
        np.testing.assert_allclose(path.overcommitment_ratio, (path.private_nav + path.unfunded) / path.total_nav)
        self.assertTrue((path.overcommitment_ratio >= path.allocation).all())

    def test_zero_target(self):
        path = self.path(target=0)
        self.assertEqual(path.commitment.sum(), 0)
        self.assertEqual(path.private_nav.sum(), 0)

    def test_public_funding_nonnegative(self):
        path = self.path(target=.8, ramp_years=1, max_commitment=10000)
        self.assertTrue((path.public_nav >= -1e-9).all())
        self.assertTrue((path.commitment >= 0).all())
        self.assertIn("funding_limited", path.status.tolist())

    def test_scaling(self):
        small = self.path()
        large = self.path(initial_public_nav=2000, max_commitment=800)
        np.testing.assert_allclose(large.commitment, small.commitment * 2)

    def test_sensitivity_has_real_effect(self):
        table = sensitivity_table(1000, .25, 4, FundParameters(6, .1, (.5,), 1), 400).set_index("scenario")
        self.assertGreater(table.loc["faster_distributions", "total_commitments"], table.loc["base", "total_commitments"])
        self.assertLess(table.loc["higher_growth", "total_commitments"], table.loc["base", "total_commitments"])
        self.assertLess(table.max_target_error.max(), 1e-9)

    def test_invalid_inputs(self):
        for kwargs in [{"initial_public_nav": 0}, {"target": 1}, {"max_commitment": -1}, {"years": 0}, {"ramp_years": 0}, {"public_growth": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.path(**kwargs)

    def test_no_first_year_investment_rejected(self):
        with self.assertRaises(ValueError):
            self.path(parameters=FundParameters(call_rates=(0, .5)))

    def test_standard_fund_reports_unavoidable_ramp_overshoot(self):
        path = self.path()
        self.assertIn("above_target", path.status.tolist())
        self.assertLess(np.abs(path.target_error.iloc[7:]).max(), 1e-9)

    def test_funding_reservation_prevents_future_insolvency(self):
        path = self.path(target=.8, ramp_years=1, max_commitment=10000)
        self.assertLess(path.allocation.iloc[0], .8)
        self.assertTrue((path.public_nav >= -1e-9).all())
