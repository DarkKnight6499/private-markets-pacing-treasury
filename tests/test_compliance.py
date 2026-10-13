import _bootstrap
import unittest
from dataclasses import replace
from pm_pacing.compliance import Holding, Exposure, check_trade, overall_status
from pm_pacing.data import synthetic_exposures, stylized_rules


class ComplianceTests(unittest.TestCase):
    def setUp(self):
        self.trade = Exposure("fund", "manager", "buyout", 2026, "NA", 0, 10, (Holding("issuer", 1, "technology", "US"),))
        self.rule = {"id": "cap", "kind": "unfunded", "upper": .20, "warn_at": .18}

    def status(self, amount):
        return check_trade([], replace(self.trade, unfunded=amount), 100, [self.rule]).status.iloc[0]

    def test_pass_below_warning(self):
        self.assertEqual(self.status(17.999), "pass")

    def test_warn_at_warning(self):
        self.assertEqual(self.status(18), "warn")

    def test_warn_at_limit(self):
        self.assertEqual(self.status(20), "warn")

    def test_fail_above_limit(self):
        self.assertEqual(self.status(20.001), "fail")

    def test_unfunded_additivity(self):
        result = check_trade([self.trade], replace(self.trade, fund="other"), 100, [self.rule])
        self.assertAlmostEqual(result.measured.iloc[0], .2)

    def test_manager_aggregates_funds(self):
        rule = {"id": "manager", "kind": "concentration", "dimension": "manager", "upper": .15}
        result = check_trade([self.trade], replace(self.trade, fund="other"), 100, [rule])
        self.assertEqual(result.status.iloc[0], "fail")
        self.assertAlmostEqual(result.measured.iloc[0], .2)

    def test_fund_aggregates_topups(self):
        rule = {"id": "fund", "kind": "concentration", "dimension": "fund", "upper": .15}
        self.assertEqual(check_trade([self.trade], self.trade, 100, [rule]).status.iloc[0], "fail")

    def test_issuer_lookthrough(self):
        rule = {"id": "issuer", "kind": "concentration", "dimension": "issuer", "upper": .15}
        other = replace(self.trade, fund="other", manager="other")
        self.assertEqual(check_trade([self.trade], other, 100, [rule]).status.iloc[0], "fail")

    def test_sector_exclusion(self):
        rule = {"id": "sector", "kind": "exclusion", "dimension": "sector", "excluded": ["coal"]}
        bad = replace(self.trade, holdings=(Holding("issuer", 1, "coal", "US"),))
        self.assertEqual(check_trade([], bad, 100, [rule]).status.iloc[0], "fail")
        self.assertEqual(check_trade([], self.trade, 100, [rule]).status.iloc[0], "pass")

    def test_jurisdiction_exclusion(self):
        rule = {"id": "jurisdiction", "kind": "exclusion", "dimension": "jurisdiction", "excluded": ["X"]}
        bad = replace(self.trade, holdings=(Holding("issuer", 1, "technology", "X"),))
        self.assertEqual(check_trade([], bad, 100, [rule]).status.iloc[0], "fail")

    def test_policy_dimensions(self):
        for dimension, category in [("strategy", "buyout"), ("vintage", 2026), ("geography", "NA")]:
            rule = {"id": dimension, "kind": "band", "dimension": dimension, "category": category, "lower": .05, "upper": .2}
            with self.subTest(dimension=dimension):
                self.assertEqual(check_trade([], self.trade, 100, [rule]).status.iloc[0], "pass")
                self.assertEqual(check_trade([], replace(self.trade, unfunded=0), 100, [rule]).status.iloc[0], "fail")

    def test_lower_band_inclusive(self):
        rule = {"id": "band", "kind": "band", "dimension": "strategy", "category": "buyout", "lower": .1, "upper": .2}
        self.assertEqual(check_trade([], self.trade, 100, [rule]).status.iloc[0], "pass")

    def test_no_nav_denominator_dilution(self):
        result = check_trade([], replace(self.trade, nav=10), 100, [self.rule])
        self.assertAlmostEqual(result.measured.iloc[0], .1)

    def test_complete_lookthrough_required(self):
        with self.assertRaises(ValueError):
            replace(self.trade, holdings=(Holding("issuer", .5, "technology", "US"),))

    def test_rules_must_be_explicit(self):
        for rules in [[], [self.rule, self.rule]]:
            with self.assertRaises(ValueError):
                check_trade([], self.trade, 100, rules)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            check_trade([], self.trade, 100, [{"id": "x", "kind": "unknown"}])

    def test_overall_severity(self):
        result = check_trade(synthetic_exposures(), self.trade, 1000, stylized_rules())
        self.assertIn(overall_status(result), {"pass", "warn", "fail"})
        failed = check_trade([], replace(self.trade, unfunded=30), 100, [self.rule])
        self.assertEqual(overall_status(failed), "fail")

    def test_tight_band_has_valid_default_warning(self):
        rule = {"id": "band", "kind": "band", "dimension": "strategy", "category": "buyout", "lower": .19, "upper": .20}
        result = check_trade([], replace(self.trade, unfunded=19.5), 100, [rule])
        self.assertEqual(result.status.iloc[0], "warn")
