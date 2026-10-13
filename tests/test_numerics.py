import _bootstrap
import unittest
import numpy as np
from pm_pacing.numerics import bisect, finite_vector


class NumericsTests(unittest.TestCase):
    def test_known_root(self):
        self.assertAlmostEqual(bisect(lambda x: x*x - 2, 0, 2), np.sqrt(2), places=10)

    def test_left_endpoint(self):
        self.assertEqual(bisect(lambda x: x, 0, 1), 0)

    def test_right_endpoint(self):
        self.assertEqual(bisect(lambda x: x - 1, 0, 1), 1)

    def test_negative_slope(self):
        self.assertAlmostEqual(bisect(lambda x: 3 - x, 1, 4), 3)

    def test_unbracketed_rejected(self):
        with self.assertRaises(ValueError):
            bisect(lambda x: x*x + 1, -1, 1)

    def test_nonfinite_function_rejected(self):
        with self.assertRaises(ValueError):
            bisect(lambda x: np.nan, 0, 1)

    def test_iteration_failure(self):
        with self.assertRaises(RuntimeError):
            bisect(lambda x: x*x - 2, 0, 2, max_iterations=1)

    def test_invalid_controls(self):
        for tolerance in [0, -1, np.nan]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                bisect(lambda x: x, -1, 1, tolerance=tolerance)

    def test_bad_vectors(self):
        for value in [[], [[1]], [np.nan], [np.inf]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                finite_vector(value, "input")
