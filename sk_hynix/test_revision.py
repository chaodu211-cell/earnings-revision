import unittest

import pandas as pd

from build import monthly_metrics


class MonthlyDerivativeTest(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-10-19", tz="Asia/Shanghai")
        self.points = [{"date": str(d.date()), "eps": float(100 + 10 * i), "high": 300, "low": 0}
                       for i, d in enumerate(pd.date_range("2025-10-31", "2026-09-30", freq="M"))]

    def test_linear_eps_has_constant_first_and_zero_second_derivative(self):
        got = monthly_metrics(self.points, self.now)
        self.assertIsNone(got[0]["d_eps"])
        self.assertIsNone(got[1]["dd_eps"])
        self.assertTrue(all(r["d_eps"] == 10 for r in got[1:]))
        self.assertTrue(all(r["dd_eps"] == 0 for r in got[2:]))

    def test_quadratic_eps_has_expected_second_derivative(self):
        points = [{**p, "eps": float(i*i)} for i, p in enumerate(self.points)]
        got = monthly_metrics(points, self.now)
        self.assertEqual(got[-1]["d_eps"], 21)
        self.assertTrue(all(r["dd_eps"] == 2 for r in got[2:]))

    def test_newer_and_partial_month_values_do_not_rewrite_history(self):
        before = monthly_metrics(self.points, self.now)
        after = monthly_metrics(self.points + [{"date": "2026-10-16", "eps": 99999}], self.now)
        self.assertEqual(before, after)
        extended = monthly_metrics(self.points + [{"date": "2026-10-30", "eps": 220}],
                                   pd.Timestamp("2026-11-10", tz="Asia/Shanghai"))
        self.assertEqual(before, extended[:-1])

    def test_missing_month_breaks_adjacent_differences_without_interpolation(self):
        points = [p for p in self.points if p["date"] != "2026-06-30"]
        got = monthly_metrics(points, self.now)
        missing = next(r for r in got if r["date"] == "2026-06-30")
        self.assertIsNone(missing["observed_date"])
        self.assertIsNone(missing["eps"])
        self.assertIsNone(missing["d_eps"])
        self.assertIsNone(missing["dd_eps"])
        july = next(r for r in got if r["date"] == "2026-07-31")
        august = next(r for r in got if r["date"] == "2026-08-31")
        self.assertIsNone(july["d_eps"])
        self.assertIsNone(august["dd_eps"])
        self.assertEqual(got[-1]["d_eps"], 10)
        self.assertEqual(got[-1]["dd_eps"], 0)

    def test_negative_eps_and_zero_crossing_are_valid(self):
        points = [{**p, "eps": v} for p, v in zip(self.points, [-20, -10, 5, 25])]
        got = monthly_metrics(points, self.now)
        self.assertEqual([r["d_eps"] for r in got[1:]], [10, 15, 20])
        self.assertEqual([r["dd_eps"] for r in got[2:]], [5, 5])


if __name__ == "__main__":
    unittest.main()
