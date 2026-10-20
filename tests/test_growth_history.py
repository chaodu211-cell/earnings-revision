"""Growth availability must not change when later report records are appended."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import next_growth, time_roll


def calculate(records, index):
    frame = pd.DataFrame(records, columns=['date', 'g_next'])
    frame['date'] = pd.to_datetime(frame.date)
    with patch('analyze.pd.read_csv', return_value=frame):
        return next_growth(pd.DatetimeIndex(index))


class GrowthHistoryTests(unittest.TestCase):
    def test_no_year_start_backfill_and_cross_year_carry(self):
        index = pd.to_datetime(['2022-12-16', '2023-01-06', '2023-06-02',
                                '2023-06-09', '2024-01-05', '2024-03-01', '2024-03-08'])
        growth, flags = calculate([('2022-12-15', 5.3), ('2023-06-09', 12.3),
                                   ('2023-12-15', 11.5), ('2024-03-08', 13.2)], index)
        self.assertEqual(list(growth.round(6)), [.053, .053, .053, .123, .115, .115, .132])
        self.assertFalse(flags.any())

    def test_first_eligible_record_does_not_fill_earlier_weeks(self):
        growth, flags = calculate([('2014-12-19', 99), ('2016-09-08', 13.5)],
                                  pd.to_datetime(['2015-12-25', '2016-01-01',
                                                  '2016-09-02', '2016-09-09']))
        self.assertTrue(growth.iloc[:3].isna().all())
        self.assertAlmostEqual(growth.iloc[3], .135)
        self.assertFalse(flags.any())

    def test_thursday_report_available_friday(self):
        growth, _ = calculate([('2024-03-07', 13.2)],
                              pd.date_range('2024-03-06', '2024-03-09'))
        self.assertTrue(growth.iloc[:2].isna().all())
        self.assertEqual(list(growth.iloc[2:]), [.132, .132])

    def test_sparse_queries_retain_intermediate_records(self):
        growth, _ = calculate([('2024-03-08', 13.2), ('2024-03-15', 13.4)],
                              pd.to_datetime(['2024-03-22', '2024-04-05']))
        self.assertEqual(list(growth), [.134, .134])

    def test_future_records_cannot_change_past_growth_or_roll(self):
        index = pd.date_range('2022-12-16', '2024-07-05', freq='W-FRI')
        early = [('2022-12-15', 5.3), ('2023-06-09', 12.3)]
        extended = early + [('2023-12-15', 11.5), ('2024-03-08', 13.2)]
        before, _ = calculate(early, index)
        after, _ = calculate(extended, index)
        cutoff = pd.Timestamp('2023-12-08')
        pd.testing.assert_series_equal(before.loc[:cutoff], after.loc[:cutoff])
        pd.testing.assert_series_equal(time_roll(index, before).rolling(13).sum().loc[:cutoff],
                                       time_roll(index, after).rolling(13).sum().loc[:cutoff])

    def test_unordered_records_in_same_week_use_latest_report(self):
        growth, _ = calculate([('2024-03-08', 13.4), ('2024-03-07', 13.2)],
                              pd.to_datetime(['2024-03-08']))
        self.assertEqual(growth.iloc[0], .134)

    def test_no_eligible_records_leave_all_values_missing(self):
        growth, flags = calculate([('2014-12-19', 99)],
                                  pd.date_range('2016-01-01', periods=3, freq='W-FRI'))
        self.assertTrue(growth.isna().all())
        self.assertFalse(flags.any())


if __name__ == '__main__':
    unittest.main()
