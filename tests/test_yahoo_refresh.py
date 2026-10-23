"""本次采集必须进入本次页面，失败不能冒充成功或污染正式历史。"""
import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import breadth
import early
import yahoo_snapshot
from snapshot_store import snapshot_paths

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 20, 19, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def row(ticker):
    r = {"ticker": ticker, "price": 100, "shares": 1000,
         "currency": "USD", "eps_ccy": "USD", "0y_end": "2026-12-31"}
    for period in ("0y", "+1y"):
        for lag in yahoo_snapshot.LAGS:
            r[f"{period}_{lag}"] = 10 if lag == "current" else 9
    return r


class YahooRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "yahoo").mkdir()
        (self.root / "yahoo_local").mkdir()
        self.sectors = self.root / "sectors.json"
        self.sectors.write_text(json.dumps({"AAPL": "IT", "MSFT": "IT"}))

    def collect(self, fetch, now=NOW):
        with contextlib.ExitStack() as stack:
            for name, value in {"BASE": str(self.root), "OUT": str(self.root / "yahoo"),
                                "SECTORS": str(self.sectors)}.items():
                stack.enter_context(patch.object(yahoo_snapshot, name, value))
            clock = stack.enter_context(patch.object(yahoo_snapshot, "datetime", wraps=datetime))
            clock.now.return_value = now
            mocked = stack.enter_context(patch.object(yahoo_snapshot, "fetch", side_effect=fetch))
            stack.enter_context(patch.object(sys, "argv", ["yahoo_snapshot.py", "--local"]))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yahoo_snapshot.main()
            return mocked.call_count, output.getvalue()

    def test_local_fetch_is_read_by_breadth_realtime_and_early(self):
        fixture = ROOT / "yahoo" / "2026-10-16.csv"
        shutil.copyfile(fixture, self.root / "yahoo" / fixture.name)
        shutil.copyfile(fixture, self.root / "yahoo_local" / "2026-10-20.csv")
        with patch.object(breadth, "BASE", str(self.root)), patch.object(early, "BASE", str(self.root)), \
             patch.object(early, "LOG", str(self.root / "no_log.csv")), \
             patch.object(early, "prices", return_value=(None, False)):
            data = breadth.breadth_data()
            self.assertEqual(data["latest"]["date"], "2026-10-20")
            self.assertEqual(data["daily"][-1]["date"], "2026-10-20")
            self.assertEqual(str(max(breadth.index_7d()).date()), "2026-10-20")
            self.assertEqual(early.page_data()["date"], "2026-10-20")

    def test_official_same_day_wins_and_local_never_enters_formal_history(self):
        for folder in ("yahoo", "yahoo_local"):
            self.write_snapshot(self.root / folder / "2026-10-20.csv")
        self.write_snapshot(self.root / "yahoo_local" / "2026-10-19.csv")
        paths = snapshot_paths(self.root)
        self.assertEqual([Path(p).name for p in paths], ["2026-10-19.csv", "2026-10-20.csv"])
        self.assertEqual(Path(paths[-1]).parent.name, "yahoo")
        self.assertEqual(len(snapshot_paths(self.root, include_local=False)), 1)

    def write_snapshot(self, path):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row("AAPL")))
            writer.writeheader()
            writer.writerow(row("AAPL"))

    def test_bad_official_does_not_hide_valid_local_same_day(self):
        (self.root / "yahoo" / "2026-10-20.csv").write_text("ticker,price\nAAPL,100\n")
        local = self.root / "yahoo_local" / "2026-10-20.csv"
        self.write_snapshot(local)
        self.assertEqual(snapshot_paths(self.root), [str(local)])

    def test_success_then_repeat_does_not_fetch_or_overwrite(self):
        calls, output = self.collect(row)
        self.assertEqual(calls, 2)
        self.assertIn("SNAPSHOT_STATUS downloaded 2026-10-20", output)
        path = self.root / "yahoo_local" / "2026-10-20.csv"
        saved = path.read_bytes()
        calls, output = self.collect(lambda _: self.fail("同日不应重抓"))
        self.assertEqual(calls, 0)
        self.assertIn("SNAPSHOT_STATUS existing", output)
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(list((self.root / "yahoo").iterdir()), [])
        self.assertFalse((self.root / "yahoo_agg.csv").exists())

    def test_network_failure_preserves_history_and_creates_no_snapshot(self):
        old = self.root / "yahoo" / "2026-10-16.csv"
        old.write_text("ticker\nAAPL\n")
        with self.assertRaisesRegex(SystemExit, "连接检查失败"):
            self.collect(lambda _: None)
        self.assertEqual(old.read_text(), "ticker\nAAPL\n")
        self.assertEqual(list((self.root / "yahoo_local").iterdir()), [])

    def test_missing_earnings_module_is_rejected_and_corrupt_local_can_retry(self):
        path = self.root / "yahoo_local" / "2026-10-20.csv"
        path.write_text("ticker,price\nAAPL,100\n")
        with self.assertRaisesRegex(SystemExit, "完整年度 EPS"):
            self.collect(lambda ticker: {"ticker": ticker, "price": 100})
        self.assertEqual(path.read_text(), "ticker,price\nAAPL,100\n")
        self.collect(row)
        with path.open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_late_intraday_run_does_not_consume_next_beijing_date(self):
        late = datetime(2026, 10, 21, 1, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        calls, output = self.collect(lambda _: self.fail("盘中闸门应先于联网"), late)
        self.assertEqual(calls, 0)
        self.assertIn("SNAPSHOT_STATUS market_hours", output)
        self.assertEqual(list((self.root / "yahoo_local").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
