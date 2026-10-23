#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股因子模型的数据：股票池、日线（复权收盘 + 成交量）、历史财报（公告时间、EPS 预期、实际 EPS）。
全部缓存在 factor_model/_cache/（不进 git）。

股票池：Nasdaq 选股器全量美股，现市值 ≥ 3 亿美元的普通股（去掉权证、单位、优先股、票据、基金）。
  回测和打分时再按当时的股价 ≥ 5、近 60 日成交额中位数 ≥ 2000 万美元过滤（见 engine.universe_mask），
  所以 2023 年的 PLTR、2025 年的 SNDK 这类当时还不是标普成分股的公司都在里面。
  幸存者偏差：清单是今天还在交易的公司，已退市、被收购的拿不到。
日线：yfinance，2010 年起，auto_adjust（拆股 + 分红复权）。增量更新只补最后几天。
财报：yfinance get_earnings_dates（Yahoo 财报日历），每只最多 100 条（约 25 年）。
  增量更新只重拉「上次记录里已到期的那次财报还没有实际 EPS」的公司，外加清单里新出现的公司。

用法：python3 factor_model/data.py [--universe] [--prices] [--earnings] [--full]
      不带参数 = 三样都增量更新
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))
from analyze import drop_partial   # 美股收盘前不用 Yahoo 给的当天盘中价

CACHE = os.path.join(BASE, "_cache")
UNI = os.path.join(CACHE, "universe.csv")
PXC = os.path.join(CACHE, "px_close.pkl")
PXV = os.path.join(CACHE, "px_vol.pkl")
EARN = os.path.join(CACHE, "earn_raw.csv")
BENCH = ["SPY", "QQQ", "IWM", "RSP"]
SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
NOT_COMMON = re.compile(r"\b(Warrant|Warrants|Unit|Units|Right|Rights|Preferred|Pfd|Notes|Debenture|ETF|Fund|"
                        r"Depositary Shares, each representing|Series [A-Z] )", re.I)


def _proxy():
    if not os.environ.get("CI"):
        os.environ.setdefault("https_proxy", "http://127.0.0.1:15236")


def update_universe(min_cap=3e8):
    import urllib.request
    _proxy()
    req = urllib.request.Request(SCREENER, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    rows = json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]["rows"]
    df = pd.DataFrame(rows)
    df["cap"] = pd.to_numeric(df.marketCap, errors="coerce")
    u = df[(df.cap >= min_cap) & ~df.name.str.contains(NOT_COMMON) & ~df.symbol.str.contains(r"\^")].copy()
    u["yf"] = u.symbol.str.replace("/", "-", regex=False).str.strip()
    os.makedirs(CACHE, exist_ok=True)
    u[["yf", "name", "cap", "country", "ipoyear", "industry", "sector"]].to_csv(UNI, index=False)
    print(f"data: 股票池 {len(u)} 只 → {os.path.relpath(UNI, BASE)}")
    return u


def universe():
    if not os.path.exists(UNI):
        return update_universe()
    return pd.read_csv(UNI)


def _download(tickers, start):
    import yfinance as yf
    def get(ts, threads):
        d = yf.download(ts, start=str(start), auto_adjust=True, progress=False, threads=threads)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    closes, vols = [], []
    for i in range(0, len(tickers), 250):
        ch = tickers[i:i + 250]
        for k in range(3):
            try:
                d = get(ch, True)
                break
            except Exception as e:
                print(f"data: 下载重试 {i}: {e}", file=sys.stderr)
                time.sleep(20)
        else:
            continue
        c, v = d["Close"], d["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(ch[0]), v.to_frame(ch[0])
        empty = [t for t in ch if t not in c or c[t].notna().sum() == 0]
        if empty:   # 并发下载时 Yahoo 会随机打回一部分，单线程补一遍
            try:
                d2 = get(empty, False)
                c2, v2 = d2["Close"], d2["Volume"]
                if isinstance(c2, pd.Series):
                    c2, v2 = c2.to_frame(empty[0]), v2.to_frame(empty[0])
                c, v = c.combine_first(c2), v.combine_first(v2)
            except Exception as e:
                print(f"data: 补下载失败：{e}", file=sys.stderr)
        closes.append(c.astype("float32"))
        vols.append(v.astype("float32"))
        print(f"data: 日线 {min(i + 250, len(tickers))}/{len(tickers)}", flush=True)
    C = pd.concat(closes, axis=1)
    V = pd.concat(vols, axis=1)
    return C.loc[:, ~C.columns.duplicated()], V.loc[:, ~V.columns.duplicated()]


def update_prices(full=False):
    """首次（或 --full）从 2010 年拉全量；之后只拉最近 15 天并覆盖（复权因子变了就整只重拉）。"""
    _proxy()
    tick = sorted(set(universe().yf) | set(BENCH))
    if full or not os.path.exists(PXC):
        C, V = _download(tick, "2010-01-01")
    else:
        C, V = pd.read_pickle(PXC), pd.read_pickle(PXV)
        new = [t for t in tick if t not in C.columns]
        start = (C.index[-1] - pd.Timedelta(days=15)).date()
        c2, v2 = _download(sorted(set(tick) - set(new)), start)
        # 分红 / 拆股后复权价整段会变：重叠日的价差 > 0.5% 的股票整只重拉
        ov = C.index.intersection(c2.index)
        drift = (c2.loc[ov] / C.loc[ov, c2.columns] - 1).abs().max()
        redo = sorted(set(drift[drift > 0.005].index) | set(new))
        if redo:
            c3, v3 = _download(redo, "2010-01-01")
            C = C.drop(columns=[t for t in redo if t in C.columns]).join(c3, how="outer")
            V = V.drop(columns=[t for t in redo if t in V.columns]).join(v3, how="outer")
        C = c2.combine_first(C)
        V = v2.combine_first(V)
    C, V = drop_partial(C), drop_partial(V)
    C.sort_index().to_pickle(PXC)
    V.sort_index().to_pickle(PXV)
    print(f"data: 日线 {C.shape[1]} 只，截至 {C.index[-1].date()}")
    return C, V


def _earn_one(t):
    import yfinance as yf
    for k in range(4):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=100)
            if e is None or len(e) == 0:
                return pd.DataFrame(columns=["ticker", "dt", "est", "act"])
            e = e.reset_index()
            dt = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert("America/New_York")
            return pd.DataFrame({"ticker": t, "dt": dt.dt.strftime("%Y-%m-%d %H:%M"),
                                 "est": e["EPS Estimate"], "act": e["Reported EPS"]})
        except Exception as ex:
            if "No earnings dates" in str(ex) or "delisted" in str(ex):
                return pd.DataFrame(columns=["ticker", "dt", "est", "act"])
            time.sleep(15 * (k + 1))
    return None


def update_earnings(full=False, workers=4):
    _proxy()
    tick = list(universe().yf)
    old = pd.read_csv(EARN) if os.path.exists(EARN) and not full else pd.DataFrame(columns=["ticker", "dt", "est", "act"])
    if len(old):
        dt = pd.to_datetime(old.dt)
        now = pd.Timestamp.now()
        # 已到公告时间（再过一天让 Yahoo 填上实际 EPS）、但还没有实际 EPS 的那家重拉
        due = old[(dt <= now - pd.Timedelta(hours=20)) & (dt >= now - pd.Timedelta(days=60)) & old.act.isna()].ticker
        todo = sorted(set(due) | (set(tick) - set(old.ticker)))
    else:
        todo = tick
    print(f"data: 财报 {len(todo)} 只要拉", flush=True)
    got, lock, n = [], threading.Lock(), [0]
    def one(t):
        r = _earn_one(t)
        with lock:
            n[0] += 1
            if n[0] % 200 == 0:
                print(f"data: 财报 {n[0]}/{len(todo)}", flush=True)
        return t, r
    with ThreadPoolExecutor(workers) as ex:
        for t, r in ex.map(one, todo):
            if r is not None:
                got.append((t, r))
    done = {t for t, _ in got}
    new = pd.concat([old[~old.ticker.isin(done)]] + [r for _, r in got if len(r)], ignore_index=True)
    new.sort_values(["ticker", "dt"]).to_csv(EARN, index=False)
    print(f"data: 财报记录 {len(new)} 条，{new.ticker.nunique()} 只（本次更新 {len(done)} 只）")
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--earnings", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    every = not (a.universe or a.prices or a.earnings)
    os.makedirs(CACHE, exist_ok=True)
    if a.universe or (every and not os.path.exists(UNI)):
        update_universe()
    if a.prices or every:
        update_prices(a.full)
    if a.earnings or every:
        update_earnings(a.full)
