#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上修初期名单 + 前瞻跟踪（2026-10-05 起）。

目的：在一家公司盈利快速上修的初期就找到它，而不是等上修已经被股价充分定价之后（上修前 5 的 MU、MPC 入选时股价早已大涨）。
个股修正历史拿不到（Yahoo 快照 2026-10-02 起），没法回测，所以每天记下名单、事后看表现——前瞻检验，规则先不改。

规则（最新一份 Yahoo 成分股快照，远期 EPS 口径同 breadth.py / 纯修正，不含时间滚动）：
  ① 近 30 天远期 EPS 上修 ≥ 3%
  ② 之前 60 天（90 → 30 天前）上修 ≤ 2%            —— 上修刚开始，不是已经修了很久
  ③ 近 30 天上调预测的分析师 ≥ 3 人，且 ≥ 下调人数 × 2（本财年 + 下一财年合计）
  ④ 近 30 天股价涨幅 < 近 30 天 EPS 上修幅度        —— 股价还没跟上
  （90 天股价涨幅只列出、不过滤：用户 2026-10-05 要求先不加）

股价：yfinance 日线（_members_px.csv 缓存，不进 git）。快照日 D 是北京时间，那时美股 D 日还没开盘：
  条件 ④ 用 D 日之前最后一个收盘；入选后的表现从 D 日（含）起第一个收盘算起，对照标普500（^GSPC）。
记录：early_log.csv（快照日, 股票, 入选时各项指标），只在 GitHub Actions 上写（CI=true 或 REV_RECORD=1），随快照一起提交；
  同一快照重算结果相同（指标都按快照日之前的数据），所以重跑只会覆盖同样的行。
用法：python3 early.py               # 打印最新名单和跟踪摘要（本地不写记录）
      python3 early.py --record-all  # 仓库里每份快照都筛一遍写进 early_log.csv（补记）
"""
import csv, glob, os, sys, time
from datetime import date

import numpy as np
import pandas as pd

import breadth
from analyze import drop_partial

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "early_log.csv")
PX = os.path.join(BASE, "_members_px.csv")
RULE = {"r30": 0.03, "pre": 0.02, "up": 3, "ratio": 2}
COLS = ["date", "ticker", "rank", "sector", "r30", "r_pre", "up", "dn", "up7", "p30", "p90", "n_analysts"]


def prices(tickers, start):
    """日线收盘（复权）。缓存 3 小时；REV_FORCE_FETCH（Actions）时每次重下，失败退回缓存。返回 (DataFrame, 是否新下载)。"""
    tickers = sorted(set(tickers) | {"^GSPC"})
    fresh = (os.path.exists(PX) and time.time() - os.path.getmtime(PX) < 3 * 3600 and not os.environ.get("REV_FORCE_FETCH"))
    if fresh:
        px = pd.read_csv(PX, index_col=0, parse_dates=True)
        if set(tickers) <= set(px.columns) and px.index.min() <= pd.Timestamp(start):
            return drop_partial(px), False
    try:
        if not os.environ.get("CI"):
            os.environ.setdefault("https_proxy", "http://127.0.0.1:15236")
        import yfinance as yf
        px = yf.download(tickers, start=str(start), auto_adjust=True, progress=False, threads=True)["Close"]
        px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
        if px.notna().sum().gt(20).sum() < 0.8 * len(tickers):   # 被限流时常常大半是空列
            raise RuntimeError(f"只拿到 {px.notna().sum().gt(20).sum()}/{len(tickers)} 只")
        px = drop_partial(px)   # 美股收盘前 Yahoo 会把当天盘中价当一根日线
        px.to_csv(PX)
        return px, True
    except Exception as e:
        print(f"early: 股价下载失败，用缓存：{e}", file=sys.stderr)
        return (drop_partial(pd.read_csv(PX, index_col=0, parse_dates=True)), False) if os.path.exists(PX) else (None, False)


def _close_before(p, d):
    """d 之前（不含）最后一个收盘。"""
    s = p[p.index < d].dropna()
    return s.iloc[-1] if len(s) else np.nan


def screen(day, g, raw, px):
    raw = raw.assign(co=raw.ticker.replace(breadth.CO))
    an = raw.groupby("co")[["0y_up30", "+1y_up30", "0y_dn30", "+1y_dn30", "0y_up7", "+1y_up7", "0y_n"]].max()
    g = g.set_index("co").join(an)
    g["r30"] = g.E_current / g.E_30daysAgo - 1
    g["r_pre"] = g.E_30daysAgo / g.E_90daysAgo - 1
    g["up"] = g["0y_up30"].fillna(0) + g["+1y_up30"].fillna(0)
    g["dn"] = g["0y_dn30"].fillna(0) + g["+1y_dn30"].fillna(0)
    g["up7"] = g["0y_up7"].fillna(0) + g["+1y_up7"].fillna(0)
    d = pd.Timestamp(day)
    def chg(co, k):
        if px is None or co not in px:
            return np.nan
        p = px[co]
        return _close_before(p, d) / _close_before(p, d - pd.Timedelta(days=k)) - 1
    g["p30"] = [chg(c, 30) for c in g.index]
    g["p90"] = [chg(c, 90) for c in g.index]
    m = ((g.E_90daysAgo > 0) & (g.E_30daysAgo > 0) & (g.r30 >= RULE["r30"]) & (g.r_pre <= RULE["pre"])
         & (g.up >= RULE["up"]) & (g.up >= RULE["ratio"] * g.dn) & (g.p30 < g.r30))
    s = g[m].copy()
    s["gap"] = s.r30 - s.p30
    s = s.sort_values("gap", ascending=False)
    s["n_analysts"] = s["0y_n"]
    return s.reset_index().rename(columns={"co": "ticker"})


def record(day, s):
    old = list(csv.DictReader(open(LOG))) if os.path.exists(LOG) else []
    old = [r for r in old if r["date"] != str(day)]
    r4 = lambda v: "" if pd.isna(v) else round(float(v), 4)
    new = [{"date": str(day), "ticker": r.ticker, "rank": int(r["rank"]), "sector": r.sector,
            "r30": r4(r.r30), "r_pre": r4(r.r_pre), "up": int(r.up), "dn": int(r.dn), "up7": int(r.up7),
            "p30": r4(r.p30), "p90": r4(r.p90), "n_analysts": "" if pd.isna(r.n_analysts) else int(r.n_analysts)}
           for _, r in s.iterrows()]
    rows = sorted(old + new, key=lambda r: (r["date"], r["ticker"]))
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"early: 记下 {day} 名单 {len(new)} 家 → early_log.csv")


def track(log, px):
    """每只股票按第一次入选算：从入选日（含）起第一个收盘建仓，到最新收盘；对照标普500 同期。"""
    if log.empty or px is None:
        return [], {}
    out = []
    spx = px["^GSPC"].dropna()
    for t, h in log.groupby("ticker"):
        d0 = pd.Timestamp(h.date.min())
        row = {"ticker": t, "first": str(d0.date()), "last": h.date.max(), "times": int(len(h)),
               "rank": int(h["rank"].iloc[0]), "sector": h.sector.iloc[0] if isinstance(h.sector.iloc[0], str) else ""}
        p = px[t].dropna() if t in px else pd.Series(dtype=float)
        after = p[p.index >= d0]
        sa = spx[spx.index >= d0]
        if len(after) and len(sa):
            e0, e1, b0 = after.iloc[0], p.iloc[-1], sa.loc[after.index[0]] if after.index[0] in sa.index else sa.iloc[0]
            ret, bret = e1 / e0 - 1, spx.iloc[-1] / b0 - 1
            row.update({"entry": str(after.index[0].date()), "days": int(len(after) - 1),
                        "ret": round(100 * ret, 2), "spx": round(100 * bret, 2), "ex": round(100 * (ret - bret), 2)})
        out.append(row)
    out.sort(key=lambda r: (r["first"], r["ticker"]), reverse=True)
    done = [r for r in out if r.get("days")]
    summ = {"n": len(out), "n_live": len(done)}
    if done:
        ex = np.array([r["ex"] for r in done])
        summ.update({"ex_mean": round(float(ex.mean()), 2), "ex_med": round(float(np.median(ex)), 2),
                     "win": int((ex > 0).sum()), "days_med": int(np.median([r["days"] for r in done]))})
    return out, summ


def page_data():
    """make_chart.py 调：最新快照的名单（CI 上顺带记录）+ 历史入选的跟踪。没有快照返回 None。"""
    paths = sorted(glob.glob(os.path.join(BASE, "yahoo", "20*.csv")))
    if not paths:
        return None
    ends = {}
    for p in paths[:-1]:          # 只为纠正倒退的财年标签（见 breadth.companies）
        breadth.companies(p, ends)
    day, g = breadth.companies(paths[-1], ends)
    raw = pd.read_csv(paths[-1])
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    start = min(pd.Timestamp(day) - pd.Timedelta(days=130),
                pd.Timestamp(log.date.min()) - pd.Timedelta(days=5) if len(log) else pd.Timestamp(day))
    px, _ = prices(list(g.co) + list(log.ticker.unique()), start.date())
    s = screen(day, g, raw, px)
    if px is not None and (os.environ.get("CI") == "true" or os.environ.get("REV_RECORD") == "1"):
        record(day, s)
        log = pd.read_csv(LOG)
    first = log.groupby("ticker").date.min().to_dict() if len(log) else {}
    pc = lambda v: None if pd.isna(v) else round(100 * float(v), 2)
    lst = [{"ticker": r.ticker, "rank": int(r["rank"]), "sector": r.sector, "r30": pc(r.r30), "r_pre": pc(r.r_pre),
            "up": int(r.up), "dn": int(r.dn), "up7": int(r.up7), "p30": pc(r.p30), "p90": pc(r.p90),
            "first": first.get(r.ticker, str(day))} for _, r in s.iterrows()]
    tr, summ = track(log, px)
    return {"date": str(day), "price_ok": px is not None, "n_screened": int(len(g)), "list": lst, "track": tr,
            "summary": summ, "log_start": str(log.date.min()) if len(log) else None,
            "rule": {"r30": 100 * RULE["r30"], "pre": 100 * RULE["pre"], "up": RULE["up"], "ratio": RULE["ratio"]}}


def record_all():
    """把仓库里每一份快照都筛一遍、记下来（补记漏掉的日子；结果和当天算的一样）。"""
    paths = sorted(glob.glob(os.path.join(BASE, "yahoo", "20*.csv")))
    ends, snaps = {}, []
    for p in paths:
        snaps.append((*breadth.companies(p, ends), pd.read_csv(p)))
    start = pd.Timestamp(snaps[0][0]) - pd.Timedelta(days=130)
    px, _ = prices(set().union(*[set(g.co) for _, g, _ in snaps]), start.date())
    if px is None:
        sys.exit("early: 没有股价，不能记录")
    for day, g, raw in snaps:
        record(day, screen(day, g, raw, px))


if __name__ == "__main__":
    import json
    if "--record-all" in sys.argv:
        record_all()
    d = page_data()
    print(json.dumps({k: d[k] for k in ("date", "price_ok", "n_screened", "summary", "log_start")}, ensure_ascii=False))
    for r in d["list"]:
        print(r)
