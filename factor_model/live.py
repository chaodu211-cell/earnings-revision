#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股模型的当天名单 + 分析师修正确认 + 组合调仓。

  1. 更新缓存（日线补最近几天、刚公布财报的公司重拉财报；--no-update 跳过）
  2. 按 model.py 的规则给全部可投股票打分，列出前 50
  3. 前 50 逐只拉 Yahoo earningsTrend，算分析师修正（没有进回测——历史拿不到，只作确认标签、前瞻记录）：
       远期 EPS = 本财年 × w + 下一财年 × (1 − w)，w = 本财年剩余天数 / 365，回看时 w 不变（纯修正，同 breadth.py）
       rev90  远期 EPS 90 天修正；dEP90 = 同一变化 ÷ 股价（亏损、微利公司看它）；广度 = (上调 − 下调) / (上调 + 下调)，近 30 天
       标签：上修（rev90 ≥ +3% 或 dEP90 ≥ +0.5%，且广度 ≥ 0）/ 下修（rev90 ≤ −3% 或 dEP90 ≤ −0.5% 或广度 ≤ −0.3）/ 持平 / 无覆盖（< 3 位分析师）
  4. 组合：portfolio.csv 里最近一次的持仓按缓冲规则（排名 ≤ 100 留下，空位按排名补满 30 只）算出调仓建议；
     只有加 --rebalance 才把新组合写进 portfolio.csv（回测是每月最后一个交易日调一次）

输出：picks/<日期>.csv（前 50 及各项指标）、picks_log.csv（每次运行追加前 50，前瞻检验用）、portfolio.csv（调仓记录）
用法：python3 factor_model/live.py [--no-update] [--rebalance] [--top 50]
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import data
import engine as en
import model

PICKS = os.path.join(BASE, "picks")
LOG = os.path.join(BASE, "picks_log.csv")
PORT = os.path.join(BASE, "portfolio.csv")
URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
raw = lambda x: x.get("raw") if isinstance(x, dict) else x


def fetch_trend(sym, yd):
    for k in range(3):
        try:
            r = yd.get_raw_json(URL.format(sym=sym), params={"modules": "earningsTrend,price"})["quoteSummary"]["result"][0]
            break
        except Exception:
            time.sleep(2 * (k + 1))
    else:
        return None
    out = {"ticker": sym, "price": raw(r["price"].get("regularMarketPrice"))}
    for t in r.get("earningsTrend", {}).get("trend", []):
        p = t.get("period")
        if p not in ("0y", "+1y"):
            continue
        out[f"{p}_end"] = t.get("endDate")
        for lag in ("current", "30daysAgo", "90daysAgo"):
            out[f"{p}_{lag}"] = raw(t.get("epsTrend", {}).get(lag))
        rv = t.get("epsRevisions", {})
        out[f"{p}_up30"], out[f"{p}_dn30"] = raw(rv.get("upLast30days")), raw(rv.get("downLast30days"))
        out[f"{p}_n"] = raw(t.get("earningsEstimate", {}).get("numberOfAnalysts"))
    return out


def revision(r, today):
    out = {"rev30": np.nan, "rev90": np.nan, "dEP90": np.nan, "breadth": np.nan, "analysts": r.get("0y_n") or 0, "fwd_pe": np.nan}
    end = r.get("0y_end")
    if not end or any(r.get(f"{p}_{l}") is None for p in ("0y", "+1y") for l in ("current", "30daysAgo", "90daysAgo")):
        return out
    w = min(max((date.fromisoformat(end) - today).days / 365, 0), 1)
    E = {l: w * r[f"0y_{l}"] + (1 - w) * r[f"+1y_{l}"] for l in ("current", "30daysAgo", "90daysAgo")}
    px = r.get("price") or np.nan
    up = sum(r.get(f"{p}_up30") or 0 for p in ("0y", "+1y"))
    dn = sum(r.get(f"{p}_dn30") or 0 for p in ("0y", "+1y"))
    out.update(rev30=E["current"] / E["30daysAgo"] - 1 if E["30daysAgo"] > 0 else np.nan,
               rev90=E["current"] / E["90daysAgo"] - 1 if E["90daysAgo"] > 0 else np.nan,
               dEP90=(E["current"] - E["90daysAgo"]) / px, breadth=(up - dn) / (up + dn) if up + dn else 0.0,
               fwd_pe=px / E["current"] if E["current"] > 0 else np.nan)
    return out


def tag(x):
    if not x["analysts"] or x["analysts"] < 3 or pd.isna(x["dEP90"]):
        return "无覆盖"
    up = (x["rev90"] >= 0.03) if pd.notna(x["rev90"]) else (x["dEP90"] >= 0.005)
    dn = (x["rev90"] <= -0.03) if pd.notna(x["rev90"]) else (x["dEP90"] <= -0.005)
    if dn or x["breadth"] <= -0.3:
        return "下修"
    if up and x["breadth"] >= 0:
        return "上修"
    return "持平"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--rebalance", action="store_true")
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()
    if not a.no_update:
        data.update_prices()
        data.update_earnings()
    C, V, E = model.load()
    day = C.index[-1]
    idx = model.weekly_index(C)
    idx = idx[-6:].union(pd.DatetimeIndex([day]))
    M = model.compute(C, V, E, idx)
    s = M["score"].loc[day].dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    top = list(s.index[:a.top])
    u = data.universe().set_index("yf")
    T = M["events"]
    last = T.sort_values("dt").groupby("ticker").last()
    df = pd.DataFrame({"rank": rank[top], "score": s[top], "mom6": M["P"]["mom6"].loc[day, top],
                       "E": M["E"].loc[day, top], "ear": M["EP"]["ear"].loc[day, top], "sue_a": M["EP"]["sue_a"].loc[day, top],
                       "sue_t": M["EP"]["sue_t"].loc[day, top], "hi52": M["P"]["hi52"].loc[day, top],
                       "last_report": last.dt.reindex(top).dt.strftime("%Y-%m-%d"), "eps_pos": last.act.reindex(top) > 0,
                       "name": u.name.reindex(top).str.replace(r" (Common Stock|Class A Common Stock|Ordinary Shares).*$", "", regex=True),
                       "industry": u.industry.reindex(top), "cap_bn": u.cap.reindex(top) / 1e9}, index=pd.Index(top, name="ticker"))
    # 一周前的排名（看是新进还是老面孔）
    prev_day = idx[idx < day - pd.Timedelta(days=5)].max() if (idx < day - pd.Timedelta(days=5)).any() else None
    if prev_day is not None:
        ps = M["score"].loc[prev_day].dropna().sort_values(ascending=False)
        df["rank_1w"] = pd.Series(np.arange(1, len(ps) + 1), index=ps.index).reindex(top)
    data._proxy()
    from yfinance.data import YfData
    yd = YfData()
    with ThreadPoolExecutor(6) as ex:
        got = [g for g in ex.map(lambda t: fetch_trend(t, yd), top) if g]
    today = date.today()
    R = pd.DataFrame({g["ticker"]: revision(g, today) for g in got}).T
    df = df.join(R)
    df["修正"] = [tag(df.loc[t]) if t in R.index else "无覆盖" for t in df.index]

    # 组合：缓冲规则
    port = pd.read_csv(PORT) if os.path.exists(PORT) else pd.DataFrame(columns=["date", "ticker"])
    held = list(port[port.date == port.date.max()].ticker) if len(port) else []
    new = model.rebalance(M["score"].loc[day], held)
    out_, in_ = [t for t in held if t not in new], [t for t in new if t not in held]
    df["持仓"] = ["是" if t in new else "" for t in df.index]

    os.makedirs(PICKS, exist_ok=True)
    df.round(4).to_csv(os.path.join(PICKS, f"{day.date()}.csv"))
    log = df.reset_index()[["ticker", "rank", "score", "mom6", "E", "rev90", "breadth", "修正"]].assign(date=str(day.date()))
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old):
        old = old[old.date != str(day.date())]
    pd.concat([old, log], ignore_index=True).to_csv(LOG, index=False)
    if a.rebalance:
        rows = pd.DataFrame({"date": str(day.date()), "ticker": new})
        pd.concat([port[port.date != str(day.date())], rows], ignore_index=True).to_csv(PORT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 28)
    show = df[["rank", "rank_1w", "mom6", "E", "ear", "hi52", "eps_pos", "修正", "rev90", "breadth", "analysts", "fwd_pe", "cap_bn", "name"]] \
        if "rank_1w" in df else df
    print(f"截至 {day.date()}：可投 {int(M['mask'].loc[day].sum())} 只，过财报门槛 {len(s)} 只")
    print(show.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\n修正标签：{df['修正'].value_counts().to_dict()}")
    if held:
        print(f"按缓冲规则调仓：卖出 {out_ or '无'}；买入 {in_ or '无'}" + ("（已写入 portfolio.csv）" if a.rebalance else "（没加 --rebalance，未写入）"))
    else:
        print(f"还没有组合记录；按规则首个组合是排名前 {len(new)} 的这些" + ("，已写入 portfolio.csv" if a.rebalance else "，加 --rebalance 才写入"))


if __name__ == "__main__":
    main()
