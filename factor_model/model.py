#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股因子模型 v1（2026-10-14）—— 动量排序 + 财报门槛

  可投   股价 ≥ 5、近 60 日成交额中位数 ≥ 2000 万美元、上市满 63 个交易日（engine.universe_mask）
  门槛   财报分 E ≥ 0.5：E = 最近一次财报的 ear、sue_a、sue_t 三个当天百分位的平均（缺的跳过，三个都缺不可选）
  排序   6 个月涨幅 mom6 在当天可投股票里的百分位
  组合   每月最后一个交易日调仓，等权 30 只：已持有的只要排名 ≤ 100 就留着（缓冲，少换手、让赢家跑），
         空出来的位置按排名从高往低补

为什么是「排序 + 门槛」而不是几个因子分数取平均：取平均选出来的是各项都不错的「全能股」，
  右尾（一年内翻倍的比例）比纯动量少一半；先过基本面门槛、再按动量排，两段时期都比纯动量好。
  对比、参数敏感性见 python3 factor_model/model.py backtest。

用法：python3 factor_model/model.py backtest     # 回测表（约 2 分钟）
"""
import os, sys, time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import data
import engine as en

N_HOLD, BUFFER, GATE = 30, 100, 0.5
COST = 0.002        # 每单位换手扣 0.2%（买卖各约 0.1%）


def load():
    C, V = en.clean_prices(pd.read_pickle(data.PXC), pd.read_pickle(data.PXV))
    E = en.clean_events(pd.read_csv(data.EARN))
    return C, V, E


def weekly_index(C, start="2011-01-01"):
    """每周最后一个交易日（最后一周到最新一天为止）"""
    wk = C.index.to_series().groupby(C.index.to_period("W-FRI")).max()
    idx = pd.DatetimeIndex(wk.values)
    return idx[idx >= start]


def compute(C, V, E, idx):
    """在日期 idx 上算出：mask（可投）、各因子、财报分 E、打分 score（门槛外为空）"""
    stocks = [c for c in C.columns if c not in en.BENCH]
    Cs, Vs = C[stocks], V[stocks]
    mask = en.universe_mask(Cs, Vs).reindex(idx).fillna(False)
    P = en.price_factors(Cs, idx)
    T = en.event_table(E, C, V)
    EP = en.event_panels(T, idx, stocks)
    pc = lambda df: en.pct(df, mask)
    Ev = en.nmean(pc(EP["ear"]), pc(EP["sue_a"]), pc(EP["sue_t"]), min_n=1)
    mom6 = pc(P["mom6"])
    return dict(mask=mask, P=P, EP=EP, E=Ev, mom6=mom6, score=mom6.where(Ev >= GATE),
                mom12_1=pc(P["mom12_1"]), events=T)


def rebalance(score_row, held, n=N_HOLD, buffer=BUFFER):
    """一次调仓：held 里排名 ≤ buffer 的留下，空位按打分从高到低补满 n 只"""
    s = score_row.dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    keep = [t for t in held if t in rank.index and rank[t] <= buffer]
    keep = sorted(keep, key=lambda t: rank[t])
    new = [t for t in s.index if t not in keep][:max(n - len(keep), 0)]
    return (keep + new)[:n]


# ---------------- 回测 ----------------

def month_ends(idx, start="2012-01-01"):
    me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values)
    return me[me >= start]


def run(score, C, me, n=N_HOLD, buffer=BUFFER, cost=COST):
    """月度调仓，持有到下一个调仓日（真实持有期收益），返回 (月收益, 换手, 各期持仓)"""
    px = C.reindex(me)
    hold = px.shift(-1) / px - 1
    s = score.reindex(me)
    rets, turns, book, held = [], [], {}, []
    for d in me[:-1]:
        row = s.loc[d]
        if row.notna().sum() < n:
            rets.append(np.nan); turns.append(np.nan); continue
        pick = rebalance(row, held, n, buffer)
        tv = 1 - len(set(held) & set(pick)) / n
        rets.append(hold.loc[d, pick].fillna(0.0).mean() - cost * tv)     # 下个调仓日没价格的（极少）按 0 计
        turns.append(tv); book[d] = pick; held = pick
    return pd.Series(rets, index=me[:-1]), pd.Series(turns, index=me[:-1]), book


def stats(r, bench):
    r = r.dropna()
    b = bench.reindex(r.index)
    ex = r - b
    yrs = len(r) / 12
    eq = (1 + r).cumprod()
    return dict(months=len(r), cagr=(1 + r).prod() ** (1 / yrs) - 1, spy=(1 + b).prod() ** (1 / yrs) - 1,
                ex_ann=ex.mean() * 12, ex_t=ex.mean() / ex.std() * np.sqrt(len(ex)),
                vol=r.std() * np.sqrt(12), maxdd=(eq / eq.cummax() - 1).min(), win=(ex > 0).mean())


def backtest():
    t0 = time.time()
    C, V, E = load()
    idx = weekly_index(C)
    M = compute(C, V, E, idx)
    me = month_ends(idx)
    spy = C["SPY"].reindex(me)
    bench = spy.shift(-1) / spy - 1
    px = C.reindex(me)
    uni = (px.shift(-1) / px - 1)[list(M["mask"].columns)].where(M["mask"].reindex(me)).mean(axis=1)
    models = {"模型：mom6 | E≥0.5，缓冲 100": (M["score"], BUFFER),
              "同上，不留缓冲（每月取前 30）": (M["score"], N_HOLD),
              "纯 6 个月动量": (M["mom6"], BUFFER),
              "经典 12-1 动量": (M["mom12_1"], BUFFER)}
    periods = {"2012–2018": ("2012", "2018"), "2019–2026": ("2019", "2026"), "全部": ("2012", "2026")}
    rows = []
    for name, (sc, buf) in models.items():
        r, t, _ = run(sc, C, me, buffer=buf)
        for pn, (a, b) in periods.items():
            rows.append(dict(model=name, period=pn, turnover=t.loc[a:b].mean(), **stats(r.loc[a:b], bench)))
    for pn, (a, b) in periods.items():
        rows.append(dict(model="股票池等权", period=pn, **stats(uni.loc[a:b], bench)))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"可投股票（平均每月）：{M['mask'].reindex(me).sum(axis=1).mean():.0f} 只；"
          f"过财报门槛：{(M['score'].reindex(me).notna().sum(axis=1)).mean():.0f} 只；{time.time() - t0:.0f}s")
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    R.to_csv(os.path.join(BASE, "backtest_summary.csv"), index=False)

    print("\n门槛敏感性（动量窗口 × E 门槛，缓冲 100；年化超额 / t）")
    for mname in ("mom6", "mom12_1"):
        for th in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
            sc = M[mname] if th == 0 else M[mname].where(M["E"] >= th)
            r, _, _ = run(sc, C, me)
            a, b = stats(r.loc["2012":"2018"], bench), stats(r.loc["2019":"2026"], bench)
            print(f"  {mname:8s} E≥{th:.1f}  2012–18 {a['ex_ann']:+.3f} (t {a['ex_t']:.2f})   2019–26 {b['ex_ann']:+.3f} (t {b['ex_t']:.2f})")
    return R


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest()
    else:
        print(__doc__)
