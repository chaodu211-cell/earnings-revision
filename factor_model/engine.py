# -*- coding: utf-8 -*-
"""
选股因子：价格因子 + 财报事件因子。全部按当时能看到的数据算（point-in-time）：
  · 日期 t 的价格因子只用到 t 收盘为止的价格；
  · 一次财报的事件因子从「公告日后一个交易日」收盘起才可用（窗口 d−1 收盘 → d+1 收盘，
    盘前、盘后公布都包含在内，不需要知道确切时间）。

价格因子
  mom6     近 126 个交易日涨幅（模型的排序因子）
  mom12_1  经典 12-1 动量（t−252 → t−21），回测对照用
  hi52     收盘 ÷ 近 252 日最高收盘（上市不满一年用上市以来），只展示
财报事件因子（最近一次财报；离公告超过 130 天没有新财报就作废）
  ear      公告窗口超额收益：股票 − SPY，d−1 → d+1
  sue_a    分析师意外：(实际 EPS − 一致预期) ÷ 公告前股价（按价格标准化，EPS 接近 0 也不失真）
  sue_t    时间序列意外：EPS 同比变化 ÷ 之前 8 次同比变化的标准差（Bernard-Thomas SUE）
  evol     公告两天的最大成交量 ÷ 公告前均量，只展示
"""
import warnings

import numpy as np
import pandas as pd

F32 = np.float32
BENCH = ("SPY", "QQQ", "IWM", "RSP")


def clean_prices(C, V):
    C = C[C["SPY"].notna()]
    return C.ffill(limit=3), V.reindex(C.index)   # 偶发缺口只补 3 天，停牌 / 退市不补


def universe_mask(C, V, min_px=5.0, min_adv=20e6, min_hist=63):
    """当时可投：股价 ≥ 5、近 60 日成交额中位数 ≥ 2000 万美元、上市满 63 个交易日"""
    adv = (C * V).astype(F32).rolling(60, min_periods=40).median()
    return (C >= min_px) & (adv >= min_adv) & (C.notna().cumsum() >= min_hist)


def price_factors(C, idx):
    s = lambda df: df.reindex(idx).astype(F32)
    return {"mom6": s(C / C.shift(126) - 1),
            "mom12_1": s(C.shift(21) / C.shift(252) - 1),
            "hi52": s(C / C.rolling(252, min_periods=63).max())}


def clean_events(E):
    E = E.copy()
    E["dt"] = pd.to_datetime(E["dt"])
    E = E[E.act.notna()].sort_values(["ticker", "dt"])
    gap = E.groupby("ticker").dt.diff().dt.days
    E = E[~(gap.shift(-1).fillna(999) < 25)]        # 同一季度的重复记录（相隔 < 25 天）只留后一条
    return E.reset_index(drop=True)


def event_table(E, C, V):
    """每次财报一行：ear、sue_a、sue_t、evol、act，avail = 可用日期（公告日后一个交易日）"""
    idx = C.index
    spy = C["SPY"].values
    cols = {t: i for i, t in enumerate(C.columns)}
    Cv, Vv = C.values, V.values
    rows = []
    for t, g in E.groupby("ticker"):
        if t not in cols:
            continue
        j, hist = cols[t], []
        for dt, est, act in zip(g.dt, g.est, g.act):
            p = idx.searchsorted(dt.normalize())        # 公告日当天或之后第一个交易日
            prev = [a0 for d0, a0 in hist if 300 <= (dt - d0).days <= 430]   # 去年同季
            hist.append((dt, act))
            if p < 61 or p + 1 >= len(idx):
                continue
            c0, c1 = Cv[p - 1, j], Cv[p + 1, j]
            if not (c0 > 0 and c1 > 0):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # 公告前成交量全空
                vb = np.nanmean(Vv[p - 60:p - 5, j])
            rows.append(dict(ticker=t, dt=dt, avail=idx[p + 1], act=act, est=est,
                             ear=c1 / c0 - spy[p + 1] / spy[p - 1],
                             sue_a=(act - est) / c0 if pd.notna(est) else np.nan,
                             evol=np.nanmax(Vv[p:p + 2, j]) / vb if vb and vb > 0 else np.nan,
                             diff4=act - prev[-1] if prev else np.nan))
    T = pd.DataFrame(rows)
    sd = T.groupby("ticker").diff4.transform(lambda d: d.shift(1).rolling(8, min_periods=4).std())
    T["sue_t"] = T.diff4 / sd.replace(0, np.nan)
    return T


def event_panels(T, idx, cols, fields=("ear", "sue_a", "sue_t", "evol", "act"), max_age_days=130):
    """把每次财报的值铺成面板：每个日期取当时最近一次财报的值，离公告超过 max_age_days 天作废"""
    T = T.sort_values("avail")
    full = idx.union(pd.DatetimeIndex(T.avail.unique()))
    when = T.pivot_table(index="avail", columns="ticker", values="dt", aggfunc="last").reindex(full).ffill()
    age = when.rsub(pd.Series(full, index=full), axis=0).apply(lambda c: c.dt.days)
    fresh = (age <= max_age_days).reindex(idx).reindex(columns=cols).fillna(False)
    out = {}
    for f in fields:
        p = T.pivot_table(index="avail", columns="ticker", values=f, aggfunc="last")
        out[f] = p.reindex(full).ffill().reindex(idx).reindex(columns=cols).where(fresh).astype(F32)
    return out


def pct(df, mask):
    """当天可投股票里的百分位（0~1，越大越好），不可投的为空"""
    return df.where(mask).rank(axis=1, pct=True).astype(F32)


def nmean(*dfs, min_n=1):
    """逐格平均，忽略空值；可用的个数少于 min_n 的格子为空"""
    st = np.stack([d.values for d in dfs]).astype(np.float64)
    n = np.isfinite(st).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全空的格子
        m = np.nanmean(st, axis=0)
    m[n < min_n] = np.nan
    return pd.DataFrame(m, index=dfs[0].index, columns=dfs[0].columns).astype(F32)
