#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「盈利修正速度」能不能当最优拟合的因子（2026-10-02）。只做事先定好的候选，不扫描、不挑选。

因子：实时口径的超额修正 x13（realtime.py，周五周报 → 下一个交易日起可用，逐日沿用）。
  上修越快读数越高（基本面利好被充分定价 → 偏热），下修越猛读数越低（利空在出清 → 偏冷）。
  两种刻度：REVp = 252 日滚动分位（和红点其他因子同口径）；REVa = 固定锚点 −6% / 0 / +6% → 0 / 50 / 100。

红点：现行 ALT_W 五因子 + 修正因子，占比 10% / 20% / 33%（其余等比缩小）。门槛按训练窗（2017-10-18 起）
      对齐到 67 天（ALT_DESIGN_RATE），连 3 日。看红点日之后 30 个交易日 QQQ 相对全部交易日的超额、事件数、抓顶。
      留出段只能用 2014-07 ~ 2017-10（修正因子 2013-07 才有，滚动分位再要一年）：长面板因子缓存
      _long_factors.csv（「市值跑赢等权」仍是改 RSP 之前的口径）+ 同频门槛。
蓝点：BLUE_W 四因子 + 修正因子（读数低 = 冷），占比 20% / 33%；门槛重标到与现行同样的触发天数，VIX ≥ 30 闸门不动。

生产因子用 make_alt_page.patch 同一条路径从 raw/ 重算（缓存 _prod_panel.csv，删掉即重算）。
用法：python3 factor_test.py
"""
import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
import engine as E
import alt_engine as A
from compare_weights import zigzag, BEAR

PANEL = os.path.join(BASE, "_prod_panel.csv")
TR0 = pd.Timestamp("2017-10-18")
HO = (pd.Timestamp("2014-07-01"), pd.Timestamp("2017-10-17"))


def production_panel():
    if os.path.exists(PANEL):
        return pd.read_csv(PANEL, index_col=0, parse_dates=True)
    rawdf, meta, spy, _ = E.build_indicators()
    dirs = E.direction(spy, rawdf.index)
    _, adj, _, _ = E.compose(rawdf, dirs)
    lev_pct, _ = E.leverage_monitor(rawdf)
    lev_pct, _, lev_raw = A.lev_single(rawdf.index, lev_pct)
    parts = A.alt_inputs(rawdf, adj, lev_pct, spy, rawdf.index)
    rf = rawdf.copy()
    for k, v in {"turnover": "_turnover_daily", "advancing": "_advancing_daily", "leverage": "_leverage_daily"}.items():
        rf[k] = rawdf[v]
    _, adjf, _, _ = E.compose(rf, dirs, fast=True)
    if lev_raw is not None:
        adjf = A.blue_lev_splice(adjf, dirs, lev_raw["ratio_daily"], fast=True)
    idx = rawdf.index
    df = pd.DataFrame({**{k: parts[k] for k in A.ALT_W}, **{"b_" + k: adjf[k] for k in A.BLUE_W}}, index=idx)
    df["vix"] = E.load_vix(idx)
    df["qqq"] = E.load("QQQ")["close"].reindex(idx)
    df.to_csv(PANEL)
    return df


def revision_factor(idx):
    rt = pd.read_csv(os.path.join(BASE, "_weekly_rt.csv"), index_col=0, parse_dates=True)
    x = rt.x13.dropna()
    x.index = x.index + pd.Timedelta(days=1)          # 周五周报 → 下一个交易日起可用
    xd = x.reindex(x.index.union(idx)).ffill().reindex(idx)
    return {"REVp": E.rolling_pct(xd), "REVa": E.abs_map(100 * xd, (-6.0, 0.0, 6.0))}, xd


def red_temp(F, rev=None, share=0.0):
    w = dict(A.ALT_W)
    T = sum(F[k] * v for k, v in w.items()) / sum(w.values())
    if rev is not None and share > 0:
        T = (1 - share) * T + share * rev
    return T


def evaluate_red(T, px, idx_win, n_days=67, th=None):
    t3 = T.rolling(3).min()
    s = t3[idx_win].dropna().sort_values(ascending=False)
    if th is None:
        th = (s.iloc[n_days - 1] + s.iloc[n_days]) / 2
    hot = (t3 > th) & idx_win
    f30 = px.shift(-30) / px - 1
    f, b = f30[hot].dropna(), f30[idx_win].dropna()
    ev = A.events(hot)
    tops = [z for z in zigzag(px[idx_win].dropna(), 0.11) if z[0] not in BEAR]
    ix = px.index
    got = [pk for pk, _, _ in tops if hot.iloc[max(0, ix.searchsorted(pk) - 40): ix.searchsorted(pk) + 6].any()]
    return {"th": th, "edge": 100 * (f.mean() - b.mean()), "days": int(hot.sum()), "events": len(ev),
            "tops": f"{len(got)}/{len(tops)}", "got": [f"{t:%Y-%m}" for t in got],
            "ev_f30": [round(100 * f30[a], 1) if pd.notna(f30[a]) else None for a, _ in ev], "firsts": [a for a, _ in ev], "hot": hot}


def main():
    F = production_panel()
    idx = F.index
    REV, xd = revision_factor(idx)
    px = F.qqq
    win = pd.Series(idx >= TR0, index=idx)
    SA = win & (idx <= "2021-12-31")
    SB = pd.Series(idx >= "2023-01-03", index=idx)

    # 基线核对：重算的红点要和线上一致
    T0 = red_temp(F)
    base = evaluate_red(T0, px, win, th=A.ALT_TH)
    print(f"基线核对：门槛 {A.ALT_TH} 训练窗红点 {base['days']} 天（设计 67）")

    f30 = px.shift(-30) / px - 1
    # 因子本身：和之后 30 日 QQQ 的秩相关
    for nm, r in REV.items():
        ok = win & r.notna() & f30.notna()
        print(f"{nm} 与之后 30 日 QQQ 的秩相关（训练窗，逐日）：{r[ok].rank().corr(f30[ok].rank()):+.3f}；"
              f"与现行红点温度的相关 {r[ok].corr(T0[ok]):+.2f}")

    # ---------- 红点 ----------
    print("\n红点（门槛对齐到训练窗 67 天；超额 = 红点日之后 30 日 QQQ − 全部交易日；负 = 红点之后跌得更多）")
    L = pd.read_csv(os.path.join(ROOT, "_long_factors.csv"), index_col=0, parse_dates=True)
    ndx = pd.read_csv(os.path.join(ROOT, "_tdc_history_long.csv"), parse_dates=["Date"], index_col="Date")["ndx"]
    REVL, _ = revision_factor(L.index)
    pxl = ndx.reindex(L.index).ffill()
    how = pd.Series((L.index >= HO[0]) & (L.index <= HO[1]), index=L.index)
    rows = []
    cands = [("现行（五因子）", None, 0.0)] + [(f"+{nm} {int(round(sh * 100))}%", nm, sh)
                                          for nm in ("REVp", "REVa") for sh in (0.10, 0.20, 0.33)]
    for nm, rk, sh in cands:
        T = red_temp(F, REV[rk] if rk else None, sh).where(T0.notna())
        r = evaluate_red(T, px, win)
        a = evaluate_red(T, px, SA, th=r["th"]); b = evaluate_red(T, px, SB, th=r["th"])
        # 留出段：长面板缓存 + 同频门槛
        TL = red_temp(L, REVL[rk] if rk else None, sh)
        t3 = TL.rolling(3).min()
        thl = t3[how].dropna().quantile(1 - r["days"] / win.sum())
        h = evaluate_red(TL, pxl, how, th=thl)
        rows.append((nm, r, a, b, h))
        print(f"  {nm:12s} 门槛 {r['th']:5.1f} | 训练窗 {r['edge']:+6.2f}pp {r['days']:3d}天 {r['events']:2d}次 抓顶 {r['tops']:>4s}"
              f" | 2018-21 {a['edge']:+6.2f}pp  2023起 {b['edge']:+6.2f}pp | 留出 14-07~17-10 {h['edge']:+6.2f}pp {h['days']:3d}天 抓顶 {h['tops']}")
    print("\n  各候选抓到的顶（训练窗）：")
    for nm, r, *_ in rows:
        print(f"    {nm:12s} {' '.join(r['got'])}")
    print("\n  各候选的红点事件（首日 → 之后 30 日 QQQ）：")
    for nm, r, *_ in rows:
        print(f"    {nm:12s} " + "  ".join(f"{d:%Y-%m-%d} {v:+.1f}%" if v is not None else f"{d:%Y-%m-%d} —"
                                           for d, v in zip(r["firsts"], r["ev_f30"])))

    # ---------- 蓝点 ----------
    print("\n蓝点（VIX ≥ 30 闸门不动；门槛重标到与现行同样的触发天数；看蓝点日之后 30 日 QQQ）")
    B0 = sum(F["b_" + k] * v for k, v in A.BLUE_W.items()) / sum(A.BLUE_W.values())
    gate = (F.vix >= E.VIX_COLD) & win
    cold0 = (B0 < A.BLUE_TH) & gate
    n_b = int(cold0.sum())
    allg = f30[gate].dropna()
    print(f"  训练窗 VIX ≥ 30 共 {int(gate.sum())} 天，之后 30 日平均 {100 * allg.mean():+.2f}%；现行蓝点 {n_b} 天")
    for nm, rk, sh in [("现行（四因子）", None, 0.0)] + [(f"+{k} {int(round(s * 100))}%", k, s) for k in ("REVp", "REVa") for s in (0.20, 0.33)]:
        B = B0 if rk is None else (1 - sh) * B0 + sh * REV[rk]
        if rk is None:
            cold = cold0
            th = A.BLUE_TH
        else:
            v = B[gate].dropna().sort_values()
            th = (v.iloc[n_b - 1] + v.iloc[n_b]) / 2
            cold = (B < th) & gate
        f = f30[cold].dropna()
        ev = A.events(cold)
        print(f"  {nm:12s} 门槛 {th:5.1f} | {int(cold.sum()):2d}天 {len(ev)}次 | 之后 30 日 {100 * f.mean():+6.2f}%，"
              f"为负 {100 * (f < 0).mean():3.0f}% | 事件首日 " + " ".join(f"{a:%Y-%m-%d}" for a, _ in ev))


if __name__ == "__main__":
    main()
