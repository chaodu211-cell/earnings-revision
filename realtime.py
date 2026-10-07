#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时口径：模拟每周五拿到周报那一刻能算出来的纯修正（x13）和二阶导 → _weekly_rt.csv

和 analyze.py（清洗口径，研究用）的区别：
  清洗口径  笔误按「前后各 2 期中位数」剔除、缺周线性插值——都用到了之后的数据，历史标记比当时看到的干净
  实时口径  每周只用当周及以前的周报；缺周沿用上一周；加「待核实」防护（只看过去 + 下一周确认）：
            远期 EPS 和上一个已确认值相比单周变动 > 2% → 先标「待核实」，本周仍用上一周的值；
            下一个有数据的周离原值更近 → 判笔误作废；离可疑值更近 → 判真实变动，补认（晚一周）。
            （不用固定回归带：2021-06-17 笔误之后隔一周的值比原值高 1.4%，强势上修期正常增长，按「回到 1% 以内」会误判成真实）
            （不加「指数没怎么动」这个条件：2022-06-17 那期可疑值出在指数单周跌 8.7% 的一周，加了就漏掉）
            2026-10-07 起：有当周 Yahoo 成分股快照时先拿它当场判（yahoo_verdict）——FactSet 单周变动和 Yahoo 同期 7 天纯修正
            + 一周时间滚动相差 ≤ 1 个百分点 → 真实、当周采用；相差 ≥ 2.5 个百分点且 Yahoo 自己动不到 1% → 笔误、当周作废；
            其余照旧等下一期。Yahoo 快照 2026-10-02 起才有，之前的待核实周不受影响。

每周两套值：
  x13 / a{n}             当周实际能算出的；待核实的周和没有数据的周不更新（空值，阶段维持上周）——画标记、算阶段用这个
  eps_final / x13_final  核实后的值（笔误作废、缺周事后插值）——只用来算之后各周的「N 周前」和滚动均值
用法：python3 realtime.py
"""
import math, os
import numpy as np
import pandas as pd

from analyze import next_growth, time_roll

BASE = os.path.dirname(os.path.abspath(__file__))
JUMP, CARRY = 0.02, 3   # 待核实阈值、缺周最多插值几周
YH_REAL, YH_TYPO = 0.01, 0.025   # 和 Yahoo 同期变动差多少算真实 / 算笔误（见 yahoo_verdict）
NS = (4, 8, 13)


def weekly_raw():
    d = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"]).set_index("date")
    e = d.fwd_eps.dropna().copy()
    e.index = e.index + pd.to_timedelta((4 - e.index.weekday) % 7, unit="D")  # 周四出的算到当周五
    e = e[~e.index.duplicated(keep="last")]
    grid = pd.date_range(e.index.min(), e.index.max(), freq="W-FRI")
    return e.reindex(grid)


def yahoo_verdict(jump, y7, roll):
    """FactSet 单周变动 jump（对数）和 Yahoo 同期 7 天纯修正 y7 + 一周时间滚动 roll 比：
    差 ≤ 1 个百分点 → 真实；差 ≥ 2.5 个百分点且 Yahoo 自己动得不到 1% → 笔误；其余说不清，照旧等下一期。"""
    gap = abs(jump - y7 - roll)
    if gap <= YH_REAL:
        return "real"
    if gap >= YH_TYPO and abs(y7) < 0.01:
        return "typo"
    return None


def guard(raw, yahoo=None, roll=None):
    """逐周模拟。返回 当周有没有新读数（ok）、核实后的值（final，缺周 / 笔误事后插值）、提示文字（flag）。
    yahoo：{快照日: 全指数 7 天纯修正}（breadth.index_7d，2026-10-02 起）；roll：每周的时间滚动（对数）。
    单周变动 > 2% 时，如果有当周（周五，没有就周四）的 Yahoo 快照、且上一个确认值正好是上周的，先拿 Yahoo 当场判；判不了才等下一期。"""
    yahoo, roll = yahoo or {}, roll if roll is not None else pd.Series(dtype=float)
    final, ok, flag = {}, {}, {}
    acc, acc_t, pend = None, None, None     # 最近确认值及其周、待核实（周, 值）
    for t, e in raw.items():
        ok[t], final[t] = False, np.nan
        if math.isnan(e):                   # 没有周报 / 周报没写市盈率：本周不更新
            continue
        if pend is not None:                # 先核实上一个待核实
            pt, pv = pend
            if abs(math.log(e / acc)) < abs(math.log(e / pv)):
                flag[pt] += f"；{t:%m-%d} 离原值更近 → 判笔误作废"
            else:
                final[pt], acc, acc_t = pv, pv, pt
                flag[pt] += f"；{t:%m-%d} 离可疑值更近 → 判真实变动，补认"
            pend = None
        if acc is None or abs(math.log(e / acc)) <= JUMP:
            final[t], acc, acc_t, ok[t] = e, e, t, True
            continue
        jump = math.log(e / acc)
        msg = f"待核实：远期 EPS 单周 {100 * jump:+.1f}%（{acc:.2f} → {e:.2f}）"
        y7 = next((yahoo[d] for d in (t, t - pd.Timedelta(days=1)) if d in yahoo), None)
        rw = roll.get(t, np.nan)
        v = yahoo_verdict(jump, y7, rw) if y7 is not None and acc_t == t - pd.Timedelta(days=7) and not math.isnan(rw) else None
        if v == "real":
            final[t], acc, acc_t, ok[t] = e, e, t, True
            flag[t] = msg + f"；Yahoo 同期 {100 * y7:+.1f}%（加时间滚动 {100 * rw:+.1f}%）→ 判真实变动，当周采用"
        elif v == "typo":
            flag[t] = msg + f"；Yahoo 同期只 {100 * y7:+.1f}% → 判笔误作废"
        else:
            pend = (t, e)                   # 本周不更新，等下一个有数据的周核实
            flag[t] = msg + (f"（Yahoo 同期 {100 * y7:+.1f}%，判不了）" if y7 is not None else "")
    out = pd.DataFrame({"eps_raw": raw, "ok": pd.Series(ok)})
    # 缺周、作废的笔误：事后按前后已确认值插值（最多 CARRY 周）。只在「N 周前」「滚动均值」里用到，
    # 那时缺口早已结束（N ≥ 4 > CARRY），所以仍然只用到当时已有的数据
    out["eps_final"] = pd.Series(final).interpolate(limit=CARRY, limit_area="inside")
    out["flag"] = pd.Series(flag)
    return out


def signals(w):
    lf = np.log(w.eps_final)
    r13f = lf.diff(13)
    g, _ = next_growth(w.index)
    roll13 = time_roll(w.index, g).rolling(13).sum()   # 预测不动时 13 周自然涨的部分（和清洗口径同一算法）
    w["x13_final"] = r13f - roll13
    # 当周有新读数才更新；待核实 / 缺周记空值（页面上那周没有柱子，阶段维持上周）
    w["r13"] = (np.log(w.eps_raw) - lf.shift(13)).where(w.ok)
    w["x13"] = w.r13 - roll13                                   # 纯修正
    for n in NS:
        w[f"a{n}"] = w.x13 - w.x13_final.shift(n)                # 当周值 − N 周前（已核实）
    return w


def main():
    import breadth
    raw = weekly_raw()
    g, _ = next_growth(raw.index)
    w = signals(guard(raw, yahoo=breadth.index_7d(), roll=time_roll(raw.index, g)))
    w.to_csv(os.path.join(BASE, "_weekly_rt.csv"))
    f = w[w.flag.notna()]
    print(f"待核实 {len(f)} 周：")
    for t, r in f.iterrows():
        print(f"  {t.date()}  {r.flag}")
    clean = pd.read_csv(os.path.join(BASE, "_weekly_signals.csv"), index_col=0, parse_dates=True).x13
    d = (w.x13 - clean).abs().dropna()
    print(f"与清洗口径 x13 的差：平均 {100 * d.mean():.3f}pp，最大 {100 * d.max():.2f}pp（{d.idxmax().date()}）")


if __name__ == "__main__":
    main()
