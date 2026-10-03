#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时口径：模拟每周五拿到周报那一刻能算出来的超额修正和二阶导 → _weekly_rt.csv

和 analyze.py（清洗口径，研究用）的区别：
  清洗口径  笔误按「前后各 2 期中位数」剔除、缺周线性插值——都用到了之后的数据，历史标记比当时看到的干净
  实时口径  每周只用当周及以前的周报；缺周沿用上一周；加「待核实」防护（只看过去 + 下一周确认）：
            远期 EPS 和上一个已确认值相比单周变动 > 2% → 先标「待核实」，本周仍用上一周的值；
            下一个有数据的周离原值更近 → 判笔误作废；离可疑值更近 → 判真实变动，补认（晚一周）。
            （不用固定回归带：2021-06-17 笔误之后隔一周的值比原值高 1.4%，强势上修期正常增长，按「回到 1% 以内」会误判成真实）
            （不加「指数没怎么动」这个条件：2022-06-17 那期可疑值出在指数单周跌 8.7% 的一周，加了就漏掉）

每周两套值：
  x13 / a{n}             当周实际能算出的；待核实的周和没有数据的周不更新（空值，阶段维持上周）——画标记、算阶段用这个
  eps_final / x13_final  核实后的值（笔误作废、缺周事后插值）——只用来算之后各周的「N 周前」和滚动均值
用法：python3 realtime.py
"""
import math, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JUMP, CARRY = 0.02, 3   # 待核实阈值、缺周最多插值几周
NS = (4, 8, 13)


def weekly_raw():
    d = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"]).set_index("date")
    e = d.fwd_eps.dropna().copy()
    e.index = e.index + pd.to_timedelta((4 - e.index.weekday) % 7, unit="D")  # 周四出的算到当周五
    e = e[~e.index.duplicated(keep="last")]
    grid = pd.date_range(e.index.min(), e.index.max(), freq="W-FRI")
    return e.reindex(grid)


def guard(raw):
    """逐周模拟。返回 当周有没有新读数（ok）、核实后的值（final，缺周 / 笔误事后插值）、提示文字（flag）。"""
    final, ok, flag = {}, {}, {}
    acc, pend = None, None                  # 最近确认值、待核实（周, 值）
    for t, e in raw.items():
        ok[t], final[t] = False, np.nan
        if math.isnan(e):                   # 没有周报 / 周报没写市盈率：本周不更新
            continue
        if pend is not None:                # 先核实上一个待核实
            pt, pv = pend
            if abs(math.log(e / acc)) < abs(math.log(e / pv)):
                flag[pt] += f"；{t:%m-%d} 离原值更近 → 判笔误作废"
            else:
                final[pt], acc = pv, pv
                flag[pt] += f"；{t:%m-%d} 离可疑值更近 → 判真实变动，补认"
            pend = None
        if acc is None or abs(math.log(e / acc)) <= JUMP:
            final[t], acc, ok[t] = e, e, True
        else:
            pend = (t, e)                   # 本周不更新，等下一个有数据的周核实
            flag[t] = f"待核实：远期 EPS 单周 {100 * math.log(e / acc):+.1f}%（{acc:.2f} → {e:.2f}）"
    out = pd.DataFrame({"eps_raw": raw, "ok": pd.Series(ok)})
    # 缺周、作废的笔误：事后按前后已确认值插值（最多 CARRY 周）。只在「N 周前」「滚动均值」里用到，
    # 那时缺口早已结束（N ≥ 4 > CARRY），所以仍然只用到当时已有的数据
    out["eps_final"] = pd.Series(final).interpolate(limit=CARRY, limit_area="inside")
    out["flag"] = pd.Series(flag)
    return out


def signals(w):
    lf = np.log(w.eps_final)
    r13f = lf.diff(13)
    mean = r13f.rolling(156, min_periods=104).mean().shift(1)
    w["x13_final"] = r13f - mean
    # 当周有新读数才更新；待核实 / 缺周记空值（页面上那周没有柱子，阶段维持上周）
    w["r13"] = (np.log(w.eps_raw) - lf.shift(13)).where(w.ok)
    w["x13"] = w.r13 - mean
    for n in NS:
        w[f"a{n}"] = w.x13 - w.x13_final.shift(n)                # 当周值 − N 周前（已核实）
    return w


def main():
    w = signals(guard(weekly_raw()))
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
