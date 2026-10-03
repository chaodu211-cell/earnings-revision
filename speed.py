#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正「速度拐点」× 纳指（2014 起，超额修正要 3 年历史）

四个阶段（每周一格）：
                二阶导 > 0（加快）     二阶导 < 0（放慢）
  超额修正 > 0   上修加速              上修减速   ← 「上修见顶」拐点在这里开始
  超额修正 < 0   下修放缓   ← 「下修见底」拐点        下修加速

二阶导 = 超额修正 − N 周前的超额修正（默认 N = 8）。周度噪声大（不去噪一年翻 9 次），用滞回去噪：
二阶导 > +h 才算转入加快，< −h 才算转入放慢，夹在中间维持原状态（默认 h = 1 个百分点）。

两种拐点：
  事后拐点：超额修正的局部高点 / 低点（前后各 13 周内最高 / 最低、且突出度 ≥ 2 个百分点）——当时不知道
  实时拐点：滞回状态翻转那一周——当时就知道，但比事后拐点晚

输出 _speed.json；python3 speed.py [--n 8] [--h 1.0]
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analyze import nw_t, drop_partial

BASE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["上修加速", "上修减速", "下修加速", "下修放缓"]


def load(n, h, src="clean"):
    """src = clean：清洗口径（analyze.py，研究用）；rt：实时口径（realtime.py，每周当时能算出的值，二阶导已按当周值算好）"""
    f = "_weekly_signals.csv" if src == "clean" else "_weekly_rt.csv"
    w = pd.read_csv(os.path.join(BASE, f), index_col=0, parse_dates=True)
    w = w[w.x13.first_valid_index():].copy()
    nq = drop_partial(pd.read_csv(os.path.join(BASE, "_nasdaq.csv"), index_col=0, parse_dates=True))["^IXIC"].dropna()
    daily = nq.reindex(pd.date_range(nq.index.min(), nq.index.max())).ffill()
    g = w.index
    w["px"] = daily.reindex(g).values
    last = daily.index.max()
    for k in (4, 13, 26):
        fut = daily.reindex(g + pd.Timedelta(weeks=k)).values
        w[f"f{k}"] = np.log(fut / w.px.values)
        w.loc[g + pd.Timedelta(weeks=k) > last, f"f{k}"] = np.nan
    w["p13"] = np.log(w.px).diff(13)
    w["a"] = w.x13.diff(n) if src == "clean" else w[f"a{n}"]
    # 滞回去噪：+1 加快 / −1 放慢
    st, cur = [], np.nan
    for v in w.a:
        if pd.notna(v):
            if v > h / 100:
                cur = 1
            elif v < -h / 100:
                cur = -1
            elif np.isnan(cur):
                cur = 1 if v >= 0 else -1
        st.append(cur)
    w["acc"] = st
    up = w.x13 > 0
    w["phase"] = np.select([up & (w.acc > 0), up & (w.acc < 0), ~up & (w.acc < 0), ~up & (w.acc > 0)], PHASES, None)
    w.loc[w.acc.isna() | w.x13.isna(), "phase"] = None
    if src == "rt":  # 实时口径：待核实 / 缺周不更新，阶段维持上周
        w["phase"] = w.phase.ffill(limit=4)
    return w, daily


def phase_table(w):
    rows = {}
    base = {k: w[f"f{k}"].mean() for k in (4, 13, 26)}
    for ph in PHASES:
        s = w[w.phase == ph]
        r = {"weeks": int(len(s)), "share": round(len(s) / w.phase.notna().sum(), 3),
             "prior13_pct": round(100 * s.p13.mean(), 2)}
        for k in (4, 13, 26):
            f = s[f"f{k}"].dropna()
            r[f"f{k}_pct"] = round(100 * f.mean(), 2)
            r[f"f{k}_hit"] = round(float((f > 0).mean()), 2)
            # 相对全样本平均的超额，用 Newey-West 算 t（重叠收益）
            d = w[[f"f{k}"]].dropna().assign(dum=lambda x: (w.phase.reindex(x.index) == ph).astype(float))
            b, t = nw_t(d[f"f{k}"].values, d.dum.values, k)
            r[f"f{k}_vs_all_t"] = round(float(t[1]), 2)
        rows[ph] = r
    rows["全部"] = {f"f{k}_pct": round(100 * v, 2) for k, v in base.items()}
    return rows


def path(w, dates, lo=-26, hi=26):
    """事件前后纳指累计涨跌（以事件周为 0），返回均值、中位数和逐个事件。"""
    if not dates:
        return {"k": list(range(lo, hi + 1)), "mean": [], "median": [], "n": [], "each": []}
    lp = np.log(w.px)
    idx = {d: i for i, d in enumerate(w.index)}
    paths = []
    for d in dates:
        i = idx[d]
        row = []
        for k in range(lo, hi + 1):
            j = i + k
            row.append(100 * (lp.iloc[j] - lp.iloc[i]) if 0 <= j < len(w) and pd.notna(lp.iloc[j]) else np.nan)
        paths.append(row)
    a = np.array(paths, dtype=float)
    return {"k": list(range(lo, hi + 1)),
            "mean": [None if np.isnan(v) else round(float(v), 2) for v in np.nanmean(a, axis=0)],
            "median": [None if np.isnan(v) else round(float(v), 2) for v in np.nanmedian(a, axis=0)],
            "n": [int(v) for v in np.sum(~np.isnan(a), axis=0)],
            "each": [[None if np.isnan(v) else round(float(v), 2) for v in r] for r in a]}


def realtime_events(w, gap=13):
    """滞回状态翻转的那一周；同类事件至少隔 gap 周。"""
    ev = {"上修见顶": [], "下修见底": [], "上修再加速": [], "下修再加剧": []}
    prev = None
    for d, r in w.iterrows():
        if pd.isna(r.acc) or pd.isna(r.x13):
            continue
        if prev is not None and r.acc != prev:
            typ = ("上修见顶" if r.acc < 0 else "上修再加速") if r.x13 > 0 else ("下修见底" if r.acc > 0 else "下修再加剧")
            if not ev[typ] or (d - ev[typ][-1]).days >= gap * 7:
                ev[typ].append(d)
        prev = r.acc
    return ev


def expost_turns(w, prom=0.02, dist=13):
    x = w.x13.interpolate(limit_area="inside")
    ok = x.notna()
    xs = x[ok]
    pk, _ = find_peaks(xs.values, prominence=prom, distance=dist)
    tr, _ = find_peaks(-xs.values, prominence=prom, distance=dist)
    return list(xs.index[pk]), list(xs.index[tr])


def nasdaq_extreme_lag(w, d, kind, win=26):
    """事后拐点前后 win 周内纳指的最高（kind=peak）/最低点，返回相差周数（负 = 纳指先到）。"""
    i = w.index.get_loc(d)
    seg = w.px.iloc[max(0, i - win):i + win + 1]
    j = seg.idxmax() if kind == "peak" else seg.idxmin()
    return int(round((j - d).days / 7)), str(j.date())


def event_rows(w, dates):
    out = []
    for d in dates:
        r = w.loc[d]
        out.append({"date": str(d.date()), "x13": round(100 * r.x13, 2), "a": round(100 * r.a, 2),
                    "prior13": round(100 * r.p13, 1) if pd.notna(r.p13) else None,
                    **{f"f{k}": (round(100 * r[f"f{k}"], 1) if pd.notna(r[f"f{k}"]) else None) for k in (4, 13, 26)}})
    return out


def summarize(rows):
    s = {}
    for k in (4, 13, 26):
        v = [r[f"f{k}"] for r in rows if r[f"f{k}"] is not None]
        s[f"f{k}_mean"] = round(float(np.mean(v)), 2) if v else None
        s[f"f{k}_hit"] = round(float(np.mean([x > 0 for x in v])), 2) if v else None
        s[f"f{k}_n"] = len(v)
    v = [r["prior13"] for r in rows if r["prior13"] is not None]
    s["prior13_mean"] = round(float(np.mean(v)), 2) if v else None
    return s


def _run_start(ph):
    g = ph.ne(ph.shift()).cumsum()
    return ph.index[g == g.iloc[-1]][0]


def run(n=8, h=1.0, verbose=True, src="clean"):
    w, daily = load(n, h, src)
    res = {"n": n, "h": h, "span": [str(w.index[0].date()), str(w.index[-1].date())]}
    res["phases"] = phase_table(w)
    acc = w.acc.dropna()
    res["flips_per_year"] = round(float((acc != acc.shift()).sum() / (len(acc) / 52.18)), 1)

    ev = realtime_events(w)
    res["realtime"] = {}
    for typ, ds in ev.items():
        rows = event_rows(w, ds)
        res["realtime"][typ] = {"events": rows, "summary": summarize(rows), "path": path(w, ds)}

    # 事后高低点本来就是事后看的，一律用清洗口径（数据最干净）
    # 事后高低点本来就是事后看的，一律用清洗口径（数据最干净）；事件行里的超额修正、二阶导也取清洗口径——
    # 实时口径在待核实周是空值（如 2026-06-12），会显示成 NaN、被当成「不是正值」漏掉
    wx = w if src == "clean" else load(n, h, "clean")[0]
    pk, tr = expost_turns(wx)
    ex = {}
    for kind, ds, name in (("peak", pk, "修正速度高点"), ("trough", tr, "修正速度低点")):
        rows = event_rows(wx, ds)
        for r, d in zip(rows, ds):
            r["nq_lag_weeks"], r["nq_extreme"] = nasdaq_extreme_lag(w, d, kind)
            # 实时信号晚多少周：事后拐点之后第一个对应的实时事件
            rt = ev["上修见顶"] + ev["下修再加剧"] if kind == "peak" else ev["下修见底"] + ev["上修再加速"]
            later = sorted(x for x in rt if x >= d)
            r["realtime_delay_weeks"] = int((later[0] - d).days / 7) if later and (later[0] - d).days <= 26 * 7 else None
        ex[name] = {"events": rows, "summary": summarize(rows), "path": path(w, ds)}
    res["expost"] = ex
    res["base"] = {f"f{k}_mean": round(100 * w[f"f{k}"].mean(), 2) for k in (4, 13, 26)}
    res["current"] = {"date": str(w.index[-1].date()), "phase": w.phase.dropna().iloc[-1],
                      "x13": round(100 * w.x13.iloc[-1], 2), "a": round(100 * w.a.iloc[-1], 2),
                      "since": str(_run_start(w.phase.dropna()).date())}
    if verbose and src == "clean":  # 只存主参数那一组（稳健性循环不覆盖）
        w[["x13", "a", "acc", "phase", "px"]].to_csv(os.path.join(BASE, "_speed_weekly.csv"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--h", type=float, default=1.0)
    a = ap.parse_args()
    res = run(a.n, a.h)
    # 稳健性：换窗口 / 换滞回带宽，看两类实时拐点的结论变不变
    rob = []
    for n in (4, 8, 13):
        for h in (0.5, 1.0, 1.5, 2.0):
            r = run(n, h, verbose=False)
            row = {"n": n, "h": h, "flips_per_year": r["flips_per_year"]}
            for typ in ("上修见顶", "下修见底"):
                s = r["realtime"][typ]["summary"]
                row[typ] = {"count": s["f13_n"], "f13": s["f13_mean"], "hit13": s["f13_hit"], "f26": s["f26_mean"], "prior13": s["prior13_mean"]}
            for ph in PHASES:
                row[ph + "_f13"] = r["phases"][ph]["f13_pct"]
            rob.append(row)
    res["robustness"] = rob
    with open(os.path.join(BASE, "_speed.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("n", "h", "span", "flips_per_year", "phases", "base", "current")}, ensure_ascii=False, indent=1))
    for typ, v in res["realtime"].items():
        print(typ, v["summary"])
        for e in v["events"]:
            print("   ", e)
    for typ, v in res["expost"].items():
        print(typ, v["summary"])
        for e in v["events"]:
            print("   ", e)
    for r in rob:
        print(r)


if __name__ == "__main__":
    main()
