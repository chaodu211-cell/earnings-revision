#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盈利预期修正周期 × 标普500 表现（FactSet 周报，2011-04 起）

信号（每周，周报周五发布、数据截至周四收盘；按周五收盘进出）：
  r13   远期 12 个月 EPS 13 周对数变化（含「时间往前滚」的机械增长）
  x13   r13 减去它此前 3 年（156 周）的均值 → 相对常态的超额修正，去掉平均滚动增长
        （攒满 2 年才有均值，即 2013-07 起；2011-07 ~ 2013-07 用第一个均值往前补，见 baseline()）
  r4    4 周变化，同理 x4
对照：p13 = 标普 13 周价格动量

输出：_analysis.json（数字），终端打印摘要
用法：python3 analyze.py
"""
import json, os, sys, time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def drop_partial(df):
    """美股收盘（美东 16:15）前，Yahoo 会把当天盘中价当一根日线给出来——不用它。"""
    now = pd.Timestamp.now(tz="America/New_York")
    if now.hour * 60 + now.minute < 16 * 60 + 15:
        df = df[df.index < now.normalize().tz_localize(None)]
    return df


def spx():
    path = os.path.join(BASE, "_spx.csv")
    # 缓存 3 小时内有效（按文件修改时间的秒数比，避免时区换算出错）。
    # REV_FORCE_FETCH=1（GitHub Actions）时不看缓存新旧、每次重下；下载失败才退回缓存。
    fresh = (os.path.exists(path) and time.time() - os.path.getmtime(path) < 3 * 3600
             and not os.environ.get("REV_FORCE_FETCH"))
    if not fresh:
        try:
            import yfinance as yf
            h = yf.Ticker("^GSPC").history(start="2008-01-01", auto_adjust=False)
            if len(h) < 2000:  # 被限流时 yfinance 常返回空表而不报错，不能拿它覆盖缓存
                raise RuntimeError(f"只拿到 {len(h)} 行")
            h["Close"].tz_localize(None).rename("close").to_csv(path)
        except Exception as e:
            if not os.path.exists(path):
                raise
            print(f"PRICE_FALLBACK ^GSPC 下载失败，改用缓存 _spx.csv：{e}", file=sys.stderr)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    return drop_partial(s)


def nw_t(y, X, lag):
    """OLS + Newey-West t 值（重叠收益要用）。"""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, b / np.sqrt(np.diag(V))


def baseline(r):
    """r 此前 3 年（156 周，至少 104 周）的均值，不含当周。
    2013-07 之前还攒不够 104 周：用第一个均值（= 2011-07 ~ 2013-07 这两年 r 的平均，13 周口径 1.32%）往前补，
    这样超额修正从 2011-07 就有、2013-07 接上时不跳。这段基准是事后值（2011 年当时并不知道），页面上标明。"""
    b = r.rolling(156, min_periods=104).mean().shift(1)
    first = b.first_valid_index()
    if first is not None:
        b.loc[:first] = b.loc[first]
    return b


def load():
    d = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"]).set_index("date")
    d = d[d.fwd_eps_clean.notna()]
    # 周频网格（周五），缺的周（节假日没出周报）最多线性补 3 周
    grid = pd.date_range(d.index.min() - pd.Timedelta(days=d.index.min().weekday() - 4), d.index.max(), freq="W-FRI")
    eps = d.fwd_eps_clean.copy()
    eps.index = eps.index + pd.to_timedelta((4 - eps.index.weekday) % 7, unit="D")  # 周四出的算到当周五
    eps = eps[~eps.index.duplicated(keep="last")].reindex(grid).interpolate(limit=3, limit_area="inside")
    px = spx().reindex(pd.date_range("2008-01-01", grid.max() + pd.Timedelta(days=200))).ffill()
    w = pd.DataFrame({"eps": eps, "px": px.reindex(grid)})
    w["guide_pos_share"] = (d.guide_pos / (d.guide_pos + d.guide_neg)).reindex(w.index, method="ffill", limit=2)
    le, lp = np.log(w.eps), np.log(w.px)
    w["r4"], w["r13"] = le.diff(4), le.diff(13)
    w["x13"] = w.r13 - baseline(w.r13)
    w["x4"] = w.r4 - baseline(w.r4)
    w["p13"] = lp.diff(13)
    pxf = px.reindex(grid)
    for h in (4, 13, 26):
        fut = px.reindex(grid + pd.Timedelta(weeks=h)).values
        w[f"f{h}"] = np.log(fut / pxf.values)
    w.loc[w.index + pd.Timedelta(weeks=26) > px.dropna().index.max(), "f26"] = np.nan
    w.loc[w.index + pd.Timedelta(weeks=13) > px.dropna().index.max(), "f13"] = np.nan
    w.loc[w.index + pd.Timedelta(weeks=4) > px.dropna().index.max(), "f4"] = np.nan
    return w


def main():
    w = load()
    res = {"span": [str(w.index.min().date()), str(w.index.max().date())], "weeks": int(w.eps.notna().sum())}

    # 1) 预测回归：未来 h 周收益 ~ 信号（单变量 / 加价格动量）
    reg = {}
    for sig in ("r4", "r13", "x4", "x13", "p13", "guide_pos_share"):
        for h in (4, 13, 26):
            s = w[[sig, f"f{h}"]].dropna()
            if len(s) < 100:
                continue
            z = (s[sig] - s[sig].mean()) / s[sig].std()
            b, t = nw_t(s[f"f{h}"].values, z.values, h)
            reg[f"{sig}->f{h}"] = {"n": len(s), "beta_per_sd_pct": round(100 * b[1], 2), "t": round(t[1], 2),
                                   "corr": round(float(np.corrcoef(s[sig], s[f"f{h}"])[0, 1]), 3)}
    for h in (4, 13, 26):
        s = w[["x13", "p13", f"f{h}"]].dropna()
        Z = ((s[["x13", "p13"]] - s[["x13", "p13"]].mean()) / s[["x13", "p13"]].std()).values
        b, t = nw_t(s[f"f{h}"].values, Z, h)
        reg[f"x13+p13->f{h}"] = {"n": len(s), "beta_x13": round(100 * b[1], 2), "t_x13": round(t[1], 2),
                                  "beta_p13": round(100 * b[2], 2), "t_p13": round(t[2], 2)}
    # 分前后两段看是否稳定（2011-04~2018 / 2019~今）
    for sig in ("r13", "x13", "guide_pos_share"):
        for lo, hi in (("2011", "2018"), ("2019", "2026")):
            s = w.loc[lo:hi, [sig, "f13"]].dropna()
            if len(s) < 60:
                continue
            z = (s[sig] - s[sig].mean()) / s[sig].std()
            b, t = nw_t(s.f13.values, z.values, 13)
            reg[f"{sig}->f13 [{lo}-{hi}]"] = {"n": len(s), "beta_per_sd_pct": round(100 * b[1], 2), "t": round(t[1], 2)}
    res["regress"] = reg

    # 2) 价格领先还是修正领先：corr(13 周 EPS 修正 在 t, 13 周价格变化 在 t+k)
    ll = {}
    for k in range(-26, 27, 2):
        a = w.r13
        b = np.log(w.px).diff(13).shift(-k)
        ok = a.notna() & b.notna()
        ll[k] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)
    res["leadlag_r13_vs_p13_shift"] = ll

    # 3) 状态：超额修正为正 / 为负 时的未来收益
    st = {}
    for sig in ("x13", "r13"):
        for h in (4, 13, 26):
            s = w[[sig, f"f{h}"]].dropna()
            up, dn = s[s[sig] > 0][f"f{h}"], s[s[sig] <= 0][f"f{h}"]
            st[f"{sig}>0 f{h}"] = {"share_time": round(len(up) / len(s), 3),
                                   "mean_up_pct": round(100 * up.mean(), 2), "mean_dn_pct": round(100 * dn.mean(), 2),
                                   "hit_up": round(float((up > 0).mean()), 3), "hit_dn": round(float((dn > 0).mean()), 3)}
    # 按五分位
    s = w[["x13", "f13"]].dropna()
    q = pd.qcut(s.x13, 5, labels=False)
    st["x13_quintile_f13_mean_pct"] = [round(100 * v, 2) for v in s.groupby(q).f13.mean()]
    res["state"] = st

    # 4) 简单择时：x13>0 持有标普、否则空仓（下周五收盘执行，不计利息和成本）
    px = w.px
    wk = np.log(px).diff().shift(-1)  # 本周五 → 下周五
    tim = {}
    for sig in ("x13", "r13", "p13"):
        s = pd.DataFrame({"pos": (w[sig] > 0).astype(float).where(w[sig].notna()), "ret": wk}).dropna()
        s["pos"] = s.pos.shift(1).fillna(0)  # 信号周五晚上才拿到 → 下一周才生效，保守
        for name, r in (("strategy", s.pos * s.ret), ("buyhold", s.ret)):
            eq = np.exp(r.cumsum())
            yrs = len(r) / 52.18
            tim[f"{sig}:{name}"] = {"cagr_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 2),
                                    "maxdd_pct": round(100 * (eq / eq.cummax() - 1).min(), 1),
                                    "in_market": round(float(s.pos.mean()), 3) if name == "strategy" else 1.0}
    res["timing"] = tim

    # 5) 上修周期清单：x13 由负转正、且持续 ≥ 4 周
    ep = []
    sgn = (w.x13 > 0).astype(int).where(w.x13.notna())
    start = None
    for dt, v in sgn.items():
        if v == 1 and start is None:
            start = dt
        elif v == 0 and start is not None:
            if (dt - start).days >= 28:
                ep.append((start, dt))
            start = None
    if start is not None:
        ep.append((start, None))
    rows = []
    for a, b in ep:
        r = {"start": str(a.date()), "end": str(b.date()) if b is not None else "进行中",
             "weeks": int(((b or w.index.max()) - a).days / 7),
             "px_chg_during_pct": round(100 * (w.px.get(b, w.px.iloc[-1]) / w.px[a] - 1), 1),
             "px_chg_prior13w_pct": round(100 * w.p13[a], 1) if pd.notna(w.p13[a]) else None,
             "f13_from_start_pct": round(100 * w.f13[a], 1) if pd.notna(w.f13[a]) else None}
        rows.append(r)
    res["upcycles"] = rows

    # 5b) 大跌之后：修正转正时，指数已经从底部涨了多少（跌幅 >15% 的几次底）
    pxd_all = spx()
    tp = []
    for trough in ("2011-10-03", "2016-02-11", "2018-12-24", "2020-03-23", "2022-10-12", "2025-04-08"):
        t0 = pd.Timestamp(trough)
        after = w.loc[t0:]
        row = {"trough": trough}
        lowk = w.r13.loc[t0 - pd.Timedelta(weeks=26):t0 + pd.Timedelta(weeks=52)]
        if lowk.notna().any():
            row["r13_low"] = str(lowk.idxmin().date())
            row["r13_low_weeks_after_px_low"] = int((lowk.idxmin() - t0).days / 7)
        for sig in ("r13", "x13"):
            hit = after[after[sig] > 0]
            first_neg = w.loc[t0 - pd.Timedelta(weeks=26):t0, sig]
            if len(hit) and first_neg.notna().any() and (first_neg < 0).any():
                d1 = hit.index[0]
                row[f"{sig}_turns_up"] = str(d1.date())
                row[f"{sig}_weeks"] = int((d1 - t0).days / 7)
                row[f"{sig}_px_from_low_pct"] = round(100 * (pxd_all[:d1].iloc[-1] / pxd_all[t0] - 1), 1)
        tp.append(row)
    res["troughs"] = tp

    # 6) 季度口径（固定季度，不含滚动增长）：季内自下而上 EPS 修正 vs 当季 / 下季指数涨跌
    d = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"])
    d = d[d.q.notna()]
    d["qcal"] = d.date.dt.year.astype(str) + "Q" + d.date.dt.quarter.astype(str)
    d = d[d.q == d.qcal]  # 只要「进行中的季度」那句
    qq = d.sort_values("date").groupby("q").last()
    qq = qq[pd.to_datetime(qq.date).dt.month.isin([3, 6, 9, 12])]  # 季末那个月的读数才算「季内修正」终值
    pxd = spx()
    out = []
    for q, r in qq.iterrows():
        y, n = int(q[:4]), int(q[-1])
        q0 = pd.Timestamp(y, 3 * n - 2, 1) - pd.Timedelta(days=1)
        q1 = pd.Timestamp(y, 3 * n, 1) + pd.offsets.MonthEnd(0)
        q2 = q1 + pd.offsets.QuarterEnd(1)
        p = lambda t: pxd[:t].iloc[-1]
        out.append({"q": q, "q_rev_pct": round(100 * r.q_rev, 2),
                    "px_same_q_pct": round(100 * (p(q1) / p(q0) - 1), 2),
                    "px_next_q_pct": round(100 * (p(q2) / p(q1) - 1), 2) if q2 <= pxd.index.max() else None})
    qdf = pd.DataFrame(out)
    ok = qdf.dropna()
    res["quarterly"] = {"n": len(qdf), "rows": out,
                        "corr_same_q": round(float(qdf.q_rev_pct.corr(qdf.px_same_q_pct)), 3),
                        "corr_next_q": round(float(ok.q_rev_pct.corr(ok.px_next_q_pct)), 3),
                        "next_q_when_rev_above_median": round(float(ok[ok.q_rev_pct > ok.q_rev_pct.median()].px_next_q_pct.mean()), 2),
                        "next_q_when_rev_below_median": round(float(ok[ok.q_rev_pct <= ok.q_rev_pct.median()].px_next_q_pct.mean()), 2)}

    last = w.dropna(subset=["eps"]).iloc[-1]
    res["latest"] = {"date": str(w.dropna(subset=["eps"]).index[-1].date()), "fwd_eps": round(last.eps, 2),
                     "r4_pct": round(100 * last.r4, 2), "r13_pct": round(100 * last.r13, 2),
                     "x13_pct": round(100 * last.x13, 2) if pd.notna(last.x13) else None,
                     "x13_pctile": round(float((w.x13.dropna() < last.x13).mean()), 3) if pd.notna(last.x13) else None,
                     "guide_pos_share": round(last.guide_pos_share, 3) if pd.notna(last.guide_pos_share) else None}
    w.to_csv(os.path.join(BASE, "_weekly_signals.csv"))
    with open(os.path.join(BASE, "_analysis.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
