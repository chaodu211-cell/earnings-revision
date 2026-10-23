#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「上修速度见顶零点」能不能并进最优拟合红点（2026-10-23 复测，结论：不加，见 说明.md）。

零点 = 页面默认口径：实时纯修正 x13 > 0、8 周二阶导 a8 由正转负（照抄 make_chart 的 markSets）。
周五周报 → 下一个交易日起生效。红点温度取上级目录 data_alt.json 的 temperature_sell（页面上 2017-10-18 起），
训练窗 2017-10-18 ~ 2026-09-30，并入后门槛一律同频重标到 67 天，看之后 30 个交易日 QQQ。

要用上级目录（best-weights 仓库）的 data_alt.json、raw/QQQ.csv、策略回测/_signals_long.csv，只能本地跑。
用法：python3 peak_factor_test.py [1-8 …]（不给编号跑全部；4、5、7 较慢）
"""
import json, os, sys
import numpy as np
import pandas as pd

REV = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(REV)   # 上级 最优拟合/
sys.path.insert(0, HERE)
FWD = 30
TR0, TR1 = pd.Timestamp("2017-10-18"), pd.Timestamp("2026-09-30")
TARGET = 67
BASE_TH = 82.5


def load_qqq():
    df = pd.read_csv(f"{HERE}/raw/QQQ.csv", header=None, names=["date", "close", "rawclose", "volume"],
                     parse_dates=["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()["close"]


def load_prod():
    d = json.load(open(f"{HERE}/data_alt.json"))
    s = d["series"]
    idx = pd.to_datetime(s["dates"])
    T = pd.Series(s["temperature_sell"], index=idx, dtype=float)
    hot = pd.Series(d["alerts"]["flags"]["hot"], index=idx).astype(bool)
    hot_ad = pd.Series(d["alerts"]["flags"]["hot_ad"], index=idx).astype(bool)
    return T, hot, hot_ad


def weekly(src="rt"):
    f = "_weekly_rt.csv" if src == "rt" else "_weekly_signals.csv"
    return pd.read_csv(os.path.join(REV, f), index_col=0, parse_dates=True)


def zero_events(x, a):
    """照抄页面 markSets() 的零点：二阶导符号（空值记 0，不更新 prev）由正转负且纯修正 > 0"""
    up, dn, prev = [], [], 0
    for t in x.index:
        av, xv = a.get(t, np.nan), x.get(t, np.nan)
        s = 0 if pd.isna(av) else int(np.sign(av))
        if s < 0 and prev > 0 and pd.notna(xv) and xv > 0:
            up.append(t)
        if s > 0 and prev < 0 and pd.notna(xv) and xv < 0:
            dn.append(t)
        if s != 0:
            prev = s
    return up, dn


def conf_events(x, a, h, rt=True):
    """确认点：滞回去噪后状态翻转（speed.load 的做法），阶段 ffill(limit=4)"""
    st, cur = [], np.nan
    for v in a.values:
        if pd.notna(v):
            if v > h / 100: cur = 1
            elif v < -h / 100: cur = -1
            elif np.isnan(cur): cur = 1 if v >= 0 else -1
        st.append(cur)
    acc = pd.Series(st, index=a.index)
    up_ = x > 0
    ph = pd.Series(np.select([up_ & (acc > 0), up_ & (acc < 0), ~up_ & (acc < 0), ~up_ & (acc > 0)],
                             ["A", "B", "C", "D"], ""), index=a.index).replace("", np.nan)
    ph[acc.isna() | x.isna()] = np.nan
    if rt:
        ph = ph.ffill(limit=4)
    up, prev = [], 0
    for t in x.index:
        c = ph.get(t)
        s = 1 if c in ("A", "D") else -1 if c in ("B", "C") else 0
        xv = x.get(t)
        if s < 0 and prev > 0 and pd.notna(xv) and xv > 0:
            up.append(t)
        if s != 0:
            prev = s
    return up


def swing_events(x, R=2, A=4, L=2):
    """大波段（页面 swings()），x 用百分数"""
    out, mode, ext, last, hi, lo = [], None, None, None, None, None
    for t, v in x.dropna().items():
        cur = (t, v)
        if mode is None:
            if hi is None or v > hi[1]: hi = cur
            if lo is None or v < lo[1]: lo = cur
            if v <= hi[1] - R: out.append(("top", hi, cur, np.nan)); last = hi[1]; mode = "down"; ext = cur
            elif v >= lo[1] + R: out.append(("bot", lo, cur, np.nan)); last = lo[1]; mode = "up"; ext = cur
            continue
        if mode == "up":
            if v > ext[1]: ext = cur
            elif v <= ext[1] - R: out.append(("top", ext, cur, ext[1] - last)); last = ext[1]; mode = "down"; ext = cur
        else:
            if v < ext[1]: ext = cur
            elif v >= ext[1] + R: out.append(("bot", ext, cur, last - ext[1])); last = ext[1]; mode = "up"; ext = cur
    return [e[2][0] for e in out if e[0] == "top" and e[1][1] >= L and e[3] >= A]


def eff_pos(idx, dates):
    """周五周报 → 下一个交易日起生效"""
    return np.searchsorted(idx.values, np.array(pd.to_datetime(dates), dtype="datetime64[ns]"), side="right")


def factor(idx, dates, K, mode="hold", off=0.0):
    F = np.full(len(idx), float(off))
    for p in eff_pos(idx, dates):
        if p >= len(idx):
            continue
        n = min(K, len(idx) - p)
        if mode == "hold":
            F[p:p + n] = 100.0
        else:  # 线性衰减：触发当天 100，K 天后回到 off
            v = off + (100.0 - off) * (1 - np.arange(n) / K)
            F[p:p + n] = np.maximum(F[p:p + n], v)
    return F


def min3(v):
    s = pd.Series(v)
    return s.rolling(3, min_periods=3).min().values


def calib(m3, win, target=TARGET):
    v = np.sort(m3[win & np.isfinite(m3)])[::-1]
    return (v[target - 1] + v[target]) / 2


def tops(px):
    c = px.values; n = len(c); cand = []
    for i in range(63, n - 1):
        if c[i] >= np.nanmax(c[i - 63:i + 1]) and np.nanmin(c[i:min(n, i + 64)]) / c[i] - 1 <= -0.11:
            cand.append(i)
    out = []
    for i in cand:
        if out and i - out[-1] <= 63:
            if c[i] > c[out[-1]]: out[-1] = i
        else:
            out.append(i)
    return out


def groups(m):
    i = np.flatnonzero(m)
    if len(i) == 0:
        return []
    g, cur = [], [i[0]]
    for a, b in zip(i[:-1], i[1:]):
        if b - a > FWD:
            g.append(cur); cur = []
        cur.append(b)
    g.append(cur)
    return g


class Ctx:
    def __init__(self):
        self.T, self.hot, self.hot_ad = load_prod()
        q = load_qqq()
        self.qall = q
        self.idx = self.T.index
        self.q = q.reindex(self.idx)
        fwd = q.shift(-FWD) / q - 1
        self.fwd = fwd.reindex(self.idx).values
        self.win = np.asarray((self.idx >= TR0) & (self.idx <= TR1))
        ok = self.win & np.isfinite(self.fwd)
        self.base = np.nanmean(self.fwd[ok])
        # 下跌顶（scan_window.tops，QQQ 全历史上算，再落到页面日期）
        tp = tops(q)
        self.tops = [q.index[i] for i in tp if TR0 <= q.index[i] <= TR1]
        self.top_pos = [self.idx.get_loc(t) for t in self.tops]
        # 熊市反弹：不是过去一年最高收盘
        hi252 = q.rolling(252, min_periods=200).max()
        self.bear = {t for t in self.tops if q[t] < hi252[t] - 1e-9}

    def evaluate(self, m):
        m = m & self.win
        ok = m & np.isfinite(self.fwd)
        g = groups(m)
        ev = [self.fwd[x[0]] for x in g if np.isfinite(self.fwd[x[0]])]
        caught = [t for t, p in zip(self.tops, self.top_pos) if m[max(0, p - 30):p + 6].any()]
        return dict(days=int(m.sum()), mean=100 * np.nanmean(self.fwd[ok]) if ok.any() else np.nan,
                    neg=100 * np.mean(self.fwd[ok] < 0) if ok.any() else np.nan,
                    mg=100 * (np.nanmean(self.fwd[ok]) - self.base) if ok.any() else np.nan,
                    ev=len(g), ev_mean=100 * np.mean(ev) if ev else np.nan,
                    caught=caught)

    def red(self, Tn, th=None):
        m3 = min3(Tn)
        if th is None:
            th = calib(m3, self.win)
        return (m3 > th), th


def step1():
    """复现现行红点、列出各口径的上修速度见顶事件"""
    C = Ctx()
    T = C.T.values
    m, th = C.red(T, BASE_TH)
    print("复现红点：与页面 hot 一致的天数", int((m == C.hot.values).sum()), "/", len(m), " 不一致", int((m != C.hot.values).sum()))
    r = C.evaluate(m)
    print("现行红点 训练窗:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'caught'})
    print("基准 30 日 QQQ：%.2f%%" % (100 * C.base))
    print("训练窗下跌顶：", [(str(t.date()), '熊市反弹' if t in C.bear else '') for t in C.tops])
    print("现行抓到：", [str(t.date()) for t in r['caught']])
    mc, thc = C.red(T)
    print("同频重标门槛（应≈82.5）：%.3f" % thc, int((mc & C.win).sum()))

    w = weekly("rt")
    for n in (4, 8, 13):
        up, dn = zero_events(w.x13, w[f"a{n}"])
        print(f"\n零点 a{n}: 上修速度见顶 {len(up)} 次（{w.x13.first_valid_index().date()}~{w.index[-1].date()}，每年 {len(up)/((w.index[-1]-w.x13.first_valid_index()).days/365.25):.1f} 次）")
        if n == 8:
            print([str(d.date()) for d in up])
    for h in (0.5, 1.0):
        up = conf_events(w.x13, w.a8, h)
        print(f"确认点 a8 h={h}: {len(up)} 次", [str(d.date()) for d in up])
    sw = swing_events(100 * w.x13)
    print("大波段:", [str(d.date()) for d in sw])


def step2():
    """零点单独看：之后 30 / 63 日 QQQ、63 日内最大回撤"""
    C = Ctx()
    q = C.qall
    fq = {k: (q.shift(-k) / q - 1) for k in (30, 63)}
    # 之后 63 日最大回撤
    lo63 = q[::-1].rolling(64, min_periods=1).min()[::-1]
    dd63 = lo63 / q - 1
    w = weekly("rt")
    up, _ = zero_events(w.x13, w.a8)
    idx = q.index
    red_pos = np.flatnonzero(C.hot.values)
    red_days = C.idx[red_pos]
    print("日期(周报)  生效日      纯修正  二阶导  当日红点温度 30日内温度最高  30日QQQ  63日QQQ  63日内最大回撤  前后红点")
    rows = []
    for d in up:
        p = np.searchsorted(idx.values, np.datetime64(d), side="right")
        e = idx[p]
        T0 = C.T.get(e, np.nan)
        seg = C.T[(C.T.index >= e)].iloc[:30]
        near = [r for r in red_days if abs((r - e).days) <= 45]
        rows.append(dict(d=d, e=e, x=100*w.x13[d], a=100*w.a8[d], T0=T0, Tmax=seg.max() if len(seg) else np.nan,
                         f30=100*fq[30][e], f63=100*fq[63][e], dd=100*dd63[e], red=len(near)))
        r = rows[-1]
        print(f"{d.date()}  {e.date()}  {r['x']:5.2f}%  {r['a']:5.2f}  {r['T0']:6.1f}  {r['Tmax']:6.1f}  {r['f30']:6.1f}%  {r['f63']:6.1f}%  {r['dd']:6.1f}%  {'有' if near else ''}")
    R = pd.DataFrame(rows)
    # 同期基准：2016-12 起全部交易日
    s0 = w.x13.first_valid_index()
    allf30 = fq[30][(idx > s0)].dropna(); allf63 = fq[63][(idx > s0)].dropna(); alldd = dd63[(idx > s0)]
    print(f"\n零点 {len(R)} 次：30 日均 {R.f30.mean():.2f}%（为负 {100*(R.f30<0).mean():.0f}%）vs 全部日子 {100*allf30.mean():.2f}%（为负 {100*(allf30<0).mean():.0f}%）")
    print(f"            63 日均 {R.f63.mean():.2f}%（为负 {100*(R.f63<0).mean():.0f}%）vs {100*allf63.mean():.2f}%（{100*(allf63<0).mean():.0f}%）")
    print(f"            63 日内最大回撤均 {R.dd.mean():.2f}% vs {100*alldd.mean():.2f}%；回撤≥10% 的比例 {100*(R.dd<=-10).mean():.0f}% vs {100*(alldd<=-0.10).mean():.0f}%")
    print(f"            触发日红点温度均 {R.T0.mean():.1f}，30 日内最高均 {R.Tmax.mean():.1f}")
    # 随机日子对照：同样 21 个日子随机抽，30 日均值分布
    rng = np.random.default_rng(20261023)
    pool = allf30.values
    sims = np.array([rng.choice(pool, len(R), replace=False).mean() for _ in range(20000)])
    print(f"随机抽 {len(R)} 天的 30 日均值 ≤ 实际的比例 p = {(sims <= R.f30.mean()/100).mean():.3f}")
    pool = allf63.values
    sims = np.array([rng.choice(pool, len(R), replace=False).mean() for _ in range(20000)])
    print(f"随机抽 {len(R)} 天的 63 日均值 ≤ 实际的比例 p = {(sims <= R.f63.mean()/100).mean():.3f}")


def step3():
    """并进红点温度的网格（K 天读 100 / 其余 0 或 50 / 线性衰减 / 按阶段），门槛同频重标到 67 天"""
    C = Ctx()
    T0 = C.T.values
    idx = C.idx
    w = weekly("rt")
    UP, _ = zero_events(w.x13, w.a8)

    # 阶段状态（不去噪）：x13>0 且 a8<0，按周报生效日铺到日度
    def phase_state(x, a):
        st = ((x > 0) & (a < 0)).astype(float)
        st[x.isna() | a.isna()] = np.nan
        st = st.ffill(limit=4)  # 待核实/缺周维持上周
        p = eff_pos(idx, st.index)
        out = np.full(len(idx), np.nan)
        for (t, v), pp in zip(st.items(), p):
            if pp < len(idx):
                out[pp:] = v  # 后面的会覆盖
        return np.nan_to_num(out, nan=0.0)
    PH = phase_state(w.x13, w.a8)

    def run(F, wt):
        Tn = (1 - wt) * T0 + wt * F
        m, th = C.red(Tn)
        r = C.evaluate(m)
        r["th"] = th
        return r, m

    base = C.evaluate(C.red(T0, BASE_TH)[0])
    def fmt(tag, r):
        miss = [t for t in C.tops if t not in r["caught"] and t not in C.bear]
        newc = [str(t.date()) for t in r["caught"] if t not in base["caught"]]
        lost = [str(t.date()) for t in base["caught"] if t not in r["caught"]]
        return (f"{tag:<28} 门槛{r['th']:6.2f} {r['days']:3d}天 {r['ev']:2d}次  30日{r['mean']:6.2f}% 为负{r['neg']:4.0f}% 边际{r['mg']:7.2f}pp "
                f"事件首日{r['ev_mean']:6.2f}%  抓顶{len([t for t in r['caught'] if t not in C.bear])}/7"
                + (f" 新抓{newc}" if newc else "") + (f" 丢{lost}" if lost else ""))

    print(fmt("现行", dict(base, th=BASE_TH)))
    res = []
    for mode, off in (("hold", 0), ("hold", 50), ("decay", 0), ("decay", 50)):
        for K in (10, 20, 30, 45, 63):
            F = factor(idx, UP, K, mode, off)
            for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
                r, m = run(F, wt)
                res.append((mode, off, K, wt, r))
    for off in (0, 50):
        F = np.where(PH > 0, 100.0, float(off))
        for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
            r, m = run(F, wt)
            res.append(("phase", off, 0, wt, r))

    print("\n—— 用户方案：触发后 K 天读 100、其余读 0 ——")
    for mode, off, K, wt, r in res:
        if mode == "hold" and off == 0:
            print(fmt(f"hold K={K:<3} 权重{int(wt*100):>2}%", r))
    print("\n—— 其余读 50（中性）——")
    for mode, off, K, wt, r in res:
        if mode == "hold" and off == 50:
            print(fmt(f"hold50 K={K:<3} 权重{int(wt*100):>2}%", r))
    print("\n—— 线性衰减 ——")
    for mode, off, K, wt, r in res:
        if mode == "decay":
            print(fmt(f"decay{off} K={K:<3} 权重{int(wt*100):>2}%", r))
    print("\n—— 阶段状态（上修减速期间读 100）——")
    for mode, off, K, wt, r in res:
        if mode == "phase":
            print(fmt(f"phase{off} 权重{int(wt*100):>2}%", r))

    best = min(res, key=lambda z: z[4]["mg"])
    print("\n边际最好的一组：", best[:4], round(best[4]["mg"], 2))


def step4():
    """安慰剂：零点时间整体循环平移（≥26 周），同样并入、重标"""
    C = Ctx()
    T0 = C.T.values; idx = C.idx
    w = weekly("rt")
    UP, _ = zero_events(w.x13, w.a8)
    wk = pd.date_range(w.x13.first_valid_index(), w.index[-1], freq="W-FRI")
    N = len(wk)
    pos = np.array([wk.get_loc(d) for d in UP])

    CFG = [("hold", 0, 20, 0.10), ("hold", 0, 30, 0.10), ("hold", 0, 30, 0.15), ("hold", 50, 63, 0.05),
           ("hold", 50, 20, 0.10), ("decay", 50, 10, 0.05)]
    GRID = [(m, o, K, wt) for m, o in (("hold", 0), ("hold", 50), ("decay", 0), ("decay", 50))
            for K in (10, 20, 30, 45, 63) for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33)]

    def score(dates, cfg):
        mode, off, K, wt = cfg
        F = factor(idx, dates, K, mode, off)
        m, th = C.red((1 - wt) * T0 + wt * F)
        r = C.evaluate(m)
        return r["mg"], len([t for t in r["caught"] if t not in C.bear])

    act = {c: score(UP, c) for c in CFG}
    act_best = min(score(UP, g)[0] for g in GRID)
    shifts = [s for s in range(26, N - 26)]
    sim = {c: [] for c in CFG}; sim_best = []
    for s in shifts:
        d = list(wk[(pos + s) % N])
        for c in CFG:
            sim[c].append(score(d, c))
        if s % 4 == 0:  # 网格最优较慢，每 4 周取一次
            sim_best.append(min(score(d, g)[0] for g in GRID))
    print(f"平移 {len(shifts)} 种（26 周以上），网格最优对照 {len(sim_best)} 种")
    for c in CFG:
        mg = np.array([x[0] for x in sim[c]]); ct = np.array([x[1] for x in sim[c]])
        p = np.mean(mg <= act[c][0])
        print(f"{c}: 实际边际 {act[c][0]:.2f}pp 抓顶 {act[c][1]}/7 | 随机平移 边际中位 {np.median(mg):.2f}pp、"
              f"10~90% [{np.percentile(mg,10):.2f}, {np.percentile(mg,90):.2f}]，抓顶中位 {np.median(ct):.0f}；p(随机≤实际) = {p:.2f}")
    sb = np.array(sim_best)
    print(f"网格最优：实际 {act_best:.2f}pp；随机平移的网格最优 中位 {np.median(sb):.2f}pp，p = {np.mean(sb <= act_best):.2f}")


def step5():
    """换事件定义、热端拆分、更长尺度、2011-16 单独看"""
    C = Ctx()
    T0 = C.T.values; idx = C.idx
    w = weekly("rt")
    wk = pd.date_range(w.x13.first_valid_index(), w.index[-1], freq="W-FRI"); N = len(wk)
    DEF = {}
    for n in (4, 8, 13):
        DEF[f"零点 a{n}"] = zero_events(w.x13, w[f"a{n}"])[0]
    DEF["确认点 h=0.5"] = conf_events(w.x13, w.a8, 0.5)
    DEF["确认点 h=1"] = conf_events(w.x13, w.a8, 1.0)
    DEF["大波段"] = swing_events(100 * w.x13)

    def score(dates, cfg):
        mode, off, K, wt = cfg
        F = factor(idx, dates, K, mode, off)
        m, th = C.red((1 - wt) * T0 + wt * F)
        r = C.evaluate(m)
        return r["mg"], len([t for t in r["caught"] if t not in C.bear])

    print("(b) 换事件定义：用户方案（K=30 天读 100、其余 0、权重 10%）与中性版（其余 50），现行 −11.29pp、5/7")
    for name, ev in DEF.items():
        pos = np.array([wk.get_loc(d) for d in ev])
        for cfg in (("hold", 0, 30, 0.10), ("hold", 50, 30, 0.10)):
            a = score(ev, cfg)
            sims = [score(list(wk[(pos + s) % N]), cfg)[0] for s in range(26, N - 26, 3)]
            print(f"  {name:<12} {len(ev):2d}次 off={cfg[1]:<2}: 边际 {a[0]:7.2f}pp 抓顶 {a[1]}/7；随机平移中位 {np.median(sims):7.2f}pp，p = {np.mean(np.array(sims) <= a[0]):.2f}")

    print("\n(a) 热端拆分：连 3 日温度 ≥ 70 的日子（训练窗），按「过去 30 个交易日内有没有零点 a8」分组")
    m3 = min3(T0)
    hot = (m3 >= 70) & C.win & np.isfinite(C.fwd)
    F = factor(idx, DEF["零点 a8"], 30, "hold", 0) > 0
    for lo, hi in ((70, 82.5), (82.5, 101)):
        band = hot & (m3 >= lo) & (m3 < hi)
        for lab, mk in (("有零点", band & F), ("没有", band & ~F)):
            g = groups(mk)
            print(f"  温度 {lo}~{hi} {lab}: {mk.sum():3d} 天/{len(g):2d} 段，之后 30 日 {100*np.nanmean(C.fwd[mk]):6.2f}%（为负 {100*np.mean(C.fwd[mk]<0):3.0f}%）" if mk.sum() else f"  温度 {lo}~{hi} {lab}: 0 天")

    print("\n(c) 单独看更长尺度：零点 a8 之后 63 日 QQQ，随机平移对照，分时段")
    q = C.qall
    f63 = q.shift(-63) / q - 1; f30 = q.shift(-30) / q - 1
    def after(dates, f):
        p = eff_pos(q.index, dates); p = p[p < len(q)]
        return f.values[p]
    ev = DEF["零点 a8"]; pos = np.array([wk.get_loc(d) for d in ev])
    for lab, f in (("30 日", f30), ("63 日", f63)):
        a = np.nanmean(after(ev, f))
        sims = np.array([np.nanmean(after(list(wk[(pos + s) % N]), f)) for s in range(26, N - 26)])
        base = np.nanmean(f[(q.index >= wk[0])].values)
        print(f"  {lab}: 零点后 {100*a:.2f}%，全部日子 {100*base:.2f}%；随机平移中位 {100*np.median(sims):.2f}%，p = {np.mean(sims <= a):.2f}")
    for y0, y1 in (("2016-12-01", "2020-12-31"), ("2021-01-01", "2022-12-31"), ("2023-01-01", "2026-10-31")):
        e = [d for d in ev if pd.Timestamp(y0) <= d <= pd.Timestamp(y1)]
        v = after(e, f63); v = v[np.isfinite(v)]
        b = f63[(q.index >= y0) & (q.index <= y1)].dropna()
        print(f"  {y0[:4]}~{y1[:4]}: {len(v)} 次，63 日 {100*v.mean():6.2f}% vs 同期全部 {100*b.mean():6.2f}%")

    print("\n(d) 2011-2016 样本外（超额修正口径，清洗数据）：零点 a8 之后 QQQ")
    ws = weekly("clean")
    xe = ws.x13_excess
    ae = xe - xe.shift(8)
    ev2 = [d for d in zero_events(xe, ae)[0] if d < pd.Timestamp("2016-12-01")]
    print("  事件：", [str(d.date()) for d in ev2])
    wk2 = pd.date_range(xe.first_valid_index(), pd.Timestamp("2016-11-25"), freq="W-FRI"); N2 = len(wk2)
    pos2 = np.array([wk2.get_loc(d) for d in ev2])
    for lab, f in (("30 日", f30), ("63 日", f63)):
        a = np.nanmean(after(ev2, f))
        sims = np.array([np.nanmean(after(list(wk2[(pos2 + s) % N2]), f)) for s in range(26, N2 - 26)])
        base = np.nanmean(f[(q.index >= wk2[0]) & (q.index <= wk2[-1])].values)
        print(f"  {lab}: 零点后 {100*a:.2f}%（{len(ev2)} 次，为负 {100*np.mean(after(ev2, f) < 0):.0f}%），同期全部日子 {100*base:.2f}%；随机平移中位 {100*np.median(sims):.2f}%，p = {np.mean(sims <= a):.2f}")


def step6():
    """13 周零点「改善」来自哪几天、用户方案的等价门槛、逐顶对照"""
    C = Ctx()
    T0 = C.T.values; idx = C.idx
    w = weekly("rt")
    m0, _ = C.red(T0, BASE_TH)
    print("1) 13 周零点 / 去噪 0.5 确认点的「改善」从哪来（K=30、其余 50、权重 10%）")
    for name, ev in (("零点 a13", zero_events(w.x13, w.a13)[0]), ("确认点 h=0.5", conf_events(w.x13, w.a8, 0.5)), ("零点 a8", zero_events(w.x13, w.a8)[0])):
        F = factor(idx, ev, 30, "hold", 50)
        m, th = C.red(0.9 * T0 + 0.1 * F)
        add = idx[m & ~m0 & C.win]; rem = idx[~m & m0 & C.win]
        f = lambda ds: ", ".join(f"{d:%Y-%m-%d}({100*C.fwd[idx.get_loc(d)]:+.1f}%)" for d in ds)
        print(f"  {name}: 新增 {len(add)} 天 [{f(add)}]")
        print(f"  {' '*len(name)}  去掉 {len(rem)} 天 [{f(rem)}]")

    print("\n2) 用户方案的覆盖面与等价门槛（零点 a8，K 天读 100、其余 0）")
    up8 = zero_events(w.x13, w.a8)[0]
    for K, wt in ((20, 0.10), (30, 0.10), (30, 0.15)):
        F = factor(idx, up8, K, "hold", 0)
        on = (F > 0) & C.win
        m, th = C.red((1 - wt) * T0 + wt * F)
        print(f"  K={K} 权重{int(wt*100)}%：训练窗 {100*on.sum()/C.win.sum():.0f}% 的交易日处在「读 100」；门槛 {th:.1f} ⇒ 原温度上 窗口内 > {(th-100*wt)/(1-wt):.1f}、窗口外 > {th/(1-wt):.1f}"
              f"；红点 {int((m&C.win&(F>0)).sum())} 天在窗口内、{int((m&C.win&(F==0)).sum())} 天在窗口外")

    print("\n3) 每个下跌顶前后的零点（a8，按生效日，交易日计）")
    q = C.qall
    eff = [q.index[p] for p in eff_pos(q.index, up8) if p < len(q)]
    for t in C.tops:
        pt = q.index.get_loc(t)
        before = [e for e in eff if 0 <= pt - q.index.get_loc(e) <= 63]
        after_ = [e for e in eff if 0 < q.index.get_loc(e) - pt <= 63]
        red_hit = m0[max(0, idx.get_loc(t) - 30):idx.get_loc(t) + 6].any()
        b = ", ".join(f"{e:%Y-%m-%d}(顶前{pt - q.index.get_loc(e)}天)" for e in before) or "—"
        a = ", ".join(f"{e:%Y-%m-%d}(顶后{q.index.get_loc(e) - pt}天)" for e in after_) or "—"
        print(f"  {t:%Y-%m-%d}{'(熊市反弹)' if t in C.bear else ''} 红点{'抓到' if red_hit else '没抓到'} | 顶前 63 日内: {b} | 顶后 63 日内: {a}")


def step7():
    """2011-16 长面板样本外并入（_signals_long.csv，同频重标）"""
    C = Ctx()
    L = pd.read_csv(f"{HERE}/策略回测/_signals_long.csv", index_col=0, parse_dates=True)
    # 一致性：长表 2017-10 以后与页面温度
    j = L.temp_red.reindex(C.T.index).dropna()
    print("长表 vs 页面温度（重叠段）最大差 %.2f，平均绝对差 %.3f" % ((j - C.T.reindex(j.index)).abs().max(), (j - C.T.reindex(j.index)).abs().mean()))
    q = C.qall
    ws = weekly("clean")
    xe = ws.x13_excess; ae = xe - xe.shift(8)
    for start in ("2011-07-01", "2013-07-01"):
        W0, W1 = pd.Timestamp(start), pd.Timestamp("2016-11-30")
        S = L.loc[(L.index >= W0 - pd.Timedelta(days=10)) & (L.index <= W1)].copy()
        idx = S.index; T = S.temp_red.values
        fwd = (q.shift(-FWD) / q - 1).reindex(idx).values
        win = np.asarray((idx >= W0) & (idx <= W1)) & np.isfinite(T)
        n_target = int(round(67 / 2249 * win.sum()))
        base = np.nanmean(fwd[win])
        tp = [q.index[i] for i in tops(q) if W0 <= q.index[i] <= W1]
        tpp = [idx.get_loc(t) for t in tp]
        def ev(Tn):
            m3 = min3(Tn)
            v = np.sort(m3[win & np.isfinite(m3)])[::-1]
            th = (v[n_target - 1] + v[n_target]) / 2
            m = (m3 > th) & win
            ok = m & np.isfinite(fwd)
            caught = [t for t, p in zip(tp, tpp) if m[max(0, p - 30):p + 6].any()]
            return 100 * (np.nanmean(fwd[ok]) - base), len(groups(m)), caught, th
        ups = [d for d in zero_events(xe, ae)[0] if W0 - pd.Timedelta(days=60) <= d <= W1]
        wk = pd.date_range(W0, W1, freq="W-FRI"); N = len(wk)
        pos = np.array([wk.get_loc(d) for d in ups if d >= wk[0]])
        b = ev(T)
        print(f"\n样本外 {start}~2016-11（同频 {n_target} 天，基准 30 日 {100*base:.2f}%）；顶：{[str(t.date()) for t in tp]}")
        print(f"  零点（超额修正 a8）：{[str(d.date()) for d in ups]}")
        print(f"  现行（同频重标门槛 {b[3]:.1f}）：边际 {b[0]:.2f}pp，{b[1]} 次，抓到 {[str(t.date()) for t in b[2]]}")
        for cfg in (("hold", 0, 20, 0.10), ("hold", 0, 30, 0.10), ("hold", 0, 30, 0.15), ("hold", 50, 30, 0.10), ("hold", 50, 63, 0.05)):
            mode, off, K, wt = cfg
            F = factor(idx, ups, K, mode, off)
            r = ev((1 - wt) * T + wt * F)
            sims = []
            for s in range(26, N - 26, 2):
                F2 = factor(idx, list(wk[(pos + s) % N]), K, mode, off)
                sims.append(ev((1 - wt) * T + wt * F2)[0])
            sims = np.array(sims)
            print(f"  {cfg}: 边际 {r[0]:.2f}pp，{r[1]} 次，抓到 {[str(t.date()) for t in r[2]]}；随机平移中位 {np.median(sims):.2f}pp，p = {np.mean(sims <= r[0]):.2f}")


def step8():
    """顺带：估值抬升（13 周远期市盈率变化分位）当因子"""
    C = Ctx()
    T0 = C.T.values; idx = C.idx
    ws = weekly("clean")
    # 13 周估值变化 = 13 周对数涨跌 − 13 周远期 EPS 对数变化（恒等式里的「估值」那块，标普口径）
    v13 = (ws.p13 - ws.r13).dropna()
    p = eff_pos(idx, v13.index)
    V = pd.Series(np.nan, index=idx)
    for t, pp in zip(v13.index, p):
        if pp < len(idx): V.iloc[pp:] = v13[t]
    # 252 日滚动分位（与红点其它因子同口径）
    Vp = V.rolling(252, min_periods=200).apply(lambda a: 100 * (a[:-1] < a[-1]).mean() + 50 * (a[:-1] == a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    ok = C.win & np.isfinite(C.fwd) & np.isfinite(Vp.values)
    from scipy.stats import spearmanr
    print("估值 13 周变化分位 与 之后 30 日 QQQ 的秩相关（训练窗）：%.3f；与红点温度相关 %.2f" % (spearmanr(Vp.values[ok], C.fwd[ok])[0], np.corrcoef(Vp.values[ok], T0[ok])[0, 1]))
    for lo, hi in ((0, 20), (20, 50), (50, 80), (80, 90), (90, 101)):
        mk = ok & (Vp.values >= lo) & (Vp.values < hi)
        print(f"  分位 {lo:>2}~{hi:<3}: {mk.sum():4d} 天，之后 30 日 {100*np.nanmean(C.fwd[mk]):5.2f}%")
    print("下跌顶前 30 个交易日内估值分位最高 / 原始 13 周估值变化：")
    for t, pt in zip(C.tops, C.top_pos):
        seg = Vp.values[max(0, pt - 30):pt + 1]; raw = V.values[max(0, pt - 30):pt + 1]
        hit = C.red(T0, BASE_TH)[0][max(0, pt - 30):pt + 6].any()
        print(f"  {t:%Y-%m-%d}{'(熊市反弹)' if t in C.bear else '          '} 红点{'抓到' if hit else '没抓到'}  分位最高 {np.nanmax(seg):5.1f}  估值 13 周 {100*np.nanmax(raw):+5.1f}%")
    base_m = C.evaluate(C.red(T0, BASE_TH)[0])
    wk = v13.index; N = len(wk)
    for wt in (0.10, 0.15, 0.20):
        Tn = (1 - wt) * T0 + wt * np.nan_to_num(Vp.values, nan=50)
        m, th = C.red(Tn); r = C.evaluate(m)
        # 安慰剂：整条周度序列循环平移
        sims = []
        for s in range(26, N - 26, 3):
            vs = pd.Series(np.roll(v13.values, s), index=wk)
            Vs = pd.Series(np.nan, index=idx)
            for t, pp in zip(wk, p):
                if pp < len(idx): Vs.iloc[pp:] = vs[t]
            Vsp = Vs.rolling(252, min_periods=200).apply(lambda a: 100 * (a[:-1] < a[-1]).mean() + 50 * (a[:-1] == a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
            ms, _ = C.red((1 - wt) * T0 + wt * np.nan_to_num(Vsp.values, nan=50))
            sims.append(C.evaluate(ms)["mg"])
        sims = np.array(sims)
        print(f"并入估值分位 权重{int(wt*100)}%：门槛 {th:.1f}，边际 {r['mg']:.2f}pp（现行 {base_m['mg']:.2f}），抓顶 {len([x for x in r['caught'] if x not in C.bear])}/7 "
              f"{[str(x.date()) for x in r['caught']]}；随机平移中位 {np.median(sims):.2f}pp，p = {np.mean(sims <= r['mg']):.2f}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or range(1, 9)):
        print(f"\n========== {n}. " + globals()[f"step{n}"].__doc__ + " ==========")
        globals()[f"step{n}"]()
