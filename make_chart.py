#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 修正速度与纳指.html：
  上半：纳指（综合 / 纳斯达克100）+ 超额修正（速度）+ 超额修正的二阶导（加速度），三图共用时间轴、缩放、十字线联动；
        纳指底色 = 去噪后的四个阶段，标出实时拐点。
  下半：速度拐点 × 纳指（speed.py）：四阶段之后的纳指表现、拐点前后平均走势、事件清单；窗口 N × 去噪阈值 h 共 12 组全部内嵌，页面上切换。
单文件、离线可开（ECharts 5.6.0 内嵌，vendor/echarts.min.js，Apache-2.0；对数轴自定义刻度要 ≥5.6）。

二阶导 = 超额修正本周值 − N 周前的值（N = 4 / 8 / 13，页面上切换）。超额修正本身已是远期 EPS 的变化率（一阶），
它的变化就是 EPS 的加速度：正 = 上修在加快（或下修在变缓），负 = 上修在减速（或下修在加剧）。

先跑 analyze.py（生成 _weekly_signals.csv、_analysis.json），再跑本脚本：
    python3 make_chart.py            # 纳指日线超过 1 天没更新会重新拉 Yahoo（走代理 15236）
"""
import json, os, sys, time
import numpy as np
import pandas as pd

from analyze import nw_t, drop_partial
import speed

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "修正速度与纳指.html")
NS = (4, 8, 13)


def nasdaq():
    path = os.path.join(BASE, "_nasdaq.csv")
    # 缓存 3 小时内有效；REV_FORCE_FETCH=1（GitHub Actions）时每次重下，下载失败才退回缓存
    fresh = (os.path.exists(path) and time.time() - os.path.getmtime(path) < 3 * 3600
             and not os.environ.get("REV_FORCE_FETCH"))
    if not fresh:
        if not os.environ.get("CI"):  # 本地走 Veee；GitHub Actions 上直连
            os.environ.setdefault("https_proxy", "http://127.0.0.1:15236")
        try:
            import yfinance as yf
            cols = {}
            for s in ("^IXIC", "^NDX"):
                h = yf.Ticker(s).history(start="2011-01-01", auto_adjust=False)["Close"]
                if len(h) < 2000:  # 被限流时 yfinance 常返回空表而不报错，不能拿它覆盖缓存
                    raise RuntimeError(f"{s} 只拿到 {len(h)} 行")
                h.index = h.index.tz_localize(None).normalize()
                cols[s] = h
            pd.DataFrame(cols).to_csv(path)
        except Exception as e:
            if not os.path.exists(path):
                raise
            print(f"PRICE_FALLBACK 纳指下载失败，改用缓存 _nasdaq.csv：{e}", file=sys.stderr)
    return drop_partial(pd.read_csv(path, index_col=0, parse_dates=True))


def runs(s):
    """s > 0 的连续区段 → [[起, 止], ...]"""
    out, start, prev = [], None, None
    for d, v in s.dropna().items():
        if v > 0 and start is None:
            start = d
        elif v <= 0 and start is not None:
            out.append([start.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")])
            start = None
        prev = d
    if start is not None:
        out.append([start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")])
    return out


def nasdaq_tests(w, px):
    """和纳指综合比：谁领先（错开 k 周的相关最高点）、对未来 4/13/26 周涨跌的预测力（Newey-West t）。"""
    g = w.index
    daily = px.reindex(pd.date_range(px.index.min(), px.index.max())).ffill()
    p = daily.reindex(g)
    lp = np.log(p)
    out = {}
    for sig in ["x13"] + [f"a{n}" for n in NS]:
        ll = {}
        for k in range(-20, 21, 2):
            b = lp.diff(13).shift(-k)
            ok = w[sig].notna() & b.notna()
            ll[k] = float(np.corrcoef(w[sig][ok], b[ok])[0, 1])
        peak = max(ll, key=ll.get)
        ts = {}
        for h in (4, 13, 26):
            fut = daily.reindex(g + pd.Timedelta(weeks=h)).values
            f = pd.Series(np.log(fut / p.values), index=g)
            f[g + pd.Timedelta(weeks=h) > daily.index.max()] = np.nan
            s = pd.DataFrame({"x": w[sig], "f": f}).dropna()
            z = (s.x - s.x.mean()) / s.x.std()
            ts[h] = round(float(nw_t(s.f.values, z.values, h)[1][1]), 2)
        out[sig] = {"peak_k": peak, "peak_corr": round(ll[peak], 2), "t": ts}
    return out


HS = (0.0, 0.5, 1.0, 1.5, 2.0)  # 0 = 不去噪：状态翻转就是二阶导穿过 0 的那一周
CODE = {"上修加速": "A", "上修减速": "B", "下修加速": "C", "下修放缓": "D"}


def speed_data(weeks):
    """两种口径 × 每组 (N, h) 跑一遍 speed.run：阶段串（对齐 weekly 行）、阶段表、拐点事件与前后路径。"""
    out = {}
    for src in ("rt", "clean"):
      for n in NS:
        for h in HS:
            r = speed.run(n, h, verbose=(src == "clean" and n == 8 and h == 1.0), src=src)
            w, _ = speed.load(n, h, src)
            ph = w.phase.reindex(weeks)
            out[f"{src}|{n}|{h}"] = {
                "phase": "".join(CODE.get(v, "-") if isinstance(v, str) else "-" for v in ph),
                "phases": r["phases"], "current": r["current"], "flips": r["flips_per_year"],
                "rt": {k: {"events": v["events"], "summary": v["summary"], "mean": v["path"]["mean"], "each": v["path"]["each"]}
                       for k, v in r["realtime"].items()},
                "ex": {k: {"events": v["events"], "summary": v["summary"], "mean": v["path"]["mean"], "each": v["path"]["each"]}
                       for k, v in r["expost"].items()},
            }
    return out


def main():
    w = pd.read_csv(os.path.join(BASE, "_weekly_signals.csv"), index_col=0, parse_dates=True)
    nq = nasdaq()
    for n in NS:
        w[f"a{n}"] = w.x13.diff(n)  # 在完整周网格上差分，缺周处自然是空值

    wk = w[w.x13.notna()]
    pct = lambda v: None if pd.isna(v) else round(100 * float(v), 2)
    rt = pd.read_csv(os.path.join(BASE, "_weekly_rt.csv"), index_col=0, parse_dates=True).reindex(wk.index)
    # 每行：日期, x13, r13, eps, a4, a8, a13（清洗口径） | x13, a4, a8, a13（实时口径）, 待核实提示
    weekly = [[d.strftime("%Y-%m-%d"), pct(r.x13), pct(r.r13), round(r.eps, 2)] + [pct(r[f"a{n}"]) for n in NS]
              + [pct(rt.x13[d])] + [pct(rt[f"a{n}"][d]) for n in NS]
              + [rt.flag[d] if isinstance(rt.flag[d], str) else None]
              for d, r in wk.iterrows()]
    bands = {"x13": runs(wk.x13), **{f"a{n}": runs(wk[f"a{n}"]) for n in NS}}

    px = {k: [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in nq[k].dropna().items()]
          for k in ("^IXIC", "^NDX")}
    last = wk.iloc[-1]
    data = {"weekly": weekly, "bands": bands, "px": px, "ns": list(NS),
            "latest": {"date": wk.index[-1].strftime("%Y-%m-%d"), "x13": pct(last.x13),
                       "pctile": round(100 * float((wk.x13 < last.x13).mean())), "r13": pct(last.r13),
                       "eps": round(float(last.eps), 2), **{f"a{n}": pct(last[f"a{n}"]) for n in NS}},
            "tests": nasdaq_tests(wk, nq["^IXIC"].dropna())}

    data["speed"] = speed_data(wk.index)
    fw = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"])
    data["fresh"] = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                     "factset": fw.date.max().strftime("%Y-%m-%d"),
                     "nasdaq": nq["^IXIC"].dropna().index.max().strftime("%Y-%m-%d")}

    echarts = open(os.path.join(BASE, "vendor", "echarts.min.js")).read()
    html = TEMPLATE.replace("/*ECHARTS*/", echarts).replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    open(OUT, "w").write(html)
    print(OUT, f"{len(html) / 1e6:.1f} MB, weekly {len(weekly)}")
    print(json.dumps(data["tests"], ensure_ascii=False))


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>修正速度与纳指</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ee; --border: #e6e5e0;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8984;
  --pos: #2a78d6; --neg: #eb6834; --price: #3d3c39; --band: rgba(42,120,214,0.10);
  --pA: rgba(42,120,214,0.22); --pB: rgba(42,120,214,0.08); --pC: rgba(235,104,52,0.22); --pD: rgba(235,104,52,0.09);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242422; --border: #34332f;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --pos: #3987e5; --neg: #d95926; --price: #e4e3dc; --band: rgba(57,135,229,0.16);
    --pA: rgba(57,135,229,0.30); --pB: rgba(57,135,229,0.12); --pC: rgba(217,89,38,0.30); --pD: rgba(217,89,38,0.13);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242422; --border: #34332f;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --pos: #3987e5; --neg: #d95926; --price: #e4e3dc; --band: rgba(57,135,229,0.16);
  --pA: rgba(57,135,229,0.30); --pB: rgba(57,135,229,0.12); --pC: rgba(217,89,38,0.30); --pD: rgba(217,89,38,0.13);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.5 -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 16px 28px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 34px 0 4px; }
h3 { font-size: 14px; font-weight: 600; margin: 18px 0 6px; color: var(--ink-2); }
.sub { color: var(--ink-2); font-size: 13px; margin: 0 0 14px; }
.fresh { color: var(--ink-3); font-size: 12px; margin: -8px 0 12px; }
.fresh.stale { color: var(--ink); background: var(--surface-2); border-left: 3px solid var(--neg); padding: 8px 12px; border-radius: 6px; margin: 0 0 14px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-bottom: 14px; }
.tile { background: var(--surface-2); border-radius: 10px; padding: 10px 12px; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .n { color: var(--ink-3); font-size: 12px; }
.bar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 6px; }
.grp { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-3); font-size: 12px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { background: transparent; color: var(--ink-2); border: 0; padding: 5px 10px; font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
.grp { flex-wrap: wrap; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--surface-2); color: var(--ink); font-weight: 600; }
label.chk { color: var(--ink-2); font-size: 13px; display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin: 4px 0 0; min-height: 18px; }
.legend i { display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.legend .mk { font-style: normal; font-size: 11px; margin-right: 4px; }
#chart { width: 100%; height: 800px; }
#evchart { width: 100%; height: 380px; }
.tbl-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
td .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
td.muted, .muted { color: var(--ink-3); }
tr.cur td { font-weight: 600; }
.note { color: var(--ink-2); font-size: 13px; background: var(--surface-2); border-radius: 10px; padding: 10px 14px; margin: 10px 0; }
.note p { margin: 4px 0; }
.foot { color: var(--ink-3); font-size: 12px; margin-top: 8px; }
.foot p { margin: 3px 0; }
@media (max-width: 640px) { #chart { height: 700px; } #evchart { height: 320px; } .tile .v { font-size: 19px; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>修正速度与纳指</h1>
  <p class="sub">上：纳指（底色 = 修正所处阶段）；中：标普500 远期 EPS 的超额修正（一阶，速度）；下：超额修正的 N 周变化（二阶导，加速度）。拖动或滚轮缩放，三图联动。</p>

  <p class="fresh" id="fresh"></p>
  <div class="tiles" id="tiles"></div>

  <div class="bar">
    <span class="grp">数据口径
      <span class="seg" id="src">
        <button data-v="rt" aria-pressed="true">实时（当时可知）</button><button data-v="clean" aria-pressed="false">清洗（研究用）</button>
      </span>
    </span>
    <span class="seg" id="idx">
      <button data-v="^IXIC" aria-pressed="true">纳斯达克综合</button>
      <button data-v="^NDX" aria-pressed="false">纳斯达克100</button>
    </span>
    <span class="seg" id="scale">
      <button data-v="log" aria-pressed="true">对数</button>
      <button data-v="value" aria-pressed="false">线性</button>
    </span>
    <span class="grp">二阶导窗口
      <span class="seg" id="win">
        <button data-v="4" aria-pressed="false">4周</button><button data-v="8" aria-pressed="true">8周</button><button data-v="13" aria-pressed="false">13周</button>
      </span>
    </span>
    <span class="grp">去噪阈值（0 = 不去噪）
      <span class="seg" id="hys">
        <button data-v="0" aria-pressed="false">0</button><button data-v="0.5" aria-pressed="false">0.5</button><button data-v="1" aria-pressed="true">1</button><button data-v="1.5" aria-pressed="false">1.5</button><button data-v="2" aria-pressed="false">2</button>
      </span>
    </span>
  </div>
  <div class="bar">
    <span class="grp">纳指底色
      <span class="seg" id="band">
        <button data-v="phase" aria-pressed="true">四阶段</button><button data-v="acc" aria-pressed="false">加速期</button><button data-v="x13" aria-pressed="false">上修期</button><button data-v="none" aria-pressed="false">无</button>
      </span>
    </span>
    <span class="grp">速度见顶标记
      <span class="seg" id="marks">
        <button data-v="swing" aria-pressed="true">大波段</button><button data-v="zero" aria-pressed="false">零点</button><button data-v="conf" aria-pressed="false">确认点</button><button data-v="ex" aria-pressed="false">事后真实点</button><button data-v="none" aria-pressed="false">无</button>
      </span>
    </span>
    <span class="seg" id="range">
      <button data-v="1">1年</button><button data-v="3">3年</button><button data-v="5">5年</button>
      <button data-v="10">10年</button><button data-v="0" aria-pressed="true">全部</button>
    </span>
  </div>
  <div class="legend" id="legend"></div>

  <div id="chart"></div>

  <h2>速度拐点 × 纳指</h2>
  <p class="sub" id="secsub"></p>
  <div class="note" id="takeaway"></div>

  <h3>四个阶段之后，纳指的表现</h3>
  <div class="tbl-wrap"><table id="phtbl"></table></div>
  <p class="foot">「26 周 t」= 该阶段相对全部周的超额，Newey-West 调整重叠收益；|t| ≥ 2 才算显著。阶段按周归属，同一段行情会连续贡献很多周，独立样本远少于周数。</p>

  <h3>拐点前后纳指的平均走势（事件周 = 0）</h3>
  <div class="bar">
    <span class="seg" id="evmode">
      <button data-v="sw" aria-pressed="true">大波段确认</button><button data-v="rt" aria-pressed="false">实时拐点（按上面的阈值）</button><button data-v="ex" aria-pressed="false">事后拐点（速度真正的高低点）</button>
    </span>
    <label class="chk"><input type="checkbox" id="each" checked> 显示每个事件</label>
  </div>
  <div id="evchart"></div>

  <h3>事件清单</h3>
  <div class="tbl-wrap"><table id="evtbl"></table></div>

  <div class="foot">
    <p>超额修正 = 远期 12 个月 EPS 的 13 周变化 − 它此前 3 年（156 周）的均值，扣掉「时间往前滚」的平均机械增长；需要 3 年历史，所以从 2013 年中起。
       二阶导 = 本周超额修正 − N 周前的超额修正。</p>
    <p><b>两种数据口径</b>：<b>实时</b>（默认）= 每周五拿到周报时能算出的值。远期 EPS 和上一个已确认值相比单周变动 > 2% 先标「待核实」（中图 ◇），
       那周不更新信号；下一个有数据的周离原值更近判笔误作废，离可疑值更近判真实变动并补认（晚一周）；周报没给远期市盈率的周同样不更新。
       2011 年以来共拦下 20 周：5 次笔误（2011 年 3 次、2021-06-17、2022-06-17）全部作废，15 次真实大幅变动（2018 年减税、2020 年疫情、2026 年强势上修等）晚一周补认。
       <b>清洗</b> = 研究用：笔误按前后各 2 期中位数剔除、缺周线性插值，用到了之后的数据，历史标记比当时看到的略干净。</p>
    <p><b>大波段（默认的标记和事件）</b>：只标快速抬升 / 快速下跌之后的速度拐点。超额修正从最近一个低点涨了 ≥ 4 个百分点、顶部 ≥ +2%，
       之后从顶部回落满 2 个百分点 → 确认「上修速度见顶」（蓝 ▼）；跌了 ≥ 4 个百分点、谷底 ≤ −2%，之后反弹满 2 个百分点 → 确认「下修速度见顶」（橙 ▲，下修最猛的时候过去了）。
       2 取的是略高于单周噪声的 95% 分位（1.5），4 约等于超额修正一个标准差（4.4），按噪声定、没按收益挑。确认时就知道这一段真正的顶 / 底在哪周（中图圆圈）。
       2013 年以来上修 / 下修各 5 次（原来按二阶导确认是 12 / 8 次），确认滞后中位数 7 周。代价：高位平台上的第二个驼峰不算「快速抬升」，
       比如 2021-07-23（9.56%，距前一个低点只涨了 3.9 个百分点）不会标。</p>
    <p><b>三种「拐点」</b>（以 2026 年这次上修见顶为例）：
       ① 速度真实高点 = 超额修正真正的峰值，严格意义上二阶导 = 0 的点（2026-06-12，9.33%）；要等之后几周回落才能确认，事后才知道。
       ② 二阶导零点 = 页面上的二阶导由正转负（2026-07-03）。二阶导用的是「本周 − N 周前」，要等当前值跌回 N 周前的水平才到 0，
       所以比 ① 晚，窗口越长越晚（第一次穿 0 的历史中位数：4 周窗口晚 2 周、8 周晚 3 周、13 周晚 4 周）；
       代价是窗口越短翻得越勤（每年约 12.6 / 9.7 / 6.9 次），假信号多（如 2026-01-23 转负后，速度 2 月又创新高）。
       ③ 确认点 = 去噪后的状态翻转：二阶导 > +阈值 才算转入加快、< −阈值 才算转入放慢（下图虚线），再晚几周
       （2026-07-24；阈值 1 时中位数比 ① 晚：4 周窗口 4 周、8 周 7 周、13 周 9 周），但一年只翻 3 次左右。
       阈值选 0 时 ③ 就等于 ②。图上每次穿 0 / 每次翻转都标出来；下方事件研究里同类事件至少相隔 13 周（去重）。
       事后高低点 = 前后 13 周内的最高 / 最低、突出度 ≥ 2 个百分点。</p>
    <p>EPS 是<b>标普500</b>的（FactSet《Earnings Insight》周报，远期 EPS = 收盘价 ÷ 远期市盈率），没有纳指自己的历史一致预期；纳指收盘价来自 Yahoo；拐点统计用纳斯达克综合。</p>
  </div>
</div>

<script>/*ECHARTS*/</script>
<script>
const D = /*DATA*/;
const $ = s => document.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 1) => v == null ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const day = t => new Date(t).toISOString().slice(0, 10);
const state = { src: "rt", idx: "^IXIC", scale: "log", win: 8, h: 1, band: "phase", marks: "swing", evmode: "sw", each: true, ticks: [] };
const NAME = { "^IXIC": "纳斯达克综合", "^NDX": "纳斯达克100" };
// weekly 行里的列：清洗口径 x13 在 1、二阶导在 4/5/6；实时口径 x13 在 7、二阶导在 8/9/10；待核实提示在 11
const COL = { clean: { x: 1, a: { 4: 4, 8: 5, 13: 6 } }, rt: { x: 7, a: { 4: 8, 8: 9, 13: 10 } } }, FLAG = 11;
const XI = () => COL[state.src].x, AC = () => COL[state.src].a[state.win];
const PH = { A: "上修加速", B: "上修减速", C: "下修加速", D: "下修放缓" };
const PHC = { "上修加速": "A", "上修减速": "B", "下修加速": "C", "下修放缓": "D" };
const key = () => `${state.src}|${state.win}|${state.h.toFixed(1)}`;
const S = () => D.speed[key()];

const tsOf = s => Date.parse(s + "T00:00:00Z");
const WK = D.weekly.map(r => [tsOf(r[0]), ...r.slice(1)]);
const X0 = WK[0][0] - 7 * 864e5;
const PX = {}; for (const k in D.px) PX[k] = D.px[k].map(r => [tsOf(r[0]), r[1]]).filter(r => r[0] >= X0);
const X1 = Math.max(WK[WK.length - 1][0], ...Object.values(PX).map(a => a[a.length - 1][0])) + 3 * 864e5;
function before(arr, t) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][0] <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
const pxAt = t => { const i = before(PX[state.idx], t); return i >= 0 ? PX[state.idx][i][1] : null; };

function tiles() {
  const L = D.latest, p = PX[state.idx], last = p[p.length - 1], cur = S().current;
  const i13 = before(p, last[0] - 91 * 864e5), ch13 = 100 * (last[1] / p[i13][1] - 1);
  let li = WK.length - 1; while (li > 0 && WK[li][XI()] == null) li--;
  const lx = WK[li][XI()], a = WK[li][AC()], xs = WK.map(r => r[XI()]).filter(v => v != null);
  const pct = Math.round(100 * xs.filter(v => v < lx).length / xs.length);
  const pend = state.src === "rt" && WK[WK.length - 1][FLAG] && !WK[WK.length - 1][FLAG].includes("；");
  const t = [
    ["超额修正（速度）", fmt(lx), `${day(WK[li][0])}，${D.weekly[0][0].slice(0, 4)} 年以来 ${pct}% 分位` + (pend ? `；最新一周待核实` : "")],
    [`二阶导（${state.win} 周，加速度）`, fmt(a), a == null ? "" : a > 0 ? "上修在加快" : "上修在放慢（或下修加剧）"],
    [`当前阶段（${state.src === "rt" ? "实时" : "清洗"}口径${state.h > 0 ? "、去噪后" : "、不去噪"}）`, cur.phase, `自 ${cur.since} 起`],
    [`${NAME[state.idx]} 13 周涨跌`, fmt(ch13), `${day(last[0])} 收 ${last[1].toLocaleString()}`],
  ];
  $("#tiles").innerHTML = t.map(([k, v, n]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`).join("");
}

function legend() {
  const sw = c => `<i style="background:${css(c)}"></i>`;
  let h = "";
  if (state.band === "phase") h += ["A", "B", "C", "D"].map(c => `<span>${sw("--p" + c)}${PH[c]}</span>`).join("");
  else if (state.band === "acc") h += `<span>${sw("--band")}二阶导去噪后为正（加快）</span>`;
  else if (state.band === "x13") h += `<span>${sw("--band")}超额修正为正（上修期）</span>`;
  if (state.marks !== "none") { const [a, b] = MARK_TXT[state.marks];
    h += `<span><b class="mk" style="color:${css("--pos")}">▼</b>${a}</span><span><b class="mk" style="color:${css("--neg")}">▲</b>${b}</span>`; }
  if (state.marks === "zero" && state.h > 0) h += `<span class="muted">（零点固定按不去噪算；底色仍按阈值 ${state.h}）</span>`;
  if (state.marks === "swing") { const Z = swings();
    h += `<span class="muted">中图圆圈 = 这一段真正的顶 / 底；当前处在${Z.mode === "down" ? "回落段" : "上升段"}，段内${Z.mode === "down" ? "最低" : "最高"} ${Z.ext.v.toFixed(2)}%（${day(Z.ext.t)}）</span>`; }
  if (state.src === "rt") h += `<span><b class="mk" style="color:${css("--ink")}">◇</b>待核实周（中图；远期 EPS 单周变动 > 2%，下一个有数据的周核实）</span>`;
  $("#legend").innerHTML = h;
}

// 阶段串 → 连续区段
function phaseRuns(test) {
  const ph = S().phase, out = [];
  let st = null, code = null;
  for (let i = 0; i <= WK.length; i++) {
    const c = i < WK.length ? test(ph[i]) : null;
    if (c !== code) {
      if (code) out.push([st, i < WK.length ? WK[i][0] : WK[WK.length - 1][0] + 7 * 864e5, code]);
      st = i < WK.length ? WK[i][0] : null; code = c;
    }
  }
  return out;
}
function bandData() {
  if (state.band === "none") return [];
  let runs;
  if (state.band === "phase") runs = phaseRuns(c => (c && c !== "-") ? c : null).map(r => [r[0], r[1], css("--p" + r[2])]);
  else if (state.band === "acc") runs = phaseRuns(c => (c === "A" || c === "D") ? "y" : null).map(r => [r[0], r[1], css("--band")]);
  else runs = WK.reduce((acc, r, i) => {
      const on = r[XI()] > 0, last = acc[acc.length - 1];
      if (on && (!last || last.closed)) acc.push({ s: r[0], e: r[0] + 7 * 864e5 });
      else if (on) last.e = r[0] + 7 * 864e5;
      else if (last) last.closed = true;
      return acc; }, []).map(b => [b.s, b.e, css("--band")]);
  return runs.map(([a, b, c]) => [{ xAxis: a, itemStyle: { color: c } }, { xAxis: b }]);
}
// 只标「速度见顶」——按修正的猛烈程度理解：
//   上修速度见顶 = 超额修正为正、二阶导由正转负（上修最快的时候过去了）→ 蓝色 ▼，标在价格上方
//   下修速度见顶 = 超额修正为负、二阶导由负转正（下修最猛的时候过去了，即超额修正的谷底）→ 橙色 ▲，标在价格下方
// 不标「速度见底」：上修中重新加快、下修中重新加剧。
// 三种口径：零点 = 二阶导穿过 0（不去噪）；确认点 = 去噪后状态翻转；事后 = 超额修正真正的局部高点（正）/ 低点（负）
// 图上每一次都标出来；下方事件研究另按「同类至少隔 13 周」去重
// 大波段（折返）：超额修正从最近一个低点涨了 ≥ A、且顶部 ≥ +L% 之后，从顶部回落满 R → 确认「上修速度见顶」；
// 跌了 ≥ A、且谷底 ≤ −L% 之后，从谷底反弹满 R → 确认「下修速度见顶」（下修最猛的时候过去了）。
// R 取 2：略高于单周噪声的 95% 分位（1.5 个百分点）；A 取 4：约一个标准差（4.4）。按噪声定，不按收益挑。
// 只用当时已有的读数（待核实 / 缺周跳过）；确认那周就知道顶 / 底在哪周（回落前的极值）。
const SW = { R: 2, A: 4, L: 2 };
function swings() {
  const col = XI(), out = [];
  let mode = null, ext = null, last = null, hi = null, lo = null;
  for (let i = 0; i < WK.length; i++) {
    const v = WK[i][col], t = WK[i][0];
    if (v == null) continue;
    const cur = { t, v };
    if (mode === null) {
      if (!hi || v > hi.v) hi = cur;
      if (!lo || v < lo.v) lo = cur;
      if (v <= hi.v - SW.R) { out.push({ type: "top", p: hi, c: cur, amp: NaN }); last = hi.v; mode = "down"; ext = cur; }
      else if (v >= lo.v + SW.R) { out.push({ type: "bot", p: lo, c: cur, amp: NaN }); last = lo.v; mode = "up"; ext = cur; }
      continue;
    }
    if (mode === "up") {
      if (v > ext.v) ext = cur;
      else if (v <= ext.v - SW.R) { out.push({ type: "top", p: ext, c: cur, amp: ext.v - last }); last = ext.v; mode = "down"; ext = cur; }
    } else {
      if (v < ext.v) ext = cur;
      else if (v >= ext.v + SW.R) { out.push({ type: "bot", p: ext, c: cur, amp: last - ext.v }); last = ext.v; mode = "up"; ext = cur; }
    }
  }
  return { tops: out.filter(e => e.type === "top" && e.p.v >= SW.L && e.amp >= SW.A),
           bots: out.filter(e => e.type === "bot" && e.p.v <= -SW.L && e.amp >= SW.A), mode, ext };
}

function markSets() {
  if (state.marks === "none") return null;
  const pt = (t, x13) => ({ t, px: pxAt(t), x13, date: day(t) });
  let up = [], dn = [];
  if (state.marks === "swing") {
    const Z = swings();
    return { up: Z.tops.map(e => pt(e.c.t, e.c.v)).filter(p => p.px != null),
             dn: Z.bots.map(e => pt(e.c.t, e.c.v)).filter(p => p.px != null),
             piv: [...Z.tops, ...Z.bots].map(e => [e.p.t, e.p.v]) };
  }
  if (state.marks === "ex") {
    up = S().ex["修正速度高点"].events.filter(e => e.x13 > 0).map(e => pt(tsOf(e.date), e.x13));
    dn = S().ex["修正速度低点"].events.filter(e => e.x13 < 0).map(e => pt(tsOf(e.date), e.x13));
  } else {
    const sign = state.marks === "zero"
      ? i => { const v = WK[i][AC()]; return v == null ? 0 : Math.sign(v); }
      : i => { const c = S().phase[i]; return c === "A" || c === "D" ? 1 : c === "B" || c === "C" ? -1 : 0; };
    let prev = 0;
    for (let i = 0; i < WK.length; i++) {
      const s = sign(i), x = WK[i][XI()];
      if (s < 0 && prev > 0 && x > 0) up.push(pt(WK[i][0], x));
      if (s > 0 && prev < 0 && x < 0) dn.push(pt(WK[i][0], x));
      if (s !== 0) prev = s;
    }
  }
  return { up: up.filter(p => p.px != null), dn: dn.filter(p => p.px != null) };
}
const MARK_TXT = {
  swing: ["上修速度见顶确认（大波段：涨 ≥ 4 个百分点、顶 ≥ +2% 后回落 2 个百分点）", "下修速度见顶确认（大波段：跌 ≥ 4 个百分点、底 ≤ −2% 后反弹 2 个百分点）"],
  zero: ["上修速度见顶（超额修正 > 0，二阶导由正转负）", "下修速度见顶（超额修正 < 0，二阶导由负转正）"],
  conf: ["上修速度见顶确认（超额修正 > 0，二阶导跌破 −阈值）", "下修速度见顶确认（超额修正 < 0，二阶导升破 +阈值）"],
  ex: ["上修速度真实高点（超额修正峰值，事后）", "下修速度真实高点（超额修正谷底，事后）"],
};

const chart = echarts.init($("#chart"), null, { renderer: "canvas" });
const evchart = echarts.init($("#evchart"), null, { renderer: "canvas" });

function bars(col, pos, neg) {
  return WK.filter(r => r[col] != null).map(r => ({ value: [r[0], r[col]], itemStyle: { color: r[col] > 0 ? pos : neg } }));
}

function option() {
  const ink = css("--ink"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), border = css("--border");
  const pos = css("--pos"), neg = css("--neg");
  const name = NAME[state.idx];
  const M = markSets();
  const mUp = M ? M.up.map(p => [p.t, p.px, p.date]) : [], mDn = M ? M.dn.map(p => [p.t, p.px, p.date]) : [];
  // 事后高 / 低点同时标在「速度」那张图上，能直接看到是超额修正的峰 / 谷
  const sTop = M && M.piv ? M.piv : M && state.marks === "ex" ? [...M.up, ...M.dn].map(p => [p.t, p.x13]) : [];
  const ax = { axisLine: { lineStyle: { color: border } }, axisTick: { show: false },
    axisLabel: { color: ink2, fontSize: 11 }, splitLine: { lineStyle: { color: border } } };
  const G = [{ top: 26, height: "40%" }, { top: "52%", height: "16%" }, { top: "75%", height: "16%" }]
    .map(g => ({ left: 64, right: 24, ...g }));
  const xa = i => ({ type: "time", gridIndex: i, min: X0, max: X1, ...ax, splitLine: { show: false },
    axisLabel: i < 2 ? { show: false } : { color: ink2, fontSize: 11 } });
  const pctAxis = (i, cap) => ({ type: "value", gridIndex: i, ...ax, splitNumber: 3,
    ...(cap ? { min: v => Math.max(Math.floor(v.min), -cap), max: v => Math.min(Math.ceil(v.max), cap) } : {}),
    axisLabel: { color: ink2, fontSize: 11, formatter: v => v + "%" } });
  // 二阶导全图会被 2020 年的 ±20~30% 尖峰压扁：纵轴截在 95 分位附近（放大到局部时按可见数据自适应）
  const av = WK.map(r => r[AC()]).filter(v => v != null).map(Math.abs).sort((a, b) => a - b);
  const cap = Math.ceil(av[Math.floor(av.length * 0.95)] / 2) * 2;
  const line = (y, dash) => ({ yAxis: y, lineStyle: { color: ink3, width: 1, type: dash ? "dashed" : "solid" } });
  const label = (text, top) => ({ text, left: 64, top, textStyle: { color: ink2, fontSize: 12, fontWeight: 500 } });
  const mark = (data, down, color) => ({ type: "scatter", xAxisIndex: 0, yAxisIndex: 0, data, symbol: "triangle", symbolRotate: down ? 180 : 0,
    symbolSize: 10, symbolOffset: [0, down ? -10 : 10], z: 5, tooltip: { show: false },
    itemStyle: { color, borderColor: css("--surface"), borderWidth: 1 } });
  return {
    animation: false,
    backgroundColor: "transparent",
    title: [label(name, 4), label("超额修正（速度，%）", "48.5%"),
            label(`二阶导：超额修正 ${state.win} 周变化（加速度，%；${state.h > 0 ? `虚线 = ±${state.h} 去噪阈值；` : ""}全图纵轴截在 ±${cap}%）`, "71.5%")],
    grid: G,
    axisPointer: { link: [{ xAxisIndex: "all" }], label: { backgroundColor: ink2 } },
    tooltip: {
      trigger: "axis", axisPointer: { type: "line", lineStyle: { color: ink3 } },
      backgroundColor: css("--surface"), borderColor: border, textStyle: { color: ink, fontSize: 12 },
      formatter: ps => {
        const t = ps[0].axisValue, iw = before(WK, t), px = pxAt(t);
        let h = `<b>${day(t)}</b><br>${name}：${px != null ? px.toLocaleString() : "—"}`;
        if (iw >= 0) {
          const w = WK[iw], x = w[XI()], a = w[AC()], ph = PH[S().phase[iw]];
          const dot = v => `<span style="color:${v > 0 ? pos : neg}">●</span>`;
          h += (x == null && state.src === "rt" ? `<br><span style="color:${ink3}">本周没有新读数（${w[FLAG] ? "待核实" : "周报未给远期市盈率"}），信号维持上周</span>` : "")
             + `<br>${x == null ? "" : dot(x)} 超额修正：<b>${fmt(x, 2)}</b>`
             + `<br>${a == null ? "" : dot(a)} 二阶导（${state.win}周）：<b>${fmt(a, 2)}</b>`
             + (ph ? `<br>阶段：<b>${ph}</b>` : "")
             + `<br>远期EPS 13周：${fmt(w[2])}　远期EPS：${w[3]}`
             + (w[FLAG] && state.src === "rt" ? `<br><span style="color:${ink2}">◇ ${w[FLAG]}</span>` : "")
             + `<br><span style="color:${ink3}">修正读数截至 ${day(w[0])} 周报（${state.src === "rt" ? "实时口径" : "清洗口径"}）</span>`;
        }
        return h;
      },
    },
    xAxis: [xa(0), xa(1), xa(2)],
    yAxis: [
      { type: state.scale, gridIndex: 0, scale: true, ...ax,
        min: v => state.scale === "log" ? v.min * 0.96 : v.min - (v.max - v.min) * 0.04,
        max: v => state.scale === "log" ? v.max * 1.03 : v.max + (v.max - v.min) * 0.03,
        axisLabel: { color: ink2, fontSize: 11, showMinLabel: state.scale === "log", showMaxLabel: state.scale === "log",
                     customValues: state.scale === "log" ? state.ticks : undefined,
                     formatter: v => Math.round(v).toLocaleString() },
        axisTick: { show: false, customValues: state.scale === "log" ? state.ticks : undefined } },
      pctAxis(1), pctAxis(2, cap),
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2], filterMode: "filter" },
      { type: "slider", xAxisIndex: [0, 1, 2], bottom: 6, height: 20, filterMode: "filter",
        borderColor: border, fillerColor: css("--band"), handleStyle: { color: css("--surface"), borderColor: ink3 },
        textStyle: { color: ink2 }, dataBackground: { lineStyle: { color: ink3 }, areaStyle: { color: border } },
        labelFormatter: v => day(v).slice(0, 7) },
    ],
    series: [
      { name, type: "line", xAxisIndex: 0, yAxisIndex: 0, data: PX[state.idx], showSymbol: false,
        lineStyle: { width: 1.6, color: css("--price") }, itemStyle: { color: css("--price") },
        markArea: { silent: true, data: bandData() } },
      mark(mUp, true, pos),
      mark(mDn, false, neg),
      { type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: sTop, symbol: "circle", symbolSize: 8, z: 5, tooltip: { show: false },
        itemStyle: { color: "transparent", borderColor: ink2, borderWidth: 2 } },
      { type: "scatter", xAxisIndex: 1, yAxisIndex: 1, symbol: "diamond", symbolSize: 9, z: 6, tooltip: { show: false },
        data: state.src === "rt" ? WK.filter(r => r[FLAG]).map(r => [r[0], 0]) : [],
        itemStyle: { color: css("--surface"), borderColor: ink, borderWidth: 1.4 } },
      { name: "超额修正", type: "bar", xAxisIndex: 1, yAxisIndex: 1, barMaxWidth: 6, data: bars(XI(), pos, neg),
        markLine: { silent: true, symbol: "none", label: { show: false }, data: [line(0)] } },
      { name: "二阶导", type: "bar", xAxisIndex: 2, yAxisIndex: 2, barMaxWidth: 6, data: bars(AC(), pos, neg),
        markLine: { silent: true, symbol: "none", label: { show: false }, data: state.h > 0 ? [line(0), line(state.h, true), line(-state.h, true)] : [line(0)] } },
    ],
  };
}

// ---------- 第二部分：速度拐点 × 纳指 ----------
function section() {
  const s = S(), P = s.phases, cur = s.current.phase;
  $("#secsub").textContent = `${state.src === "rt" ? "实时口径" : "清洗口径"}；当前参数：二阶导窗口 ${state.win} 周、${state.h > 0 ? `去噪阈值 ${state.h} 个百分点` : "不去噪（拐点 = 二阶导穿过 0）"}（每年翻转约 ${s.flips} 次）。改上面的按钮，下面的数字全部跟着变。纳斯达克综合，${D.weekly[0][0].slice(0, 4)} 起。`;

  const cell = (m, hit) => `${fmt(m)}<span class="muted"> · ${hit == null ? "—" : Math.round(hit * 100) + "%"}</span>`;
  let h = `<tr><th>阶段</th><th>占时间</th><th>之前 13 周</th><th>之后 4 周（均值 · 胜率）</th><th>之后 13 周</th><th>之后 26 周</th><th>26 周 t</th></tr>`;
  for (const ph of ["上修加速", "上修减速", "下修加速", "下修放缓"]) {
    const r = P[ph];
    h += `<tr class="${ph === cur ? "cur" : ""}"><td><span class="sw" style="background:${css("--p" + PHC[ph])}"></span>${ph}${ph === cur ? "（当前）" : ""}</td>
      <td>${Math.round(r.share * 100)}%</td><td>${fmt(r.prior13_pct)}</td>
      <td>${cell(r.f4_pct, r.f4_hit)}</td><td>${cell(r.f13_pct, r.f13_hit)}</td><td>${cell(r.f26_pct, r.f26_hit)}</td>
      <td>${r.f26_vs_all_t > 0 ? "+" : ""}${r.f26_vs_all_t}</td></tr>`;
  }
  h += `<tr><td class="muted">全部周</td><td class="muted">100%</td><td></td><td class="muted">${fmt(P["全部"].f4_pct)}</td><td class="muted">${fmt(P["全部"].f13_pct)}</td><td class="muted">${fmt(P["全部"].f26_pct)}</td><td></td></tr>`;
  $("#phtbl").innerHTML = h;

  // 要点（数字跟参数走）
  const rt = s.rt, up = rt["上修见顶"].summary, dn = rt["下修见底"].summary;
  const exL = s.ex["修正速度低点"].events.map(e => e.nq_lag_weeks).filter(v => v != null);
  const med = a => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
  const delays = s.ex["修正速度低点"].events.map(e => e.realtime_delay_weeks).filter(v => v != null);
  $("#takeaway").innerHTML = `
    <p><b>下修见底：纳指先见底。</b>纳指低点平均比修正速度的真实低点早约 ${-med(exL)} 周（中位数），实时确认又要再晚约 ${med(delays)} 周；
       等到「下修见底」信号出来，纳指在之前 13 周已平均涨了 ${fmt(dn.prior13_mean)}。但之后 26 周仍平均 ${fmt(dn.f26_mean)}（胜率 ${Math.round(dn.f26_hit * 100)}%，${dn.f26_n} 次），并没有涨完。
       「下修放缓」阶段整体之后 26 周 ${fmt(P["下修放缓"].f26_pct)}，强于平均——但主要来自 2020、2023 两段，剔除后接近平均。</p>
    <p><b>上修见顶：不是纳指的顶。</b>信号出现后 13 周纳指平均 ${fmt(up.f13_mean)}（${up.f13_n} 次）。整个「上修减速」阶段之后 26 周平均 ${fmt(P["上修减速"].f26_pct)}，
       明显弱于全部周的 ${fmt(P["全部"].f26_pct)}——但这完全来自 2018 年和 2021 下半年～2022 上半年两段，剔除后反而高于平均。</p>
    <p class="muted">两条阶段差异在 12 组参数下方向都一致，但各自只靠两段行情撑着（「剔除后」的核对用的是 8 周 / 阈值 1）。12 年里独立的修正周期只有几轮，当作背景，不当信号。</p>
    <p><b>现在：</b>${cur}（自 ${s.current.since} 起）。${peakText()}</p>`;
  evRender();
}

// 之后 52 周纳指综合的最大回撤（不足 52 周的标「迄今」）
const NQ = PX["^IXIC"];
function maxDD(t, weeks = 52) {
  const i0 = before(NQ, t), end = t + weeks * 7 * 864e5;
  let pk = -Infinity, dd = 0, last = null;
  for (let i = Math.max(i0, 0); i < NQ.length && NQ[i][0] <= end; i++) {
    pk = Math.max(pk, NQ[i][1]); dd = Math.min(dd, NQ[i][1] / pk - 1); last = NQ[i][0];
  }
  return { dd: 100 * dd, full: last != null && end - last < 6 * 864e5 };
}
let DD_MED = null;
function peakText() {
  if (DD_MED == null) {
    const v = WK.map(r => maxDD(r[0])).filter(x => x.full).map(x => x.dd).sort((a, b) => a - b);
    DD_MED = v[Math.floor(v.length / 2)];
  }
  const pk = [...S().ex["修正速度高点"].events].sort((a, b) => b.x13 - a.x13);
  const top = pk.slice(0, 4).map(e => { const m = maxDD(tsOf(e.date)); return `${e.date}（${e.x13.toFixed(2)}%，之后 52 周最大回撤 ${m.dd.toFixed(1)}%${m.full ? "" : "，迄今"}）`; });
  const low = pk.filter(e => e.x13 < 2).map(e => ({ e, m: maxDD(tsOf(e.date)) })).filter(x => x.m.full).sort((a, b) => a.m.dd - b.m.dd)[0];
  return `超额修正历次最高的速度峰值：${top.join("、")}；全部周「之后 52 周最大回撤」的中位数是 ${DD_MED.toFixed(1)}%。`
    + (low ? `但低位峰值之后也有过 ${low.m.dd.toFixed(1)}%（${low.e.date}），峰值高低本身不是可靠的预警，只能当背景。` : "");
}

// 大波段事件：在页面里按纳斯达克综合现算前后走势（确认周的周报周五发布 → 下一个交易日为第 0 周）
function swEvents() {
  const Z = swings(), last = NQ[NQ.length - 1][0];
  const at = t => { const i = before(NQ, t); return i >= 0 ? NQ[i][1] : null; };
  const one = e => {
    const t0 = e.c.t + 864e5, p0 = at(t0);
    const r = k => { const t1 = t0 + k * 7 * 864e5; if (t1 > last + 864e5) return null; const p = at(t1); return p0 && p ? 100 * (p / p0 - 1) : null; };
    const pb = at(t0 - 13 * 7 * 864e5);
    return { date: day(e.c.t), piv: day(e.p.t), pv: e.p.v, amp: e.amp, lag: Math.round((e.c.t - e.p.t) / (7 * 864e5)),
             prior13: p0 && pb ? 100 * (p0 / pb - 1) : null, f4: r(4), f13: r(13), f26: r(26),
             path: Array.from({ length: 53 }, (_, i) => r(i - 26)) };
  };
  const grp = L => { const ev = L.map(one);
    return { events: ev, each: ev.map(e => e.path),
             mean: Array.from({ length: 53 }, (_, k) => { const v = ev.map(e => e.path[k]).filter(x => x != null); return v.length ? v.reduce((a, b) => a + b) / v.length : null; }) }; };
  return { up: grp(Z.tops), dn: grp(Z.bots) };
}

function evRender() {
  const s = S(), ink2 = css("--ink-2"), ink3 = css("--ink-3"), border = css("--border"), pos = css("--pos"), neg = css("--neg");
  const sw = state.evmode === "sw" ? swEvents() : null;
  // 颜色和图上的标记一致：上修那一类蓝、下修那一类橙
  const types = sw ? [["上修速度见顶", sw.up, pos], ["下修速度见顶", sw.dn, neg]]
              : state.evmode === "rt" ? [["上修见顶", s.rt["上修见顶"], pos], ["下修见底", s.rt["下修见底"], neg]]
              : [["修正速度高点", s.ex["修正速度高点"], pos], ["修正速度低点", s.ex["修正速度低点"], neg]];
  const ks = Array.from({ length: 53 }, (_, i) => String(i - 26));
  const series = [];
  for (const [name, g, color] of types) {
    if (state.each) g.each.forEach((row, i) => series.push({ name: name + "（单个）", type: "line", data: row, showSymbol: false, silent: true,
      lineStyle: { width: 1, color, opacity: 0.22 }, itemStyle: { color }, tooltip: { show: false }, z: 1 }));
    series.push({ name: `${name} 平均（${g.events.length} 次）`, type: "line", data: g.mean, showSymbol: false, lineStyle: { width: 2.5, color }, itemStyle: { color }, z: 3 });
  }
  evchart.setOption({
    animation: false, backgroundColor: "transparent",
    grid: { left: 52, right: 20, top: 34, bottom: 34 },
    legend: { top: 0, left: 0, textStyle: { color: ink2, fontSize: 12 }, data: series.filter(x => x.z === 3).map(x => x.name), itemWidth: 16, itemHeight: 3 },
    tooltip: { trigger: "axis", backgroundColor: css("--surface"), borderColor: border, textStyle: { color: css("--ink"), fontSize: 12 },
      formatter: ps => `第 ${+ps[0].axisValue > 0 ? "+" : ""}${ps[0].axisValue} 周<br>` +
        ps.filter(p => p.seriesName.includes("平均")).map(p => `${p.marker}${p.seriesName.split(" ")[0]}：<b>${fmt(p.value)}</b>`).join("<br>") },
    xAxis: { type: "category", data: ks, axisLine: { lineStyle: { color: border } }, axisTick: { show: false },
      axisLabel: { color: ink2, fontSize: 11, interval: 3, formatter: v => (+v > 0 ? "+" : "") + v }, name: "相对事件周", nameLocation: "middle", nameGap: 22, nameTextStyle: { color: ink3, fontSize: 11 } },
    yAxis: { type: "value", axisLabel: { color: ink2, fontSize: 11, formatter: v => v + "%" }, splitLine: { lineStyle: { color: border } } },
    series: [...series, { type: "line", data: [], markLine: { silent: true, symbol: "none", label: { show: false },
      lineStyle: { color: ink3, type: "solid", width: 1 }, data: [{ xAxis: "0" }, { yAxis: 0 }] } }],
  }, true);

  // 事件表
  if (sw) {
    let h = `<tr><th>确认周</th><th>类型</th><th>真正的顶 / 底</th><th>此前涨 / 跌</th><th>确认晚</th><th>纳指之前 13 周</th><th>之后 4 周</th><th>之后 13 周</th><th>之后 26 周</th></tr>`;
    const rows = [];
    for (const [name, g, color] of types) for (const e of g.events) rows.push([e, name, color]);
    rows.sort((a, b) => a[0].date < b[0].date ? 1 : -1);
    for (const [e, name, color] of rows)
      h += `<tr><td>${e.date}</td><td><span class="sw" style="background:${color}"></span>${name}</td><td>${e.piv}（${fmt(e.pv, 2)}）</td>
        <td>${e.amp.toFixed(1)} 个百分点</td><td>${e.lag} 周</td><td>${fmt(e.prior13)}</td><td>${fmt(e.f4)}</td><td>${fmt(e.f13)}</td><td>${fmt(e.f26)}</td></tr>`;
    $("#evtbl").innerHTML = h;
    return;
  }
  const ex = state.evmode === "ex";
  let h = `<tr><th>日期</th><th>类型</th><th>超额修正</th><th>二阶导</th><th>纳指之前 13 周</th><th>之后 4 周</th><th>之后 13 周</th><th>之后 26 周</th>${ex ? "<th>纳指极值相差</th><th>实时确认晚</th>" : ""}</tr>`;
  const rows = [];
  for (const [name, g, color] of types) for (const e of g.events) rows.push([e, name, color]);
  rows.sort((a, b) => a[0].date < b[0].date ? 1 : -1);
  for (const [e, name, color] of rows) {
    h += `<tr><td>${e.date}</td><td><span class="sw" style="background:${color}"></span>${name}</td><td>${fmt(e.x13, 2)}</td><td>${fmt(e.a, 2)}</td>
      <td>${fmt(e.prior13)}</td><td>${fmt(e.f4)}</td><td>${fmt(e.f13)}</td><td>${fmt(e.f26)}</td>
      ${ex ? `<td>${e.nq_lag_weeks == null ? "—" : (e.nq_lag_weeks > 0 ? "晚 " : e.nq_lag_weeks < 0 ? "早 " : "") + Math.abs(e.nq_lag_weeks) + " 周"}</td><td>${e.realtime_delay_weeks == null ? "—" : e.realtime_delay_weeks + " 周"}</td>` : ""}</tr>`;
  }
  $("#evtbl").innerHTML = h;
}

// 对数轴默认只在 10 的整数次幂标刻度：按当前可见区间挑 4~8 个整数刻度
function niceLogTicks(lo, hi) {
  const sets = [[1], [1, 2, 5], [1, 1.5, 2, 3, 5, 7], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]];
  let best = [];
  for (const m of sets) {
    const t = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
      for (const k of m) { const v = k * 10 ** e; if (v >= lo && v <= hi) t.push(v); }
    best = t;
    if (t.length >= 4) break;
  }
  return best;
}
function updateTicks() {
  if (state.scale !== "log") return;
  const dz = chart.getOption()?.dataZoom?.[0], p = PX[state.idx];
  const a = dz?.startValue ?? p[0][0], b = dz?.endValue ?? p[p.length - 1][0];
  let lo = Infinity, hi = -Infinity;
  for (const r of p) if (r[0] >= a && r[0] <= b) { lo = Math.min(lo, r[1]); hi = Math.max(hi, r[1]); }
  state.ticks = isFinite(lo) ? niceLogTicks(lo * 0.96, hi * 1.03) : [];
  chart.setOption({ yAxis: [{ axisLabel: { customValues: state.ticks }, axisTick: { customValues: state.ticks } }] });
}

let quiet = false;  // 程序触发的缩放不清掉区间按钮的高亮
function render(keepZoom = true) {
  const z = keepZoom ? chart.getOption()?.dataZoom?.[0] : null;
  chart.setOption(option(), true);
  if (z) { quiet = true; chart.dispatchAction({ type: "dataZoom", start: z.start, end: z.end }); quiet = false; }
  updateTicks();
  tiles();
  legend();
  section();
}

function setRange(years) {
  const end = PX[state.idx][PX[state.idx].length - 1][0];
  quiet = true;
  if (+years === 0) chart.dispatchAction({ type: "dataZoom", start: 0, end: 100 });
  else chart.dispatchAction({ type: "dataZoom", startValue: end - years * 365.25 * 864e5, endValue: end });
  quiet = false;
}

function seg(id, fn) {
  $(id).addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    fn(b.dataset.v);
  });
}
seg("#idx", v => { state.idx = v; render(); });
seg("#src", v => { state.src = v; render(); });
seg("#scale", v => { state.scale = v; render(); });
seg("#win", v => { state.win = +v; render(); });
seg("#hys", v => { state.h = +v; render(); });
seg("#band", v => { state.band = v; render(); });
seg("#range", v => setRange(v));
seg("#evmode", v => { state.evmode = v; evRender(); });
seg("#marks", v => { state.marks = v; render(); });
$("#each").addEventListener("change", e => { state.each = e.target.checked; evRender(); });
chart.on("datazoom", () => {
  updateTicks();
  if (!quiet) $("#range").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", "false"));
});

// 数据新鲜度：纳指超过 4 天、FactSet 周报超过 9 天没更新就提示
(function () {
  const F = D.fresh, today = Date.parse(new Date().toISOString().slice(0, 10) + "T00:00:00Z");
  const ago = d => Math.round((today - Date.parse(d + "T00:00:00Z")) / 864e5);
  const a = ago(F.nasdaq), b = ago(F.factset), stale = a > 4 || b > 9;
  const el = $("#fresh");
  el.className = "fresh" + (stale ? " stale" : "");
  el.innerHTML = (stale ? "⚠ 数据可能不是最新：" : "数据：")
    + `纳指截至 ${F.nasdaq}（${a} 天前），FactSet 周报截至 ${F.factset}（${b} 天前）；页面生成于 ${F.generated}。`
    + (stale ? "　双击 <b>盈利修正/更新修正页并打开.command</b> 刷新。" : "");
})();

render(false);
window.addEventListener("resize", () => { chart.resize(); evchart.resize(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
