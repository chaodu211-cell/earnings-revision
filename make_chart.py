#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 修正速度与纳指.html：
  上半：纳指（综合 / 纳斯达克100）+ 纯修正（速度）+ 纯修正的二阶导（加速度），三图共用时间轴、缩放、十字线联动；
        纯修正 = 远期 EPS 13 周变化 − 预测不动时随时间自然涨的部分（analyze.time_roll），2016-04 起；
        纳指上标出速度见顶的红蓝三角（点击看详情）；最下面一张把标普500 13 周涨跌拆成 盈利（远期 EPS）+ 估值（远期市盈率），
        右上角写当前是盈利抬升还是估值抬升。
  图下：上修广度（breadth.py，Yahoo 成分股快照）——上修前 5 的公司、按市值分组、每 30 天前 10 大 vs 其余；快照满 5 天后加每日走势。阶段 / 确认点 / 事后高低点取自 speed.py，两种口径 × 窗口 N × 去噪阈值 h 全部内嵌，页面上切换。
        （原来的下半部分「速度拐点 × 纳指」收益率分析 2026-10-04 按用户要求删了；统计结论仍在 说明.md、speed.py。）
单文件、离线可开（ECharts 5.6.0 内嵌，vendor/echarts.min.js，Apache-2.0；对数轴自定义刻度要 ≥5.6）。

二阶导 = 纯修正本周值 − N 周前的值（N = 4 / 8 / 13，页面上切换）。纯修正本身已是远期 EPS 的变化率（一阶），
它的变化就是 EPS 的加速度：正 = 上修在加快（或下修在变缓），负 = 上修在减速（或下修在加剧）。

先跑 analyze.py（生成 _weekly_signals.csv、_analysis.json），再跑本脚本：
    python3 make_chart.py            # 纳指日线超过 1 天没更新会重新拉 Yahoo（走代理 15236）
"""
import json, os, sys, time
import numpy as np
import pandas as pd

from analyze import drop_partial
import breadth
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


HS = (0.0, 0.5, 1.0, 1.5, 2.0)  # 0 = 不去噪：状态翻转就是二阶导穿过 0 的那一周
CODE = {"上修加速": "A", "上修减速": "B", "下修加速": "C", "下修放缓": "D"}


def speed_data(weeks):
    """两种口径 × 每组 (N, h) 跑一遍 speed.run：阶段串（对齐 weekly 行）、阶段表、拐点事件。
    页面现在只用 phase / current / ex 的日期和纯修正（字段名仍叫 x13）；阶段表、事件和 summary 照旧内嵌，给读这份页面数据的每日任务用
    （「盈利修正-每日更新」）。拐点前后的逐周路径（mean / each）只给已删的事件走势图用，不再内嵌。"""
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
                "rt": {k: {"events": v["events"], "summary": v["summary"]} for k, v in r["realtime"].items()},
                "ex": {k: {"events": v["events"], "summary": v["summary"]} for k, v in r["expost"].items()},
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
    # 每行：日期, x13（纯修正）, r13, eps, a4, a8, a13（清洗口径） | x13, a4, a8, a13（实时口径）, 待核实提示
    #       | 扣掉的时间滚动 roll13, 明年增速预期, 明年增速是否事后补（两种口径共用）
    #       | 标普500 13 周涨跌（对数，周五收盘）, 实时口径的远期 EPS 13 周变化 —— 涨跌拆成 盈利 + 估值 用
    weekly = [[d.strftime("%Y-%m-%d"), pct(r.x13), pct(r.r13), round(r.eps, 2)] + [pct(r[f"a{n}"]) for n in NS]
              + [pct(rt.x13[d])] + [pct(rt[f"a{n}"][d]) for n in NS]
              + [rt.flag[d] if isinstance(rt.flag[d], str) else None]
              + [pct(r.roll13), pct(r.g_next), 1 if r.g_next_la else 0]
              + [pct(r.p13), pct(rt.r13[d])]
              for d, r in wk.iterrows()]

    px = {k: [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in nq[k].dropna().items()]
          for k in ("^IXIC", "^NDX")}
    last = wk.iloc[-1]
    data = {"weekly": weekly, "px": px,
            "latest": {"date": wk.index[-1].strftime("%Y-%m-%d"), "x13": pct(last.x13),
                       "pctile": round(100 * float((wk.x13 < last.x13).mean())), "r13": pct(last.r13),
                       "eps": round(float(last.eps), 2), **{f"a{n}": pct(last[f"a{n}"]) for n in NS}}}

    data["speed"] = speed_data(wk.index)
    data["breadth"] = breadth.breadth_data()   # 上修广度（Yahoo 成分股快照）
    fw = pd.read_csv(os.path.join(BASE, "factset_weekly.csv"), parse_dates=["date"])
    data["fresh"] = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                     "factset": fw.date.max().strftime("%Y-%m-%d"),
                     "nasdaq": nq["^IXIC"].dropna().index.max().strftime("%Y-%m-%d")}

    echarts = open(os.path.join(BASE, "vendor", "echarts.min.js")).read()
    html = TEMPLATE.replace("/*ECHARTS*/", echarts).replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    open(OUT, "w").write(html)
    print(OUT, f"{len(html) / 1e6:.1f} MB, weekly {len(weekly)}")


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
  --earn: #1a9a74; --val: #8a63d2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242422; --border: #34332f;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --pos: #3987e5; --neg: #d95926; --price: #e4e3dc; --band: rgba(57,135,229,0.16);
    --earn: #35bf95; --val: #a586ea;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242422; --border: #34332f;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --pos: #3987e5; --neg: #d95926; --price: #e4e3dc; --band: rgba(57,135,229,0.16);
  --earn: #35bf95; --val: #a586ea;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.5 -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 20px 16px 28px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
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
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin: 4px 0 0; min-height: 18px; }
.legend i { display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.legend .mk { font-style: normal; font-size: 11px; margin-right: 4px; }
#chart { width: 100%; height: 1000px; }
.muted { color: var(--ink-3); }
h2 { font-size: 17px; font-weight: 600; margin: 26px 0 4px; }
h3 { font-size: 13px; font-weight: 600; margin: 0 0 6px; color: var(--ink-2); }
.br-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 14px 22px; margin-top: 10px; }
.br-grid > div { min-width: 0; }
.tbl-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 7px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
#brchart { width: 100%; height: 250px; }
.note { color: var(--ink-2); font-size: 13px; background: var(--surface-2); border-radius: 10px; padding: 10px 14px; margin: 10px 0; }
.note p { margin: 4px 0; }
@media (max-width: 640px) { #chart { height: 880px; } .tile .v { font-size: 19px; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>修正速度与纳指</h1>
  <p class="sub">上：纳指；中：标普500 远期 EPS 的纯修正（分析师 13 周里真正改了多少，一阶，速度）；再下：纯修正的 N 周变化（二阶导，加速度）；
     最下：标普500 13 周涨跌拆成盈利和估值两块，右上角写当前是盈利抬升还是估值抬升。拖动或滚轮缩放，三图联动。</p>

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
  <p class="fresh" id="swingnote" style="margin:6px 0 0"></p>
  <div class="note" id="markinfo" style="display:none"></div>

  <div id="chart"></div>

  <section id="breadth" style="display:none">
    <h2>上修广度：集中在大公司还是全市场</h2>
    <div class="note" id="br-verdict"></div>
    <p class="fresh" id="br-sub" style="margin:4px 0 0"></p>
    <div class="br-grid">
      <div><h3 id="br-top-h">上修前 5 的公司（按贡献的上修金额）</h3><div class="tbl-wrap"><table id="br-top"></table></div></div>
      <div><h3>按市值分组（近 90 天）</h3><div class="tbl-wrap"><table id="br-size"></table></div></div>
      <div><h3>上修广度走势（每个点 = 截至那天的近 30 天）</h3><div id="brchart"></div>
        <p class="fresh" id="br-trendnote" style="margin:4px 0 0"></p></div>
    </div>
  </section>

</div>

<script>/*ECHARTS*/</script>
<script>
const D = /*DATA*/;
const $ = s => document.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 1) => v == null ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const day = t => new Date(t).toISOString().slice(0, 10);
const state = { src: "rt", idx: "^IXIC", scale: "log", win: 8, h: 1, marks: "swing", ticks: [] };
const NAME = { "^IXIC": "纳斯达克综合", "^NDX": "纳斯达克100" };
// weekly 行里的列：清洗口径 x13 在 1、二阶导在 4/5/6；实时口径 x13 在 7、二阶导在 8/9/10；待核实提示在 11
const COL = { clean: { x: 1, a: { 4: 4, 8: 5, 13: 6 } }, rt: { x: 7, a: { 4: 8, 8: 9, 13: 10 } } }, FLAG = 11;
const XI = () => COL[state.src].x, AC = () => COL[state.src].a[state.win];
// 标普 13 周涨跌在 15；远期 EPS 13 周变化：清洗口径在 2、实时口径在 16。涨跌 = 盈利 + 估值
const EI = () => state.src === "rt" ? 16 : 2;
function decomp(r) {
  const p = r[15], e = r[EI()];
  if (p == null || e == null) return null;
  const v = p - e;
  const kind = p >= 0 ? (e >= v ? "盈利抬升" : "估值抬升") : (e <= v ? "盈利下修拖累" : "估值收缩拖累");
  return { p, e, v, kind, earn: kind.startsWith("盈利") };
}
function lastDecomp() {
  for (let i = WK.length - 1; i >= 0; i--) { const d = decomp(WK[i]); if (d) return { ...d, t: WK[i][0], i }; }
  return null;
}
const PH = { A: "上修加速", B: "上修减速", C: "下修加速", D: "下修放缓" };
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
    ["纯修正（速度）", fmt(lx), `${day(WK[li][0])}，${D.weekly[0][0].slice(0, 4)} 年以来 ${pct}% 分位` + (pend ? `；最新一周待核实` : "")],
    [`二阶导（${state.win} 周，加速度）`, fmt(a), a == null ? "" : a > 0 ? "上修在加快" : "上修在放慢（或下修加剧）"],
    [`当前阶段（${state.src === "rt" ? "实时" : "清洗"}口径${state.h > 0 ? "、去噪后" : "、不去噪"}）`, cur.phase, `自 ${cur.since} 起`],
    [`${NAME[state.idx]} 13 周涨跌`, fmt(ch13), `${day(last[0])} 收 ${last[1].toLocaleString()}`],
  ];
  const dc = lastDecomp();
  if (dc) t.push([`标普500 13 周涨跌拆分（${day(dc.t)} 周报）`, `<span style="color:${css(dc.earn ? "--earn" : "--val")}">${dc.kind}</span>`,
                  `涨跌 ${fmt(dc.p)} = 盈利 ${fmt(dc.e)} + 估值 ${fmt(dc.v)}`]);
  $("#tiles").innerHTML = t.map(([k, v, n]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`).join("");
}

function legend() {
  const sw = c => `<i style="background:${css(c)}"></i>`;
  let h = "";
  if (state.marks !== "none") { const [a, b] = MARK_TXT[state.marks];
    h += `<span><b class="mk" style="color:${css("--pos")}">▼</b>${a}</span><span><b class="mk" style="color:${css("--neg")}">▲</b>${b}</span>`; }
  if (state.marks === "zero" && state.h > 0) h += `<span class="muted">（零点固定按不去噪算，和去噪阈值无关）</span>`;
  if (state.marks === "swing") h += `<span class="muted">中图圆圈 = 这一段真正的顶 / 底；只看纯修正，和二阶导窗口、去噪阈值无关</span>`;
  if (state.marks !== "none") h += `<span class="muted">点三角看确认日期和实际见顶 / 见底日期</span>`;
  if (state.src === "rt") h += `<span><b class="mk" style="color:${css("--ink")}">◇</b>待核实周（中图；远期 EPS 单周变动 > 2%，下一个有数据的周核实）</span>`;
  $("#legend").innerHTML = h;
  swingNote();
}

// 大波段进度：上一个确认点 → 之后的极值 → 最新读数 → 还差多少确认下一个拐点
function swingNote() {
  const el = $("#swingnote");
  if (state.marks !== "swing") { el.style.display = "none"; return; }
  const Z = swings(), p = Z.all[Z.all.length - 1], f = v => v.toFixed(2) + "%";
  if (!p || !Z.now) { el.style.display = "none"; return; }
  const top = p.type === "top", marked = top ? Z.tops.includes(p) : Z.bots.includes(p);
  const what = top ? "速度见顶" : "速度见底";
  const lead = `上一个确认：${marked ? (top ? "▼ 上修速度见顶" : "▲ 下修速度见顶") : what + "（不够大波段条件，没标箭头）"}，`
    + `${top ? "顶" : "底"}在 ${day(p.p.t)}（${f(p.p.v)}），${day(p.c.t)} 确认。`;
  let rest;
  if (Z.mode === "down") {
    const need = Z.ext.v + SW.R, up = Z.now.v - Z.ext.v;
    const willMark = Z.ext.v <= -SW.L && (Z.last - Z.ext.v) >= SW.A;
    rest = `之后最低 ${f(Z.ext.v)}（${day(Z.ext.t)}），最新 ${f(Z.now.v)}（${day(Z.now.t)}）`
      + (Z.now.t > Z.ext.t ? `，已从低点回升 ${up.toFixed(2)} 个百分点` : "，仍在创新低")
      + `；回升到 ${f(need)} 就确认这一段回落见底`
      + (willMark ? "，会标 ▲ 下修速度见顶。" : `（底在 −${SW.L}% 以上${(Z.last - Z.ext.v) < SW.A ? "、跌幅不足 " + SW.A + " 个百分点" : ""}，不标箭头）。`);
  } else {
    const need = Z.ext.v - SW.R, dn = Z.ext.v - Z.now.v;
    const willMark = Z.ext.v >= SW.L && (Z.ext.v - Z.last) >= SW.A;
    rest = `之后最高 ${f(Z.ext.v)}（${day(Z.ext.t)}），最新 ${f(Z.now.v)}（${day(Z.now.t)}）`
      + (Z.now.t > Z.ext.t ? `，已从高点回落 ${dn.toFixed(2)} 个百分点` : "，仍在创新高")
      + `；回落到 ${f(need)} 就确认这一段见顶`
      + (willMark ? "，会标 ▼ 上修速度见顶。" : `（顶在 +${SW.L}% 以下${(Z.ext.v - Z.last) < SW.A ? "、涨幅不足 " + SW.A + " 个百分点" : ""}，不标箭头）。`);
  }
  el.style.display = "";
  el.textContent = "大波段进度（" + (state.src === "rt" ? "实时口径" : "清洗口径") + "）：" + lead + rest;
}

// 只标「速度见顶」——按修正的猛烈程度理解：
//   上修速度见顶 = 纯修正为正、二阶导由正转负（上修最快的时候过去了）→ 蓝色 ▼，标在价格上方
//   下修速度见顶 = 纯修正为负、二阶导由负转正（下修最猛的时候过去了，即纯修正的谷底）→ 橙色 ▲，标在价格下方
// 不标「速度见底」：上修中重新加快、下修中重新加剧。
// 三种口径：零点 = 二阶导穿过 0（不去噪）；确认点 = 去噪后状态翻转；事后 = 纯修正真正的局部高点（正）/ 低点（负）
// 图上每一次都标出来
// 大波段（折返）：纯修正从最近一个低点涨了 ≥ A、且顶部 ≥ +L% 之后，从顶部回落满 R → 确认「上修速度见顶」；
// 跌了 ≥ A、且谷底 ≤ −L% 之后，从谷底反弹满 R → 确认「下修速度见顶」（下修最猛的时候过去了）。
// R 取 2：略高于单周噪声的 95% 分位（1.5 个百分点）；A 取 4：约一个标准差（4.4）。按噪声定，不按收益挑。
// 只用当时已有的读数（待核实 / 缺周跳过）；确认那周就知道顶 / 底在哪周（回落前的极值）。
const SW = { R: 2, A: 4, L: 2 };
function swings() {
  const col = XI(), out = [];
  let mode = null, ext = null, last = null, hi = null, lo = null, now = null;
  for (let i = 0; i < WK.length; i++) {
    const v = WK[i][col], t = WK[i][0];
    if (v == null) continue;
    const cur = { t, v };
    now = cur;
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
           bots: out.filter(e => e.type === "bot" && e.p.v <= -SW.L && e.amp >= SW.A),
           all: out, mode, ext, last, now };
}

function markSets() {
  if (state.marks === "none") return null;
  const pt = (t, x13) => ({ t, px: pxAt(t), x13, date: day(t) });
  let up = [], dn = [];
  if (state.marks === "swing") {
    const Z = swings();
    const det = e => ({ ...pt(e.c.t, e.c.v), piv: day(e.p.t), pv: e.p.v, amp: e.amp, lag: Math.round((e.c.t - e.p.t) / (7 * 864e5)) });
    return { up: Z.tops.map(det).filter(p => p.px != null),
             dn: Z.bots.map(det).filter(p => p.px != null),
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
  zero: ["上修速度见顶（纯修正 > 0，二阶导由正转负）", "下修速度见顶（纯修正 < 0，二阶导由负转正）"],
  conf: ["上修速度见顶确认（纯修正 > 0，二阶导跌破 −阈值）", "下修速度见顶确认（纯修正 < 0，二阶导升破 +阈值）"],
  ex: ["上修速度真实高点（纯修正峰值，事后）", "下修速度真实高点（纯修正谷底，事后）"],
};

const chart = echarts.init($("#chart"), null, { renderer: "canvas" });

function bars(col, pos, neg) {
  return WK.filter(r => r[col] != null).map(r => ({ value: [r[0], r[col]], itemStyle: { color: r[col] > 0 ? pos : neg } }));
}

function option() {
  const ink = css("--ink"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), border = css("--border");
  const pos = css("--pos"), neg = css("--neg");
  const name = NAME[state.idx];
  const M = markSets();
  LASTM = M;
  const mUp = M ? M.up.map((p, i) => [p.t, p.px, i]) : [], mDn = M ? M.dn.map((p, i) => [p.t, p.px, i]) : [];
  // 事后高 / 低点同时标在「速度」那张图上，能直接看到是纯修正的峰 / 谷
  const sTop = M && M.piv ? M.piv : M && state.marks === "ex" ? [...M.up, ...M.dn].map(p => [p.t, p.x13]) : [];
  const ax = { axisLine: { lineStyle: { color: border } }, axisTick: { show: false },
    axisLabel: { color: ink2, fontSize: 11 }, splitLine: { lineStyle: { color: border } } };
  const G = [{ top: 50, height: "34.5%" }, { top: "46%", height: "12%" }, { top: "64.5%", height: "12%" }, { top: "83%", height: "10%" }]
    .map(g => ({ left: 64, right: 24, ...g }));
  const xa = i => ({ type: "time", gridIndex: i, min: X0, max: X1, ...ax, splitLine: { show: false },
    axisLabel: i < 3 ? { show: false } : { color: ink2, fontSize: 11 } });
  // 涨跌拆分：2020 年 ±30% 会把全图压扁，纵轴同样截在 95 分位附近
  const earn = css("--earn"), val = css("--val");
  const DC = WK.map(r => [r[0], decomp(r)]);
  const dv = DC.flatMap(([, d]) => d ? [Math.abs(d.p), Math.abs(d.e), Math.abs(d.v)] : []).sort((a, b) => a - b);
  const capD = Math.ceil(dv[Math.floor(dv.length * 0.95)] / 5) * 5;
  const dcNow = lastDecomp();
  const pctAxis = (i, cap) => ({ type: "value", gridIndex: i, ...ax, splitNumber: 3,
    ...(cap ? { min: v => Math.max(Math.floor(v.min), -cap), max: v => Math.min(Math.ceil(v.max), cap) } : {}),
    axisLabel: { color: ink2, fontSize: 11, formatter: v => v + "%" } });
  // 二阶导全图会被 2020 年的 ±20~30% 尖峰压扁：纵轴截在 95 分位附近（放大到局部时按可见数据自适应）
  const av = WK.map(r => r[AC()]).filter(v => v != null).map(Math.abs).sort((a, b) => a - b);
  const cap = Math.ceil(av[Math.floor(av.length * 0.95)] / 2) * 2;
  const line = (y, dash) => ({ yAxis: y, lineStyle: { color: ink3, width: 1, type: dash ? "dashed" : "solid" } });
  const label = (text, top) => ({ text, left: 64, top, textStyle: { color: ink2, fontSize: 12, fontWeight: 500 } });
  const mark = (data, down, color) => ({ name: down ? "mark-up" : "mark-dn", type: "scatter", xAxisIndex: 0, yAxisIndex: 0, data,
    symbol: "triangle", symbolRotate: down ? 180 : 0, symbolSize: 12, symbolOffset: [0, down ? -11 : 11], z: 5, tooltip: { show: false },
    cursor: "pointer", emphasis: { scale: 1.5 },
    itemStyle: { color, borderColor: css("--surface"), borderWidth: 1 } });
  return {
    animation: false,
    backgroundColor: "transparent",
    title: [label(name, 4), label("纯修正（速度，%：13 周里分析师真正改了多少，已扣掉时间往前滚的部分）", "43%"),
            label(`二阶导：纯修正 ${state.win} 周变化（加速度，%；${state.h > 0 ? `虚线 = ±${state.h} 去噪阈值；` : ""}全图纵轴截在 ±${cap}%）`, "61.5%"),
            { text: `标普500 13 周涨跌拆分（对数 %）：{e|■} 盈利（远期 EPS）  {v|■} 估值（远期市盈率）  — 线 = 指数涨跌（全图纵轴截在 ±${capD}%）`, left: 64, top: "80%",
              textStyle: { color: ink2, fontSize: 12, fontWeight: 500, rich: { e: { color: earn, fontSize: 12 }, v: { color: val, fontSize: 12 } } } },
            ...(dcNow ? [{ text: `当前：{k|${dcNow.kind}}`, subtext: `标普 13 周 ${fmt(dcNow.p)} = 盈利 ${fmt(dcNow.e)} + 估值 ${fmt(dcNow.v)}`,
              right: 28, top: 4, itemGap: 2,
              textStyle: { color: ink2, fontSize: 13, fontWeight: 500, rich: { k: { color: dcNow.earn ? earn : val, fontSize: 15, fontWeight: 700 } } },
              subtextStyle: { color: ink3, fontSize: 11 } }] : [])],
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
             + `<br>${x == null ? "" : dot(x)} 纯修正：<b>${fmt(x, 2)}</b>`
             + `<br>${a == null ? "" : dot(a)} 二阶导（${state.win}周）：<b>${fmt(a, 2)}</b>`
             + (ph ? `<br>阶段：<b>${ph}</b>` : "")
             + `<br>远期EPS：${w[3]}`
             + (w[FLAG] && state.src === "rt" ? `<br><span style="color:${ink2}">◇ ${w[FLAG]}</span>` : "")
             + `<br><span style="color:${ink3}">修正读数截至 ${day(w[0])} 周报（${state.src === "rt" ? "实时口径" : "清洗口径"}）</span>`
             + (w[12] != null ? `<br><span style="color:${ink3}">= 远期 EPS 13 周 ${fmt(w[2], 2)} − 时间滚动 ${fmt(w[12], 2)}（按明年增速预期 ${w[13].toFixed(1)}%${w[14] ? "，当年周报还没写、用之后的说法补" : ""}）</span>` : "");
          const dc = decomp(w);
          if (dc) h += `<br><b style="color:${dc.earn ? css("--earn") : css("--val")}">${dc.kind}</b>：标普 13 周 ${fmt(dc.p)} = 盈利 ${fmt(dc.e)}`
                     + (x != null && w[12] != null ? `（分析师上调 ${fmt(x)} + 时间滚动 ${fmt(w[12])}）` : "") + ` + 估值 ${fmt(dc.v)}`;
        }
        return h;
      },
    },
    xAxis: [xa(0), xa(1), xa(2), xa(3)],
    yAxis: [
      { type: state.scale, gridIndex: 0, scale: true, ...ax,
        min: v => state.scale === "log" ? v.min * 0.96 : v.min - (v.max - v.min) * 0.04,
        max: v => state.scale === "log" ? v.max * 1.03 : v.max + (v.max - v.min) * 0.03,
        axisLabel: { color: ink2, fontSize: 11, showMinLabel: state.scale === "log", showMaxLabel: state.scale === "log",
                     customValues: state.scale === "log" ? state.ticks : undefined,
                     formatter: v => Math.round(v).toLocaleString() },
        axisTick: { show: false, customValues: state.scale === "log" ? state.ticks : undefined } },
      pctAxis(1), pctAxis(2, cap), pctAxis(3, capD),
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2, 3], filterMode: "filter" },
      { type: "slider", xAxisIndex: [0, 1, 2, 3], bottom: 6, height: 20, filterMode: "filter",
        borderColor: border, fillerColor: css("--band"), handleStyle: { color: css("--surface"), borderColor: ink3 },
        textStyle: { color: ink2 }, dataBackground: { lineStyle: { color: ink3 }, areaStyle: { color: border } },
        labelFormatter: v => day(v).slice(0, 7) },
    ],
    series: [
      { name, type: "line", xAxisIndex: 0, yAxisIndex: 0, data: PX[state.idx], showSymbol: false,
        lineStyle: { width: 1.6, color: css("--price") }, itemStyle: { color: css("--price") } },
      mark(mUp, true, pos),
      mark(mDn, false, neg),
      { type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: sTop, symbol: "circle", symbolSize: 8, z: 5, tooltip: { show: false },
        itemStyle: { color: "transparent", borderColor: ink2, borderWidth: 2 } },
      { type: "scatter", xAxisIndex: 1, yAxisIndex: 1, symbol: "diamond", symbolSize: 9, z: 6, tooltip: { show: false },
        data: state.src === "rt" ? WK.filter(r => r[FLAG]).map(r => [r[0], 0]) : [],
        itemStyle: { color: css("--surface"), borderColor: ink, borderWidth: 1.4 } },
      { name: "纯修正", type: "bar", xAxisIndex: 1, yAxisIndex: 1, barMaxWidth: 6, data: bars(XI(), pos, neg),
        markLine: { silent: true, symbol: "none", label: { show: false }, data: [line(0)] } },
      { name: "二阶导", type: "bar", xAxisIndex: 2, yAxisIndex: 2, barMaxWidth: 6, data: bars(AC(), pos, neg),
        markLine: { silent: true, symbol: "none", label: { show: false }, data: state.h > 0 ? [line(0), line(state.h, true), line(-state.h, true)] : [line(0)] } },
      { name: "盈利", type: "bar", stack: "dc", xAxisIndex: 3, yAxisIndex: 3, barMaxWidth: 6, itemStyle: { color: earn },
        data: DC.filter(([, d]) => d).map(([t, d]) => [t, d.e]),
        markLine: { silent: true, symbol: "none", label: { show: false }, data: [line(0)] } },
      { name: "估值", type: "bar", stack: "dc", xAxisIndex: 3, yAxisIndex: 3, barMaxWidth: 6, itemStyle: { color: val },
        data: DC.filter(([, d]) => d).map(([t, d]) => [t, d.v]) },
      { name: "标普13周涨跌", type: "line", xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, z: 4,
        lineStyle: { color: ink, width: 1.2 }, itemStyle: { color: ink }, data: DC.filter(([, d]) => d).map(([t, d]) => [t, d.p]) },
    ],
  };
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
  const mi = document.getElementById("markinfo"); if (mi) mi.style.display = "none";
  const z = keepZoom ? chart.getOption()?.dataZoom?.[0] : null;
  chart.setOption(option(), true);
  if (z) { quiet = true; chart.dispatchAction({ type: "dataZoom", start: z.start, end: z.end }); quiet = false; }
  updateTicks();
  tiles();
  legend();
  brRender();
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
seg("#range", v => setRange(v));
seg("#marks", v => { state.marks = v; render(); });
// ---------- 上修广度 ----------
let brchart = null;
function brRender() {
  const B = D.breadth, box = $("#breadth");
  if (!B || !B.latest) { box.style.display = "none"; return; }
  box.style.display = "";
  const L = B.latest, f = v => fmt(v, 1), pc = v => v == null ? "—" : Math.round(v) + "%";
  let verdict;
  if (L.share10 == null) verdict = `<b>整体在下修</b>：近 90 天全指数 ${f(L.all)}，前 10 大 ${f(L.top10)}，其余 ${f(L.rest)}。`;
  else {
    const ratio = L.share10 / L.w10;
    const head = ratio >= 1.5 ? "偏集中在大公司" : ratio <= 0.67 ? "中小公司上修更多" : "大小公司同步上修";
    verdict = `<b>${head}</b>：市值前 10 大占指数盈利 ${pc(L.w10)}，贡献了 ${pc(L.share10)} 的上修金额（${ratio.toFixed(1)} 倍）；`
      + `其余 ${L.n - 10} 家${L.rest > 0 ? "也在上修" : "在下修"} ${f(L.rest)}，${pc(L.up)} 的公司上修超过 1%、${pc(L.dn)} 下修超过 1%。`;
  }
  const W = L.win, prev = (W[0].rest + W[1].rest) / 2, d = W[2].rest - prev;
  verdict += `<br>最近 30 天其余公司 ${f(W[2].rest)}，之前两个 30 天 ${f(W[0].rest)}、${f(W[1].rest)}——`
    + (d <= -0.3 ? "<b>广度在收窄</b>（上修越来越靠大公司）。" : d >= 0.3 ? "<b>广度在变宽</b>。" : "大致持平。")
    + ` 上修前 5：${L.lead.map(x => x.co).join("、")}，合计占 ${pc(L.lead5_share)}。`;
  $("#br-verdict").innerHTML = verdict;
  $("#br-sub").textContent = `Yahoo 成分股快照 ${L.date}，${L.n} 家公司；全指数近 90 天纯修正 ${f(L.all)}（和中图同一口径）。前 10 大：${L.top10_names.join(" ")}。`;
  $("#br-top-h").textContent = `${L.lead_dir}前 5 的公司（按贡献的${L.lead_dir}金额）`;
  $("#br-top").innerHTML = `<tr><th>公司</th><th>市值排名</th><th>行业</th><th>90 天修正</th><th>占全部${L.lead_dir}金额</th></tr>`
    + L.lead.map(x => `<tr><td><b>${x.co}</b></td><td>${x.rank}</td><td>${x.sector || "—"}</td><td>${f(x.rev)}</td><td>${pc(x.share)}</td></tr>`).join("")
    + `<tr><td class="muted">合计</td><td></td><td></td><td></td><td class="muted">${pc(L.lead5_share)}</td></tr>`;
  $("#br-size").innerHTML = `<tr><th>市值排名</th><th>占指数盈利</th><th>修正</th><th>中位数</th><th>上修>1%</th><th>占上修金额</th></tr>`
    + L.size.map(x => `<tr><td>${x.name}</td><td>${pc(x.w)}</td><td>${f(x.rev)}</td><td>${f(x.med)}</td><td>${pc(x.up)}</td><td>${pc(x.share)}</td></tr>`).join("");
  // 广度走势：前 10 大 / 其余公司的近 30 天修正（左轴），全部公司净广度（灰柱，右轴）
  const T = B.trend, el = $("#brchart");
  brchart = brchart || echarts.init(el, null, { renderer: "canvas" });
  const ink2 = css("--ink-2"), ink3 = css("--ink-3"), border = css("--border"), xs = T.map(t => tsOf(t.date));
  const ser = (k) => T.map((t, i) => [xs[i], t[k]]);
  brchart.setOption({
    animation: false, backgroundColor: "transparent",
    grid: { left: 44, right: 44, top: 34, bottom: 26 },
    legend: { top: 0, left: 0, itemWidth: 14, itemHeight: 8, textStyle: { color: ink2, fontSize: 11 } },
    tooltip: { trigger: "axis", backgroundColor: css("--surface"), borderColor: border, textStyle: { color: css("--ink"), fontSize: 12 },
      formatter: ps => `截至 ${day(ps[0].axisValue)} 的近 30 天<br>` + ps.map(q => `${q.marker}${q.seriesName}：<b>${q.seriesName.startsWith("净广度") ? (q.value[1] > 0 ? "+" : "") + Math.round(q.value[1]) + " 个百分点" : fmt(q.value[1], 2)}</b>`).join("<br>") },
    xAxis: { type: "time", splitNumber: 3, axisLabel: { color: ink2, fontSize: 11, hideOverlap: true, formatter: v => day(v).slice(5) },
             axisLine: { lineStyle: { color: border } }, splitLine: { show: false } },
    yAxis: [{ type: "value", axisLabel: { color: ink2, fontSize: 11, formatter: v => v + "%" }, splitLine: { lineStyle: { color: border } } },
            { type: "value", axisLabel: { color: ink3, fontSize: 11 }, splitLine: { show: false } }],
    series: [
      { name: "净广度（右轴）", type: "bar", yAxisIndex: 1, barWidth: 10, data: ser("net"), itemStyle: { color: ink3, opacity: 0.45 } },
      { name: "前 10 大", type: "line", data: ser("top10"), symbolSize: 5, lineStyle: { width: 2, color: css("--pos") }, itemStyle: { color: css("--pos") } },
      { name: "其余公司", type: "line", data: ser("rest"), symbolSize: 5, lineStyle: { width: 2, color: css("--earn") }, itemStyle: { color: css("--earn") } },
    ],
  }, true);
  $("#br-trendnote").textContent = `线 = 近 30 天修正；灰柱 = 净广度：近 30 天修正 > +0.5% 的公司占比 − < −0.5% 的占比（全部 ${L.n} 家）。`
    + (T.length > 2 ? `前两个点取自 ${T[2].date} 那份快照里 60、30 天前的预期，之后每个交易日添一个点。` : "");
}

// 点击红蓝三角 → 详情框：信号、确认日期、实际见顶 / 见底日期、之后纳指
let LASTM = null;
function showMark(kind, i) {
  const M = LASTM, el = $("#markinfo");
  if (!M) return;
  const up = kind === "mark-up", p = (up ? M.up : M.dn)[i];
  if (!p) return;
  const ink3 = css("--ink-3"), color = css(up ? "--pos" : "--neg"), sym = up ? "▼" : "▲";
  const name = { swing: up ? "上修速度见顶确认（大波段）" : "下修速度见顶确认（大波段）",
                 zero: up ? "上修速度见顶（二阶导零点）" : "下修速度见顶（二阶导零点）",
                 conf: up ? "上修速度见顶确认（二阶导跌破 −阈值）" : "下修速度见顶确认（二阶导升破 +阈值）",
                 ex: up ? "上修速度真实高点（事后）" : "下修速度真实低点（事后）" }[state.marks];
  const wk = n => Math.round(n / (7 * 864e5)), word = up ? "见顶" : "见底";
  let lines = [];
  if (state.marks === "ex") {
    lines.push(`实际${word}日期：<b>${p.date}</b>（纯修正 ${fmt(p.x13, 2)}）。这是事后才能确认的真实${up ? "高点" : "低点"}，当时并不知道。`);
  } else {
    lines.push(`确认日期：<b>${p.date}</b>（这一周周五的周报；当时纯修正 ${fmt(p.x13, 2)}）`);
    if (p.piv) {
      lines.push(`实际${word}日期：<b>${p.piv}</b>（纯修正 ${fmt(p.pv, 2)}）——确认晚了 ${p.lag} 周；此前${up ? "涨" : "跌"}了 ${p.amp.toFixed(1)} 个百分点`);
    } else {
      // 零点 / 确认点没有自带顶底：取确认日之前 26 周内最近的事后真实点
      const L = S().ex[up ? "修正速度高点" : "修正速度低点"].events.filter(e => e.date <= p.date && tsOf(p.date) - tsOf(e.date) <= 26 * 7 * 864e5);
      const e = L[L.length - 1];
      lines.push(e ? `实际${word}日期（事后看，确认日之前最近的真实${up ? "高点" : "低点"}）：<b>${e.date}</b>（纯修正 ${fmt(e.x13, 2)}）——确认晚了 ${wk(tsOf(p.date) - tsOf(e.date))} 周`
                   : `确认日之前 26 周内没有事后真实${up ? "高点" : "低点"}（多半是小周期的翻转）`);
    }
  }
  // 之后纳指（以确认周周五收盘为基准）
  const P = PX[state.idx], at = t => { const j = before(P, t); return j >= 0 ? P[j][1] : null; };
  const t0 = tsOf(p.date), p0 = at(t0), last = P[P.length - 1][0];
  const r = k => { const t1 = t0 + k * 7 * 864e5; if (t1 > last) return "—"; const v = at(t1); return v && p0 ? fmt(100 * (v / p0 - 1)) : "—"; };
  lines.push(`${NAME[state.idx]}：当周收盘 ${p0 ? p0.toLocaleString() : "—"}；之后 4 / 13 / 26 周 ${r(4)} / ${r(13)} / ${r(26)}`);
  el.innerHTML = `<p style="margin:0 0 4px"><b style="color:${color}">${sym} ${name}</b>`
    + `<span style="float:right;cursor:pointer;color:${ink3}" onclick="this.parentNode.parentNode.style.display='none'">✕ 关闭</span></p>`
    + lines.map(x => `<p>${x}</p>`).join("");
  el.style.display = "";
}
chart.on("click", p => { if (p.seriesName === "mark-up" || p.seriesName === "mark-dn") showMark(p.seriesName, p.data[2]); });

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
window.addEventListener("resize", () => { chart.resize(); brchart && brchart.resize(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
