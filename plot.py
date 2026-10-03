#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""画 _盈利修正周期.png：读 _weekly_signals.csv 和 _analysis.json（先跑 analyze.py）。"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

w = pd.read_csv(os.path.join(BASE, "_weekly_signals.csv"), index_col=0, parse_dates=True)
res = json.load(open(os.path.join(BASE, "_analysis.json")))
w = w[w.eps.first_valid_index():]

fig = plt.figure(figsize=(12, 11), facecolor="#fcfcfb")
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 0.8], hspace=0.38)


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)


# 1) 远期 EPS 与指数，同一基期指数化（对数轴）
ax = fig.add_subplot(gs[0])
style(ax)
e = w.eps.dropna()
base = e.index[0]
ax.plot(e.index, 100 * e / e.iloc[0], color=BLUE, lw=2, label="远期12个月EPS（FactSet）")
p = w.px.loc[base:].dropna()
ax.plot(p.index, 100 * p / p.iloc[0], color=GRAY, lw=1.6, label="标普500")
ax.set_yscale("log")
ax.set_yticks([100, 150, 200, 300, 400, 600])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title(f"远期 EPS 与标普500（{base:%Y-%m} = 100，对数刻度）", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
ax.text(e.index[-1], 100 * e.iloc[-1] / e.iloc[0], f"  {e.iloc[-1]:.0f}美元", color=INK2, fontsize=9, va="center")

# 2) 超额修正 x13：正 = 比此前 3 年常态上修得多
ax = fig.add_subplot(gs[1], sharex=ax)
style(ax)
x = 100 * w.x13.dropna()
ax.bar(x.index, x.where(x > 0, 0), width=6, color=BLUE, label="上修快于常态")
ax.bar(x.index, x.where(x <= 0, 0), width=6, color=ORANGE, label="慢于常态 / 下修")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylim(max(x.min(), -15) - 1, min(x.max(), 15) + 1)
ax.set_title("超额修正：远期EPS 13周变化 减去 此前3年同口径均值（%，纵轴截在 ±15）", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2, ncol=2)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# 3) 谁领先谁
ax = fig.add_subplot(gs[2])
style(ax)
ll = {int(k): v for k, v in res["leadlag_r13_vs_p13_shift"].items()}
ks = sorted(ll)
ax.bar(ks, [ll[k] for k in ks], width=1.6, color=[GRAY if k < 0 else BLUE for k in ks])
ax.axvline(0, color=INK2, lw=0.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("错开周数 k：负 = 价格在前（过去的涨跌），正 = 价格在后（未来的涨跌）", color=INK2, fontsize=9)
ax.set_title("13周EPS修正 与 错开 k 周的13周指数涨跌 的相关系数", loc="left", color=INK, fontsize=12)

fig.text(0.01, 0.005, "数据：FactSet《Earnings Insight》周报（远期EPS = 收盘价 ÷ 远期市盈率），标普500 来自 Yahoo。",
         color=INK2, fontsize=8)
out = os.path.join(BASE, "_盈利修正周期.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(out)
