#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上修广度：近 90 天的盈利预期上修是集中在大市值公司，还是全市场都在上修（Yahoo 成分股快照，yahoo/YYYY-MM-DD.csv）。

口径和页面中图的纯修正一致：个股远期 EPS = 本财年 × w + 下一财年 × (1 − w)，w 按快照当天定、回看时不变，所以不含时间滚动。
  盈利额 = 股本 × 远期 EPS；修正 = Σ盈利额(今天) / Σ盈利额(N 天前) − 1
  双重股权合并成一家公司（GOOG→GOOGL、FOX→FOXA、NWS→NWSA）；Yahoo 两个代码给的市值都是全公司的，所以市值用 股本 × 股价
  「前 10 大」按市值；「上修前 5」按贡献的上修金额（股本 × 远期 EPS 的 90 天变化）

make_chart.py 调 breadth_data() 内嵌进页面；快照由 GitHub Actions 每个交易日拍一份（页面用到前一天那份）。
单独跑：python3 breadth.py  → 打印最新一份的摘要
"""
import glob, json, os
from datetime import date

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SECTORS = os.environ.get("REV_SECTORS") or os.path.join(os.path.dirname(BASE), "sectors.json")
CO = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}
LAGS = ("current", "30daysAgo", "60daysAgo", "90daysAgo")
SIZE = [(1, 10, "前 10"), (11, 50, "11–50"), (51, 100, "51–100"), (101, 250, "101–250"), (251, 9999, "251 以后")]


def companies(path):
    """一份快照 → 每家公司一行：市值、各回看点的盈利额、行业、市值排名。"""
    day = date.fromisoformat(os.path.basename(path)[:10])
    d = pd.read_csv(path)
    d = d[(d.currency == "USD") & (d.eps_ccy.isna() | (d.eps_ccy == "USD")) & d.shares.notna() & d.price.notna()].copy()
    w = ((pd.to_datetime(d["0y_end"]) - pd.Timestamp(day)).dt.days / 365).clip(0, 1)
    cols = []
    for lag in LAGS:
        d[f"E_{lag}"] = d.shares * (w * d[f"0y_{lag}"] + (1 - w) * d[f"+1y_{lag}"])
        cols.append(f"E_{lag}")
    d = d.dropna(subset=cols)
    d["cap"] = d.shares * d.price
    d["co"] = d.ticker.replace(CO)
    g = d.groupby("co")[["cap"] + cols].sum().reset_index()
    sec = json.load(open(SECTORS)) if os.path.exists(SECTORS) else {}
    g["sector"] = g.co.map(sec).fillna("")
    g = g.sort_values("cap", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    g["dE"] = g.E_current - g.E_90daysAgo
    g["rev"] = np.where(g.E_90daysAgo > 0, g.E_current / g.E_90daysAgo - 1, np.nan)
    return day, g


def _rev(g, m, a="E_90daysAgo", b="E_current"):
    base = g.loc[m, a].sum()
    return float((g.loc[m, b].sum() - base) / base) if base > 0 else None


def summarize(day, g):
    p = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(100 * float(v), 2)
    tot, top = g.dE.sum(), g["rank"] <= 10
    r = g.rev.dropna()
    out = {"date": str(day), "n": int(len(g)),
           "all": p(_rev(g, g["rank"] > 0)), "top10": p(_rev(g, top)), "rest": p(_rev(g, ~top)),
           "w10": p(g.loc[top, "E_90daysAgo"].sum() / g.E_90daysAgo.sum()),
           "share10": p(g.loc[top, "dE"].sum() / tot) if tot > 0 else None,
           "up": p((r > 0.01).mean()), "dn": p((r < -0.01).mean()), "med": p(r.median()),
           "top10_names": list(g.loc[top, "co"])}
    # 上修前 5（整体净下修时改列下修前 5）
    up = tot > 0
    lead = g.sort_values("dE", ascending=not up).head(5)
    out["lead_dir"] = "上修" if up else "下修"
    out["lead"] = [{"co": c, "rank": int(k), "sector": s, "rev": p(v), "share": p(e / tot) if tot else None}
                   for c, k, s, v, e in zip(lead.co, lead["rank"], lead.sector, lead.rev, lead.dE)]
    out["lead5_share"] = p(lead.dE.sum() / tot) if tot else None
    out["size"] = []
    for a, b, name in SIZE:
        m = g["rank"].between(a, b)
        rr = g.loc[m, "rev"].dropna()
        out["size"].append({"name": name, "w": p(g.loc[m, "E_90daysAgo"].sum() / g.E_90daysAgo.sum()), "rev": p(_rev(g, m)),
                            "med": p(rr.median()), "up": p((rr > 0.01).mean()), "share": p(g.loc[m, "dE"].sum() / tot) if tot else None})
    out["win"] = []
    for a, b, name in (("E_90daysAgo", "E_60daysAgo", "90→60 天前"), ("E_60daysAgo", "E_30daysAgo", "60→30 天前"),
                       ("E_30daysAgo", "E_current", "近 30 天")):
        rr = (g[b] / g[a] - 1)[~top & (g[a] > 0)]
        out["win"].append({"name": name, "all": p(_rev(g, g["rank"] > 0, a, b)), "top10": p(_rev(g, top, a, b)),
                           "rest": p(_rev(g, ~top, a, b)), "net": p((rr > 0.005).mean() - (rr < -0.005).mean())})
    return out


def breadth_data():
    """全部快照 → {"hist": 每天一行, "latest": 最新一份的明细}；没有快照返回 None。"""
    paths = sorted(glob.glob(os.path.join(BASE, "yahoo", "20*.csv")))
    if not paths:
        return None
    hist, latest = [], None
    for path in paths:
        try:
            s = summarize(*companies(path))
        except Exception as e:  # 个别快照坏了不影响页面
            print(f"breadth: 跳过 {os.path.basename(path)}：{e}")
            continue
        hist.append({k: s[k] for k in ("date", "all", "top10", "rest", "share10", "up", "dn")})
        latest = s
    return {"hist": hist, "latest": latest} if latest else None


if __name__ == "__main__":
    b = breadth_data()
    print(json.dumps(b["latest"], ensure_ascii=False, indent=1))
