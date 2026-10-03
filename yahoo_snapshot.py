#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标普500成分股一致预期快照（Yahoo quoteSummary: earningsTrend + price）。

Yahoo 每只股票给出 当前 / 7 / 30 / 60 / 90 天前 的一致预期 EPS，以及近 7 / 30 天
上调、下调预期的分析师人数。所以拍一次快照就能算出「过去 90 天的修正」，
每天存一份，历史从今天开始累积（Yahoo 不给更早的）。

汇总口径：
  远期EPS(个股) = 本财年 × w + 下一财年 × (1 − w)，w = 本财年剩余天数 / 365
    （和 FactSet 的远期 12 个月 EPS 同一个思路；回看时沿用今天的 w，
      所以算出来的是纯修正，不含「时间往前滚」带来的机械增长）
  修正幅度 = Σ 股本 × 远期EPS(今天) / Σ 股本 × 远期EPS(N 天前) − 1   （盈利额加权）
  修正广度 = (上调人数 − 下调人数) / (上调人数 + 下调人数)，近 30 天，本财年 + 下一财年

用法：
    python3 yahoo_snapshot.py               # 拉全部成分股（名单取 ../sectors.json）
    python3 yahoo_snapshot.py --tickers NVDA MSFT

输出：yahoo/YYYY-MM-DD.csv（逐股），yahoo_agg.csv（每天一行汇总，同日重跑覆盖），
      yahoo_sector.csv（每天每个 GICS 行业一行）
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

from yfinance.data import YfData

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "yahoo")
AGG = os.path.join(BASE, "yahoo_agg.csv")
# 成分股名单：本地取上级目录（最优拟合）的 sectors.json；GitHub Actions 上用 REV_SECTORS 指到下载下来的那份
SECTORS = os.environ.get("REV_SECTORS") or os.path.join(os.path.dirname(BASE), "sectors.json")
URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
LAGS = ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo")

yd = YfData()


def raw(x):
    return x.get("raw") if isinstance(x, dict) else x


def fetch(sym):
    for k in range(3):
        try:
            j = yd.get_raw_json(URL.format(sym=sym),
                                params={"modules": "earningsTrend,price,defaultKeyStatistics"})
            r = j["quoteSummary"]["result"][0]
            break
        except Exception:
            time.sleep(2 * (k + 1))
    else:
        return None
    row = {"ticker": sym,
           "price": raw(r["price"].get("regularMarketPrice")),
           "mcap": raw(r["price"].get("marketCap")),
           "shares": raw(r.get("defaultKeyStatistics", {}).get("sharesOutstanding")),
           "currency": r["price"].get("currency")}
    for t in r.get("earningsTrend", {}).get("trend", []):
        p = t.get("period")
        if p not in ("0q", "+1q", "0y", "+1y"):
            continue
        row[f"{p}_end"] = t.get("endDate")
        for lag in LAGS:
            row[f"{p}_{lag}"] = raw(t.get("epsTrend", {}).get(lag))
        rv = t.get("epsRevisions", {})
        row[f"{p}_up7"], row[f"{p}_up30"] = raw(rv.get("upLast7days")), raw(rv.get("upLast30days"))
        row[f"{p}_dn7"], row[f"{p}_dn30"] = raw(rv.get("downLast7Days")), raw(rv.get("downLast30days"))
        row[f"{p}_n"] = raw(t.get("earningsEstimate", {}).get("numberOfAnalysts"))
        row["eps_ccy"] = t.get("epsTrend", {}).get("epsTrendCurrency")
    return row


def fwd(row, lag, today):
    """远期 12 个月 EPS，权重按今天的本财年剩余天数定。"""
    a, b, end = row.get(f"0y_{lag}"), row.get(f"+1y_{lag}"), row.get("0y_end")
    if a is None or b is None or not end:
        return None
    left = (datetime.strptime(end, "%Y-%m-%d").date() - today).days
    w = min(max(left / 365.0, 0.0), 1.0)
    return w * a + (1 - w) * b


def aggregate(rows, today, top=5):
    out = {"date": str(today), "n": 0}
    contrib = []
    num = {lag: 0.0 for lag in LAGS}
    up = dn = 0
    rising = falling = cnt = 0
    for r in rows:
        # 个股 EPS 币种和报价币种不一致（少数外币报表）时不进盈利额加总
        if not r.get("shares") or r.get("eps_ccy") not in (None, "USD") or r.get("currency") != "USD":
            continue
        f = {lag: fwd(r, lag, today) for lag in LAGS}
        if any(v is None for v in f.values()):
            continue
        out["n"] += 1
        for lag in LAGS:
            num[lag] += r["shares"] * f[lag]
        contrib.append((r["ticker"], r["shares"] * (f["current"] - f["90daysAgo"])))
        for p in ("0y", "+1y"):
            up += r.get(f"{p}_up30") or 0
            dn += r.get(f"{p}_dn30") or 0
        if f["30daysAgo"]:
            cnt += 1
            chg = f["current"] / f["30daysAgo"] - 1 if f["30daysAgo"] > 0 else 0
            rising += chg > 0.001
            falling += chg < -0.001
    for lag, d in (("7daysAgo", 7), ("30daysAgo", 30), ("60daysAgo", 60), ("90daysAgo", 90)):
        out[f"rev{d}"] = round(num["current"] / num[lag] - 1, 6) if num[lag] else None
    out["breadth30"] = round((up - dn) / (up + dn), 4) if up + dn else None
    out["up30"], out["dn30"] = up, dn
    out["co_breadth30"] = round((rising - falling) / cnt, 4) if cnt else None
    out["fwd_eps_dollars_bn"] = round(num["current"] / 1e9, 2)
    # 集中度：90 天上修金额里，贡献最大的前 5 家占多少（净变动为负时不算）
    net = sum(c for _, c in contrib)
    contrib.sort(key=lambda x: -x[1])
    out["top5_share_rev90"] = round(sum(c for _, c in contrib[:top]) / net, 3) if net > 0 else None
    out["top5_rev90"] = " ".join(t for t, _ in contrib[:top])
    return out


def by_sector(rows, today):
    sec = json.load(open(SECTORS))
    groups = {}
    for r in rows:
        groups.setdefault(sec.get(r["ticker"], "其他"), []).append(r)
    out = []
    for name, rs in sorted(groups.items()):
        a = aggregate(rs, today)
        out.append({"date": a["date"], "sector": name, "n": a["n"], "rev30": a["rev30"], "rev90": a["rev90"],
                    "breadth30": a["breadth30"], "co_breadth30": a["co_breadth30"],
                    "fwd_eps_dollars_bn": a["fwd_eps_dollars_bn"], "top5_rev90": a["top5_rev90"]})
    return out


def upsert(path, new_rows, key):
    old = []
    if os.path.exists(path):
        dates = {r["date"] for r in new_rows}
        old = [r for r in csv.DictReader(open(path)) if r["date"] not in dates]
    allr = sorted(old + new_rows, key=lambda r: tuple(str(r[k]) for k in key))
    cols = list(new_rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    syms = a.tickers or sorted(json.load(open(SECTORS)))
    today = date.today()
    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(fetch, syms) if r]
    print(f"fetched {len(rows)}/{len(syms)}")
    if not a.tickers and len(rows) < 0.9 * len(syms):
        # 快照历史只增不改：被限流拿到半截时宁可当天缺一份，也不写进去
        sys.exit(f"只拿到 {len(rows)}/{len(syms)} 只，疑似被 Yahoo 限流——不写快照")
    os.makedirs(OUT, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "ticker", k))
    with open(os.path.join(OUT, f"{today}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows, today)
    print(json.dumps(agg, ensure_ascii=False))
    if a.tickers:
        return
    upsert(AGG, [agg], ("date",))
    upsert(os.path.join(BASE, "yahoo_sector.csv"), by_sector(rows, today), ("date", "sector"))

if __name__ == "__main__":
    main()
