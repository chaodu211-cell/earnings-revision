#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 txt/*.txt（FactSet《Earnings Insight》周报文字）抽出每周的关键数字 → factset_weekly.csv

抽取字段：
  fpe            远期 12 个月市盈率（周报正文，一位小数）
  px             周报引用的标普收盘价（周四收盘）；周报没写的，用 Yahoo ^GSPC 的周四收盘补
  fwd_eps        远期 12 个月 EPS：周报直接写了美元数就用它，否则 = px / fpe
  fwd_eps_qtd    「本季度初以来远期 EPS 变了多少 %」（周报原句）
  q, q_eps, q_eps0, q_rev
                 季度自下而上 EPS：本季度（或刚结束的季度）的现值、季初值、变动 %
  q_growth, q_growth0
                 本季度预期同比增速（现在 / 季初）
  guide_q, guide_neg, guide_pos
                 季度 EPS 指引：负面家数 / 正面家数
  g_next         明年（周报年份 + 1）全年 EPS 预期增速 %（「For CY 2027, analysts are projecting earnings growth of 15.8%」）；
                 用来算「预测不动时远期 EPS 随时间自然涨多少」（analyze.time_roll）。2016 年以前基本没写，2015 年全年没有
用法：python3 parse_factset.py                 # 从 txt/ 全量重建
      python3 parse_factset.py --incremental   # 只追加 CSV 里还没有的周报（GitHub Actions 用）
"""
import csv, glob, os, re
from datetime import date, datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
NUM = r"(-?\d[\d,]*\.?\d*)"


def clean(t):
    t = t.replace("’", "'").replace("–", "-").replace("−", "-")
    # 页眉页脚：只删版权那一行和「FactSet Research Systems Inc. www.factset.com」——
    # 不能跨行非贪婪匹配到 www.factset.com：2016 年的版式里第一个网址在页底，会把整页正文吞掉
    t = re.sub(r"Copyright ©[^\n]*", " ", t)
    t = re.sub(r"FactSet Research Systems Inc\.\s+www\.factset\.com", " ", t)
    return re.sub(r"\s+", " ", t)


def f(x):
    return float(x.replace(",", "")) if x is not None else None


def first(pats, t, flags=re.I):
    for p in pats:
        m = re.search(p, t, flags)
        if m:
            return m
    return None


def parse_fpe(t):
    # 跳过「X 年平均远期市盈率」「某行业远期市盈率」这类句子
    for m in re.finditer(r"(?:forward 12-month|12-month forward) P/E ratio(?: for the S&P 500)?(?: on that date)?"
                         r" (?:is|was|of)(?: now)?(?: only)? (\d{1,2}\.\d)", t, re.I):
        before = t[max(0, m.start() - 40):m.start()].lower()
        if "average" in before or "sector" in before or "year" in before:
            continue
        return float(m.group(1))
    return None


def parse_px(t):
    m = first([r"closing price \((\d[\d,]*\.\d+)\)",
               r"closing price of (\d[\d,]*\.\d+)",
               r"closing price for the S&P 500 was (\d[\d,]*\.\d+)"], t)
    return f(m.group(1)) if m else None


def parse_fwd_eps(t):
    m = first([r"forward 12-month EPS estimate \(\$(\d[\d,]*\.\d+)\)",
               r"forward 12-month EPS estimate of \$(\d[\d,]*\.\d+)"], t)
    return f(m.group(1)) if m else None


def parse_fwd_qtd(t):
    m = re.search(r"the forward 12-month EPS estimate has (increased|decreased|risen|fallen|declined)"
                  r" by (\d+\.?\d*)%", t, re.I)
    if not m:
        return None
    v = float(m.group(2))
    return v if m.group(1).lower() in ("increased", "risen") else -v


def parse_q_eps(t):
    m = re.search(r"The Q(\d) bottom-up EPS estimate \([^)]*\) (?:has )?"
                  r"(increased|decreased|risen|fallen|declined|dropped|was unchanged|remained unchanged)"
                  r"(?: by (\d+\.?\d*)%)? \(to \$(\d+\.\d+) from \$(\d+\.\d+)\)", t, re.I)
    if not m:
        return None
    return int(m.group(1)), float(m.group(4)), float(m.group(5))


def parse_growth(t):
    cur = first([r"For Q(\d) (\d{4}),? the (?:estimated|blended)[^%]{0,80}? (?:earnings )?growth rate"
                 r"[^%]{0,30}? is (-?\d+\.?\d*)%"], t)
    start = first([r"On \w+ \d+, the estimated[^%]{0,80}?growth rate[^%]{0,40}?for Q(\d) (\d{4})"
                   r" was (-?\d+\.?\d*)%"], t)
    return cur, start


def parse_guidance(t):
    m = re.search(r"For Q(\d) (\d{4}),? (\d+) (?:S&P 500 )?companies have issued negative EPS guidance"
                  r" and (\d+) (?:S&P 500 )?companies have issued positive EPS guidance", t, re.I)
    return m


G_NEXT = [re.compile(p) for p in (
    r"For (?:all of |CY ?)(20\d\d),? (?:analysts are (?:projecting|predicting|expecting|calling for)|analysts (?:predict|project|expect)"
    r"|the (?:projected|estimated) earnings growth rate is)(?: \(year-over- ?year\))?(?: earnings growth(?: rate)? of| earnings to grow by)? (-?\d+(?:\.\d+)?)%",
    r"earnings growth to return in CY ?(20\d\d) \((-?\d+(?:\.\d+)?)%",
    r"growth in earnings in CY ?(20\d\d) \(\+?(-?\d+(?:\.\d+)?)%")]


def parse_g_next(t, year):
    """明年全年 EPS 预期增速（%），周报没写返回 None。全部周报里同一期没有出现过两个不同的值（2026-10-04 核对）。"""
    for pat in G_NEXT:
        for m in pat.finditer(t):
            if int(m.group(1)) == year + 1:
                return float(m.group(2))
    return None


def qyear(qn, day):
    """周报里只写 Q3，补上年份：取最近一个『已开始』的同号季度。"""
    y = day.year
    start = date(y, 3 * (qn - 1) + 1, 1)
    if start > day:
        y -= 1
    return f"{y}Q{qn}"


def gspc_thursday():
    """Yahoo ^GSPC 收盘，按日期查；周报数据截至周四收盘。"""
    import yfinance as yf
    h = yf.Ticker("^GSPC").history(start="2011-01-01", auto_adjust=False)
    return {d.date(): float(c) for d, c in zip(h.index, h["Close"])}


def parse_one(path):
    day = datetime.strptime(os.path.basename(path)[:10], "%Y-%m-%d").date()
    t = clean(open(path).read())
    r = {"date": str(day), "fpe": parse_fpe(t), "px": parse_px(t),
         "fwd_eps_stated": parse_fwd_eps(t), "fwd_eps_qtd": parse_fwd_qtd(t), "g_next": parse_g_next(t, day.year)}
    q = parse_q_eps(t)
    if q:
        r["q"], r["q_eps"], r["q_eps0"] = qyear(q[0], day), q[1], q[2]
        r["q_rev"] = round(q[1] / q[2] - 1, 5)
    cur, start = parse_growth(t)
    if cur:
        r["q_growth_q"], r["q_growth"] = f"{cur.group(2)}Q{cur.group(1)}", float(cur.group(3))
    if start:
        r["q_growth0_q"], r["q_growth0"] = f"{start.group(2)}Q{start.group(1)}", float(start.group(3))
    g = parse_guidance(t)
    if g:
        r["guide_q"], r["guide_neg"], r["guide_pos"] = f"{g.group(2)}Q{g.group(1)}", int(g.group(3)), int(g.group(4))
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true",
                    help="只解析 factset_weekly.csv 里还没有的周报，追加后重算笔误标记"
                         "（GitHub Actions 用：那边 txt/ 里只有新抓的几期）")
    a = ap.parse_args()
    out_path = os.path.join(BASE, "factset_weekly.csv")
    old = list(csv.DictReader(open(out_path))) if a.incremental and os.path.exists(out_path) else []
    known = {r["date"] for r in old}
    rows = [parse_one(p) for p in sorted(glob.glob(os.path.join(BASE, "txt", "*.txt")))
            if os.path.basename(p)[:10] not in known]

    # 周报没写收盘价的，用 ^GSPC 补：取周报日之前最近的周四（不含当天）往前找交易日
    px = {}
    if rows:
        try:
            px = gspc_thursday()
        except Exception as e:
            print("Yahoo ^GSPC 拉取失败：", e)
    for r in rows:
        day = datetime.strptime(r["date"], "%Y-%m-%d").date()
        d = day - timedelta(days=1)
        while d.weekday() > 3 and d > day - timedelta(days=7):
            d -= timedelta(days=1)
        for _ in range(5):
            if d in px:
                break
            d -= timedelta(days=1)
        r["px_yahoo"] = round(px[d], 2) if d in px else None
        # 周报里的「收盘价」偶尔是别的数（12 月那期会抓到年底目标价）：和 Yahoo 差 >1.5% 就用 Yahoo
        ok = r["px"] and r["px_yahoo"] and abs(r["px"] / r["px_yahoo"] - 1) < 0.015
        r["px_used"] = r["px"] if ok or not r["px_yahoo"] else r["px_yahoo"]
        if r["fwd_eps_stated"]:
            r["fwd_eps"] = r["fwd_eps_stated"]
        elif r["fpe"] and r["px_used"]:
            r["fwd_eps"] = round(r["px_used"] / r["fpe"], 2)
    if a.incremental:
        print(f"增量：已有 {len(old)} 期，新增 {len(rows)} 期" + (f"（{'、'.join(r['date'] for r in rows)}）" if rows else ""))
    rows = sorted(old + rows, key=lambda r: r["date"])  # 增量时旧行原样保留（CSV 里读出来的字符串）

    # 周报偶有笔误（如 2021-06-17 市盈率写成 22.4，前后两周都是 21.3）：
    # 偏离前后 5 期中位数 >2% 的点标出来、置空。连续的单边下修（2020-03~04）中位数会跟着走，不会误伤。
    # 窗口含后两期，所以新的一期进来后，前两期的标记也要重算——增量时一样对全部行重算。
    import math, statistics
    num = lambda v: None if v in (None, "") else float(v)
    vals = [num(r.get("fwd_eps")) for r in rows]
    for i, r in enumerate(rows):
        r["fwd_eps_clean"], r["flag"] = r.get("fwd_eps"), ""
        if vals[i] is None:
            continue
        win = [v for v in vals[max(0, i - 2):i + 3] if v is not None]
        if len(win) >= 4:
            med = statistics.median(win)
            if abs(math.log(vals[i] / med)) > 0.02:
                r["fwd_eps_clean"], r["flag"] = None, f"outlier vs median {med:.2f}"

    cols = ["date", "px", "px_yahoo", "px_used", "fpe", "fwd_eps_stated", "fwd_eps", "fwd_eps_clean", "flag", "fwd_eps_qtd",
            "q", "q_eps", "q_eps0", "q_rev", "q_growth_q", "q_growth", "q_growth0_q", "q_growth0",
            "guide_q", "guide_neg", "guide_pos", "g_next"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    for c in ("fpe", "px", "fwd_eps_stated", "fwd_eps", "fwd_eps_clean", "flag", "fwd_eps_qtd", "q_eps", "q_growth", "guide_neg", "g_next"):
        print(f"{c:15s} {sum(1 for r in rows if r.get(c) not in (None, ''))}/{n}")

if __name__ == "__main__":
    main()
