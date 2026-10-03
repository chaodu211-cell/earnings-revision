#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactSet《Earnings Insight》周报抓取 —— 只存抽出的文字，不存 PDF。

来源：
  2011-04 ~ 2016-12  FactSet 旧网址已下线，从 Internet Archive 取（清单见 _wayback_cdx.txt）
  2016-12 ~ 今       advantage.factset.com/hubfs/... 直接下载，按周五（节假日退到周四/周三）试

网络：走 Veee 本地代理 127.0.0.1:15236（直连不通）。EI_PROXY 置空 = 直连（GitHub Actions 上这样跑）。

用法：
    python3 fetch_factset.py            # 增量：已有 txt/<日期>.txt 的跳过
    python3 fetch_factset.py --new      # 只抓 2016-12 以后的
    python3 fetch_factset.py --old      # 只抓存档部分
    python3 fetch_factset.py --since-csv  # 只抓 factset_weekly.csv 最新一期之后的周（GitHub Actions 用：那边没有 txt/ 全量）

输出：txt/YYYY-MM-DD.txt（日期 = 周报落款日），_missing.txt（没找到的周）
"""
import argparse, os, re, sys, time, urllib.error, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import fitz  # PyMuPDF

BASE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(BASE, "txt")
PROXY = os.environ.get("EI_PROXY", "http://127.0.0.1:15236")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
HUB = ("https://advantage.factset.com/hubfs/Website/Resources%20Section/"
       "Research%20Desk/Earnings%20Insight/EarningsInsight_{d}{s}.pdf")
NEW_START = date(2016, 12, 16)
STATUS = Counter()  # 每次请求的结果：404 = 还没发布；403 多半是被挡了（GitHub Actions 上用来区分）

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY} if PROXY else {}))


def get(url, tries=3, timeout=90):
    k = waited = 0
    while k < tries:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with opener.open(req, timeout=timeout) as r:
                b = r.read()
                STATUS[200] += 1
                return b
        except urllib.error.HTTPError as e:
            STATUS[e.code] += 1
            if e.code in (403, 404, 410):
                return None
            if e.code == 429 and waited < 1800:  # 限流：等够了再试，不算一次尝试（最多累计等 30 分钟）
                wait = min(max(int(e.headers.get("Retry-After") or 0), 60), 300)
                time.sleep(wait)
                waited += wait
                continue
            k += 1
            time.sleep(3 * k)
        except Exception:
            STATUS["err"] += 1
            k += 1
            time.sleep(3 * k)
    return None


def save_text(pdf_bytes, day):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text() for p in doc)
    with open(os.path.join(TXT, f"{day:%Y-%m-%d}.txt"), "w") as f:
        f.write(text)


def have(day):
    return os.path.exists(os.path.join(TXT, f"{day:%Y-%m-%d}.txt"))


# ---------- 2016-12 以后：官网 ----------
def try_week(friday):
    """一周内按 周五 → 周四 → 周三 → 下周一 试；后缀按 '' → A → B 试。"""
    days = [friday + timedelta(days=o) for o in (0, -1, -2, 3)]
    days = [d for d in days if d <= date.today()]
    for day in days:  # 先查本地，任何一天有了就不发请求
        if have(day):
            return day, "cached"
    for day in days:
        for s in ("", "A", "B"):
            b = get(HUB.format(d=f"{day:%m%d%y}", s=s), tries=2, timeout=60)
            if b and b[:4] == b"%PDF":
                save_text(b, day)
                return day, f"hub{s}"
    return friday, None


def csv_last_date():
    """factset_weekly.csv 里最新一期的日期。"""
    with open(os.path.join(BASE, "factset_weekly.csv")) as fh:
        last = [l for l in fh if l[:4].isdigit()][-1]
    return date.fromisoformat(last[:10])


def fetch_new(since=None):
    fridays = []
    d = NEW_START
    while d <= date.today():
        fridays.append(d)
        d += timedelta(days=7)
    if since:  # 只试 CSV 最新一期之后的周
        fridays = [d for d in fridays if d > since]
    # 上次就没找到的周（节假日停刊）不再反复试，只重试最近 3 周
    known = set()
    mp = os.path.join(BASE, "_missing_new.txt")
    if os.path.exists(mp):
        known = {l.strip() for l in open(mp) if l.strip()}
    recent = date.today() - timedelta(days=21)
    skipped = [d for d in fridays if str(d) in known and d < recent]
    fridays = [d for d in fridays if d not in skipped]
    miss = skipped[:]
    with ThreadPoolExecutor(2) as ex:
        for fri, (day, how) in zip(fridays, ex.map(try_week, fridays)):
            if how is None:
                miss.append(fri)
            print(f"{fri}  ->  {day} {how}", flush=True)
    return miss


# ---------- 2011-04 ~ 2016-12：Internet Archive ----------
def wayback_list():
    best = {}
    for line in open(os.path.join(BASE, "_wayback_cdx.txt")):
        parts = line.split()
        if len(parts) < 4 or parts[3] != "application/pdf":
            continue
        url, ts = parts[0], parts[1]
        m = re.search(r"earningsinsight_?(\d{1,2})\.(\d{1,2})\.(\d{2})", url, re.I)
        if not m:
            continue
        try:
            day = date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            continue
        # 同一天有多个快照时取最早的
        if day not in best or ts < best[day][0]:
            best[day] = (ts, url)
    return dict(sorted(best.items()))


def fetch_old():
    miss = []
    for day, (ts, url) in wayback_list().items():
        if have(day):
            continue
        b = get(f"https://web.archive.org/web/{ts}id_/{url}", tries=4, timeout=120)
        if b and b[:4] == b"%PDF":
            save_text(b, day)
            print(f"{day} wayback ok", flush=True)
        else:
            miss.append(day)
            print(f"{day} wayback FAIL", flush=True)
        time.sleep(1.5)  # 存档站限流
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--since-csv", action="store_true",
                    help="只抓 factset_weekly.csv 最新一期之后的周（隐含 --new）")
    a = ap.parse_args()
    since = csv_last_date() if a.since_csv else None
    if since:
        print(f"since {since}（factset_weekly.csv 最新一期）", flush=True)
    both = not (a.new or a.old or a.since_csv)
    os.makedirs(TXT, exist_ok=True)
    for tag, run in (("new", a.new or a.since_csv or both), ("old", a.old or both)):
        if not run:
            continue
        miss = fetch_new(since) if tag == "new" else fetch_old()
        mp = os.path.join(BASE, f"_missing_{tag}.txt")
        if since and os.path.exists(mp):  # 只试了最近几周：保留原清单，并上新缺的
            miss = sorted(set(miss) | {date.fromisoformat(l.strip()) for l in open(mp) if l.strip()})
        with open(mp, "w") as f:
            f.write("\n".join(str(d) for d in sorted(miss)))
        print(f"{tag} done, missing {len(miss)}")
    print("HTTP_STATUS", " ".join(f"{k}:{v}" for k, v in sorted(STATUS.items(), key=str)) or "none", flush=True)


if __name__ == "__main__":
    main()
