#!/usr/bin/env python3
"""SK 海力士固定财年 EPS 修正图；不改动原有指数模型。

python3 sk_hynix/build.py --refresh  # 更新公开源，再生成离线 HTML 和 PNG
python3 sk_hynix/build.py            # 从保存的原始数据重建
"""
import argparse
import ast
import json
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
CN = ZoneInfo("Asia/Shanghai")
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://navercomp.wisereport.co.kr/"}
SOURCES = [
    {"name": "FnGuide / WiseReport：固定财年 EPS 一致预期", "url": "https://navercomp.wisereport.co.kr/v2/company/c1050001.aspx?cmp_cd=000660"},
    {"name": "FnGuide：交叉核验", "url": "https://wcomp.fnguide.com/CompanyInfo/Consensus?cmp_cd=000660"},
    {"name": "Naver：韩国上市股价", "url": "https://stock.naver.com/domestic/stock/000660/price"},
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def parse_wise(value):
    c = value["chart1"]
    c = json.loads(c) if isinstance(c, str) else c
    if c.get("item_name") != "EPS" or c.get("item_unit") != "원":
        raise ValueError("数据项不是韩元 EPS，停止生成")
    if not len(c["categories"]) == len(c["avg"]) == len(c["min_max"]):
        raise ValueError("EPS 日期、均值、上下限长度不符")
    points = []
    for d, v, bounds in zip(c["categories"], c["avg"], c["min_max"]):
        if v is None:
            continue
        lo, hi = bounds
        if lo is not None and hi is not None and not lo <= v <= hi:
            raise ValueError("EPS 均值不在供应商的分析师范围内")
        points.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "eps": v, "low": lo, "high": hi})
    dates = [p["date"] for p in points]
    if dates != sorted(set(dates)) or len(points) < 6:
        raise ValueError("EPS 日期乱序、重复或数据不足")
    return points


def parse_price(text, now):
    rows = ast.literal_eval(text.strip())
    df = pd.DataFrame(rows[1:], columns=["date", "open", "high", "low", "close", "volume", "foreign_pct"])
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
    # 韩国 15:30 收市，留到16:30以后才采用当日完整日线。
    korea = now.tz_convert("Asia/Seoul")
    cutoff = pd.Timestamp(korea.date())
    valid = df.date <= cutoff if korea.hour * 60 + korea.minute >= 16 * 60 + 30 else df.date < cutoff
    df = df.loc[valid & df.close.gt(0)].sort_values("date")
    if df.date.duplicated().any() or len(df) < 150:
        raise ValueError("日线重复或历史不足")
    return [[d.strftime("%Y-%m-%d"), float(p)] for d, p in zip(df.date, df.close)]


def monthly_metrics(points, now):
    """直接对同财年EPS做相邻月份向后差分；保留缺月，不插值、不回填。"""
    d = pd.DataFrame(points)
    dates = pd.to_datetime(d.date)
    periods = dates.dt.to_period("M")
    # 月末可能因周末/休市落在28/29/30日。当前月和中途终止的点只画EPS。
    complete = (periods < now.tz_localize(None).to_period("M")) & ((dates.dt.days_in_month - dates.dt.day) <= 7)
    d = d.loc[complete].copy()
    if d.empty:
        return []
    d["period"] = pd.to_datetime(d.date).dt.to_period("M")
    d = d.drop_duplicates("period", keep="last").set_index("period")
    grid = pd.period_range(d.index.min(), d.index.max(), freq="M")
    g = d.reindex(grid)
    # 直接对EPS金额求月差；负EPS和过零仍有效。每个目标财年独立计算。
    eps = pd.to_numeric(g.eps)
    g["d_eps"] = eps.diff()
    g["dd_eps"] = g.d_eps.diff()
    clean = lambda v: None if pd.isna(v) else float(v)
    return [{"date": r.date if pd.notna(r.date) else period.end_time.strftime("%Y-%m-%d"),
             "observed_date": r.date if pd.notna(r.date) else None, "eps": clean(r.eps),
             "d_eps": clean(r.d_eps), "dd_eps": clean(r.dd_eps)}
            for period, r in g.iterrows()]


def refresh(now, years):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    RAW.mkdir(exist_ok=True, parents=True)
    manifest = {"fetched_at": now.isoformat(), "security": "000660 / SK hynix / KOSPI", "currency": "KRW", "requests": []}

    def get(url, is_json=True):
        errors = []
        for proxy in (os.environ.get("SK_PROXY", ""), "http://127.0.0.1:15236"):
            try:
                session = requests.Session()
                session.trust_env = False
                response = session.get(url, headers=HEADERS, timeout=25,
                                       proxies={"http": proxy, "https": proxy} if proxy else {})
                response.raise_for_status()
                response.encoding = "utf-8-sig"
                return response.json() if is_json else response.text
            except Exception as e:
                errors.append(str(e))
        raise RuntimeError("公开数据请求失败：" + "; ".join(errors))

    def fetch_year(year):
        url = ("https://comp.wisereport.co.kr/company/ajax/cF5002.aspx?cmp_cd=000660"
               f"&dt={now:%Y%m%d}&yymm={year}12&frq=Y&acc_cd=312000&fingubun=IFRSL&chartType=svg")
        wise = get(url)
        points = parse_wise(wise)
        fn_url = ("https://wcomp.fnguide.com/CompanyInfo/getCnsTrendChart?cmp_cd=000660"
                  f"&consol_typ=C&freq_typ=Y&select_gsym={year}12&data_typ=3")
        fn = get(fn_url)
        crosscheck(points, fn)
        return year, wise, fn, {"year": year, "wise_url": url, "fnguide_url": fn_url,
                                "first": points[0]["date"], "last": points[-1]["date"], "points": len(points)}

    # 先完成全部请求与验证，再替换数据，失败不覆盖上一次有效缓存。
    with ThreadPoolExecutor(max_workers=2) as pool:
        fetched = list(pool.map(fetch_year, years))
    price_url = ("https://api.finance.naver.com/siseJson.naver?symbol=000660&requestType=1"
                 f"&startTime=20200101&endTime={now:%Y%m%d}&timeframe=day")
    price_text = get(price_url, False)
    parse_price(price_text, now)
    for year, wise, fn, info in fetched:
        write_json(RAW / f"wise_eps_{year}.json", wise)
        write_json(RAW / f"fnguide_eps_{year}.json", fn)
        manifest["requests"].append(info)
    (RAW / "naver_price.txt").write_text(price_text)
    manifest["price_url"] = price_url
    write_json(RAW / "manifest.json", manifest)
    save_snapshot(now, fetched)


def save_snapshot(now, fetched):
    """只记录采集时实际可见的最新预测，不把月末历史插值成每周数据。"""
    snapshot = {"fetched_at": now.isoformat(), "currency": "KRW", "source": "FnGuide / WiseReport",
                "observations": [{"fiscal_year": year, **parse_wise(wise)[-1]}
                                 for year, wise, _, _ in fetched]}
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    path = RAW / "snapshots" / f"{stamp}.json"
    if not path.exists():
        write_json(path, snapshot)


def crosscheck(points, fn):
    lookup = {p["date"]: p for p in points}
    checked = 0
    for r in fn["dataset"]["data"]:
        date = pd.to_datetime(r["TRD_DT"], format="%y/%m/%d").strftime("%Y-%m-%d")
        if date not in lookup or r["VAL_AVG"] is None:
            continue
        for a, b in (("eps", "VAL_AVG"), ("low", "VAL_MIN"), ("high", "VAL_MAX")):
            if lookup[date][a] is None and r[b] is None:
                continue
            if lookup[date][a] is None or r[b] is None or abs(lookup[date][a] - r[b]) > .011:
                raise ValueError(f"FnGuide 与 WiseReport EPS 交叉验证不符：{date} {a}")
        checked += 1
    if checked < 6:
        raise ValueError("可交叉验证的EPS观测不足6个")
    return checked


def dataset(now, years):
    manifest = json.loads((RAW / "manifest.json").read_text())
    targets = []
    for year in years:
        points = parse_wise(json.loads((RAW / f"wise_eps_{year}.json").read_text()))
        fn = json.loads((RAW / f"fnguide_eps_{year}.json").read_text())
        checked = crosscheck(points, fn)
        # 缓存中的未来日期不应混入一次重建。
        points = [p for p in points if p["date"] <= now.strftime("%Y-%m-%d")]
        monthly = monthly_metrics(points, now)
        valid = [r for r in monthly if r["d_eps"] is not None]
        targets.append({"year": year, "points": points, "monthly": monthly, "latest": points[-1],
                        "derivative_latest": valid[-1] if valid else None, "crosschecked_points": checked})
    price = parse_price((RAW / "naver_price.txt").read_text(), now)
    start = min(p["date"] for t in targets for p in t["points"])
    price = [p for p in price if p[0] >= start]
    data = {"generated": now.strftime("%Y-%m-%d %H:%M") + " 北京时间", "fetched_at": manifest["fetched_at"],
            "company": "SK 海力士", "symbol": "000660", "currency": "KRW", "price": price,
            "price_asof": price[-1][0], "targets": targets, "source_links": SOURCES,
            "coverage_note": "公开历史为月末观测及最新值，约一年。图3、图4分别为图2 EPS的月度一阶、二阶差分；固定财年独立计算。当前月最新值只画EPS。",
            "method": {"derivatives": "backward_calendar_month_differences", "first": "EPS_t - EPS_t_minus_1_month",
                       "second": "EPS_t - 2*EPS_t_minus_1_month + EPS_t_minus_2_months",
                       "raw_units": ["KRW/share/month", "KRW/share/month^2"],
                       "display_divisor": 10000, "complete_months_only": True}}
    return data


def static_plot(data):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/skhynix_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager, ticker
    for path in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(4, 1, figsize=(15, 12.5), sharex=True,
                            gridspec_kw={"height_ratios": [1.25, 1.15, 1, .95]})
    fig.patch.set_facecolor("#f5f7fb")
    fig.subplots_adjust(top=.835, bottom=.12, left=.09, right=.90, hspace=.36)
    fig.text(.09, .951, "SK 海力士｜EPS 预期及其一阶、二阶导", fontsize=23, weight="bold", color="#0f172a")
    fig.text(.09, .920, "000660 · KOSPI · 韩元  /  月频一致预期 · 固定财年 · 不含年度滚动", fontsize=12, color="#475569")
    colors = ["#2563eb", "#0d9488"]
    for j, t in enumerate(data["targets"]):
        latest, derivative = t["latest"], t["derivative_latest"]
        value = f"{t['year']}年 EPS  {latest['eps']/10000:.2f}万韩元"
        value += f"   ·   一阶导 {derivative['d_eps']/10000:+.2f}万/月" if derivative else ""
        fig.text(.09 + j*.43, .877, value, fontsize=12, color=colors[j], weight="bold")
    price = pd.DataFrame(data["price"], columns=["date", "close"])
    axs[0].plot(pd.to_datetime(price.date), price.close/10000, color="#1e293b", lw=1.6)
    titles = ["01  股价走势", "02  固定财年 EPS 一致预期", "03  EPS 一阶导：相邻月份 EPS 的变化", "04  EPS 二阶导：相邻月份一阶导的变化"]
    labels = ["万韩元", "万韩元 / 股", "万韩元 / 股 / 月", "万韩元 / 股 / 月²"]
    for ax, title, label in zip(axs, titles, labels):
        ax.set_facecolor("white")
        ax.set_title(title, loc="left", fontsize=12, pad=10, color="#0f172a", weight="bold")
        ax.set_ylabel(label, color="#64748b", fontsize=10)
        ax.grid(axis="y", color="#e8edf4", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#dbe2ec")
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    for j, t in enumerate(data["targets"]):
        color = colors[j]
        p = pd.DataFrame(t["points"])
        d = pd.to_datetime(p.date)
        axs[1].plot(d, p.eps/10000, color=color, marker="o", ms=4, lw=1.8, label=f"{t['year']} 财年")
        axs[1].annotate(f"{p.eps.iloc[-1]/10000:.2f}", (d.iloc[-1], p.eps.iloc[-1]/10000),
                        xytext=(8, 0), textcoords="offset points", va="center", color=color, fontsize=10)
        m = pd.DataFrame(t["monthly"])
        for ax, key in ((axs[2], "d_eps"), (axs[3], "dd_eps")):
            ax.plot(pd.to_datetime(m.date), pd.to_numeric(m[key])/10000, marker="o", ms=4, color=color, lw=1.7)
            valid = m.loc[m[key].notna()]
            if not valid.empty:
                r = valid.iloc[-1]
                endpoints = [target["derivative_latest"][key] for target in data["targets"]
                             if target["derivative_latest"] and target["derivative_latest"][key] is not None]
                offset = 8 if r[key] >= sum(endpoints) / len(endpoints) else -10
                ax.annotate(f"{r[key]/10000:+.2f}", (pd.Timestamp(r.date), r[key]/10000),
                            xytext=(8, offset), textcoords="offset points", color=color, fontsize=9)
    axs[1].legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    for ax in axs[2:]:
        ax.axhline(0, color="#94a3b8", lw=.9, ls="--")
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axs[-1].set_xlim(pd.Timestamp(price.date.min()), pd.Timestamp(price.date.max()) + pd.Timedelta(days=15))
    eps_asof = max(t["latest"]["date"] for t in data["targets"])
    derivative_asof = max(t["derivative_latest"]["date"] for t in data["targets"] if t["derivative_latest"])
    fig.text(.09, .077, f"数据截至：股价 {data['price_asof']} ｜ EPS {eps_asof} ｜ 月度导数 {derivative_asof}", fontsize=10, color="#475569")
    fig.text(.09, .053, "一阶导 ≈ EPS本月 − EPS上月；二阶导 ≈ EPS本月 − 2×EPS上月 + EPS前月。按1个月为步长；缺月留空。", fontsize=9, color="#64748b")
    fig.text(.09, .032, "来源：FnGuide / WiseReport（EPS，按固定2026/2027财年分别计算）；Naver（日线）。10月未满月，不参与差分。", fontsize=9, color="#64748b")
    fig.savefig(BASE / "SK海力士_盈利修正.png", dpi=170, facecolor=fig.get_facecolor())
    # HTML直接包含矢量曲线。即使宿主不运行脚本，也无需图片请求即可显示。
    from matplotlib.transforms import Bbox
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(BASE / "chart_inline.svg", format="svg", facecolor=fig.get_facecolor(),
                    bbox_inches=Bbox.from_extents(.25, 1.05, 14.7, 10.75))
    plt.close(fig)


def render_html(data):
    from static_content import render_static_content
    template = (BASE / "chart_template.html").read_text()
    static = render_static_content(data)
    for key, value in static.items():
        template = template.replace(f"/*STATIC_{key.upper()}*/", value)
    svg = (BASE / "chart_inline.svg").read_text()
    svg = svg[svg.index("<svg"):]
    end = svg.index(">")
    opening = re.sub(r'\s(?:width|height)="[^"]*"', "", svg[:end])
    svg = opening.replace("<svg ", '<svg role="img" aria-label="SK海力士股价和固定财年盈利修正图" ', 1) + svg[end:]
    template = template.replace("/*STATIC_CHART*/", svg)
    echarts = (BASE.parent / "vendor" / "echarts.min.js").read_text()
    html = template.replace("/*ECHARTS*/", echarts).replace("/*DATA*/", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    (BASE / "SK海力士_盈利修正.html").write_text(html)
    # 新文件名确保再次打开时不沿用宿主缓存的空白页。
    (BASE / "SK海力士_盈利修正_兼容版.html").write_text(html)
    (BASE / "SK海力士_EPS一阶二阶导.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    now = pd.Timestamp.now(tz=CN)
    years = [now.year, now.year + 1]
    if args.refresh:
        refresh(now, years)
    data = dataset(now, years)
    write_json(BASE / "chart_data.json", data)
    rows = [{"fiscal_year": t["year"], **r} for t in data["targets"] for r in t["monthly"]]
    pd.DataFrame(rows).to_csv(BASE / "monthly_derivatives.csv", index=False)
    (BASE / "monthly_revision.csv").unlink(missing_ok=True)
    static_plot(data)
    render_html(data)
    for target in data["targets"]:
        print(json.dumps({k: target[k] for k in ("year", "latest", "derivative_latest", "crosschecked_points")}, ensure_ascii=False))
    print(BASE / "SK海力士_盈利修正.html")


if __name__ == "__main__":
    main()
