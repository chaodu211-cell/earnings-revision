"""Render the chart's readable HTML content without relying on JavaScript."""

from datetime import date
from html import escape
import math
from typing import Any, Mapping
from urllib.parse import urlsplit


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _html(value: Any) -> str:
    return escape(_text(value), quote=True)


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _num(value: Any, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}" if _number(value) else "—"


def _percent(value: Any, suffix: str = "%") -> str:
    if not _number(value) or not _number(value * 100):
        return "—"
    return f"{'+' if value > 0 else ''}{_num(value * 100)}{suffix}"


def _derivative(value: Any, unit: str) -> str:
    if not _number(value):
        return "—"
    return f"{'+' if value > 0 else ''}{_num(value / 10000)} {unit}"


def _date(value: Any) -> str:
    text = _text(value)[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return ""


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any) -> list:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, (list, tuple)) else []


def _latest_point(target: Mapping) -> Mapping:
    latest = _mapping(target.get("latest"))
    if latest:
        return latest
    points = [item for item in _records(target.get("points")) if _date(item.get("date"))]
    return max(points, key=lambda item: _date(item.get("date")), default={})


def _latest_derivative(target: Mapping, reference_month: str) -> Mapping:
    candidates = _records(target.get("monthly"))
    supplied = _mapping(target.get("derivative_latest"))
    if supplied:
        candidates.append(supplied)
    complete = [
        item for item in candidates
        if _date(item.get("date"))
        and (not reference_month or _date(item.get("date"))[:7] < reference_month)
    ]
    return max(complete, key=lambda item: _date(item.get("date")), default={})


def render_static_content(data: Mapping) -> dict[str, str]:
    """Return escaped cards, timestamp, coverage text, and allowed source links.

    EPS, prices, and monthly EPS differences are supplied in KRW. EPS differences
    are divided by 10,000 for display in ten-thousand-KRW per month or per month
    squared. The two fiscal-year cards remain visible even if the input is empty.
    """
    data = _mapping(data)
    targets = _records(data.get("targets"))
    target_by_year = {_text(target.get("year")): target for target in targets}
    latest_dates = [_date(_latest_point(target).get("date")) for target in targets]
    fallback_date = max([_date(data.get("price_asof")), *latest_dates], default="")
    reference_month = (_date(data.get("generated")) or fallback_date)[:7]

    price_rows = data.get("price")
    prices = [
        (_date(row[0]), row[1]) for row in price_rows
        if isinstance(row, (list, tuple)) and len(row) >= 2
        and _date(row[0]) and _number(row[1])
    ] if isinstance(price_rows, (list, tuple)) else []
    prices.sort(key=lambda row: row[0])
    latest_price = prices[-1] if prices else ("", None)
    previous_price = prices[-2] if len(prices) > 1 else ("", None)
    price_change = (
        latest_price[1] / previous_price[1] - 1
        if _number(previous_price[1]) and previous_price[1] > 0 else None
    )
    cards = [
        '<article class="card"><div class="card-label">最新收盘价</div>'
        f'<div class="card-value">{_html(_num(latest_price[1], 0))}<small>韩元</small></div>'
        f'<div class="card-meta">{_html(latest_price[0] or "—")}</div>'
        '<div class="card-detail"><span>较前一交易日</span>'
        f'<b>{_html(_percent(price_change))}</b></div></article>'
    ]
    for year, color in (("2026", "#2563eb"), ("2027", "#0891b2")):
        target = target_by_year.get(year, {})
        latest = _latest_point(target)
        derivative = _latest_derivative(target, reference_month)
        eps = latest.get("eps")
        latest_date = _date(latest.get("date"))
        derivative_date = _date(derivative.get("date"))
        current_month = bool(reference_month and latest_date and latest_date[:7] >= reference_month)
        meta = f'观测 {latest_date or "—"}' + (" · 当月未满月" if current_month else "")
        derivative_label = f"{derivative_date[:7]} 完整月" if derivative_date else "最新完整月"
        cards.append(
            f'<article class="card" style="border-top-color:{color}">'
            f'<div class="card-label">{_html(year)} 年 EPS 一致预期</div>'
            f'<div class="card-value" style="color:{color}">{_html(_num(eps / 10000 if _number(eps) else None))}'
            '<small>万韩元</small></div>'
            f'<div class="card-meta">{_html(meta)}</div>'
            f'<div class="card-detail"><span>{_html(derivative_label)}<br>'
            '<small>月度一阶 / 二阶差分</small></span>'
            f'<b>一阶导 {_html(_derivative(derivative.get("d_eps"), "万韩元/月"))}<br>'
            f'<small>二阶导 {_html(_derivative(derivative.get("dd_eps"), "万韩元/月²"))}</small>'
            '</b></div></article>'
        )

    sources = []
    for source in _records(data.get("source_links")):
        url = _text(source.get("url"))
        try:
            parsed = urlsplit(url)
            allowed = parsed.scheme.lower() in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            allowed = False
        if allowed:
            label = source.get("name") or url
            sources.append(
                f'<a href="{_html(url)}" target="_blank" rel="noopener noreferrer">{_html(label)}</a>'
            )

    fetched_at = _text(data.get("fetched_at"), "—")[:16].replace("T", " ")
    stamp = (
        f'股价截至 {_date(data.get("price_asof")) or latest_price[0] or "—"}'
        f'　·　数据采集 {fetched_at or "—"}'
        f'　·　页面生成 {_text(data.get("generated")) or "—"}'
    )
    return {
        "cards": "".join(cards),
        "stamp": _html(stamp),
        "coverage": _html(data.get("coverage_note") or "一致预期按可获得的月份观测展示。"),
        "sources": "".join(sources) or "FnGuide · Naver",
    }
