#!/usr/bin/env python3
"""Plot top confirmation events at h=0.5pp for base windows 8/13/26.

Uses the production page's revision convention and an eight-week acceleration
comparison. The 13-week-base markers are checked against the saved HTML itself;
8/26-week bases extend the same production formula, not the page's N selector.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/earnings-window-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze import next_growth, time_roll

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = HERE / '纳斯达克_去噪0.5确认点_8周13周26周对照'
STYLE = {8: ('#14856e', 'o', '8周'),
         13: ('#2563a6', 'v', '13周'),
         26: ('#8a4bb4', 'D', '26周')}


def confirmation_dates(revision, states):
    """Use the same phase-sign iteration as markSets in the saved HTML."""
    previous = 0
    events = []
    for date, value in revision.items():
        state = states.loc[date]
        if state < 0 and previous > 0 and value > 0:
            events.append(date)
        if pd.notna(state) and state != 0:
            previous = state
    return pd.DatetimeIndex(events)


def production_features():
    html = (BASE/'修正速度与纳指.html').read_text()
    prefix = 'const D = '
    data, _ = json.JSONDecoder().raw_decode(html[html.index(prefix)+len(prefix):])
    index = pd.DatetimeIndex([row[0] for row in data['weekly']])
    rt = pd.read_csv(BASE/'_weekly_rt.csv', index_col=0, parse_dates=True)
    growth, _ = next_growth(rt.index)
    roll = time_roll(rt.index, growth)
    log_final = np.log(rt.eps_final)
    features = pd.DataFrame(index=index)
    states = {}
    for weeks in STYLE:
        rolling = roll.rolling(weeks).sum()
        revision = (np.log(rt.eps_raw)-log_final.shift(weeks)).where(rt.ok)-rolling
        revision_final = log_final.diff(weeks)-rolling
        acceleration = revision-revision_final.shift(8)
        history, current = [], None
        for value in acceleration:
            if pd.notna(value):
                if value > .005:
                    current = 1
                elif value < -.005:
                    current = -1
                elif current is None:
                    current = 1 if value >= 0 else -1
            history.append(current)
        states[weeks] = pd.Series(history, index=rt.index, dtype=float).where(
            revision.notna()).ffill(limit=4).reindex(index)
        features[f'x{weeks}'] = revision.reindex(index)
        features[f'a{weeks}'] = acceleration.reindex(index)
    # The production 13-week formula must match its cached calculation before
    # validating marker dates against the actual HTML (including phase carries).
    assert np.nanmax(abs(features.x13-rt.x13.reindex(index))) < 1e-10
    assert np.nanmax(abs(features.a13-rt.a8.reindex(index))) < 1e-10
    phase = data['speed']['rt|8|0.5']['phase']
    web_states = pd.Series([1 if p in ('A','D') else -1 if p in ('B','C') else 0
                            for p in phase], index=index)
    web_x = pd.Series([np.nan if row[7] is None else row[7]/100
                       for row in data['weekly']], index=index)
    web_dates = confirmation_dates(web_x, web_states)
    assert confirmation_dates(features.x13, states[13]).equals(web_dates)
    return features, states, data['fresh']


def main():
    features, states, fresh = production_features()
    nq = pd.read_csv(BASE/'_nasdaq.csv', index_col=0, parse_dates=True)['^IXIC'].dropna()
    prices = nq.loc['2018-01-01':features.index.max()]
    dates = {}
    for weeks in STYLE:
        ds = confirmation_dates(features[f'x{weeks}'], states[weeks])
        dates[weeks] = ds[(ds >= prices.index.min()) & (ds <= prices.index.max())]
        assert features.loc[dates[weeks], f'x{weeks}'].gt(0).all()
        assert features.loc[dates[weeks], f'a{weeks}'].lt(-.005).all()

    font = '/System/Library/Fonts/PingFang.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.77)
    ax.axvspan(prices.index.min(), pd.Timestamp('2021-01-01'), color='#f1f3f6', zorder=0)
    ax.axvspan(pd.Timestamp('2026-05-22'), prices.index.max()+pd.Timedelta(days=60),
               color='#fff7e9', zorder=0)
    for day in ('2021-01-01', '2026-05-22'):
        ax.axvline(pd.Timestamp(day), color='#b8bec7', lw=.8, ls='--')
    ax.plot(prices.index, prices, color='#29394b', lw=1.4, zorder=2)
    points = []
    # Larger purple diamonds remain visible around the blue triangles when
    # both windows confirm on the same report date.
    for weeks in (26, 13, 8):
        color, marker, _ = STYLE[weeks]
        vals = nq.reindex(dates[weeks], method='ffill')
        ax.scatter(vals.index, vals.values, marker=marker,
                   s={8: 85, 13: 90, 26: 175}[weeks], color=color,
                   edgecolors='white', linewidths=1., zorder={26:5,13:6,8:7}[weeks])
        for date, price in vals.items():
            price_day = nq.index[nq.index.searchsorted(date, side='right')-1]
            points.append({'修正窗口_周': weeks, '确认日期': str(date.date()),
                           '对应收盘日期': str(price_day.date()),
                           '纳斯达克综合收盘': float(price),
                           '纯修正_百分点': float(features.loc[date, f'x{weeks}']*100),
                           '二阶导_百分点': float(features.loc[date, f'a{weeks}']*100)})

    # Date labels use separate vertical lanes by window, with explicit offsets
    # for dense clusters. Their connector ends always remain at the true date.
    overrides = {
        (8, '2018-03-23'): (-36, -33, 'right'),
        (8, '2018-07-13'): (18, -38, 'left'),
        (13, '2018-04-06'): (-15, 40, 'right'),
        (26, '2018-07-06'): (0, 85, 'center'),
        (13, '2018-08-31'): (20, 32, 'left'),
        (8, '2020-10-02'): (-15, -38, 'right'),
        (13, '2020-12-04'): (-15, 32, 'right'),
        (26, '2021-02-12'): (-15, 85, 'right'),
        (8, '2021-01-08'): (-18, -38, 'right'),
        (8, '2021-04-02'): (-18, -73, 'right'),
        (8, '2021-07-02'): (-15, -38, 'right'),
        (8, '2021-09-24'): (8, -73, 'left'),
        (13, '2021-07-30'): (-16, 40, 'right'),
        (13, '2021-07-09'): (-20, 75, 'right'),
        (26, '2021-09-10'): (13, 116, 'left'),
        (26, '2021-04-23'): (-12, 115, 'right'),
        (26, '2021-10-29'): (15, 85, 'left'),
        (13, '2021-10-29'): (15, 35, 'left'),
        (8, '2022-04-01'): (-15, -40, 'right'),
        (8, '2022-04-08'): (-15, -40, 'right'),
        (8, '2022-06-03'): (-10, -73, 'right'),
        (26, '2022-04-15'): (15, 85, 'left'),
        (26, '2022-05-27'): (14, 115, 'left'),
        (26, '2022-07-08'): (24, 68, 'left'),
        (13, '2022-05-13'): (15, 28, 'left'),
        (8, '2025-10-17'): (-15, -35, 'right'),
        (13, '2025-10-31'): (-20, 40, 'right'),
        (13, '2025-12-05'): (-20, 68, 'right'),
        (8, '2026-01-16'): (-15, -68, 'right'),
        (13, '2026-02-13'): (-7, 38, 'right'),
        (26, '2026-02-13'): (-7, 103, 'right'),
        (8, '2026-06-26'): (-5, -35, 'right'),
        (13, '2026-07-24'): (-7, 39, 'right'),
        (26, '2026-07-03'): (-55, 115, 'right'),
        (13, '2026-09-11'): (-7, 70, 'right'),
        (13, '2026-10-16'): (-7, 40, 'right'),
        (8, '2026-10-09'): (-5, -85, 'right'),
        (26, '2026-10-16'): (-7, 105, 'right'),
    }
    for p in points:
        weeks = p['修正窗口_周']
        date, price = pd.Timestamp(p['确认日期']), p['纳斯达克综合收盘']
        color = STYLE[weeks][0]
        dx, dy, ha = overrides.get((weeks, p['确认日期']), (0, 32, 'center'))
        ax.annotate(p['确认日期'], xy=(date, price), xytext=(dx, dy),
                    textcoords='offset points', fontsize=9.5, color=color,
                    ha=ha, va='bottom' if dy > 0 else 'top', annotation_clip=False,
                    zorder=8,
                    arrowprops={'arrowstyle':'-', 'lw':.8, 'color':color, 'alpha':.7},
                    bbox={'boxstyle':'round,pad=.14', 'fc':'white', 'ec':'none', 'alpha':.94})

    ax.set_yscale('log')
    ax.set_ylim(prices.min()*.77, prices.max()*1.32)
    ax.set_xlim(prices.index.min()-pd.Timedelta(days=40), prices.index.max()+pd.Timedelta(days=60))
    ax.yaxis.set_major_locator(ticker.FixedLocator([5000,7500,10000,15000,20000,25000,30000]))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', color='#e4e9ee', lw=.7)
    ax.grid(axis='x', color='#eef1f4', lw=.5)
    ax.set_ylabel('纳斯达克综合指数收盘价（对数轴）', labelpad=14)
    ax.tick_params(axis='both', labelsize=11, length=0, pad=8)
    for side in ('left','bottom'):
        ax.spines[side].set_color('#bdc6d0')
    for day, label in [('2019-05-01','2018—2020 历史参考段'),
                       ('2023-05-01','2021年起：上一轮留出检验区间'),
                       ('2026-08-01','近期展示')]:
        ax.text(pd.Timestamp(day), .98, label, transform=ax.get_xaxis_transform(),
                va='top', ha='center', fontsize=10, color='#8b95a1')
    fig.text(.07,.947,'纳斯达克：按网页口径重画的去噪 0.5 确认点',
             fontsize=23, weight='bold', color='#172b43')
    fig.text(.07,.9,
             f'基础修正窗口 26 / 13 / 8 周  ·  二阶导统一与 8 周前比较  ·  行情截至 {prices.index.max():%Y-%m-%d}',
             fontsize=12, color='#607083')
    handles = [Line2D([],[],color='#29394b',lw=1.5,label='纳斯达克综合')]
    for weeks in (26,13,8):
        color, marker, label = STYLE[weeks]
        handles.append(Line2D([],[],linestyle='None',marker=marker,markersize=9,
                              color=color,label=f'{label}窗口（{len(dates[weeks])}点）'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.87),
               ncol=4,frameon=False,fontsize=12,columnspacing=2.2)
    fig.text(.07,.155,'触发：纯修正 > 0，且二阶导由上行状态切换为下行状态（低于 −0.5 个百分点）。',
             fontsize=11,color='#425569')
    fig.text(.07,.113,'去噪：±0.5 个百分点内保持原状态；超过 +0.5 后才重新进入上行。点位标在当周报告确认日。',
             fontsize=11,color='#425569')
    fig.text(.07,.071,'13周组已逐点核对网页实时口径、二阶导8周、去噪0.5；8/26周为同公式扩展；均不额外加冷却。',
             fontsize=10,color='#718091')
    fig.text(.07,.035,'8/13/26指基础修正窗口；增长率只沿用截至当周已有记录，不向年初回填；清洗/实时的EPS处理差异仍保留。',
             fontsize=10,color='#718091')
    fig.savefig(str(OUTPUT)+'.png',dpi=200,facecolor='white')
    fig.savefig(str(OUTPUT)+'.svg',facecolor='white')
    plt.close(fig)
    points = sorted(points, key=lambda p:(p['确认日期'],p['修正窗口_周']))
    pd.DataFrame(points).to_csv(HERE/'去噪0.5确认点_8周13周26周点位.csv',
                                index=False,encoding='utf-8-sig')
    info = {'index':'^IXIC','base_revision_windows_weeks':[8,13,26],
            'acceleration_comparison_weeks':8, 'hysteresis_threshold_pp':.5,
            'condition':'positive revision, hysteretic acceleration state +1 to -1',
            'cooldown_weeks':0, 'basis':'production realtime formula, growth recorded as of each week; no backfill',
            'production_page_source':fresh, 'html_marker_check':'13-week base exactly matches rt|8|0.5',
            'price_end':str(prices.index.max().date()),
            'counts':{str(w):len(dates[w]) for w in STYLE}, 'points':points}
    (HERE/'去噪0.5确认点_8周13周26周图说明.json').write_text(
        json.dumps(info,ensure_ascii=False,indent=2))
    print(str(OUTPUT)+'.png')
    print(info['counts'])


if __name__ == '__main__':
    main()
