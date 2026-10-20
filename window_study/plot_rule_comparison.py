#!/usr/bin/env python3
"""Standalone Nasdaq chart: current vs calibrated 13-week swing thresholds.

Both sets use identical causal study features and a 13-week cooldown.
Markers are at the report's confirmation date, never at a backdated pivot.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/earnings-window-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
import pandas as pd

from run_study import swing_alerts

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = HERE / '纳斯达克_现行与校准规则对照'
BLUE, ORANGE, INK = '#2563a6', '#db731b', '#29394b'


def main():
    f = pd.read_csv(HERE / 'asof_features.csv', index_col=0, parse_dates=True)
    p = pd.read_csv(BASE / '_nasdaq.csv', index_col=0, parse_dates=True)['^IXIC'].dropna()
    prices = p.loc['2018-01-01':f.index.max()]
    sets = [(name, c, swing_alerts(f.x13, c)) for name, c in [('现行阈值', 1.), ('校准阈值', .25)]]
    sets = [(name, c, dates[(dates >= prices.index.min()) & (dates <= prices.index.max())])
            for name, c, dates in sets]
    font = '/System/Library/Fonts/PingFang.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(18, 8.5), facecolor='white')
    fig.subplots_adjust(left=.075, right=.98, bottom=.205, top=.77)
    ax.axvspan(prices.index.min(), pd.Timestamp('2021-01-01'), color='#f1f3f6', zorder=0)
    ax.axvspan(pd.Timestamp('2026-05-22'), prices.index.max()+pd.Timedelta(days=45),
               facecolor='#fff7e9', zorder=0)
    ax.axvline(pd.Timestamp('2021-01-01'), color='#b5bdc7', lw=.9, ls='--')
    ax.axvline(pd.Timestamp('2026-05-22'), color='#d1b478', lw=.9, ls='--')
    ax.plot(prices.index, prices, color=INK, lw=1.35, zorder=2)
    exported = []
    for name, c, dates in sets:
        vals = p.reindex(dates)
        # A Friday market holiday is mapped to the preceding available close.
        vals = vals.fillna(p.reindex(dates, method='ffill'))
        color = BLUE if c == 1. else ORANGE
        if c == 1.:
            ax.scatter(dates, vals, marker='v', s=102, color=color,
                       edgecolors='white', linewidths=1., zorder=6)
        else:
            ax.scatter(dates, vals, marker='o', s=73, facecolors='white',
                       edgecolors=color, linewidths=2., zorder=5)
        for j, (date, value) in enumerate(vals.items()):
            if c == 1.:
                offset = (7, [31, 54][j % 2])
                va, ha = 'bottom', 'left'
                # Keep nearby blue date labels apart and inside the right edge.
                if date.strftime('%Y-%m-%d') in ('2021-03-05', '2025-11-07', '2026-07-10'):
                    offset = (-9, 45 if date.year == 2026 else 54)
                    ha = 'right'
            else:
                offset = (0, [-28, -50, -72][j % 3])
                va, ha = 'top', 'center'
            if date > prices.index.max()-pd.Timedelta(days=90):
                offset = (-8, offset[1])
                ha = 'right'
            ax.annotate(date.strftime('%Y-%m-%d'), xy=(date, value), xytext=offset,
                        textcoords='offset points', color=color, fontsize=9,
                        ha=ha, va=va, annotation_clip=False,
                        arrowprops={'arrowstyle': '-', 'lw': .8, 'color': color, 'alpha': .65},
                        bbox={'boxstyle': 'round,pad=.15', 'facecolor': 'white',
                              'edgecolor': 'none', 'alpha': .9}, zorder=7)
            price_date = p.index[p.index.searchsorted(date, side='right')-1]
            exported.append({'规则': name, '阈值倍数': c,
                             '确认日期': str(date.date()), '对应收盘日期': str(price_date.date()),
                             '纳斯达克综合收盘': float(value), '13周纯修正百分比': float(f.x13[date]*100),
                             '样本区间': '校准期' if date < pd.Timestamp('2021-01-01') else
                                         '近期未完成主检验观察' if date > pd.Timestamp('2026-05-22') else '时间留出评价期'})
    ax.set_yscale('log')
    ax.set_ylim(prices.min()*.76, prices.max()*1.28)
    ax.set_xlim(prices.index.min()-pd.Timedelta(days=38), prices.index.max()+pd.Timedelta(days=45))
    ax.yaxis.set_major_locator(ticker.FixedLocator([5000, 7500, 10000, 15000, 20000, 25000, 30000]))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', color='#e4e9ee', lw=.7)
    ax.grid(axis='x', color='#eef1f4', lw=.5)
    ax.set_ylabel('纳斯达克综合指数收盘价（对数轴）', labelpad=14)
    ax.tick_params(axis='both', labelsize=11, length=0, pad=8)
    ax.spines['left'].set_color('#bdc6d0')
    ax.spines['bottom'].set_color('#bdc6d0')
    ax.text(pd.Timestamp('2019-06-01'), .98, '2018—2020 校准期',
            transform=ax.get_xaxis_transform(), color='#808a95', fontsize=10, va='top', ha='center')
    ax.text(pd.Timestamp('2023-06-01'), .98, '2021年起：参数固定后的信号',
            transform=ax.get_xaxis_transform(), color='#808a95', fontsize=10, va='top', ha='center')
    ax.text(pd.Timestamp('2026-08-10'), .98, '近期展示', transform=ax.get_xaxis_transform(),
            color='#a17e37', fontsize=10, va='top', ha='center')
    fig.text(.075, .94, '纳斯达克：现行规则与校准规则的确认点',
             fontsize=22, weight='bold', color='#172b43')
    fig.text(.075, .895,
             f'13周盈利纯修正  ·  点标在规则确认日  ·  行情截至 {prices.index.max():%Y-%m-%d}',
             fontsize=12, color='#607083')
    handles = [Line2D([], [], color=INK, lw=1.5, label='纳斯达克综合'),
               Line2D([], [], linestyle='None', marker='v', markersize=9, color=BLUE,
                      label=f'现行阈值：上涨4pp／高点2%／回落2pp（{len(sets[0][2])}点）'),
               Line2D([], [], linestyle='None', marker='o', markersize=8, markerfacecolor='white',
                      markeredgecolor=ORANGE, markeredgewidth=2,
                      label=f'校准阈值：上涨1pp／高点0.5%／回落0.5pp（{len(sets[1][2])}点）')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.07, .87), ncol=3,
               frameon=False, fontsize=10.5, handletextpad=.65, columnspacing=1.8)
    fig.text(.075, .115,
             '读图：蓝色 ▼ 为现行阈值，橙色空心 ○ 为校准阈值；日期是确认日，不是此前的盈利修正高点日。pp = 个百分点。',
             fontsize=11, color='#425569')
    fig.text(.075, .075,
             '口径：两组均用研究中逐周当时可知的13周数据重算，并设13周冷却；“现行”指现行阈值，不是逐点复制生产页面历史。',
             fontsize=10, color='#718091')
    fig.text(.075, .04,
             '灰底为校准期，浅黄底为主回测截止日之后；本图展示全部触发点，不等于此前留出检验中的3次／9次警报统计。',
             fontsize=10, color='#718091')
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=200, facecolor='white')
    fig.savefig(OUTPUT.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    pd.DataFrame(exported).sort_values(['确认日期', '规则']).to_csv(
        HERE / '规则对照点位.csv', index=False, encoding='utf-8-sig')
    info = {'price_index': '^IXIC', 'price_end': str(prices.index.max().date()),
            'marker_date': 'report confirmation date', 'basis': 'causal study x13',
            'cooldown_weeks': 13,
            'counts': {name: len(dates) for name,c,dates in sets}, 'points': exported}
    (HERE / '规则对照图说明.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(OUTPUT.with_suffix('.png'))
    print(info['counts'])


if __name__ == '__main__':
    main()
