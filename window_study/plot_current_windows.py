#!/usr/bin/env python3
"""Compare 8/13/26-week revision windows with the SAME current thresholds."""
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
OUTPUT = HERE / '纳斯达克_当前规则_8周13周26周对照'
STYLE = {8: ('#14856e', 'o', '8周'),
         13: ('#2563a6', 'v', '13周'),
         26: ('#8a4bb4', 'D', '26周')}


def main():
    features = pd.read_csv(HERE/'asof_features.csv', index_col=0, parse_dates=True)
    nq = pd.read_csv(BASE/'_nasdaq.csv', index_col=0, parse_dates=True)['^IXIC'].dropna()
    prices = nq.loc['2018-01-01':features.index.max()]
    dates = {}
    for weeks in STYLE:
        ds = swing_alerts(features[f'x{weeks}'], 1.)
        dates[weeks] = ds[(ds >= prices.index.min()) & (ds <= prices.index.max())]
    font = '/System/Library/Fonts/PingFang.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(18, 8.5), facecolor='white')
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.77)
    ax.axvspan(prices.index.min(), pd.Timestamp('2021-01-01'), color='#f1f3f6', zorder=0)
    ax.axvspan(pd.Timestamp('2026-05-22'), prices.index.max()+pd.Timedelta(days=45),
               color='#fff7e9', zorder=0)
    for day in ('2021-01-01', '2026-05-22'):
        ax.axvline(pd.Timestamp(day), color='#b8bec7', lw=.8, ls='--')
    ax.plot(prices.index, prices, color='#29394b', lw=1.4, zorder=2)
    points = []
    for weeks, (color, marker, label) in STYLE.items():
        vals = nq.reindex(dates[weeks], method='ffill')
        ax.scatter(vals.index, vals.values, marker=marker, s=85 if weeks == 8 else 110,
                   color=color, edgecolors='white', linewidths=1., zorder=5+weeks/100)
        for j, (date, price) in enumerate(vals.items()):
            if weeks == 8:
                offset, va, ha = (0, [-30, -50, -67][j % 3]), 'top', 'center'
                if date == pd.Timestamp('2026-06-26'):
                    offset, ha = (-10, -24), 'right'
            elif weeks == 13:
                offset, va, ha = (8, 32), 'bottom', 'left'
                if date.strftime('%Y-%m-%d') in ('2021-03-05', '2025-11-07', '2026-07-10'):
                    offset, ha = (-9, 48), 'right'
            else:
                offset, va, ha = (0, 77), 'bottom', 'center'
            if date >= prices.index.max()-pd.Timedelta(days=90):
                offset, ha = (-7, offset[1]), 'right'
            ax.annotate(date.strftime('%Y-%m-%d'), xy=(date, price), xytext=offset,
                        textcoords='offset points', fontsize=9.5, color=color,
                        ha=ha, va=va, annotation_clip=False, zorder=8,
                        arrowprops={'arrowstyle':'-', 'lw':.85, 'color':color, 'alpha':.65},
                        bbox={'boxstyle':'round,pad=.12', 'fc':'white', 'ec':'none', 'alpha':.9})
            price_day = nq.index[nq.index.searchsorted(date, side='right')-1]
            points.append({'修正窗口_周':weeks, '确认日期':str(date.date()),
                           '对应收盘日期':str(price_day.date()), '纳斯达克综合收盘':float(price),
                           '纯修正百分比':float(features.loc[date, f'x{weeks}']*100)})
    ax.set_yscale('log')
    ax.set_ylim(prices.min()*.76, prices.max()*1.28)
    ax.set_xlim(prices.index.min()-pd.Timedelta(days=38), prices.index.max()+pd.Timedelta(days=45))
    ax.yaxis.set_major_locator(ticker.FixedLocator([5000,7500,10000,15000,20000,25000,30000]))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', color='#e4e9ee', lw=.7)
    ax.grid(axis='x', color='#eef1f4', lw=.5)
    ax.set_ylabel('纳斯达克综合指数收盘价（对数轴）', labelpad=14)
    ax.tick_params(axis='both', labelsize=11, length=0, pad=8)
    for side in ('left','bottom'): ax.spines[side].set_color('#bdc6d0')
    for day, text in [('2019-05-01','2018—2020 历史参考段'),
                      ('2023-05-01','2021年起：上一轮留出检验区间'),
                      ('2026-08-10','近期展示')]:
        ax.text(pd.Timestamp(day), .98, text, transform=ax.get_xaxis_transform(),
                va='top', ha='center', fontsize=10, color='#8b95a1')
    fig.text(.075,.94,'纳斯达克：当前规则下的 26、13、8 周确认点',
             fontsize=22,weight='bold',color='#172b43')
    fig.text(.075,.895,
             f'相同阈值：上涨≥4个百分点、高点≥2%、回落≥2个百分点  ·  行情截至 {prices.index.max():%Y-%m-%d}',
             fontsize=12,color='#607083')
    handles=[Line2D([],[],color='#29394b',lw=1.5,label='纳斯达克综合')]
    for weeks in (26,13,8):
        color,marker,label=STYLE[weeks]
        handles.append(Line2D([],[],linestyle='None',marker=marker,markersize=9,
                              color=color,label=f'{label}窗口（{len(dates[weeks])}点）'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.07,.87),
               ncol=4,frameon=False,fontsize=12,columnspacing=2.2)
    fig.text(.075,.115,'读图：紫色 ◆ = 26周；蓝色 ▼ = 13周；绿色 ● = 8周。全部标在信号确认日，而非此前的修正高点日。',
             fontsize=11,color='#425569')
    fig.text(.075,.075,'口径：三组均用同一份研究中的当时可知数据重算、设13周冷却，仅改变基础修正窗口；当前指现行阈值。',
             fontsize=10,color='#718091')
    fig.text(.075,.04,'本图展示2018年起的全部触发点；浅黄底为主回测截止日之后的近期信号，尚未全部完成后续观察。',
             fontsize=10,color='#718091')
    fig.savefig(OUTPUT.with_suffix('.png'),dpi=200,facecolor='white')
    fig.savefig(OUTPUT.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    pd.DataFrame(points).sort_values(['确认日期','修正窗口_周']).to_csv(
        HERE/'当前规则_8周13周26周点位.csv',index=False,encoding='utf-8-sig')
    info={'index':'^IXIC','thresholds':{'rise_pp':4,'peak_pct':2,'reversal_pp':2},
          'cooldown_weeks':13,'basis':'causal study features',
          'price_end':str(prices.index.max().date()),
          'counts':{str(w):len(dates[w]) for w in STYLE},'points':points}
    (HERE/'当前规则_8周13周26周图说明.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(OUTPUT.with_suffix('.png'))
    print(info['counts'])


if __name__=='__main__': main()
