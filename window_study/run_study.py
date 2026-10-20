#!/usr/bin/env python3
"""Reproducible, chronological revision-window study; see protocol.md.

Run from any directory: python3 /absolute/path/window_study/run_study.py
Existing production files are read only. No price downloads or model changes.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from analyze import G_START, time_roll
from realtime import guard, weekly_raw
import breadth

WINDOWS = (4, 8, 13, 26, 52)
SCALES = (.25, .5, .75, 1., 1.5, 2., 3., 4.)
TRAIN_START = pd.Timestamp('2018-01-01')
TRAIN_END = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')
COOLDOWN = pd.Timedelta(weeks=13)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    """Each row is reconstructed using only the report prefix known that week."""
    raw = weekly_raw()
    fw = pd.read_csv(BASE / 'factset_weekly.csv', parse_dates=['date'])
    grow = fw.loc[fw.date.ge(G_START) & fw.g_next.notna(), ['date', 'g_next']].copy()
    grow['date'] += pd.to_timedelta((4 - grow.date.dt.weekday) % 7, unit='D')
    gs = grow.drop_duplicates('date', keep='last').set_index('date').g_next / 100
    # Crucially, do NOT backfill the first observation of a year to January.
    g = gs.reindex(raw.index).ffill()
    roll = time_roll(raw.index, g)
    rollsum = {l: roll.rolling(l, min_periods=l).sum() for l in WINDOWS}
    yh = breadth.index_7d()
    records = []
    for t in raw.index[raw.index >= G_START]:
        if pd.isna(raw[t]):
            continue
        hist = guard(raw.loc[:t], yahoo=yh, roll=roll.loc[:t])
        if not hist.loc[t, 'ok']:
            continue
        final = np.log(hist.eps_final)
        row = {'date': t, 'raw_eps': raw[t], 'g_known': g[t]}
        for l in WINDOWS:
            changes = final.diff(l)
            pure = changes - rollsum[l].reindex(changes.index)
            row[f'x{l}'] = pure.get(t, np.nan)
            row[f'raw{l}'] = changes.get(t, np.nan)
            row[f'a{l}'] = pure.diff(8).get(t, np.nan)
            row[f'raw_a{l}'] = changes.diff(8).get(t, np.nan)
        records.append(row)
    features = pd.DataFrame(records).set_index('date').reindex(raw.index)
    common = features[[f'x{l}' for l in WINDOWS]].notna().all(axis=1)
    common_start = features.index[common][0]
    features = features.loc[common_start:].copy()
    # Every window's state sees the SAME weeks, including common missing weeks.
    features.loc[~common.reindex(features.index), :] = np.nan
    features.to_csv(HERE / 'asof_features.csv', float_format='%.10g')
    return features, str(gs.index.min().date())


def swing_alerts(s, scale):
    """Same online turning-point algorithm as make_chart.swings, plus cooldown."""
    R, A, level = .02 * scale, .04 * scale, .02 * scale
    mode = None
    ext = hi = lo = last = None
    events = []
    for t, v in s.dropna().items():
        if mode is None:
            hi = v if hi is None else max(hi, v)
            lo = v if lo is None else min(lo, v)
            if v <= hi - R:
                last, mode, ext = hi, 'down', v
            elif v >= lo + R:
                last, mode, ext = lo, 'up', v
            continue
        if mode == 'up':
            if v > ext:
                ext = v
            elif v <= ext - R:
                if ext >= level and ext - last >= A:
                    if not events or t - events[-1] >= COOLDOWN:
                        events.append(t)
                last, mode, ext = ext, 'down', v
        else:
            if v < ext:
                ext = v
            elif v >= ext + R:
                last, mode, ext = ext, 'up', v
    return pd.DatetimeIndex(events)


def acceleration_alerts(s, a, scale):
    """Fixed eight-week comparison; positive revision + hysteretic down-flip."""
    state = None
    events = []
    h = .01 * scale
    for t, v in s.items():
        av = a.get(t, np.nan)
        if pd.isna(v) or pd.isna(av):
            continue
        new = 1 if av > h else -1 if av < -h else state
        if new is None:
            new = 1 if av >= 0 else -1
        if state == 1 and new == -1 and v > 0:
            if not events or t - events[-1] >= COOLDOWN:
                events.append(t)
        state = new
    return pd.DatetimeIndex(events)


def next_session(dates, index):
    """Conservative availability: execute only at the next trading close."""
    out = []
    for d in dates:
        i = index.searchsorted(d, side='right')
        if i < len(index):
            out.append((d, index[i]))
    return out


def calibrate(s, prices, acceleration=None):
    years = (TRAIN_END - TRAIN_START).days / 365.25
    candidates = []
    for c in SCALES:
        alerts = (swing_alerts(s, c) if acceleration is None
                  else acceleration_alerts(s, acceleration, c))
        act = next_session(alerts, prices.index)
        n = sum(TRAIN_START <= a <= TRAIN_END for _, a in act)
        candidates.append((abs(n - years), -c, c, n, alerts))
    best = min(candidates, key=lambda x: x[:2])
    return best[2], best[3], best[4], [{'c': x[2], 'train_alerts': x[3]} for x in candidates]


def market_peaks(s, drawdown, confirmation_weeks=13):
    """Price-only symmetric zigzag; require the decline threshold within H weeks."""
    v, ix = s.values, s.index
    hi = 0
    up = True
    lo = None
    events = []
    for i in range(1, len(s)):
        if up:
            if v[i] > v[hi]:
                hi = i
            elif v[i] <= v[hi] * (1 - drawdown):
                events.append({'peak': ix[hi], 'confirmed': ix[i],
                               'peak_price': float(v[hi]), 'threshold': drawdown,
                               'confirmation_ok': (ix[i] - ix[hi]).days <= 7 * confirmation_weeks})
                up, lo = False, i
        else:
            if v[i] < v[lo]:
                lo = i
            elif v[i] >= v[lo] * (1 + drawdown):
                up, hi = True, i
    return events


def match(alert_dates, peak_dates, lead_weeks):
    """One-to-one: process peaks chronologically and choose the latest alert."""
    used, pairs = set(), []
    for j, p in enumerate(peak_dates):
        candidates = [i for i, a in enumerate(alert_dates)
                      if i not in used and 0 < (p - a).days <= lead_weeks * 7]
        if candidates:
            i = max(candidates, key=lambda i: alert_dates[i])
            used.add(i)
            pairs.append((i, j))
    return pairs


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.96
    p, den = k / n, 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [mid-half, mid+half]


def evaluate(report_dates, prices, events, start, end, lead=8, exclude_2022=False):
    active = [(r, a) for r, a in next_session(report_dates, prices.index)
              if start <= a <= end and not (exclude_2022 and a.year == 2022)]
    # Right-tail labels are fully observed through lead weeks + 13 confirmation weeks.
    peaks = [e for e in events if e['confirmation_ok'] and start <= e['peak'] <= end + pd.Timedelta(weeks=lead)]
    if exclude_2022:
        peaks = [e for e in peaks if e['peak'].year != 2022]
    dates = [a for _, a in active]
    pairs = match(dates, [e['peak'] for e in peaks], lead)
    by_peak = {j: i for i, j in pairs}
    # Recall requires a FULL pre-peak warning opportunity inside this period.
    # Precision can still credit a valid warning of an earlier left-edge peak.
    recall_js = [j for j, e in enumerate(peaks)
                 if start + pd.Timedelta(weeks=lead) <= e['peak'] <= end]
    hits = sum(j in by_peak for j in recall_js)
    excluded_days = (max(0, (min(end, pd.Timestamp('2022-12-31'))
                            - max(start, pd.Timestamp('2022-01-01'))).days + 1)
                     if exclude_2022 else 0)
    years = max(((end-start).days-excluded_days)/365.25, 1e-9)
    rows = []
    for j in recall_js:
        e = peaks[j]
        i = by_peak.get(j)
        late = [a for a in dates if 0 <= (a-e['peak']).days <= lead*7]
        rows.append({'peak': str(e['peak'].date()), 'price': e['peak_price'],
                     'confirmed': str(e['confirmed'].date()),
                     'warning_report': str(active[i][0].date()) if i is not None else None,
                     'warning_available': str(dates[i].date()) if i is not None else None,
                     'lead_weeks': (e['peak']-dates[i]).days/7 if i is not None else None,
                     'first_late_warning': str(min(late).date()) if late else None})
    leads = [r['lead_weeks'] for r in rows if r['lead_weeks'] is not None]
    success_alerts = {i for i, _ in pairs}
    alert_rows = [{'report': str(r.date()), 'available': str(a.date()),
                   'hit_peak': str(peaks[next(j for ii,j in pairs if ii==i)]['peak'].date()) if i in success_alerts else None}
                  for i,(r,a) in enumerate(active)]
    return {'start': str(start.date()), 'end': str(end.date()), 'targets': len(recall_js),
            'caught': hits, 'recall': hits/len(recall_js) if recall_js else None,
            'recall_ci95': wilson(hits,len(recall_js)),
            'alerts': len(active), 'successful_alerts': len(pairs),
            'precision': len(pairs)/len(active) if active else None,
            'false_per_year': (len(active)-len(pairs))/years,
            'median_lead_weeks': float(np.median(leads)) if leads else None,
            'late_only_targets': sum(r['warning_available'] is None and r['first_late_warning'] is not None for r in rows),
            'events': rows, 'alert_details': alert_rows}


def permutation(alerts_by_l, prices, peaks, start, end, lead=8):
    """Joint circular shifts: preserves alert counts and spacing for all five L."""
    weekly = pd.date_range(start, end, freq='W-FRI')
    active_grid = next_session(weekly, prices.index)
    report_grid = pd.DatetimeIndex([r for r,a in active_grid if start <= a <= end])
    available_grid = [a for r,a in active_grid if start <= a <= end]
    peak_dates = [e['peak'] for e in peaks if e['confirmation_ok'] and
                  start + pd.Timedelta(weeks=lead) <= e['peak'] <= end]
    flags = np.array([[d in set(alerts_by_l[l]) for d in report_grid] for l in WINDOWS])
    def count(f):
        a = [available_grid[i] for i in np.flatnonzero(f)]
        return len(match(a,peak_dates,lead))
    actual = np.array([count(f) for f in flags])
    null = np.array([[count(np.roll(f,k)) for f in flags]
                     for k in range(26, len(report_grid)-25)])
    if not len(null):
        return {}
    return {'shifts': len(null), 'observed_hits': actual.tolist(),
            'null_mean_hits': null.mean(axis=0).tolist(),
            'p_each': ((1+(null>=actual).sum(axis=0))/(1+len(null))).tolist(),
            'p_best_of_five': float((1+(null.max(axis=1)>=actual.max()).sum())/(1+len(null))),
            'note': 'Exploratory only: circular shifts assume approximate time stationarity.'}


def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj,np.generic): return clean(obj.item())
    if isinstance(obj,float) and not math.isfinite(obj): return None
    if isinstance(obj,pd.Timestamp): return str(obj.date())
    return obj


def main():
    feats, first_growth = prepare()
    print('As-of features rebuilt:', len(feats), 'weeks, common start',feats.index[0].date(),flush=True)
    px = pd.read_csv(BASE/'_nasdaq.csv', index_col=0, parse_dates=True).sort_index()
    end_data = px.index.max()
    results, summaries, event_rows, alert_rows = [], [], [], []
    calibrated = {}
    for source in ('pure','raw'):
        for rule in ('swing','acceleration'):
            for l in WINDOWS:
                s = feats[f'x{l}' if source=='pure' else f'raw{l}']
                acc = feats[f'a{l}' if source=='pure' else f'raw_a{l}']
                c,n,dates,cands = calibrate(s,px,acc if rule=='acceleration' else None)
                calibrated[(source,rule,l)] = dates
                variants = [('calibrated',c,n,dates)]
                if source=='pure' and rule=='swing':
                    variants.append(('fixed',1.,None,swing_alerts(s,1.)))
                for variant,scale,train_n,report_dates in variants:
                    for index in ('^IXIC','^NDX'):
                        price = px[index].dropna()
                        for dd in (.10,.15):
                            peaks = market_peaks(price,dd)
                            for lead in (8,13):
                                cutoff=end_data-pd.Timedelta(weeks=13+lead)
                                cutoff=pd.date_range(end=cutoff,periods=1,freq='W-FRI')[0]
                                ranges=[('all',TRAIN_START,cutoff),('train',TRAIN_START,TRAIN_END),
                                        ('test',TEST_START,cutoff),('2021_2022',TEST_START,pd.Timestamp('2022-12-31')),
                                        ('2023_on',pd.Timestamp('2023-01-01'),cutoff)]
                                for period,start,end in ranges:
                                    if start>end: continue
                                    r=evaluate(report_dates,price,peaks,start,end,lead)
                                    meta={'source':source,'rule':rule,'variant':variant,'window':l,
                                          'scale':scale,'train_count':train_n,'index':index,
                                          'drawdown':dd,'lead_horizon':lead,'confirmation_weeks':13,'period':period}
                                    result={**meta,**r}
                                    results.append(result)
                                    summaries.append({k:v for k,v in result.items() if k not in ('events','alert_details','recall_ci95')})
                                    if source=='pure' and rule=='swing' and variant=='calibrated' and period=='test':
                                        event_rows.extend([{**meta,**e} for e in r['events']])
                                        alert_rows.extend([{**meta,**a} for a in r['alert_details']])
                    if variant=='calibrated':
                        print(source,rule,'L',l,'scale',scale,'train alerts',train_n,flush=True)
    # Label-completeness audit, added after the first run: slower 26-week declines.
    # This does not change the predeclared primary table or choose new parameters.
    for l in WINDOWS:
        for index in ('^IXIC','^NDX'):
            price=px[index].dropna()
            for dd in (.10,.15):
                peaks=market_peaks(price,dd,confirmation_weeks=26)
                for lead in (8,13):
                    cutoff=end_data-pd.Timedelta(weeks=26+lead)
                    cutoff=pd.date_range(end=cutoff,periods=1,freq='W-FRI')[0]
                    base=next(r for r in results if r['source']=='pure' and r['rule']=='swing' and
                              r['variant']=='calibrated' and r['window']==l)
                    meta={k:base[k] for k in ('source','rule','variant','window','scale','train_count')}
                    meta.update(index=index,drawdown=dd,lead_horizon=lead,confirmation_weeks=26,period='test')
                    r=evaluate(calibrated[('pure','swing',l)],price,peaks,TEST_START,cutoff,lead)
                    result={**meta,**r}
                    results.append(result)
                    summaries.append({k:v for k,v in result.items() if k not in ('events','alert_details','recall_ci95')})
                    event_rows.extend([{**meta,**e} for e in r['events']])
                    alert_rows.extend([{**meta,**a} for a in r['alert_details']])
    primary=[r for r in results if r['source']=='pure' and r['rule']=='swing' and
             r['variant']=='calibrated' and r['index']=='^IXIC' and r['drawdown']==.10 and
             r['lead_horizon']==8 and r['confirmation_weeks']==13 and r['period']=='test']
    cutoff=pd.Timestamp(primary[0]['end'])
    perm=permutation({l:calibrated[('pure','swing',l)] for l in WINDOWS},px['^IXIC'].dropna(),
                     market_peaks(px['^IXIC'].dropna(),.10),TEST_START,cutoff)
    # Leave one target out: frequency-aware event F1, no retuning of thresholds.
    loo=[]
    for peak in [e['peak'] for e in primary[0]['events']]:
        scores={}
        for r in primary:
            removed_hit=int(any(e['peak']==peak and e['warning_available'] for e in r['events']))
            caught=r['caught']-removed_hit
            nt=r['targets']-1
            precision=(r['successful_alerts']-removed_hit)/r['alerts'] if r['alerts'] else 0
            recall=caught/nt if nt else 0
            scores[r['window']]=2*precision*recall/(precision+recall) if precision+recall else 0
        best=max(scores.values())
        loo.append({'omitted_peak':peak,'best_windows':[l for l,s in scores.items() if abs(s-best)<1e-12], 'f1':scores})
    no2022=[]
    for l in WINDOWS:
        r=evaluate(calibrated[('pure','swing',l)],px['^IXIC'].dropna(),market_peaks(px['^IXIC'].dropna(),.10),
                   TEST_START,cutoff,8,exclude_2022=True)
        no2022.append({'window':l,**r})
    output={'meta':{'data_end':str(end_data.date()),'first_public_growth':first_growth,
                    'common_feature_start':str(feats.index[0].date()),
                    'input_sha256':{p:digest(BASE/p) for p in ('factset_weekly.csv','_nasdaq.csv')},
                    'study_code_sha256':digest(__file__),
                    'protocol_sha256':digest(HERE/'protocol.md'),
                    'sensitivity_26w_note':'Added after initial run, for label completeness; primary table unchanged.'},
            'results':results,'primary_permutation':perm,'leave_one_out':loo,'without_2022':no2022}
    (HERE/'results.json').write_text(json.dumps(clean(output),ensure_ascii=False,indent=2))
    pd.DataFrame(summaries).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(event_rows).to_csv(HERE/'top_events.csv',index=False)
    pd.DataFrame(alert_rows).to_csv(HERE/'warning_events.csv',index=False)
    print('\nPRIMARY (test, IXIC 10%, <=8w early):',flush=True)
    for r in primary:
        print({k:r[k] for k in ('window','scale','caught','targets','alerts','precision','false_per_year','median_lead_weeks','late_only_targets')},flush=True)
    print('PERMUTATION',perm,flush=True)


if __name__=='__main__': main()
