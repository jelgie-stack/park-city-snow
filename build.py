"""Park City snow outlook: refresh data.json.

Pulls (1) the latest 10-day ensemble forecasts (ECMWF, ECMWF AI, GFS ensemble, Canada GEM; 154 members)
from the Open-Meteo Ensemble API and (2) Thaynes Canyon SNOTEL snowpack (this season, 1991-2020 median,
2022-23) from NRCS, then writes data.json next to this script.  Usage:  python3 build.py [out_dir]
Needs: requests, numpy, pandas, pyshp, shapely.
"""
import json, os, re, sys, io, datetime as dt
import numpy as np, pandas as pd, requests

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
TZ = 'America/Denver'
SITES = {
    'summit': dict(label='Park City Mountain summit', elev_ft=9715, lat=40.612, lon=-111.554, elev_m=2960),
    'town': dict(label='Town (near resort base)', elev_ft=7000, lat=40.654, lon=-111.513, elev_m=2134),
}
MODELS = {'ecmwf_ifs025_ensemble': 'ECMWF', 'ecmwf_aifs025_ensemble': 'ECMWF AI', 'ncep_gefs025': 'GFS ensemble', 'gem_global_ensemble': 'Canada GEM'}
API_MODELS = 'ecmwf_ifs025,ecmwf_aifs025,gfs025,gem_global'
# Water bias correction for the summit, from the 2025-26 backtest at Thaynes Canyon SNOTEL (forecast/observed ratio)
BIAS = {'summit': {'ecmwf_ifs025_ensemble': 1 / 0.76, 'ecmwf_aifs025_ensemble': 1 / 0.79, 'ncep_gefs025': 1 / 0.91, 'gem_global_ensemble': 1.25},
        'town': {m: 1.0 for m in MODELS}}
TH = [1, 3, 6, 12]
S = requests.Session()


def get(url, **kw):
    import time
    for i in range(5):
        try:
            r = S.get(url, timeout=120, **kw)
            if r.ok: return r
            print(f'fetch {url} -> HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
        except Exception as e:
            print(f'fetch {url} -> {e}', file=sys.stderr)
        time.sleep(5 * 2 ** i)  # back off: 5, 10, 20, 40, 80 s
    raise RuntimeError('fetch failed: ' + url)


def slr(t):  # snow inches per inch of water by 2 m temperature (C); 0 = rain
    return np.select([t > 1.5, t > 0.5, t > -2, t > -6], [0, 5, 8, 11], 13)


def wquant(arrs, p):
    vals = np.concatenate(arrs); w = np.concatenate([np.full(len(a), 1 / len(a)) for a in arrs])
    o = np.argsort(vals); cw = np.cumsum(w[o]) / w.sum()
    return float(vals[o][min(np.searchsorted(cw, p), len(vals) - 1)])


def fmt_hour(t):
    h = t.hour % 12 or 12
    return f"{h}{'am' if t.hour < 12 else 'pm'}"


def forecast(site, today):
    c = SITES[site]
    h = get('https://ensemble-api.open-meteo.com/v1/ensemble', params=dict(
        latitude=c['lat'], longitude=c['lon'], elevation=c['elev_m'], hourly='precipitation,temperature_2m',
        models=API_MODELS, forecast_days=12, timezone=TZ)).json()['hourly']
    t = pd.to_datetime(h['time'])
    snow, water, temp = {}, {}, {}
    for m in MODELS:
        pc = sorted(k for k in h if re.fullmatch(rf'precipitation(_member\d+)?_{m}', k))
        tc = sorted(k for k in h if re.fullmatch(rf'temperature_2m(_member\d+)?_{m}', k))
        p = pd.DataFrame({k: h[k] for k in pc}, index=t).astype(float) / 25.4
        T = pd.DataFrame({k: h[k] for k in tc}, index=t).astype(float)
        T.columns = p.columns
        water[m], temp[m] = p, T
        pb = p * BIAS[site][m]
        snow[m] = pd.DataFrame(pb.values * slr(T.values), index=t, columns=p.columns).where(pb.notna())
    days = pd.date_range(today + dt.timedelta(days=1), periods=10)
    out = dict(site=site, label=c['label'], elev_ft=c['elev_ft'], days=[])
    for day in days:
        sl = slice(day, day + pd.Timedelta(hours=23))
        arrs, byM, probs, used = [], {}, {k: [] for k in TH}, []
        for m in MODELS:
            s = snow[m].loc[sl]
            if len(s) < 24 or s.notna().all(axis=1).sum() < 24: continue
            v = s.sum().values
            arrs.append(v); byM[MODELS[m]] = round(float(v.mean()), 1); used.append(MODELS[m])
            for k in TH: probs[k].append((v >= k).mean())
        if not arrs: continue
        out['days'].append(dict(iso=str(day.date()), label=day.strftime('%a %b %-d'), dow=day.strftime('%a'), dom=day.day,
                                p={str(k): round(100 * float(np.mean(probs[k]))) for k in TH},
                                p10=round(wquant(arrs, .1), 1), p50=round(wquant(arrs, .5), 1), p90=round(wquant(arrs, .9), 1),
                                mean=round(float(np.mean([a.mean() for a in arrs])), 1), by_model=byM, models=used))
    # period total
    first, last = days[0], days[-1] + pd.Timedelta(hours=23)
    arrs, byM = [], {}
    for m in MODELS:
        s = snow[m].loc[first:last]
        v = s.sum(min_count=int(0.85 * len(s))).dropna().values
        if len(v): arrs.append(v); byM[MODELS[m]] = round(float(v.mean()), 1)
    out['total'] = dict(start=str(days[0].date()), end=str(days[-1].date()),
                        p10=round(wquant(arrs, .1)), p50=round(wquant(arrs, .5)), p90=round(wquant(arrs, .9)),
                        p={str(k): round(100 * float(np.mean([(a >= k).mean() for a in arrs]))) for k in [3, 6, 12, 18, 24]},
                        by_model=byM)
    return out, snow, water, temp


def storm_blocks(snow, water, temp, day):
    d0 = pd.Timestamp(day) - pd.Timedelta(hours=6)
    blocks = []
    for b in range(12):
        a = d0 + pd.Timedelta(hours=3 * b); sl = slice(a, a + pd.Timedelta(hours=2))
        ps, ms, rain = [], [], []
        for m in MODELS:
            s = snow[m].loc[sl]
            if len(s) < 3 or s.notna().all(axis=1).sum() < 3: continue
            v = s.sum(); ps.append((v >= 0.25).mean()); ms.append(v.mean())
            p = water[m].loc[sl].sum(); T = temp[m].loc[sl].mean(); wet = p >= 0.01
            rain.append(((T > 1.5) & wet).sum() / max(wet.sum(), 1))
        if not ps: continue
        e = a + pd.Timedelta(hours=3)
        blocks.append(dict(start=a.isoformat(), label=f"{a.strftime('%a')} {fmt_hour(a)}–{fmt_hour(e)}",
                           p=round(100 * float(np.mean(ps))), mean=round(float(np.mean(ms)), 2), rain=round(100 * float(np.mean(rain)))))
    return blocks


def snotel(start, end, elems):
    u = ('https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/'
         f'814:UT:SNTL%7Cid=%22%22%7Cname/{start},{end}/{elems}')
    txt = '\n'.join(l for l in get(u).text.splitlines() if not l.startswith('#'))
    return pd.read_csv(io.StringIO(txt), parse_dates=['Date'])


STATION = dict(lat=40.6235, lon=-111.5332, elev_m=2822)  # Thaynes Canyon SNOTEL


def station_projection(today, swe0, prec0):
    """10-day projection of snowpack (SWE) and season precipitation at the SNOTEL site.
    SWE grows by water that falls as snow (2 m temp <= 1.5C); melt is ignored. Summit bias correction applied."""
    h = get('https://ensemble-api.open-meteo.com/v1/ensemble', params=dict(
        latitude=STATION['lat'], longitude=STATION['lon'], elevation=STATION['elev_m'], hourly='precipitation,temperature_2m',
        models=API_MODELS, forecast_days=12, timezone=TZ)).json()['hourly']
    t = pd.to_datetime(h['time'])
    t0 = pd.Timestamp(today)
    bounds = [t0 + pd.Timedelta(days=k) for k in range(1, 11)]
    out = {'swe': {'p10': [swe0], 'p50': [swe0], 'p90': [swe0]}, 'prec': {'p10': [prec0], 'p50': [prec0], 'p90': [prec0]}}
    cum = {}
    for m in MODELS:
        pc = sorted(k for k in h if re.fullmatch(rf'precipitation(_member\d+)?_{m}', k))
        tc = sorted(k for k in h if re.fullmatch(rf'temperature_2m(_member\d+)?_{m}', k))
        p = pd.DataFrame({k: h[k] for k in pc}, index=t).astype(float) / 25.4 * BIAS['summit'][m]
        T = pd.DataFrame({k: h[k] for k in tc}, index=t).astype(float); T.columns = p.columns
        p = p[p.index >= t0]; T = T[T.index >= t0]
        snoww = p.where(T <= 1.5, 0.0).where(p.notna())
        cum[m] = (p, snoww)
    for b in bounds:
        for var, idx in (('swe', 1), ('prec', 0)):
            arrs = []
            for m in MODELS:
                x = cum[m][idx]; x = x[x.index < b]
                if x.notna().all(axis=1).sum() < len(x) or len(x) < 24 * ((b - t0).days): continue
                arrs.append(x.sum().values)
            base = swe0 if var == 'swe' else prec0
            for q, p_ in (('p10', .1), ('p50', .5), ('p90', .9)):
                out[var][q].append(None if not arrs else round(base + wquant(arrs, p_), 1))
    return out


def season(today):
    wy = today.year + 1 if today.month >= 10 else today.year
    s0 = dt.date(wy - 1, 10, 1)
    cur = snotel(s0, today, 'WTEQ::value,WTEQ::median_1991,PREC::value,PREC::median_1991,SNWD::value')
    cur.columns = ['date', 'swe', 'swe_med', 'prec', 'prec_med', 'depth']
    med = snotel(s0, dt.date(wy, 7, 31), 'WTEQ::median_1991,PREC::median_1991')
    med.columns = ['date', 'swe_med', 'prec_med']
    y23 = snotel(dt.date(2022, 10, 1), dt.date(2023, 7, 31), 'WTEQ::value,PREC::value,SNWD::value')
    y23.columns = ['date', 'swe', 'prec', 'depth']
    n = len(med)
    clean = lambda s: [None if pd.isna(x) else round(float(x), 1) for x in s]
    last = cur.dropna(subset=['swe']).iloc[-1]
    i = len(cur.dropna(subset=['swe'])) - 1
    lbl = lambda d: d.strftime('%b %-d')
    peak_med = med.swe_med.max(); peak23 = y23.swe.max()
    proj = station_projection(last.date.date(), float(last.swe), float(last.prec))
    try:
        weeks = weekly_outlook(today, med)
    except Exception as e:
        print('weekly outlook failed', e); weeks = []
    return dict(
        weeks=weeks,
        forecast=dict(start_index=i, note='10-day ensemble projection from today; melt ignored', **proj),
        station='Thaynes Canyon SNOTEL', elev_ft=9260, water_year=f'{wy - 1}–{str(wy)[2:]}',
        day_index=i,
        today=dict(date=str(last.date.date()), label=lbl(last.date), swe=float(last.swe), swe_med=float(last.swe_med),
                   swe_pct=round(100 * last.swe / last.swe_med) if last.swe_med else None,
                   prec=float(last.prec), prec_med=float(last.prec_med),
                   prec_pct=round(100 * last.prec / last.prec_med) if last.prec_med else None,
                   depth=None if pd.isna(last.depth) else int(last.depth),
                   swe_2223=float(y23.swe.iloc[i]), prec_2223=float(y23.prec.iloc[i]), depth_2223=None if pd.isna(y23.depth.iloc[i]) else int(y23.depth.iloc[i])),
        peaks=dict(median=dict(swe=float(peak_med), label=lbl(med.date[med.swe_med.idxmax()])),
                   y2223=dict(swe=float(peak23), label=lbl(y23.date[y23.swe.idxmax()]))),
        series=dict(labels=[lbl(d) for d in med.date],
                    swe=dict(this=clean(cur.swe), median=clean(med.swe_med), y2223=clean(y23.swe[:n])),
                    prec=dict(this=clean(cur.prec), median=clean(med.prec_med), y2223=clean(y23.prec[:n]))))


CPC = 'https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/'


def cpc_outlooks():
    """NOAA CPC official outlooks (probability of above/near/below normal) at Thaynes Canyon."""
    import zipfile, tempfile, glob, shapefile
    from shapely.geometry import shape, Point
    pt = Point(STATION['lon'], STATION['lat'])
    tmp = tempfile.mkdtemp()

    def grab(name):
        d = os.path.join(tmp, name)
        if not os.path.isdir(d):
            zp = os.path.join(tmp, name + '.zip')
            with open(zp, 'wb') as f: f.write(get(CPC + name + '_latest.zip').content)
            zipfile.ZipFile(zp).extractall(d)
        return d

    def at(shp):
        r = shapefile.Reader(shp)
        flds = [x[0] for x in r.fields[1:]]
        for sr in r.iterShapeRecords():
            if shape(sr.shape.__geo_interface__).contains(pt):
                return dict(zip(flds, sr.record))
        return {}

    def lean(rec, kind):
        cat, prob = rec.get('Cat', 'EC'), rec.get('Prob')
        words = {'prcp': {'Above': 'Wetter than normal', 'Below': 'Drier than normal', 'Normal': 'Near normal', 'EC': 'No tilt (equal chances)'},
                 'temp': {'Above': 'Warmer than normal', 'Below': 'Colder than normal', 'Normal': 'Near normal', 'EC': 'No tilt (equal chances)'}}[kind]
        return dict(cat=cat if cat in ('Above', 'Below', 'Normal') else 'EC', prob=None if cat == 'EC' or prob is None else int(prob), text=words.get(cat, words['EC']))

    def dlabel(a, b): return f"{a.strftime('%b %-d')}–{b.strftime('%b %-d')}" if a.month != b.month else f"{a.strftime('%b %-d')}–{b.day}"
    out = []
    for key, nm in (('814', '8–14 day outlook'), ('wk34', 'Weeks 3–4 outlook')):
        try:
            p = at(os.path.join(grab(key + 'prcp'), key + 'prcp_latest.shp')); t = at(os.path.join(grab(key + 'temp'), key + 'temp_latest.shp'))
            rec = p or t
            out.append(dict(name=nm, period=dlabel(rec['Start_Date'], rec['End_Date']) if rec else '', issued=str(rec.get('Fcst_Date', '')),
                            precip=lean(p, 'prcp'), temp=lean(t, 'temp')))
        except Exception as e:
            print('cpc', key, e)
    try:
        dp = grab('seasprcp'); dt_ = grab('seastemp')
        mon = sorted(glob.glob(os.path.join(dp, 'lead14_*_prcp.shp')))
        try:
            mp = grab('monthupd_prcp'); mt = grab('monthupd_temp')
            mon_u = sorted(glob.glob(os.path.join(mp, 'lead14_*_prcp.shp')))
            if mon_u: mon = mon_u; dt_m = mt
            else: dt_m = dt_
        except Exception:
            dt_m = dt_
        for f in mon[:1]:
            p = at(f); t = at(f.replace(os.path.dirname(f), dt_m).replace('_prcp', '_temp'))
            rec = p or t
            out.append(dict(name='Monthly outlook', period=rec.get('Valid_Seas', os.path.basename(f).split('_')[1]), issued=str(rec.get('Fcst_Date', '')),
                            precip=lean(p, 'prcp'), temp=lean(t, 'temp')))
        for lead in ('lead1', 'lead2'):
            fs = glob.glob(os.path.join(dp, lead + '_*_prcp.shp'))
            if not fs: continue
            f = fs[0]; p = at(f); t = at(f.replace(dp, dt_).replace('_prcp', '_temp'))
            rec = p or t
            out.append(dict(name='3-month outlook', period=rec.get('Valid_Seas', os.path.basename(f).split('_')[1]), issued=str(rec.get('Fcst_Date', '')),
                            precip=lean(p, 'prcp'), temp=lean(t, 'temp')))
    except Exception as e:
        print('cpc seasonal', e)
    return out


def weekly_outlook(today, med):
    """Weeks 1-4 water totals at the SNOTEL site from the 31-member GFS ensemble (35-day runs), vs 1991-2020 normal."""
    h = get('https://ensemble-api.open-meteo.com/v1/ensemble', params=dict(
        latitude=STATION['lat'], longitude=STATION['lon'], elevation=STATION['elev_m'], hourly='precipitation',
        models='gfs05', forecast_days=35, timezone=TZ)).json()['hourly']
    t = pd.to_datetime(h['time'])
    p = pd.DataFrame({k: h[k] for k in h if k.startswith('precipitation')}, index=t).astype(float) / 25.4 * BIAS['summit']['ncep_gefs025']
    medp = med.set_index(med.date.dt.date).prec_med
    weeks = []
    for w in range(5):
        a = pd.Timestamp(today) + pd.Timedelta(days=1 + 7 * w); b = a + pd.Timedelta(days=7)
        x = p[(p.index >= a) & (p.index < b)]
        if len(x) < 24 * 7 or not x.notna().all(axis=1).all(): break
        v = x.sum().values
        try:
            normal = float(medp[b.date()] - medp[a.date()])
        except KeyError:
            normal = None
        weeks.append(dict(label=f"{a.strftime('%b %-d')}–{(b - pd.Timedelta(days=1)).strftime('%b %-d')}", week=w + 1,
                          p10=round(float(np.percentile(v, 10)), 2), p50=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                          normal=None if normal is None else round(normal, 2),
                          p_above=None if normal is None else round(100 * float((v > normal).mean()))))
    return weeks


def previous_snapshot(now, min_hours=18, fallback_hours=6):
    """Earlier data.json from git history: the newest one at least min_hours old,
    else the oldest one at least fallback_hours old. Returns (data, hours_old) or (None, None)."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        shas = subprocess.run(['git', '-C', here, 'log', '--format=%H', '-n', '60', '--', 'data.json'],
                              capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        return None, None
    snaps = []
    for sha in shas:
        try:
            txt = subprocess.run(['git', '-C', here, 'show', f'{sha}:data.json'], capture_output=True, text=True, timeout=30).stdout
            d = json.loads(txt)
            age = (now - pd.Timestamp(d['updated'])).total_seconds() / 3600
            snaps.append((age, d))
        except Exception:
            continue
    old = [x for x in snaps if x[0] >= min_hours]
    if old:
        return min(old, key=lambda x: x[0])[1], min(old, key=lambda x: x[0])[0]
    fb = [x for x in snaps if x[0] >= fallback_hours]
    if fb:
        return max(fb, key=lambda x: x[0])[1], max(fb, key=lambda x: x[0])[0]
    return None, None


def direction(now_v, then_v, abs_th, rel_th=0.15):
    d = now_v - then_v
    if abs(d) < max(abs_th, rel_th * max(abs(then_v), 1e-9)):
        return 'steady'
    return 'up' if d > 0 else 'down'


def add_trends(data, now):
    prev, hours = previous_snapshot(now)
    if not prev:
        return
    since = dict(label=prev.get('updated_label', ''), hours=round(hours))
    for site, f in data['forecast'].items():
        pf = prev.get('forecast', {}).get(site)
        if not pf: continue
        pd_ = {d['iso']: d for d in pf.get('days', [])}
        days, tn, tt = {}, 0.0, 0.0
        for d in f['days']:
            o = pd_.get(d['iso'])
            if not o: continue
            days[d['iso']] = dict(dp1=d['p']['1'] - o['p']['1'], dmean=round(d['mean'] - o['mean'], 1))
            tn += d['mean']; tt += o['mean']
        if not days: continue
        sd = f.get('storm', {}).get('day')
        storm = None
        if sd in pd_:
            nd = next(x for x in f['days'] if x['iso'] == sd)
            storm = dict(day=sd, p1_now=nd['p']['1'], p1_then=pd_[sd]['p']['1'])
        f['trend'] = dict(since, days=days, n_days=len(days), total_now=round(tn, 1), total_then=round(tt, 1),
                          dir=direction(tn, tt, 1.0), storm=storm)
    pw = prev.get('longrange', {}).get('weeks') or []
    weeks = []
    for i, w in enumerate(data['longrange'].get('weeks', [])):
        if i < len(pw) and w.get('p_above') is not None and pw[i].get('p_above') is not None:
            weeks.append(dict(week=w['week'], d_above=w['p_above'] - pw[i]['p_above'],
                              dir=direction(w['p_above'], pw[i]['p_above'], 10, 0)))
        else:
            weeks.append(None)
    if any(weeks):
        data['longrange']['trend'] = dict(since, weeks=weeks)


def main():
    now = pd.Timestamp.now(tz=TZ)
    today = now.date()
    sites = {}
    for s in SITES:
        out, snow, water, temp = forecast(s, today)
        sites[s] = (out, snow, water, temp)
    sd = max(sites['summit'][0]['days'], key=lambda r: r['mean'])['iso']
    fc = {}
    for s, (out, snow, water, temp) in sites.items():
        out['storm'] = dict(day=sd, label=pd.Timestamp(sd).strftime('%A, %b %-d'), blocks=storm_blocks(snow, water, temp, sd))
        fc[s] = out
    data = dict(updated=now.isoformat(timespec='minutes'), updated_label=now.strftime('%a %b %-d, %-I:%M %p MT'),
                forecast=fc, season=season(today),
                method=dict(members={'ECMWF': 51, 'ECMWF AI': 51, 'GFS ensemble': 31, 'Canada GEM': 21}))
    try:
        cpc = cpc_outlooks()
    except Exception as e:
        print('cpc failed', e); cpc = []
    data['longrange'] = dict(weeks=data['season'].pop('weeks'), cpc=cpc)
    try:
        add_trends(data, now)
    except Exception as e:
        print('trend failed', e)
    with open(os.path.join(OUT, 'data.json'), 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print('wrote', os.path.join(OUT, 'data.json'), 'storm day', sd, 'summit total', fc['summit']['total'])


if __name__ == '__main__':
    main()
