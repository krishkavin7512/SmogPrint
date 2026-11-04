"""Checks against the outside world. Nothing here feeds back into the model.

The model only sees pollutant numbers from before 2020. These tests ask whether its factors
behave the way real sources would: by the clock, by the calendar, by place, and in events it
never saw. Each test has a control, because a large change alone proves little.
"""
import numpy as np
import pandas as pd

from . import config


def t_stat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std(ddof=1) > 0 else 0.0


def _window(s, cols, a, b):
    return s[(s.Date >= a) & (s.Date <= b)].groupby("City")[cols].mean()


def lockdown(s, cols):
    """Difference-in-differences by city: the change into the lockdown window this year, minus the
    same seasonal change a year earlier. Earlier years are placebos: no lockdown happened then."""
    out = {}
    for y in (2020, 2019, 2018):
        this = _window(s, cols, f"{y}-03-25", f"{y}-04-30") - _window(s, cols, f"{y}-{config.BEFORE_START}", f"{y}-{config.BEFORE_END}")
        prev = _window(s, cols, f"{y-1}-03-25", f"{y-1}-04-30") - _window(s, cols, f"{y-1}-{config.BEFORE_START}", f"{y-1}-{config.BEFORE_END}")
        d = (this - prev).dropna()
        out[y] = {"n_cities": len(d), "effect": d.mean().to_dict(), "t": {c: t_stat(d[c]) for c in cols}, "falling": {c: int((d[c] < 0).sum()) for c in cols}}
    return out


def daily_trace(s, cols):
    """Average score across cities by day of year, 2019 against 2020, smoothed over a week."""
    both = s[s.Date.dt.year.isin((2019, 2020)) & (s.Date.dt.month <= 6)]
    g = both.groupby([both.Date.dt.year, both.Date.dt.dayofyear])[cols].mean()
    out = {"day": list(range(1, 182))}
    for y in (2019, 2020):
        out[y] = {c: g.loc[y][c].reindex(range(1, 182)).rolling(7, min_periods=3, center=True).mean().round(3).tolist() for c in cols}
    return out


def raw_lockdown(df):
    a = df[(df.Date >= "2019-03-25") & (df.Date <= "2019-04-30")].groupby("City")[config.POLLUTANTS].mean()
    b = df[(df.Date >= "2020-03-25") & (df.Date <= "2020-04-30")].groupby("City")[config.POLLUTANTS].mean()
    i = a.index.intersection(b.index)
    return ((b.loc[i].mean() / a.loc[i].mean() - 1) * 100).round(1).to_dict()


def _diwali_effects(s, cols, day, offset):
    """Score on a given offset from `day`, minus the same city's score 4 to 10 days either side."""
    cur = s[s.Date == day + pd.Timedelta(days=offset)].set_index("City")[cols]
    near = s[((s.Date >= day - pd.Timedelta(days=10)) & (s.Date <= day - pd.Timedelta(days=4))) |
             ((s.Date >= day + pd.Timedelta(days=4)) & (s.Date <= day + pd.Timedelta(days=10)))].groupby("City")[cols].mean()
    i = cur.index.intersection(near.index)
    return (cur.loc[i] - near.loc[i]).mean()


def diwali(s, cols, n_placebo=5000, seed=config.SEED):
    dates = pd.to_datetime(config.DIWALI)
    curve = {off: pd.concat([_diwali_effects(s, cols, d, off) for d in dates], axis=1).T.mean() for off in range(-5, 6)}
    by_year = pd.concat([_diwali_effects(s, cols, d, 1) for d in dates], axis=1).T
    by_year.index = dates.year
    real = by_year.mean()
    # Placebo: run the same calculation on ordinary October-November dates, one per year.
    pools = []
    for d in dates:
        cand = pd.date_range(f"{d.year}-10-01", f"{d.year}-11-30")
        cand = cand[np.abs((cand - d).days) > 10]
        pools.append(pd.concat([_diwali_effects(s, cols, c, 1) for c in cand], axis=1).T.to_numpy())
    rng = np.random.default_rng(seed)
    draws = np.mean([p[rng.integers(0, len(p), n_placebo)] for p in pools], axis=0)
    return {"offsets": list(range(-5, 6)), "curve": {c: [float(curve[o][c]) for o in range(-5, 6)] for c in cols},
            "by_year": {c: by_year[c].round(3).to_dict() for c in cols}, "day_after": real.to_dict(),
            "placebo_mean": dict(zip(cols, draws.mean(0))), "placebo_sd": dict(zip(cols, draws.std(0))),
            "placebo_p95": dict(zip(cols, np.quantile(draws, .95, axis=0))), "placebo_p05": dict(zip(cols, np.quantile(draws, .05, axis=0))),
            "p_value": {c: float((draws[:, j] >= real[c]).mean()) for j, c in enumerate(cols)}, "n_placebo": n_placebo}


def weekly(s, cols):
    """Day-of-week pattern with city-month averages removed, so seasons and places cancel out."""
    w = s[s.Date < config.FIT_END].dropna(subset=cols).copy()
    key = w.City + w.Date.dt.strftime("%Y%m")
    for c in cols:
        w[c] = w[c] - w.groupby(key)[c].transform("mean")
    dow = w.Date.dt.dayofweek
    return {"means": {c: w.groupby(dow)[c].mean().round(4).tolist() for c in cols},
            "sunday_t": {c: t_stat(w.loc[dow == 6, c]) for c in cols}, "sunday_mean": {c: float(w.loc[dow == 6, c].mean()) for c in cols}}


def by_hour(sh, cols):
    """Hour-of-day profiles from the hourly records. All before 2020 unless stated."""
    fit = sh[sh.Datetime < config.FIT_END]
    h, dow = fit.Datetime.dt.hour, fit.Datetime.dt.dayofweek
    week = fit[dow < 5].groupby(h[dow < 5])[cols].mean()
    sun = fit[dow == 6].groupby(h[dow == 6])[cols].mean()
    a = sh[(sh.Datetime >= "2019-03-25") & (sh.Datetime <= "2019-04-30")]
    b = sh[(sh.Datetime >= "2020-03-25") & (sh.Datetime <= "2020-04-30")]
    pa, pb = a.groupby(a.Datetime.dt.hour)[cols].mean(), b.groupby(b.Datetime.dt.hour)[cols].mean()
    profile = fit.groupby(h)[cols].mean()
    return {"profile": {c: profile[c].round(3).tolist() for c in cols}, "peak_hour": {c: int(profile[c].idxmax()) for c in cols}, "trough_hour": {c: int(profile[c].idxmin()) for c in cols},
            "weekday_minus_sunday": {c: (week[c] - sun[c]).round(3).tolist() for c in cols},
            "spring2019": {c: pa[c].round(3).tolist() for c in cols}, "lockdown2020": {c: pb[c].round(3).tolist() for c in cols}}


def seasons(s, cols):
    fit = s[s.Date < config.FIT_END]
    m = fit.Date.dt.month
    out = {}
    for name, cities in (("north", config.NORTH), ("south", config.SOUTH)):
        sel = fit.City.isin(cities)
        g = fit[sel].groupby(m[sel])[cols].mean()
        out[name] = {c: g[c].round(3).tolist() for c in cols}
    return out


def cities(s, cols):
    fit = s[s.Date < config.FIT_END]
    n = fit.groupby("City").size()
    prof = fit.groupby("City")[cols].mean()
    prof = prof[n >= config.MIN_CITY_DAYS]
    return prof, {c: prof[c].sort_values(ascending=False).round(3).to_dict() for c in cols}, n.to_dict()


def city_months(s, cols):
    fit = s[s.Date < config.FIT_END]
    g = fit.groupby(["City", fit.Date.dt.month])[cols].mean()
    return {city: {c: g.loc[city][c].reindex(range(1, 13)).round(3).tolist() for c in cols} for city in fit.City.unique()}
