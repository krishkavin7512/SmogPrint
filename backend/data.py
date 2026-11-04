"""Load the air-quality records and put them on a common scale.

Readings are log-transformed because pollution is heavily right-skewed, then standardised using
pre-2020 values only, so nothing about the lockdown leaks into the scale.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Scale:
    mean: pd.Series
    sd: pd.Series

    def apply(self, frame):
        """Log and standardise. Values below zero are sensor errors and are clipped to zero."""
        return ((np.log1p(frame[config.POLLUTANTS].clip(lower=0)) - self.mean) / self.sd).to_numpy()


def load_day():
    df = pd.read_csv(config.DATA / "city_day.csv.gz", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def load_hour():
    return pd.read_csv(config.DATA / "city_hour.csv.gz", parse_dates=["Datetime"])


def fit_scale(df):
    logged = np.log1p(df.loc[df.Date < config.FIT_END, config.POLLUTANTS].clip(lower=0))
    return Scale(logged.mean(), logged.std())


def enough(Z):
    """Rows with enough pollutants reported to be worth scoring."""
    return (~np.isnan(Z)).sum(1) >= config.MIN_READINGS


def describe(df):
    fit = df[df.Date < config.FIT_END]
    return {
        "rows": len(df), "cities": int(df.City.nunique()), "first": str(df.Date.min().date()), "last": str(df.Date.max().date()),
        "fit_rows": len(fit), "after_rows": int((df.Date >= config.FIT_END).sum()),
        "missing_pct": (df[config.POLLUTANTS].isna().mean() * 100).round(1).to_dict(),
        "rows_per_year": df.groupby(df.Date.dt.year).size().to_dict(), "days_per_city": df.groupby("City").size().sort_values(ascending=False).to_dict(),
        "skew_raw": df[config.POLLUTANTS].skew().round(2).to_dict(), "skew_log": np.log1p(df[config.POLLUTANTS].clip(lower=0)).skew().round(2).to_dict(),
        "medians": df[config.POLLUTANTS].median().round(2).to_dict(),
    }
