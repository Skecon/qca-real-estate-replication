"""Loading and preparing the Sindian real-estate transactions.

The UCI release ``Real estate valuation data set.xlsx`` is the Sindian circle of
housing supply and demand used by Yeh & Hsu (2018); its descriptive statistics
reproduce the Sindian row of the paper's Table 1 exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Mean radius of the Earth, in metres.
_EARTH_RADIUS_M = 6_371_008.8


def load_raw(path=None) -> pd.DataFrame:
    """Read the workbook and apply short column names."""
    df = pd.read_excel(path or config.RAW_XLSX)
    missing = set(config.COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"unexpected column layout, missing: {sorted(missing)}")
    return df.rename(columns=config.COLUMNS)


def add_local_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Project latitude/longitude onto a local northing/easting grid, in metres.

    The paper works with northing and easting in metres (Eq. 25), while the UCI
    release stores degrees. Over a single district an equirectangular projection
    about the sample centroid is accurate to well under a metre, and it keeps
    the quadratic surface of Eq. (25) numerically well conditioned.
    """
    df = df.copy()
    lat0 = df["lat"].mean()
    lon0 = df["lon"].mean()
    df["northing"] = np.deg2rad(df["lat"] - lat0) * _EARTH_RADIUS_M
    df["easting"] = (
        np.deg2rad(df["lon"] - lon0) * _EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    )
    df.attrs["origin"] = (lat0, lon0)
    return df


def load(path=None) -> pd.DataFrame:
    """Return the analysis frame: short names, floored age, local coordinates."""
    df = load_raw(path)
    df["age"] = df["age"].clip(lower=config.MIN_AGE)
    return add_local_coordinates(df)


def train_test_split_paper(df: pd.DataFrame, random_state: int = config.RANDOM_STATE):
    """Split 2/3 train, 1/3 test, as described in Section 4.2 of the paper."""
    test = df.sample(frac=1.0 - config.TRAIN_FRACTION, random_state=random_state)
    train = df.drop(index=test.index)
    return train.copy(), test.copy()


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the Sindian block of the paper's Table 1."""
    rows = []
    for col in ["mrt", "stores", "age", "date"]:
        s = df[col]
        rows.append(
            {
                "Independent variable": config.LABELS[col],
                "Min": s.min(),
                "Max": s.max(),
                "Mean": s.mean(),
                "Standard deviation": s.std(ddof=1),
                "Number": s.size,
            }
        )
    return pd.DataFrame(rows).set_index("Independent variable")
