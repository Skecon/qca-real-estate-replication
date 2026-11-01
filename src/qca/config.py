"""Project-wide constants and paths.

All magic numbers that come from Yeh & Hsu (2018) are collected here so that
every reference to the paper is traceable to a single place.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

RAW_XLSX = DATA_RAW / "Real estate valuation data set.xlsx"

for _p in (DATA_PROCESSED, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------
# Short, code-friendly names for the UCI column headers.
COLUMNS = {
    "No": "id",
    "X1 transaction date": "date",
    "X2 house age": "age",
    "X3 distance to the nearest MRT station": "mrt",
    "X4 number of convenience stores": "stores",
    "X5 latitude": "lat",
    "X6 longitude": "lon",
    "Y house price of unit area": "price",
}

TARGET = "price"

# Human-readable labels, matching the wording used in the paper's tables.
LABELS = {
    "mrt": "Distance to the nearest MRT station",
    "stores": "Number of convenience stores in the living circle on foot",
    "age": "House age",
    "date": "Transaction date",
    "geo": "Geographic coordinate",
}

# --------------------------------------------------------------------------
# Method settings from the paper
# --------------------------------------------------------------------------
# Section 3.2.1: the sorting quantile method splits the sample into ten portions.
N_PORTIONS = 10

# Section 5.1: factor order (descending importance) and the regression family
# selected for each adjustment coefficient. The geographic coordinate is always
# decomposed last because its importance cannot be scored by the sorting
# quantile method (Section 3.2.2).
DECOMPOSITION_ORDER = ["mrt", "stores", "age", "date"]
PAPER_CURVES = {
    "mrt": "logarithmic",
    "stores": "linear",
    "age": "logarithmic",
    "date": "linear",
}

# Section 3.3.2: the weight of the geographic coordinate cannot be scored by the
# sorting quantile method, so the paper fixes it at 3.
GEO_WEIGHT = 3.0

# Section 3.3.2: effect radii delta that the paper searched by trial and error.
EFFECT_RADII = [100.0, 5.0, 3.0, 2.0, 1.75, 1.5, 1.25]

# Section 4.2: two thirds of each circle are used for training.
TRAIN_FRACTION = 2.0 / 3.0
RANDOM_STATE = 0

# Table 1 reports a minimum house age of 0.10 year for Sindian, while the UCI
# release stores 17 of those transactions as exactly 0.0. The logarithmic
# adjustment-coefficient model is undefined at 0, so ages are floored at the
# value the paper itself reports.
MIN_AGE = 0.10
