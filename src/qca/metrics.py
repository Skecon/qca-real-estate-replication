"""Accuracy measures used in Tables 5-7 of Yeh & Hsu (2018)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate(y_true, y_pred, tolerance: float) -> float:
    """Percentage of cases whose absolute percentage error is within ``tolerance``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / np.abs(y_true)
    return 100.0 * float((ape <= tolerance).mean())


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2_explained(y_true, y_pred) -> float:
    """Coefficient of determination as 1 - SSE/SST."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(((y_true - y_pred) ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - sse / sst


def r2_correlation(y_true, y_pred) -> float:
    """Coefficient of determination as the squared Pearson correlation.

    The paper does not say which definition it uses. The two disagree whenever
    the predictions are biased or mis-scaled, and the R-squared/RMSE pairs
    printed in Tables 5-7 are only mutually consistent under this one, so both
    are reported throughout this replication.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return 100.0 * float(np.mean(np.abs(np.asarray(y_pred) - y_true) / np.abs(y_true)))


def evaluate(y_true, y_pred, mean_house_price: float | None = None) -> dict:
    """The full row of Tables 5-7 for one model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    avg = float(mean_house_price if mean_house_price is not None else y_true.mean())
    error = rmse(y_true, y_pred)
    return {
        "20% error hit rate": hit_rate(y_true, y_pred, 0.20),
        "10% error hit rate": hit_rate(y_true, y_pred, 0.10),
        "R2 (1-SSE/SST)": r2_explained(y_true, y_pred),
        "R2 (corr^2)": r2_correlation(y_true, y_pred),
        "RMSE": error,
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Avg. house price": avg,
        "RMSE/Avg. house price": error / avg,
    }


def comparison_table(results: dict, mean_house_price: float | None = None):
    """Stack several ``name -> (y_true, y_pred)`` pairs into one table."""
    return pd.DataFrame(
        {
            name: evaluate(y_true, y_pred, mean_house_price)
            for name, (y_true, y_pred) in results.items()
        }
    ).T
