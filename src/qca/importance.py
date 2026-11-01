"""The sorting quantile method and the four weighting schemes.

Section 3.2.1 (Eq. 8) and Section 3.3.2 of Yeh & Hsu (2018).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import config


def portion_indices(x, n_portions: int = config.N_PORTIONS):
    """Sort the samples by ``x`` and split them into ``n_portions`` equal groups.

    ``np.array_split`` on the sort order gives portions of equal size even when
    the factor has heavy ties (the transaction date takes only 12 distinct
    values), which quantile cut points cannot do.
    """
    order = np.argsort(np.asarray(x), kind="mergesort")
    return np.array_split(order, n_portions)


def sorting_quantile_variance(x, y, n_portions: int = config.N_PORTIONS) -> float:
    """Eq. (8): the variance of the portion means of ``y`` after sorting by ``x``.

    The larger the spread of the portion means around the grand mean, the more
    of the dependent variable the factor explains, and the more important it is.
    """
    y = np.asarray(y, dtype=float)
    means = np.array([y[idx].mean() for idx in portion_indices(x, n_portions)])
    return float(((means - y.mean()) ** 2).sum())


def factor_importance(
    df: pd.DataFrame,
    factors: Sequence[str] = tuple(config.DECOMPOSITION_ORDER),
    target: str = config.TARGET,
) -> pd.DataFrame:
    """Score and rank every factor by the sorting quantile method."""
    y = df[target].values
    rows = [
        {
            "factor": f,
            "label": config.LABELS[f],
            "variance": sorting_quantile_variance(df[f].values, y),
        }
        for f in factors
    ]
    out = pd.DataFrame(rows).set_index("factor")
    out["root_variance"] = np.sqrt(out["variance"])
    out = out.sort_values("variance", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out


def characteristic_weights(
    df: pd.DataFrame,
    method: int,
    factors: Sequence[str] = tuple(config.DECOMPOSITION_ORDER),
    target: str = config.TARGET,
    pooled: dict[str, float] | None = None,
) -> dict[str, float]:
    """Weights W_i of the characteristic variables for the case distance (Eq. 32).

    Method 1 makes the weight proportional to the root of the Eq. (8) variance,
    method 2 to the variance itself. Methods 3 and 4 are the same two rules
    applied to the average over all four circles of supply and demand; only the
    Sindian circle was released, so those two must be supplied through
    ``pooled`` -- the project passes the values the paper prints in Table 3.

    In every method the least important factor is set to 1 and the rest are
    scaled proportionally.
    """
    if method in (3, 4):
        if pooled is None:
            raise ValueError(
                "methods 3 and 4 pool the four circles; only Sindian was "
                "released, so pass the published weights through `pooled`"
            )
        weights = dict(pooled)
    elif method in (1, 2):
        imp = factor_importance(df, factors, target)
        col = "root_variance" if method == 1 else "variance"
        base = imp[col].min()
        weights = (imp[col] / base).to_dict()
    else:
        raise ValueError(f"method must be 1, 2, 3 or 4, got {method!r}")

    weights["geo"] = config.GEO_WEIGHT
    return weights


def weights_table(df: pd.DataFrame, pooled3=None, pooled4=None) -> pd.DataFrame:
    """Reproduce the Sindian rows of the paper's Table 3."""
    rows = {}
    for m, pooled in ((1, None), (2, None), (3, pooled3), (4, pooled4)):
        if m in (3, 4) and pooled is None:
            continue
        w = characteristic_weights(df, m, pooled=pooled)
        rows[f"Method {m}"] = {
            config.LABELS[f]: w[f] for f in config.DECOMPOSITION_ORDER
        }
    return pd.DataFrame(rows).T
