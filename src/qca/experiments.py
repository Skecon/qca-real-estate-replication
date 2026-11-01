"""Reusable experiments: the paper's tables, and the extensions beyond it.

Everything the notebooks and ``scripts/run_all.py`` report is produced here, so
that a table in the write-up and the same table in a notebook cannot drift
apart.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import config, data, importance, metrics
from .benchmarks import (
    MultivariateRegression,
    PaperMLP,
    SklearnAdapter,
    modern_models,
)
from .estimator import QuantitativeComparativeApproach

# Table 3 of the paper: methods 3 and 4 average over all four circles of supply
# and demand. Only Sindian was released with the paper, so the published values
# are used directly rather than recomputed.
POOLED_WEIGHTS_METHOD3 = {"mrt": 2.8, "stores": 2.4, "age": 2.4, "date": 1.0}
POOLED_WEIGHTS_METHOD4 = {"mrt": 9.1, "stores": 6.2, "age": 6.4, "date": 1.0}

POOLED = {3: POOLED_WEIGHTS_METHOD3, 4: POOLED_WEIGHTS_METHOD4}


def weights_for(train: pd.DataFrame, method: int) -> dict[str, float]:
    return importance.characteristic_weights(train, method, pooled=POOLED.get(method))


# ---------------------------------------------------------------------------
# Table 4: RMSE over the weighting-method x effect-radius grid
# ---------------------------------------------------------------------------
def radius_grid(
    train: pd.DataFrame,
    test: pd.DataFrame,
    radii: Sequence[float] = tuple(config.EFFECT_RADII),
    methods: Sequence[int] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """Reproduce the Sindian block of the paper's Table 4."""
    rows = {}
    for method in methods:
        weights = weights_for(train, method)
        model = QuantitativeComparativeApproach(weights=weights).fit(train)
        row = {}
        for radius in radii:
            model.effect_radius = radius
            row[radius] = metrics.rmse(test[config.TARGET], model.predict(test))
        rows[f"Method {method}"] = row
    out = pd.DataFrame(rows).T
    out["Min"] = out.min(axis=1)
    return out


def best_setting(grid: pd.DataFrame) -> tuple:
    """The (weighting method, effect radius) cell with the lowest test RMSE."""
    body = grid.drop(columns="Min")
    method_label = body.min(axis=1).idxmin()
    radius = body.loc[method_label].idxmin()
    return int(str(method_label).split()[-1]), float(radius)


# ---------------------------------------------------------------------------
# Tables 5-7: the three approaches on one split
# ---------------------------------------------------------------------------
def fit_all_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: int,
    radius: float,
    include_modern: bool = False,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Fit every approach on ``train`` and predict ``test``."""
    preds: dict[str, np.ndarray] = {}

    qca = QuantitativeComparativeApproach(
        weights=weights_for(train, method), effect_radius=radius
    ).fit(train)
    preds["Quantitative Comparative Approach"] = qca.predict(test)
    preds["Multivariate regression analysis"] = (
        MultivariateRegression().fit(train).predict(test)
    )
    preds["Neural network (paper spec)"] = (
        PaperMLP(random_state=random_state).fit(train).predict(test)
    )

    if include_modern:
        for name, model in modern_models(random_state).items():
            preds[name] = SklearnAdapter(model).fit(train).predict(test)
    return preds


def evaluation_table(
    test: pd.DataFrame, preds: dict[str, np.ndarray], mean_house_price: float
) -> pd.DataFrame:
    y = test[config.TARGET].to_numpy(dtype=float)
    return metrics.comparison_table(
        {name: (y, p) for name, p in preds.items()}, mean_house_price
    )


# ---------------------------------------------------------------------------
# Extension 1: repeated random splits
# ---------------------------------------------------------------------------
def repeated_splits(
    df: pd.DataFrame,
    n_repeats: int = 30,
    method: int = 1,
    radius: float = 1.5,
    include_modern: bool = True,
    tune_radius: bool = False,
    radii: Sequence[float] = tuple(config.EFFECT_RADII),
) -> pd.DataFrame:
    """Re-run the whole comparison over ``n_repeats`` independent 2/3-1/3 splits.

    The paper reports a single split. With 138 test cases a difference of a few
    tenths of an RMSE point is well inside sampling noise, so the ranking of the
    three approaches cannot be read off one split alone.

    With ``tune_radius`` the effect radius is chosen on the training set by
    cross-validation instead of being fixed, which keeps the test set out of the
    tuning decision -- the paper picks its radius on the test set.
    """
    records = []
    mean_price = float(df[config.TARGET].mean())

    for seed in range(n_repeats):
        train, test = data.train_test_split_paper(df, random_state=seed)
        chosen = (
            cv_select_radius(train, method=method, radii=radii)
            if tune_radius
            else radius
        )
        preds = fit_all_models(
            train, test, method, chosen, include_modern, random_state=seed
        )
        for name, pred in preds.items():
            row = metrics.evaluate(test[config.TARGET], pred, mean_price)
            row.update({"model": name, "seed": seed, "effect_radius": chosen})
            records.append(row)

    return pd.DataFrame(records)


def summarise_repeats(repeats: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Mean, standard deviation and a 95% interval across the splits."""
    g = repeats.groupby("model")[metric]
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "sd": g.std(ddof=1),
            "min": g.min(),
            "max": g.max(),
            "p2.5": g.quantile(0.025),
            "p97.5": g.quantile(0.975),
        }
    )
    return out.sort_values("mean")


def paired_tests(repeats: pd.DataFrame, reference: str, metric: str = "RMSE"):
    """Paired comparison of every model against ``reference`` across splits.

    The splits are shared by all models, so a paired test is the right one; the
    Wilcoxon signed-rank test avoids assuming the per-split differences are
    normal.
    """
    from scipy import stats

    wide = repeats.pivot(index="seed", columns="model", values=metric)
    rows = []
    for model in wide.columns:
        if model == reference:
            continue
        diff = (wide[model] - wide[reference]).dropna()
        _, t_p = stats.ttest_rel(wide[model], wide[reference])
        try:
            _, w_p = stats.wilcoxon(wide[model], wide[reference])
        except ValueError:  # all differences zero
            w_p = np.nan
        rows.append(
            {
                "model": model,
                "mean difference vs reference": diff.mean(),
                "wins against reference": int((diff < 0).sum()),
                "splits": len(diff),
                "paired t p-value": t_p,
                "Wilcoxon p-value": w_p,
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("mean difference vs reference")
    )


# ---------------------------------------------------------------------------
# Extension 2: choosing the effect radius honestly
# ---------------------------------------------------------------------------
def cv_select_radius(
    train: pd.DataFrame,
    method: int = 1,
    radii: Sequence[float] = tuple(config.EFFECT_RADII),
    n_folds: int = 5,
    random_state: int = 0,
) -> float:
    """Pick the effect radius by k-fold cross-validation inside the training set."""
    rng = np.random.default_rng(random_state)
    folds = np.array_split(rng.permutation(len(train)), n_folds)
    scores = {r: [] for r in radii}

    for fold in folds:
        mask = np.zeros(len(train), dtype=bool)
        mask[fold] = True
        inner_train, inner_val = train[~mask], train[mask]
        if len(inner_train) < 20 or len(inner_val) == 0:
            continue
        model = QuantitativeComparativeApproach(
            weights=weights_for(inner_train, method)
        ).fit(inner_train)
        for r in radii:
            model.effect_radius = r
            scores[r].append(
                metrics.rmse(inner_val[config.TARGET], model.predict(inner_val))
            )
    return float(min(scores, key=lambda r: np.mean(scores[r])))


# ---------------------------------------------------------------------------
# Extension 3: does the decomposition order matter?
# ---------------------------------------------------------------------------
def order_sensitivity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: int = 1,
    radius: float = 1.5,
) -> pd.DataFrame:
    """Every ordering of the four factors in the stepwise decomposition.

    Section 6 of the paper lists this as its first suggestion for future work:
    the sequence of decomposition was held fixed everywhere, and the authors
    expect that optimising it may improve prediction.
    """
    weights = weights_for(train, method)
    rows = []
    for order in itertools.permutations(config.DECOMPOSITION_ORDER):
        model = QuantitativeComparativeApproach(
            weights=weights, effect_radius=radius, order=order
        ).fit(train)
        rows.append(
            {
                "order": " > ".join(order),
                "RMSE": metrics.rmse(test[config.TARGET], model.predict(test)),
                "is paper order": list(order) == list(config.DECOMPOSITION_ORDER),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Extension 4: which regression family fits each adjustment coefficient?
# ---------------------------------------------------------------------------
def family_comparison(train: pd.DataFrame) -> pd.DataFrame:
    """Fit all five families (Eqs. 27-31) to every factor's portion points.

    The paper states its chosen families without showing the competition; this
    reruns it.
    """
    from .curves import FAMILIES, fit_curve

    rows = []
    y = train[config.TARGET].to_numpy(dtype=float)
    k_aggregate = y / y.mean()
    explained = np.ones_like(k_aggregate)

    for factor in config.DECOMPOSITION_ORDER:
        x = train[factor].to_numpy(dtype=float)
        residual = k_aggregate / explained
        idx = importance.portion_indices(x)
        x_p = np.array([x[i].mean() for i in idx])
        k_p = np.array([residual[i].mean() for i in idx])

        for family in FAMILIES:
            try:
                fitted = fit_curve(x_p, k_p, family)
            except Exception:
                continue
            rows.append(
                {
                    "factor": config.LABELS[factor],
                    "family": family,
                    "R2": fitted.r2,
                    "RMSE": fitted.rmse,
                    "paper's choice": family == config.PAPER_CURVES[factor],
                }
            )
        chosen = fit_curve(x_p, k_p, config.PAPER_CURVES[factor])
        explained = explained * chosen(x)

    return pd.DataFrame(rows)
