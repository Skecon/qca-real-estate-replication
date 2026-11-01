"""Tests for the replication.

These check the properties the method is supposed to have, not the exact
numbers of one run, so they stay meaningful if a default is retuned.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qca import config, data, experiments, importance, metrics
from qca.benchmarks import MultivariateRegression, PaperMLP
from qca.curves import FAMILIES, fit_curve
from qca.decomposition import StepwiseDecomposition
from qca.estimator import QuantitativeComparativeApproach


@pytest.fixture(scope="module")
def df():
    return data.load()


@pytest.fixture(scope="module")
def split(df):
    return data.train_test_split_paper(df, 0)


# ---------------------------------------------------------------- the data
def test_dataset_matches_paper_table1(df):
    """The released file is the Sindian circle of the paper's Table 1."""
    assert len(df) == 414
    stats = data.descriptive_statistics(df)
    mrt = stats.loc[config.LABELS["mrt"]]
    assert mrt["Mean"] == pytest.approx(1083.89, abs=0.01)
    assert mrt["Standard deviation"] == pytest.approx(1262.11, abs=0.01)
    age = stats.loc[config.LABELS["age"]]
    assert age["Mean"] == pytest.approx(17.71, abs=0.01)
    assert age["Min"] == pytest.approx(0.10)


def test_split_is_two_thirds_and_disjoint(split):
    train, test = split
    assert len(train) == 276 and len(test) == 138
    assert not set(train.index) & set(test.index)


def test_local_projection_is_metric(df):
    """One degree of latitude is about 111 km, and the origin maps to zero."""
    span_deg = df["lat"].max() - df["lat"].min()
    span_m = df["northing"].max() - df["northing"].min()
    assert span_m / span_deg == pytest.approx(111_195, rel=0.01)
    assert abs(df["northing"].mean()) < 1e-6


# ------------------------------------------------------- sorting quantile
def test_portions_are_equal_sized_and_cover_everything():
    x = np.repeat([1.0, 2.0, 3.0], 40)  # heavy ties, as the transaction date has
    idx = importance.portion_indices(x, 10)
    assert len(idx) == 10
    assert sorted(np.concatenate(idx)) == list(range(120))
    assert {len(i) for i in idx} == {12}


def test_importance_ranking_matches_the_paper(split):
    """Section 5.1: MRT distance, then stores, then age, then date."""
    train, _ = split
    assert list(importance.factor_importance(train).index) == [
        "mrt",
        "stores",
        "age",
        "date",
    ]


def test_least_important_factor_has_weight_one(split):
    train, _ = split
    for method in (1, 2):
        w = importance.characteristic_weights(train, method)
        assert w["date"] == pytest.approx(1.0)
        assert w["mrt"] > w["stores"] > w["age"] > w["date"]


def test_pooled_methods_require_published_weights(split):
    train, _ = split
    with pytest.raises(ValueError):
        importance.characteristic_weights(train, 3)


# -------------------------------------------------------------- the curves
@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_recovers_a_clean_signal(family):
    x = np.linspace(1, 10, 10)
    func, _, _ = FAMILIES[family]
    truth = func(x, 0.5, 0.8, 0.1) if family == "quadratic" else func(x, 0.5, 0.8)
    fitted = fit_curve(x, truth, family)
    assert fitted.r2 > 0.999
    assert fitted(x) == pytest.approx(truth, rel=1e-5)


def test_nonconverging_family_is_rejected_not_silently_nan():
    x = np.linspace(1, 10, 10)
    bad = np.array([np.nan] * 10)
    with pytest.raises(Exception):
        fit_curve(x, bad, "power")


# ------------------------------------------------------- the decomposition
def test_adjustment_coefficients_average_about_one(split):
    """The coefficients are dimensionless corrections around the circle mean."""
    train, _ = split
    model = StepwiseDecomposition.fit(train)
    coef = model.coefficients(train)
    for factor in config.DECOMPOSITION_ORDER:
        assert coef[factor].mean() == pytest.approx(1.0, abs=0.08)


def test_mrt_coefficient_falls_with_distance(split):
    """Paper finding (2): the coefficient is inversely proportional to distance."""
    train, _ = split
    curve = StepwiseDecomposition.fit(train).curves["mrt"]
    assert curve(100) > curve(1000) > curve(5000)
    assert curve.a < 0


def test_decomposition_reproduces_paper_table2_on_full_sample(df):
    """The paper fits on a split it does not publish, so the whole sample is the
    closest available comparison; it lands on the printed coefficients."""
    t2 = StepwiseDecomposition.fit(df).table2()
    mrt = t2.loc[config.LABELS["mrt"]]
    assert mrt["Coefficient a"] == pytest.approx(-0.23843, abs=0.005)
    assert mrt["Coefficient b"] == pytest.approx(2.527609, abs=0.02)
    age = t2.loc[config.LABELS["age"]]
    assert age["Coefficient a"] == pytest.approx(-0.06408, abs=0.005)


def test_transaction_date_intercept_contradicts_the_printed_table(df):
    """The paper prints b = -20.62480 for the transaction date with a = 0.21170.

    An adjustment coefficient is dimensionless and sits around 1, so at the mean
    transaction date of 2013.15 the fitted line must pass near 1. The printed
    pair gives about 405, which is why this replication treats the printed
    intercept as a typesetting error rather than a result to reproduce.
    """
    mean_date = float(df["date"].mean())
    printed = 0.21170 * mean_date + (-20.62480)
    assert printed > 100  # nowhere near a dimensionless coefficient

    curve = StepwiseDecomposition.fit(df).curves["date"]
    assert curve.a == pytest.approx(0.21170, abs=0.02)
    assert float(curve(mean_date)) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------- the estimator
def test_distance_to_self_is_zero_and_never_negative(split):
    train, _ = split
    model = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, 1)
    ).fit(train)
    d = model.case_distance(train)
    assert np.allclose(np.diag(d), 0.0, atol=1e-9)
    assert (d >= 0).all()


def test_case_weight_hits_the_documented_values(split):
    """Eq. (33): weight 1 at distance 0, and 1/e at the effect radius."""
    train, _ = split
    model = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, 1), effect_radius=1.5
    ).fit(train)
    assert model.case_weight(np.array([0.0]))[0] == pytest.approx(1.0)
    assert model.case_weight(np.array([1.5]))[0] == pytest.approx(0.3679, abs=1e-4)


def test_large_radius_collapses_to_the_hedonic_model(split):
    """With every case weighted alike, Eq. (35) is the decomposition model."""
    train, test = split
    model = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, 1), effect_radius=1e6
    ).fit(train)
    direct = model.decomposition.predict_price(test)
    assert model.predict(test) == pytest.approx(direct, rel=0.02)


def test_qca_beats_the_regression_benchmark(split):
    """The paper's headline claim, on the replication split."""
    train, test = split
    qca = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, 1), effect_radius=1.5
    ).fit(train)
    mra = MultivariateRegression().fit(train)
    y = test[config.TARGET]
    assert metrics.rmse(y, qca.predict(test)) < metrics.rmse(y, mra.predict(test))


def test_comparables_are_real_cases_with_explicit_corrections(split):
    """The point of the approach: a valuation traceable to real trades."""
    train, test = split
    model = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, 1), effect_radius=1.5
    ).fit(train)
    table = model.comparables(test.iloc[0], k=5)
    assert len(table) == 5
    assert set(table.index) <= set(train.index)
    assert (table["case weight"].diff().dropna() <= 1e-12).all()  # sorted
    assert np.allclose(
        table["adjusted price"],
        table[config.TARGET] * table["correction factor"],
    )


# ---------------------------------------------------------------- metrics
def test_hit_rate_and_r2_on_a_known_case():
    y = np.array([100.0, 100.0, 100.0, 100.0])
    pred = np.array([105.0, 115.0, 100.0, 125.0])  # within 10%: 2/4, within 20%: 3/4
    assert metrics.hit_rate(y, pred, 0.10) == pytest.approx(50.0)
    assert metrics.hit_rate(y, pred, 0.20) == pytest.approx(75.0)


def test_perfect_prediction_scores_perfectly():
    y = np.array([10.0, 20.0, 30.0])
    assert metrics.rmse(y, y) == 0.0
    assert metrics.r2_explained(y, y) == pytest.approx(1.0)
    assert metrics.r2_correlation(y, y) == pytest.approx(1.0)


def test_the_two_r2_definitions_differ_under_bias():
    """Why both are reported: a shifted prediction keeps a perfect correlation."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    biased = y + 5.0
    assert metrics.r2_correlation(y, biased) == pytest.approx(1.0)
    assert metrics.r2_explained(y, biased) < 1.0


# ------------------------------------------------------------- benchmarks
def test_mlp_training_error_decreases(split):
    train, _ = split
    net = PaperMLP(n_cycles=100, random_state=0).fit(train)
    assert net.history_[-1] < net.history_[0]


def test_regression_design_has_the_ten_terms_of_eq36(split):
    train, _ = split
    model = MultivariateRegression().fit(train)
    assert model.coefficients.shape == (10,)
    assert np.isfinite(model.coefficients).all()
