"""The two classical hedonic price benchmarks, plus modern ML comparators.

Section 5.3 of Yeh & Hsu (2018): a multivariate regression (Eq. 36) and a
multi-layer perceptron. The MLP is written out in NumPy rather than taken from
a library so that the paper's own hyper-parameters -- one hidden layer, five
hidden units, learning rate 0.6, 1500 learning cycles -- mean what they meant in
the paper: 1500 passes of batch back-propagation on inputs and target scaled to
[0, 1], with logistic activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

HEDONIC_FEATURES = ["date", "age", "mrt", "stores", "northing", "easting"]


# ---------------------------------------------------------------------------
# Multivariate regression analysis, Eq. (36)
# ---------------------------------------------------------------------------
def hedonic_design(df: pd.DataFrame) -> np.ndarray:
    """Eq. (36): the four scored factors, the coordinates, their squares and
    their interaction."""
    n = df["northing"].to_numpy(dtype=float)
    e = df["easting"].to_numpy(dtype=float)
    return np.c_[
        np.ones(len(df)),
        df["date"].to_numpy(dtype=float),
        df["age"].to_numpy(dtype=float),
        df["mrt"].to_numpy(dtype=float),
        df["stores"].to_numpy(dtype=float),
        n,
        e,
        n**2,
        e**2,
        n * e,
    ]


HEDONIC_TERMS = [
    "intercept", "date", "age", "mrt", "stores",
    "N", "E", "N^2", "E^2", "N*E",
]


@dataclass
class MultivariateRegression:
    """Ordinary least squares on the Eq. (36) design."""

    coefficients: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> MultivariateRegression:
        X = hedonic_design(train)
        y = train[config.TARGET].to_numpy(dtype=float)
        self.coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return hedonic_design(df) @ self.coefficients

    def summary(self) -> pd.Series:
        return pd.Series(self.coefficients, index=HEDONIC_TERMS, name="coefficient")


# ---------------------------------------------------------------------------
# Multi-layer perceptron, Section 5.3 (2)
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class PaperMLP:
    """Batch back-propagation MLP with the paper's architecture.

    Defaults are the values Section 5.3 reports as best after a number of
    trials: ``hidden_units=5``, ``learning_rate=0.6``, ``n_cycles=1500``.
    """

    hidden_units: int = 5
    learning_rate: float = 0.6
    n_cycles: int = 1500
    features: tuple = tuple(HEDONIC_FEATURES)
    random_state: int = 0

    w1: np.ndarray | None = field(default=None, repr=False)
    b1: np.ndarray | None = field(default=None, repr=False)
    w2: np.ndarray | None = field(default=None, repr=False)
    b2: np.ndarray | None = field(default=None, repr=False)
    history_: list = field(default_factory=list, repr=False)

    # -- scaling to [0, 1], as is standard for logistic-activation networks --
    def _scale_x(self, df: pd.DataFrame) -> np.ndarray:
        X = df[list(self.features)].to_numpy(dtype=float)
        return (X - self._x_min) / self._x_range

    def fit(self, train: pd.DataFrame) -> PaperMLP:
        X_raw = train[list(self.features)].to_numpy(dtype=float)
        y_raw = train[config.TARGET].to_numpy(dtype=float)

        self._x_min = X_raw.min(axis=0)
        self._x_range = np.where(np.ptp(X_raw, axis=0) == 0, 1.0, np.ptp(X_raw, axis=0))
        self._y_min, self._y_max = y_raw.min(), y_raw.max()
        self._y_range = self._y_max - self._y_min

        X = (X_raw - self._x_min) / self._x_range
        y = ((y_raw - self._y_min) / self._y_range).reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)
        n_in = X.shape[1]
        # Nguyen-Widrow style scaling keeps the logistic units out of saturation.
        self.w1 = rng.uniform(-1, 1, (n_in, self.hidden_units)) / np.sqrt(n_in)
        self.b1 = np.zeros((1, self.hidden_units))
        self.w2 = rng.uniform(-1, 1, (self.hidden_units, 1)) / np.sqrt(
            self.hidden_units
        )
        self.b2 = np.zeros((1, 1))

        n = len(X)
        for _ in range(self.n_cycles):
            # One learning cycle is one pass of pattern-by-pattern updates, the
            # convention of the back-propagation literature the paper follows.
            # Batch-averaged updates at this learning rate leave the network far
            # short of convergence after 1500 cycles.
            for i in rng.permutation(n):
                xi = X[i : i + 1]
                hidden = _sigmoid(xi @ self.w1 + self.b1)
                out = _sigmoid(hidden @ self.w2 + self.b2)

                err = out - y[i : i + 1]
                d_out = err * out * (1 - out)
                d_hidden = (d_out @ self.w2.T) * hidden * (1 - hidden)

                self.w2 -= self.learning_rate * hidden.T @ d_out
                self.b2 -= self.learning_rate * d_out
                self.w1 -= self.learning_rate * xi.T @ d_hidden
                self.b1 -= self.learning_rate * d_hidden

            hidden = _sigmoid(X @ self.w1 + self.b1)
            out = _sigmoid(hidden @ self.w2 + self.b2)
            self.history_.append(
                float(np.sqrt(np.mean((out - y) ** 2))) * self._y_range
            )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._scale_x(df)
        hidden = _sigmoid(X @ self.w1 + self.b1)
        out = _sigmoid(hidden @ self.w2 + self.b2)
        return (out.ravel() * self._y_range + self._y_min)


# ---------------------------------------------------------------------------
# Modern comparators (extension, not part of the original paper)
# ---------------------------------------------------------------------------
def modern_models(random_state: int = 0) -> dict:
    """Off-the-shelf learners used to put the 2018 results in context."""
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "OLS (linear, no polynomial terms)": make_pipeline(
            StandardScaler(), LinearRegression()
        ),
        "k-NN (k=5, standardised)": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=500, random_state=random_state, n_jobs=-1
        ),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, random_state=random_state
        ),
    }


class SklearnAdapter:
    """Give a scikit-learn estimator the same DataFrame interface as the rest."""

    def __init__(self, model, features=tuple(HEDONIC_FEATURES)):
        self.model = model
        self.features = list(features)

    def fit(self, train: pd.DataFrame):
        self.model.fit(
            train[self.features], train[config.TARGET].to_numpy(dtype=float)
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(df[self.features]), dtype=float)
