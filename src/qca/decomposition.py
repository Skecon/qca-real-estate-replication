"""Stepwise decomposition regression analysis.

Section 3.2.2 of Yeh & Hsu (2018), Eqs. (9)-(26).

The price per unit area of a property is taken to be the average price of its
circle of supply and demand multiplied by the product of dimensionless
adjustment coefficients, one per factor::

    Y_j = Ybar * k_1j * k_2j * ... * k_mj

The factors are peeled off one at a time in descending order of importance. At
each step the part of the aggregate coefficient that the factors already fitted
cannot explain is attributed to the next factor, averaged within the ten
portions, and regressed on that factor.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .curves import CurveFit, best_curve, fit_curve
from .importance import portion_indices


@dataclass
class GeoSurface:
    """Second-order polynomial adjustment coefficient of the geographic
    coordinate (Eq. 25), fitted sample-by-sample rather than by portion."""

    coefficients: np.ndarray  # c0..c5
    r2: float
    rmse: float

    @staticmethod
    def design(northing, easting) -> np.ndarray:
        n = np.asarray(northing, dtype=float)
        e = np.asarray(easting, dtype=float)
        return np.c_[np.ones_like(n), n, e, n**2, e**2, n * e]

    def __call__(self, northing, easting) -> np.ndarray:
        return self.design(northing, easting) @ self.coefficients

    @classmethod
    def fit(cls, northing, easting, k) -> GeoSurface:
        k = np.asarray(k, dtype=float)
        design = cls.design(northing, easting)
        coef, *_ = np.linalg.lstsq(design, k, rcond=None)
        resid = k - design @ coef
        sse = float((resid**2).sum())
        sst = float(((k - k.mean()) ** 2).sum())
        return cls(
            coefficients=coef,
            r2=1.0 - sse / sst if sst > 0 else np.nan,
            rmse=float(np.sqrt(sse / len(k))),
        )


@dataclass
class StepwiseDecomposition:
    """Fitted adjustment-coefficient models for one circle of supply and demand."""

    y_bar: float
    order: list[str]
    curves: dict[str, CurveFit] = field(default_factory=dict)
    geo: GeoSurface | None = None
    portion_points: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        order: Sequence[str] = tuple(config.DECOMPOSITION_ORDER),
        curves: dict[str, str] | None = None,
        use_geo: bool = True,
        select_best_family: bool = False,
        target: str = config.TARGET,
        n_portions: int = config.N_PORTIONS,
    ) -> StepwiseDecomposition:
        """Peel the factors off one at a time, in the given order.

        ``curves`` names the regression family per factor; the project default
        is the set the paper selected in Section 5.1. With
        ``select_best_family`` the family is instead chosen per factor as the
        one of the five candidates with the lowest portion-level RMSE.
        """
        curves = dict(curves or config.PAPER_CURVES)
        y = df[target].to_numpy(dtype=float)
        y_bar = float(y.mean())

        # Eq. (10): the aggregate adjustment coefficient of each sample.
        k_aggregate = y / y_bar

        model = cls(y_bar=y_bar, order=list(order))
        explained = np.ones_like(k_aggregate)  # product of the k_hat so far

        for factor in order:
            x = df[factor].to_numpy(dtype=float)
            # Eqs. (12), (16), (19), (21): what the earlier factors leave over.
            residual_k = k_aggregate / explained

            idx = portion_indices(x, n_portions)
            x_portion = np.array([x[i].mean() for i in idx])
            k_portion = np.array([residual_k[i].mean() for i in idx])

            if select_best_family:
                fitted = best_curve(x_portion, k_portion)
            else:
                fitted = fit_curve(x_portion, k_portion, curves[factor])

            model.curves[factor] = fitted
            model.portion_points[factor] = pd.DataFrame(
                {
                    "x": x_portion,
                    "k": k_portion,
                    "k_hat": fitted(x_portion),
                    "n": [len(i) for i in idx],
                }
            )
            explained = explained * fitted(x)

        if use_geo:
            # Eq. (24): everything the four scored factors leave over is
            # attributed to location.
            k_geo = k_aggregate / explained
            model.geo = GeoSurface.fit(
                df["northing"].to_numpy(), df["easting"].to_numpy(), k_geo
            )

        return model

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------
    def coefficients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjustment coefficient of every factor for every row of ``df``."""
        out = pd.DataFrame(index=df.index)
        for factor, curve in self.curves.items():
            out[factor] = curve(df[factor].to_numpy(dtype=float))
        if self.geo is not None:
            out["geo"] = self.geo(df["northing"].to_numpy(), df["easting"].to_numpy())
        return out

    def aggregate_coefficient(self, df: pd.DataFrame) -> np.ndarray:
        """Product of all adjustment coefficients -- the hedonic form of Eq. (9)."""
        return self.coefficients(df).prod(axis=1).to_numpy()

    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """Direct price prediction Ybar * prod(k_i).

        This is *not* the quantitative comparative approach; it is the
        multiplicative hedonic model implied by Eq. (9), kept as a diagnostic of
        how much the case-based comparison step adds on its own.
        """
        return self.y_bar * self.aggregate_coefficient(df)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def table2(self) -> pd.DataFrame:
        """Reproduce the Sindian block of the paper's Table 2."""
        rows = []
        for factor in self.order:
            c = self.curves[factor]
            rows.append(
                {
                    "Factor": config.LABELS[factor],
                    "Model": c.family,
                    "Coefficient a": c.a,
                    "Coefficient b": c.b,
                    "Coefficient of determination": c.r2,
                    "Root of mean square error": c.rmse,
                }
            )
        if self.geo is not None:
            rows.append(
                {
                    "Factor": config.LABELS["geo"],
                    "Model": "2nd-order polynomial",
                    "Coefficient a": np.nan,
                    "Coefficient b": np.nan,
                    "Coefficient of determination": self.geo.r2,
                    "Root of mean square error": self.geo.rmse,
                }
            )
        return pd.DataFrame(rows).set_index("Factor")
