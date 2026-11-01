"""The Quantitative Comparative Approach.

Section 3.3 of Yeh & Hsu (2018), Eqs. (32)-(35).

For every target case the estimator

1. measures the weighted, standardised distance to each comparative case
   (Eq. 32),
2. turns that distance into a case weight with a Gaussian kernel of radius
   delta (Eq. 33),
3. adjusts the observed price of each comparative case by the ratio of the
   target's adjustment coefficients to the comparative case's (Eq. 34), and
4. returns the weighted average of the adjusted prices (Eq. 35).

Steps 1-2 are ordinary kernel regression. What makes the approach a
*comparative* one is step 3: the price actually carried over from a comparative
case is its real transaction price corrected factor by factor, so an appraiser
can show the client which real trades produced the valuation and how each
difference was priced.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .decomposition import StepwiseDecomposition


@dataclass
class QuantitativeComparativeApproach:
    """Fit on the training circle, then value target cases against it."""

    weights: dict[str, float]
    effect_radius: float = 1.5
    order: Sequence[str] = tuple(config.DECOMPOSITION_ORDER)
    curves: dict[str, str] | None = None
    use_geo: bool = True
    select_best_family: bool = False

    # Fitted state
    decomposition: StepwiseDecomposition | None = field(default=None, repr=False)
    comparatives_: pd.DataFrame | None = field(default=None, repr=False)
    scale_: dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> QuantitativeComparativeApproach:
        self.decomposition = StepwiseDecomposition.fit(
            train,
            order=self.order,
            curves=self.curves,
            use_geo=self.use_geo,
            select_best_family=self.select_best_family,
        )
        self.comparatives_ = train.copy()
        # std(x_i) in Eq. (32), taken over the data set.
        for col in self._distance_columns():
            self.scale_[col] = float(train[col].std(ddof=1))
        self._comp_coef = self.decomposition.coefficients(train).to_numpy()
        self._comp_price = train[config.TARGET].to_numpy(dtype=float)
        self._comp_x = train[self._distance_columns()].to_numpy(dtype=float)
        self._w = np.array(
            [self._weight_of(col) for col in self._distance_columns()], dtype=float
        )
        return self

    # ------------------------------------------------------------------
    def _distance_columns(self):
        cols = list(self.order)
        if self.use_geo:
            # The geographic coordinate is one factor with two components; each
            # enters the distance carrying the factor's weight.
            cols += ["northing", "easting"]
        return cols

    def _weight_of(self, col: str) -> float:
        if col in ("northing", "easting"):
            return self.weights.get("geo", config.GEO_WEIGHT)
        return self.weights[col]

    # ------------------------------------------------------------------
    def case_distance(self, targets: pd.DataFrame) -> np.ndarray:
        """Eq. (32). Returns an (n_targets, n_comparatives) matrix."""
        cols = self._distance_columns()
        scale = np.array([self.scale_[c] for c in cols], dtype=float)
        target_x = targets[cols].to_numpy(dtype=float)

        # (n_targets, n_comparatives, n_features), standardised
        diff = (target_x[:, None, :] - self._comp_x[None, :, :]) / scale
        num = (self._w * diff**2).sum(axis=2)
        return np.sqrt(num / self._w.sum())

    def case_weight(self, distance: np.ndarray) -> np.ndarray:
        """Eq. (33): a Gaussian kernel of radius delta."""
        return np.exp(-((distance / self.effect_radius) ** 2))

    def adjusted_prices(self, targets: pd.DataFrame) -> np.ndarray:
        """Eq. (34). Returns an (n_targets, n_comparatives) matrix."""
        target_coef = self.decomposition.coefficients(targets).to_numpy()
        # Correction coefficient K_i = k_i(target) / k_i(comparative), Eq. (6),
        # multiplied over the factors, Eq. (7). Done in logs for stability.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = target_coef[:, None, :] / self._comp_coef[None, :, :]
        correction = np.prod(ratio, axis=2)
        return self._comp_price[None, :] * correction

    # ------------------------------------------------------------------
    def predict(self, targets: pd.DataFrame, return_detail: bool = False):
        """Eq. (35): the case-weighted average of the adjusted prices."""
        if self.decomposition is None:
            raise RuntimeError("call fit() first")
        distance = self.case_distance(targets)
        weight = self.case_weight(distance)
        adjusted = self.adjusted_prices(targets)

        finite = np.isfinite(adjusted)
        weight = np.where(finite, weight, 0.0)
        adjusted = np.where(finite, adjusted, 0.0)

        total = weight.sum(axis=1)
        # A target with no usable comparative falls back on the circle average.
        pred = np.where(
            total > 0,
            (weight * adjusted).sum(axis=1) / np.where(total > 0, total, 1.0),
            self.decomposition.y_bar,
        )
        if return_detail:
            return pred, {"distance": distance, "weight": weight, "adjusted": adjusted}
        return pred

    # ------------------------------------------------------------------
    def comparables(self, target: pd.Series, k: int = 5) -> pd.DataFrame:
        """The ``k`` nearest comparative cases behind one valuation.

        This is the practical payoff the paper argues for over a hedonic
        regression: the appraisal can be shown as real trades plus explicit,
        quantified corrections.
        """
        frame = target.to_frame().T if isinstance(target, pd.Series) else target
        pred, detail = self.predict(frame, return_detail=True)
        order = np.argsort(-detail["weight"][0])[:k]
        out = self.comparatives_.iloc[order][
            list(self.order) + [config.TARGET]
        ].copy()
        out["case distance"] = detail["distance"][0][order]
        out["case weight"] = detail["weight"][0][order]
        out["adjusted price"] = detail["adjusted"][0][order]
        out["correction factor"] = (
            detail["adjusted"][0][order] / out[config.TARGET].to_numpy()
        )
        out.attrs["estimate"] = float(pred[0])
        return out
