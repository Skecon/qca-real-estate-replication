"""The five candidate regression families for adjustment coefficients.

Equations (27)-(31) of Yeh & Hsu (2018). Each family is fitted by minimising the
sum of squared errors between the predicted and the observed adjustment
coefficient of the ten portions (Eqs. 14, 18).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


def _linear(x, a, b):
    return a * x + b


def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def _logarithmic(x, a, b):
    return a * np.log(x) + b


def _exponential(x, a, b):
    return a * np.exp(b * x)


def _power(x, a, b):
    """Eq. (31); the paper calls this the 'exponential growth' model."""
    return a * x**b


#: name -> (callable, number of parameters, whether x must be strictly positive)
FAMILIES: dict[str, tuple] = {
    "linear": (_linear, 2, False),
    "quadratic": (_quadratic, 3, False),
    "logarithmic": (_logarithmic, 2, True),
    "exponential": (_exponential, 2, False),
    "power": (_power, 2, True),
}


@dataclass
class CurveFit:
    """A fitted adjustment-coefficient model k_i = f(x_i)."""

    family: str
    params: np.ndarray
    r2: float
    rmse: float
    n_points: int
    func: Callable = field(repr=False)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if FAMILIES[self.family][2]:
            # Guard the log/power families against non-positive inputs.
            x = np.clip(x, 1e-9, None)
        return np.asarray(self.func(x, *self.params), dtype=float)

    @property
    def a(self) -> float:
        return float(self.params[0])

    @property
    def b(self) -> float:
        return float(self.params[1])


def fit_curve(x, k, family: str) -> CurveFit:
    """Least-squares fit of one family to the ten portion points."""
    func, n_params, positive = FAMILIES[family]
    x = np.asarray(x, dtype=float)
    k = np.asarray(k, dtype=float)
    if positive:
        x = np.clip(x, 1e-9, None)

    if family in ("linear", "quadratic", "logarithmic"):
        # Linear in the parameters: solve directly, no starting values needed.
        if family == "linear":
            design = np.c_[x, np.ones_like(x)]
        elif family == "quadratic":
            design = np.c_[x**2, x, np.ones_like(x)]
        else:
            design = np.c_[np.log(x), np.ones_like(x)]
        params, *_ = np.linalg.lstsq(design, k, rcond=None)
    else:
        # Non-linear: start from the log-linearised solution where possible.
        if family == "exponential":
            slope, intercept = np.polyfit(x, np.log(np.clip(k, 1e-9, None)), 1)
            p0 = [np.exp(intercept), slope]
        else:
            slope, intercept = np.polyfit(
                np.log(x), np.log(np.clip(k, 1e-9, None)), 1
            )
            p0 = [np.exp(intercept), slope]
        params, _ = curve_fit(func, x, k, p0=p0, maxfev=100_000)

    pred = func(x, *params)
    if not np.all(np.isfinite(pred)):
        raise RuntimeError(f"{family} model did not converge on these points")
    resid = k - pred
    sse = float((resid**2).sum())
    sst = float(((k - k.mean()) ** 2).sum())
    return CurveFit(
        family=family,
        params=np.asarray(params, dtype=float),
        r2=1.0 - sse / sst if sst > 0 else np.nan,
        rmse=float(np.sqrt(sse / len(k))),
        n_points=len(k),
        func=func,
    )


def best_curve(x, k, families=None) -> CurveFit:
    """Fit every candidate family and return the one with the lowest RMSE."""
    candidates = []
    for name in families or FAMILIES:
        try:
            candidates.append(fit_curve(x, k, name))
        except Exception:  # a family that cannot be fitted is simply skipped
            continue
    if not candidates:
        raise RuntimeError("no regression family could be fitted")
    return min(candidates, key=lambda c: c.rmse)
