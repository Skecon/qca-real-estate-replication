"""Replication of Yeh & Hsu (2018), *Applied Soft Computing* 65, 260-271.

"Building real estate valuation models with comparative approach through
case-based reasoning".

The package implements the paper's Quantitative Comparative Approach end to
end -- the sorting quantile method, stepwise decomposition regression, the
case-distance/case-weight kernel and the weighted average of adjusted prices --
together with the two hedonic benchmarks it is compared against.
"""
from . import (
    benchmarks,
    config,
    curves,
    data,
    decomposition,
    experiments,
    importance,
    metrics,
    plots,
)
from .decomposition import StepwiseDecomposition
from .estimator import QuantitativeComparativeApproach

__all__ = [
    "benchmarks",
    "config",
    "curves",
    "data",
    "decomposition",
    "experiments",
    "importance",
    "metrics",
    "plots",
    "StepwiseDecomposition",
    "QuantitativeComparativeApproach",
]

__version__ = "1.0.0"
