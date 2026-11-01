"""Figures replicating those in Yeh & Hsu (2018), plus diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# A single muted, colour-blind-safe palette used across every figure.
PALETTE = ["#2f5d8a", "#c2703d", "#4a7c59", "#8a5a83", "#8a7f2f", "#6b6b6b"]


def use_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "legend.frameon": False,
        }
    )


def adjustment_curve(model, factor: str, ax=None, show_points: bool = True):
    """The paper's Fig. 7: portion points and the fitted coefficient curve."""
    ax = ax or plt.gca()
    pts = model.portion_points[factor]
    curve = model.curves[factor]
    grid = np.linspace(pts["x"].min(), pts["x"].max(), 300)
    if show_points:
        ax.scatter(pts["x"], pts["k"], s=38, zorder=3, label="Portion means")
    ax.plot(grid, curve(grid), lw=2, color=PALETTE[1], label=f"{curve.family} fit")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(config.LABELS[factor])
    ax.set_ylabel("Adjustment coefficient $k$")
    ax.set_title(
        f"{config.LABELS[factor]}\n$R^2$={curve.r2:.4f}, RMSE={curve.rmse:.4f}",
        fontsize=9,
    )
    return ax


def actual_vs_predicted(y_true, y_pred, ax=None, title: str = "", lim=None):
    """The paper's Fig. 9."""
    ax = ax or plt.gca()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    hi = lim or max(y_true.max(), y_pred.max()) * 1.05
    ax.scatter(y_true, y_pred, s=22, alpha=0.75, edgecolor="none")
    ax.plot([0, hi], [0, hi], color="0.4", lw=1, ls="--")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual price per unit area")
    ax.set_ylabel("Predicted price per unit area")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def price_map(df: pd.DataFrame, values, ax=None, title: str = "", vmin=None, vmax=None):
    """Spatial distribution of price, in the spirit of the paper's Figs. 10-11."""
    ax = ax or plt.gca()
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=np.asarray(values, dtype=float),
        s=26,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    return sc


def radius_curve(table: pd.DataFrame, ax=None):
    """RMSE against effect radius for each weighting method (the paper's Table 4)."""
    ax = ax or plt.gca()
    for method in table.index:
        ax.plot(
            table.columns.astype(float), table.loc[method], marker="o", label=method
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Effect radius $\delta$ (log scale)")
    ax.set_ylabel("Test RMSE")
    ax.legend(fontsize=8)
    return ax
