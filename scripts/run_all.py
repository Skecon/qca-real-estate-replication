"""Run the whole replication and write every table and figure to results/.

    python scripts/run_all.py              # full run
    python scripts/run_all.py --quick      # fewer repeated splits, for a smoke test

Each table is saved as CSV under ``results/tables`` and each figure as PNG under
``results/figures``. The notebooks read the same functions, so the numbers in
the write-up and the numbers in the notebooks come from one code path.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qca import config, data, experiments, importance, metrics, plots  # noqa: E402
from qca.decomposition import StepwiseDecomposition  # noqa: E402
from qca.estimator import QuantitativeComparativeApproach  # noqa: E402

PAPER = {
    "table2": pd.DataFrame(
        {
            "Coefficient a": [-0.23843, 0.009452, -0.06408, 0.21170],
            "Coefficient b": [2.527609, 0.955591, 1.159766, -20.62480],
            "Coefficient of determination": [0.8820, 0.2830, 0.5238, 0.7283],
            "Root of mean square error": [0.081076, 0.018765, 0.036351, 0.013932],
        },
        index=[config.LABELS[f] for f in config.DECOMPOSITION_ORDER],
    ),
    "table3": pd.DataFrame(
        [[3.7, 2.9, 2.3, 1.0], [13.4, 8.2, 5.5, 1.0],
         [2.8, 2.4, 2.4, 1.0], [9.1, 6.2, 6.4, 1.0]],
        index=[f"Method {i}" for i in (1, 2, 3, 4)],
        columns=[config.LABELS[f] for f in config.DECOMPOSITION_ORDER],
    ),
    "table4": pd.DataFrame(
        [[7.98, 7.95, 7.77, 7.68, 7.78, 7.80, 7.82],
         [7.98, 7.83, 7.80, 8.23, 8.30, 8.35, 8.45],
         [7.98, 7.95, 7.81, 7.73, 7.74, 7.82, 7.87],
         [7.98, 7.92, 7.91, 8.12, 8.27, 8.35, 8.39]],
        index=[f"Method {i}" for i in (1, 2, 3, 4)],
        columns=config.EFFECT_RADII,
    ),
    "tables567": pd.DataFrame(
        [[80.2, 52.9, 0.683, 7.73, 37.98, 0.204],
         [74.3, 41.8, 0.530, 8.65, 37.98, 0.228],
         [78.0, 48.8, 0.627, 8.06, 37.98, 0.212]],
        index=["Quantitative Comparative Approach",
               "Multivariate regression analysis",
               "Neural network (paper spec)"],
        columns=["20% error hit rate", "10% error hit rate", "R2", "RMSE",
                 "Avg. house price", "RMSE/Avg. house price"],
    ),
}


def save(df: pd.DataFrame, name: str, note: str = "") -> pd.DataFrame:
    path = config.TABLES / f"{name}.csv"
    df.to_csv(path)
    print(f"\n### {name} {note}\n{df.round(4).to_string()}")
    return df


def savefig(fig, name: str) -> None:
    fig.savefig(config.FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"    figure -> results/figures/{name}.png")


def main(quick: bool = False) -> None:
    t0 = time.time()
    plots.use_paper_style()
    rng_repeats = 5 if quick else 30

    # ---------------------------------------------------------------- data
    df = data.load()
    train, test = data.train_test_split_paper(df, config.RANDOM_STATE)
    mean_price = float(df[config.TARGET].mean())
    print(f"Loaded {len(df)} transactions "
          f"({len(train)} train / {len(test)} test), mean price {mean_price:.2f}")

    save(data.descriptive_statistics(df), "table1_descriptive_statistics",
         "(paper Table 1, Sindian)")

    # ------------------------------------------------- factor importance
    imp = importance.factor_importance(train)
    save(imp, "importance_sorting_quantile", "(Section 5.1 ranking)")

    # ---------------------------------------- Table 2: the coefficient models
    model = StepwiseDecomposition.fit(train)
    t2 = model.table2()
    save(t2, "table2_adjustment_coefficients", "(paper Table 2, Sindian)")
    save(PAPER["table2"], "table2_paper_reference", "(as printed in the paper)")

    full_model = StepwiseDecomposition.fit(df)
    save(full_model.table2(), "table2_full_sample",
         "(fitted on all 414 rows, closest comparison to the paper)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, factor in zip(axes.ravel(), config.DECOMPOSITION_ORDER):
        plots.adjustment_curve(model, factor, ax=ax)
    fig.suptitle("Adjustment coefficient models (paper Figs. 7-8)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "fig_adjustment_curves")

    # ------------------------------------------------ Table 3: the weights
    t3 = importance.weights_table(
        train, experiments.POOLED_WEIGHTS_METHOD3, experiments.POOLED_WEIGHTS_METHOD4
    )
    save(t3, "table3_characteristic_weights", "(paper Table 3, Sindian)")
    save(PAPER["table3"], "table3_paper_reference", "(as printed in the paper)")

    # ------------------------------------------- Table 4: the radius grid
    t4 = experiments.radius_grid(train, test)
    save(t4, "table4_radius_grid", "(paper Table 4, Sindian)")
    save(PAPER["table4"], "table4_paper_reference", "(as printed in the paper)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plots.radius_curve(t4.drop(columns="Min"), ax=ax)
    ax.set_title("Test RMSE against the effect radius")
    savefig(fig, "fig_radius_grid")

    method, radius = experiments.best_setting(t4)
    print(f"\nBest cell of Table 4: weighting method {method}, effect radius {radius}")

    # --------------------------------------- Tables 5-7: the three approaches
    preds = experiments.fit_all_models(
        train, test, method, radius, include_modern=True
    )
    t567 = experiments.evaluation_table(test, preds, mean_price)
    save(t567, "tables567_model_comparison", "(paper Tables 5-7, Sindian)")
    save(PAPER["tables567"], "tables567_paper_reference", "(as printed in the paper)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    lim = float(max(test[config.TARGET].max(), 80))
    for ax, name in zip(axes, list(PAPER["tables567"].index)):
        plots.actual_vs_predicted(
            test[config.TARGET], preds[name], ax=ax,
            title=f"{name}\nRMSE {metrics.rmse(test[config.TARGET], preds[name]):.2f}",
            lim=lim,
        )
    fig.suptitle("Actual against predicted price per unit area (paper Fig. 9)")
    fig.tight_layout()
    savefig(fig, "fig_actual_vs_predicted")

    # ---------------------------------------------------- spatial maps
    qca_model = QuantitativeComparativeApproach(
        weights=experiments.weights_for(train, method), effect_radius=radius
    ).fit(train)
    fitted_all = qca_model.predict(df)
    vmin, vmax = np.percentile(df[config.TARGET], [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plots.price_map(df, df[config.TARGET], ax=axes[0],
                    title="Actual price per unit area", vmin=vmin, vmax=vmax)
    sc = plots.price_map(df, fitted_all, ax=axes[1],
                         title="Predicted price per unit area (QCA)",
                         vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=axes, label="10,000 NT$ / Ping", shrink=0.85)
    fig.suptitle("Spatial distribution over Sindian (paper Figs. 10-11)")
    savefig(fig, "fig_price_maps")

    # ================================================================
    # Extensions
    # ================================================================
    print("\n" + "=" * 70 + "\nEXTENSIONS\n" + "=" * 70)

    # 1. A wider radius grid than the paper searched.
    wide_radii = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0] + list(config.EFFECT_RADII)
    t4w = experiments.radius_grid(train, test, radii=sorted(wide_radii))
    save(t4w, "ext_radius_grid_wide", "(extends the paper's search below 1.25)")

    # 2. Repeated random splits.
    repeats = experiments.repeated_splits(
        df, n_repeats=rng_repeats, method=method, radius=radius, include_modern=True
    )
    repeats.to_csv(config.TABLES / "ext_repeated_splits_raw.csv", index=False)
    save(experiments.summarise_repeats(repeats, "RMSE"),
         "ext_repeated_splits_rmse", f"(RMSE over {rng_repeats} random splits)")
    save(experiments.summarise_repeats(repeats, "20% error hit rate"),
         "ext_repeated_splits_hitrate", f"(20% hit rate over {rng_repeats} splits)")

    reference = "Quantitative Comparative Approach"
    save(experiments.paired_tests(repeats, reference),
         "ext_paired_tests", f"(paired against {reference})")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    order = repeats.groupby("model")["RMSE"].mean().sort_values().index
    ax.boxplot(
        [repeats.loc[repeats.model == m, "RMSE"] for m in order],
        tick_labels=[m.replace(" (paper spec)", "") for m in order],
        widths=0.6,
    )
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"Test RMSE across {rng_repeats} random 2/3-1/3 splits")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_repeated_splits")

    # 3. Radius chosen by cross-validation inside the training set.
    cv_repeats = experiments.repeated_splits(
        df, n_repeats=rng_repeats, method=method, include_modern=False,
        tune_radius=True, radii=sorted(wide_radii),
    )
    save(experiments.summarise_repeats(cv_repeats, "RMSE"),
         "ext_cv_tuned_radius",
         "(effect radius cross-validated on the training set only)")

    # 4. Does the decomposition order matter?
    save(experiments.order_sensitivity(train, test, method, radius),
         "ext_decomposition_order", "(all 24 orderings; paper future work #1)")

    # 5. Which regression family actually fits best?
    save(experiments.family_comparison(train),
         "ext_regression_families", "(all five families of Eqs. 27-31)")

    # 6. A worked appraisal, the output an appraiser would actually show.
    worked = qca_model.comparables(test.iloc[0], k=5)
    save(worked, "ext_worked_appraisal",
         f"(target case {test.index[0]}, actual "
         f"{test.iloc[0][config.TARGET]:.1f}, estimate {worked.attrs['estimate']:.1f})")

    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Tables -> results/tables, figures -> results/figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use 5 repeated splits instead of 30")
    main(**vars(parser.parse_args()))
