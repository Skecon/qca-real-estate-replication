"""Generate the five replication notebooks.

Keeping the notebooks under a generator means the narrative and the code live in
a reviewable .py file, and regenerating them is one command:

    python scripts/build_notebooks.py          # write the notebooks
    python scripts/build_notebooks.py --run    # write and execute them in place
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

HEADER = """\
import sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qca import config, data, experiments, importance, metrics, plots
from qca.decomposition import StepwiseDecomposition
from qca.estimator import QuantitativeComparativeApproach
from qca.benchmarks import MultivariateRegression, PaperMLP

plots.use_paper_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
"""

CITE = """\
> **Paper** &nbsp; I-Cheng Yeh and Tzu-Kuang Hsu (2018), "Building real estate
> valuation models with comparative approach through case-based reasoning",
> *Applied Soft Computing* **65**, 260-271.
> [doi:10.1016/j.asoc.2018.01.029](https://doi.org/10.1016/j.asoc.2018.01.029)
>
> **Data** &nbsp; UCI Machine Learning Repository, *Real Estate Valuation*
> ([doi:10.24432/C5J30W](https://doi.org/10.24432/C5J30W)), CC BY 4.0 — the
> Sindian circle of supply and demand, 414 transactions.
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.rstrip())


def notebook(cells, title):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "title": title,
        }
    )
    return nb


# ===========================================================================
# 01 - data and the measurement of factor importance
# ===========================================================================
nb01 = notebook(
    [
        md(
            "# 01 · The data and the importance of each appraisal factor\n\n"
            + CITE
            + "\n---\n\n"
            "## What this notebook establishes\n\n"
            "1. The file released by UCI **is** the Sindian circle of the paper: "
            "its descriptive statistics reproduce the Sindian row of Table 1 to "
            "the second decimal.\n"
            "2. The **sorting quantile method** (Eq. 8) recovers the factor "
            "ranking the paper reports in Section 5.1 — distance to the nearest "
            "MRT station, then convenience stores, then house age, then "
            "transaction date.\n\n"
            "That ranking is not a side result. It fixes the order in which the "
            "stepwise decomposition peels the factors off in notebook 02, and it "
            "sets the weights of the case distance in notebook 03."
        ),
        code(HEADER),
        md(
            "## 1. Loading\n\n"
            "`qca.data.load` does three things: it renames the UCI columns to "
            "short names, floors house age at 0.10 year, and projects "
            "latitude/longitude onto a local northing/easting grid in metres.\n\n"
            "The flooring is worth justifying. Table 1 of the paper reports a "
            "minimum house age of 0.10 for Sindian, but the UCI release stores 17 "
            "of those transactions as exactly 0.0. The adjustment coefficient of "
            "house age is a **logarithmic** model, which is undefined at zero, so "
            "the ages are floored at the value the paper itself reports rather "
            "than at an arbitrary epsilon."
        ),
        code(
            'df = data.load()\n'
            'print(f"{len(df)} transactions, {df.shape[1]} columns")\n'
            'df.head()'
        ),
        code(
            '# How many ages the UCI file stores as exactly zero.\n'
            'raw = data.load_raw()\n'
            'print("ages recorded as 0.0 in the raw file:", int((raw["age"] == 0).sum()))\n'
            'print("minimum age after flooring:", df["age"].min())'
        ),
        md(
            "## 2. Table 1 of the paper\n\n"
            "If this matches, the dataset identification is settled."
        ),
        code(
            'table1 = data.descriptive_statistics(df)\n'
            'table1'
        ),
        code(
            'paper_table1 = pd.DataFrame(\n'
            '    {\n'
            '        "Min": [23.38, 0.00, 0.10, 2012.67],\n'
            '        "Max": [6488.02, 10.00, 43.80, 2013.58],\n'
            '        "Mean": [1083.89, 4.09, 17.71, 2013.15],\n'
            '        "Standard deviation": [1262.11, 2.95, 11.39, 0.28],\n'
            '        "Number": [414, 414, 414, 414],\n'
            '    },\n'
            '    index=table1.index,\n'
            ')\n'
            'difference = (table1 - paper_table1).abs()\n'
            'print("largest absolute difference from the printed Table 1:",\n'
            '      round(float(difference.max().max()), 4))\n'
            'difference'
        ),
        md(
            "Every cell agrees to rounding. The released file is the Sindian "
            "circle, and the replication can proceed against the published "
            "numbers rather than against a reconstruction."
        ),
        md(
            "## 3. The target variable\n\n"
            "Price is in 10,000 New Taiwan dollars per *ping* (1 ping = 3.3 m²). "
            "Note the single transaction at 117.5, far above the rest — it "
            "reappears in notebook 03 as an influential comparative case."
        ),
        code(
            'fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))\n'
            'axes[0].hist(df["price"], bins=40, color=plots.PALETTE[0])\n'
            'axes[0].set_xlabel("Price per unit area")\n'
            'axes[0].set_ylabel("Transactions")\n'
            'axes[0].set_title("Distribution of the target")\n'
            'axes[1].scatter(df["mrt"], df["price"], s=16, alpha=0.6)\n'
            'axes[1].set_xlabel(config.LABELS["mrt"])\n'
            'axes[1].set_ylabel("Price per unit area")\n'
            'axes[1].set_title("Price against the strongest single factor")\n'
            'fig.tight_layout()\n'
            'print(df["price"].describe().round(2).to_string())'
        ),
        md(
            "## 4. The sorting quantile method (Eq. 8)\n\n"
            "The paper's measure of importance is deliberately simple and "
            "non-parametric. Sort the sample by a factor, cut it into ten equal "
            "portions, take the mean price of each portion, and sum the squared "
            "deviations of those ten means from the grand mean:\n\n"
            "$$\\mathrm{Var} = \\sum_{i=1}^{10}\\left(\\bar{Y}_i - "
            "\\bar{Y}\\right)^2$$\n\n"
            "A factor that sorts the sample into portions with very different "
            "mean prices explains a lot of price; one that does not, does not. "
            "Because it only uses the ordering of the factor it is indifferent to "
            "monotone rescaling and to the functional form of the relationship — "
            "which matters here, because half of the relationships turn out to be "
            "non-linear, the most important one most of all.\n\n"
            "One implementation detail decides whether this works at all. The "
            "transaction date takes only 12 distinct values across 414 rows, so "
            "quantile cut points collapse: `pd.qcut` cannot produce ten portions "
            "from twelve heavily tied values. Splitting the *sort order* into ten "
            "equal blocks always can, and that is what `portion_indices` does."
        ),
        code(
            'print("distinct values per factor:")\n'
            'for f in config.DECOMPOSITION_ORDER:\n'
            '    print(f"  {config.LABELS[f]:<60s} {df[f].nunique():>4d}")'
        ),
        code(
            'train, test = data.train_test_split_paper(df, config.RANDOM_STATE)\n'
            'print(f"training {len(train)} / test {len(test)} (the paper\'s 2:1 split)")\n\n'
            'imp = importance.factor_importance(train)\n'
            'imp'
        ),
        md(
            "The ranking is the paper's: **MRT distance > convenience stores > "
            "house age > transaction date**, which is finding (1) of Section 6.\n\n"
            "The *levels* differ from the paper's Table 3 weights because the "
            "paper fits on a training split it never published, and because the "
            "transaction date — the denominator of every weight ratio — has the "
            "smallest and therefore the noisiest variance of the four. The ratios "
            "among the top three factors are nonetheless close to the published "
            "ones, which is the part that drives the case distance."
        ),
        code(
            'fig, ax = plt.subplots(figsize=(7.5, 3.6))\n'
            'ax.barh(\n'
            '    [config.LABELS[f] for f in imp.index][::-1],\n'
            '    imp["variance"][::-1],\n'
            '    color=plots.PALETTE[0],\n'
            ')\n'
            'ax.set_xlabel("Eq. (8) variance of the ten portion means")\n'
            'ax.set_title("Factor importance by the sorting quantile method")\n'
            'fig.tight_layout()'
        ),
        md(
            "### Why the portion means are the whole story\n\n"
            "Plotting the ten portion means directly shows what the statistic is "
            "picking up, and previews the shapes that notebook 02 has to fit."
        ),
        code(
            'fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)\n'
            'y = train["price"].to_numpy()\n'
            'for ax, factor in zip(axes, config.DECOMPOSITION_ORDER):\n'
            '    x = train[factor].to_numpy()\n'
            '    idx = importance.portion_indices(x)\n'
            '    ax.plot([x[i].mean() for i in idx], [y[i].mean() for i in idx],\n'
            '            marker="o", color=plots.PALETTE[0])\n'
            '    ax.axhline(y.mean(), color="0.6", ls=":", lw=0.9)\n'
            '    ax.set_title(config.LABELS[factor], fontsize=8)\n'
            'axes[0].set_ylabel("Mean price of the portion")\n'
            'fig.tight_layout()'
        ),
        md(
            "The MRT panel is a decaying curve, not a line — which is exactly why "
            "the paper fits a logarithmic adjustment coefficient to it in the "
            "next notebook."
        ),
    ],
    "Data and factor importance",
)


# ===========================================================================
# 02 - stepwise decomposition
# ===========================================================================
nb02 = notebook(
    [
        md(
            "# 02 · Stepwise decomposition regression (paper Table 2)\n\n"
            + CITE
            + "\n---\n\n"
            "## The idea\n\n"
            "The paper assumes a price per unit area is the circle's average "
            "price scaled by a product of dimensionless **adjustment "
            "coefficients**, one per factor (Eq. 9):\n\n"
            "$$Y_j = \\bar{Y}\\,k_{1j}k_{2j}\\cdots k_{mj}$$\n\n"
            "Each $k$ answers: *how much does this one feature push this property "
            "above or below the typical property in its market?* A coefficient of "
            "1.1 means +10%.\n\n"
            "The coefficients cannot be observed, only their product "
            "$k_j = Y_j/\\bar{Y}$ (Eq. 10). The paper separates them **greedily**: "
            "attribute the whole product to the most important factor, fit a curve, "
            "divide it out, attribute the remainder to the next factor, and so on "
            "(Eqs. 12, 16, 19, 21). The geographic coordinate is taken last, "
            "because the sorting quantile method cannot score a two-component "
            "factor.\n\n"
            "This is a **multiplicative** analogue of forward stagewise "
            "regression. The order matters, and the paper names optimising it as "
            "its first suggestion for future work — notebook 05 tests exactly that."
        ),
        code(HEADER),
        code(
            'df = data.load()\n'
            'train, test = data.train_test_split_paper(df, config.RANDOM_STATE)\n'
            'print(f"Ybar on the training circle: {train[config.TARGET].mean():.4f}")'
        ),
        md(
            "## 1. Doing one step by hand\n\n"
            "Before using the class, here is the first step written out, so that "
            "nothing in `StepwiseDecomposition` is a black box."
        ),
        code(
            'y = train[config.TARGET].to_numpy()\n'
            'k_aggregate = y / y.mean()          # Eq. (10)\n'
            'x = train["mrt"].to_numpy()\n\n'
            '# Eq. (12): the first factor is credited with the whole product.\n'
            'idx = importance.portion_indices(x)\n'
            'x_portion = np.array([x[i].mean() for i in idx])\n'
            'k_portion = np.array([k_aggregate[i].mean() for i in idx])\n\n'
            'pd.DataFrame({"mean distance (m)": x_portion,\n'
            '              "mean adjustment coefficient": k_portion,\n'
            '              "n": [len(i) for i in idx]}).round(4)'
        ),
        md(
            "The coefficient runs from well above 1 for the portion closest to an "
            "MRT station down to well below 1 for the farthest — a 2:1 spread "
            "attributable to transit access alone. Fitting Eq. (29), the "
            "logarithmic model $\\hat{k}=a\\ln x + b$, by least squares over these "
            "ten points (Eq. 14):"
        ),
        code(
            'from qca.curves import fit_curve\n\n'
            'mrt_curve = fit_curve(x_portion, k_portion, "logarithmic")\n'
            'print(f"a = {mrt_curve.a:.5f}   b = {mrt_curve.b:.5f}   "\n'
            '      f"R2 = {mrt_curve.r2:.4f}   RMSE = {mrt_curve.rmse:.6f}")\n'
            'print("paper (Table 2, Sindian): a = -0.23843  b = 2.527609  "\n'
            '      "R2 = 0.8820  RMSE = 0.081076")'
        ),
        md(
            "## 2. All four factors, plus location\n\n"
            "`StepwiseDecomposition.fit` repeats that step down the ranking, "
            "dividing out what has already been explained, and finishes with the "
            "second-order polynomial surface of Eq. (25) for the coordinates."
        ),
        code(
            'model = StepwiseDecomposition.fit(train)\n'
            'model.table2()'
        ),
        md(
            "### Against the published table\n\n"
            "The paper fits on a 2/3 training split drawn with a seed it never "
            "reports, so an exact match is not reachable. Fitting on **all 414 "
            "rows** is the closest well-defined comparison, and it is the one that "
            "should be judged."
        ),
        code(
            'full = StepwiseDecomposition.fit(df)\n\n'
            'paper = pd.DataFrame(\n'
            '    {\n'
            '        "paper a": [-0.23843, 0.009452, -0.06408, 0.21170],\n'
            '        "paper b": [2.527609, 0.955591, 1.159766, -20.62480],\n'
            '    },\n'
            '    index=[config.LABELS[f] for f in config.DECOMPOSITION_ORDER],\n'
            ')\n'
            'ours = full.table2().loc[paper.index, ["Coefficient a", "Coefficient b"]]\n'
            'ours.columns = ["ours a", "ours b"]\n'
            'comparison = pd.concat([paper, ours], axis=1)\n'
            'comparison["|diff a|"] = (comparison["ours a"] - comparison["paper a"]).abs()\n'
            'comparison'
        ),
        md(
            "Three of the four slopes land within 0.002 of the published value, "
            "and the intercepts of those three within 0.002 as well. For the most "
            "important factor the agreement is to the fourth decimal "
            "(−0.23882 against −0.23843). **The decomposition replicates.**\n\n"
            "### A reporting error in the published Table 2\n\n"
            "The transaction-date row does not agree, and it cannot: the printed "
            "pair is internally inconsistent.\n\n"
            "An adjustment coefficient is dimensionless and sits around 1 by "
            "construction. With the printed $a = 0.21170$ and $b = -20.62480$, the "
            "fitted line at the mean transaction date of 2013.15 evaluates to:"
        ),
        code(
            'mean_date = float(df["date"].mean())\n'
            'printed = 0.21170 * mean_date + (-20.62480)\n'
            'print(f"printed pair at the mean date : {printed:,.2f}")\n'
            'print(f"ours at the mean date         : {float(full.curves[\'date\'](mean_date)):,.4f}")\n'
            'print(f"intercept implied by the printed slope and k = 1: "\n'
            '      f"{1 - 0.21170 * mean_date:,.4f}")\n'
            'print(f"our intercept                 : {full.curves[\'date\'].b:,.4f}")'
        ),
        md(
            "A coefficient of ~405 would multiply every price by four hundred. "
            "The **slope** matches ours closely (0.21170 against 0.22280), and an "
            "intercept of about −425 is what that slope requires for the "
            "coefficient to pass through 1. The published −20.62480 is a "
            "typesetting error — most likely lost digits — not a different result. "
            "This replication therefore reproduces the slope and reports the "
            "intercept it implies.\n\n"
            "`tests/test_qca.py::test_transaction_date_intercept_contradicts_the_printed_table` "
            "pins this down so the claim is checkable."
        ),
        md(
            "## 3. The fitted coefficient curves (paper Figs. 7–8)\n\n"
            "Each panel is the paper's Fig. 7 for one factor: the ten portion "
            "means and the curve fitted through them."
        ),
        code(
            'fig, axes = plt.subplots(2, 2, figsize=(11, 8))\n'
            'for ax, factor in zip(axes.ravel(), config.DECOMPOSITION_ORDER):\n'
            '    plots.adjustment_curve(model, factor, ax=ax)\n'
            'fig.suptitle("Adjustment coefficient models, training circle", fontsize=12)\n'
            'fig.tight_layout()'
        ),
        md(
            "Reading these as the paper does in Section 5.1:\n\n"
            "* **MRT distance** — strongly decreasing, and the curvature is the "
            "point. Almost all of the effect is inside the first 500 m; beyond "
            "about 1 km extra distance costs little, because by then nobody is "
            "walking to the station anyway. A linear model would misprice both "
            "ends.\n"
            "* **Convenience stores** — gently increasing. Each extra store in the "
            "walking circle adds about 1%.\n"
            "* **House age** — decreasing, and again curved: depreciation is "
            "fastest in the first years.\n"
            "* **Transaction date** — increasing, which over June 2012 to May 2013 "
            "is Sindian's price boom, the market trend rather than a property "
            "feature.\n\n"
            "Every sign is the one the paper reports, and each is the sign economic "
            "reasoning predicts."
        ),
        md(
            "## 4. What the decomposition is worth on its own\n\n"
            "Multiplying the coefficients back together gives a **multiplicative "
            "hedonic model**, $\\bar{Y}\\prod_i \\hat{k}_i$. That is not the "
            "paper's approach — it has no comparative cases in it — but it is the "
            "right baseline for notebook 03, because it isolates how much the "
            "case-based comparison step adds over the coefficient models alone."
        ),
        code(
            'direct = model.predict_price(test)\n'
            'baseline = metrics.evaluate(test[config.TARGET], direct, df[config.TARGET].mean())\n'
            'pd.Series(baseline).to_frame("decomposition model only").round(4)'
        ),
        code(
            'fig, ax = plt.subplots(figsize=(4.8, 4.8))\n'
            'plots.actual_vs_predicted(test[config.TARGET], direct, ax=ax,\n'
            '                          title="Decomposition model on the test set")'
        ),
        md(
            "Keep this RMSE in mind: notebook 03 shows the full approach has to "
            "beat it to justify the extra machinery."
        ),
    ],
    "Stepwise decomposition regression",
)


# ===========================================================================
# 03 - the quantitative comparative approach
# ===========================================================================
nb03 = notebook(
    [
        md(
            "# 03 · The Quantitative Comparative Approach (paper Tables 3–5)\n\n"
            + CITE
            + "\n---\n\n"
            "## The method in four steps\n\n"
            "For a target property, against every comparative case $j$ in the same "
            "circle:\n\n"
            "1. **Distance** (Eq. 32) — a weighted, standardised Euclidean "
            "distance over the appraisal factors.\n"
            "2. **Weight** (Eq. 33) — $F_j = \\exp[-(D_j/\\delta)^2]$, a Gaussian "
            "kernel of *effect radius* $\\delta$.\n"
            "3. **Adjustment** (Eq. 34) — the comparative case's real transaction "
            "price, corrected factor by factor by "
            "$K_i = \\hat{k}_i(\\text{target})/\\hat{k}_i(\\text{comparative})$.\n"
            "4. **Weighted average** (Eq. 35) — $\\hat{Y} = \\sum_j F_jY_j / "
            "\\sum_j F_j$.\n\n"
            "Steps 1–2 are ordinary kernel regression. **Step 3 is the "
            "contribution.** A hedonic regression returns a number; this returns a "
            "number *plus the real trades it came from and the priced reason for "
            "every difference*. That is what the paper argues appraisers need to "
            "persuade a client, and it is what the traditional comparative "
            "approach does by subjective judgement."
        ),
        code(HEADER),
        code(
            'df = data.load()\n'
            'train, test = data.train_test_split_paper(df, config.RANDOM_STATE)\n'
            'mean_price = float(df[config.TARGET].mean())'
        ),
        md(
            "## 1. Weights of the characteristic variables (paper Table 3)\n\n"
            "Four schemes. Methods 1 and 2 make the weight proportional to the "
            "root of the Eq. (8) variance and to the variance itself. Methods 3 "
            "and 4 are the same two rules averaged over **all four** circles of "
            "supply and demand.\n\n"
            "Only Sindian was released with the paper, so methods 3 and 4 cannot "
            "be recomputed from this data. Rather than silently substitute "
            "something else, the project passes the **values printed in Table 3** "
            "and says so; `characteristic_weights` raises if they are not "
            "supplied. The weight of the geographic coordinate cannot be scored "
            "this way either, and the paper fixes it at 3."
        ),
        code(
            'weights = importance.weights_table(\n'
            '    train,\n'
            '    experiments.POOLED_WEIGHTS_METHOD3,\n'
            '    experiments.POOLED_WEIGHTS_METHOD4,\n'
            ')\n'
            'weights'
        ),
        code(
            'paper_t3 = pd.DataFrame(\n'
            '    [[3.7, 2.9, 2.3, 1.0], [13.4, 8.2, 5.5, 1.0],\n'
            '     [2.8, 2.4, 2.4, 1.0], [9.1, 6.2, 6.4, 1.0]],\n'
            '    index=weights.index, columns=weights.columns,\n'
            ')\n'
            'print("ratio of each weight to the house-age weight "\n'
            '      "(removes the noisy transaction-date denominator):")\n'
            'ours = weights.iloc[0] / weights.iloc[0, 2]\n'
            'theirs = paper_t3.iloc[0] / paper_t3.iloc[0, 2]\n'
            'pd.DataFrame({"ours (method 1)": ours, "paper (method 1)": theirs}).round(3)'
        ),
        md(
            "Scaled against house age instead of the transaction date the two "
            "agree closely. The weights are the same object; the published "
            "normaliser is simply the noisiest of the four."
        ),
        md(
            "## 2. The case distance and the effect radius\n\n"
            "Two implementation choices have to be made explicit, because the "
            "paper's text leaves them open.\n\n"
            "**The square root.** Equation (32) is printed as a ratio of sums. "
            "Taken literally it is a squared distance; the kernel of Eq. (33) then "
            "squares it again. This project takes the root, so $D$ is a distance "
            "in standard-deviation units and the paper's own description of the "
            "kernel holds — weight $1$ at $D=0$, $1/e \\approx 0.368$ at "
            "$D=\\delta$.\n\n"
            "**The coordinates.** The geographic coordinate is one factor with two "
            "components. Each of northing and easting enters the sum carrying the "
            "factor's weight of 3."
        ),
        code(
            'model = QuantitativeComparativeApproach(\n'
            '    weights=experiments.weights_for(train, 1), effect_radius=1.5\n'
            ').fit(train)\n\n'
            'pred, detail = model.predict(test, return_detail=True)\n'
            'D = detail["distance"]\n'
            'print(f"case distance  min {D.min():.3f}  median {np.median(D):.3f}  max {D.max():.3f}")\n'
            'print(f"case weight    min {detail[\'weight\'].min():.3f}  "\n'
            '      f"median {np.median(detail[\'weight\']):.3f}  max {detail[\'weight\'].max():.3f}")'
        ),
        code(
            'fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))\n'
            'axes[0].hist(D.ravel(), bins=60, color=plots.PALETTE[0])\n'
            'axes[0].set_xlabel("Case distance $D_j$")\n'
            'axes[0].set_ylabel("Target-comparative pairs")\n'
            'axes[0].set_title("Distribution of case distances")\n\n'
            'grid = np.linspace(0, 4, 300)\n'
            'for radius in [0.5, 1.25, 1.5, 3.0, 100.0]:\n'
            '    axes[1].plot(grid, np.exp(-((grid / radius) ** 2)),\n'
            '                 label=f"$\\\\delta$ = {radius:g}")\n'
            'axes[1].set_xlabel("Case distance $D_j$")\n'
            'axes[1].set_ylabel("Case weight $F_j$")\n'
            'axes[1].set_title("The kernel of Eq. (33)")\n'
            'axes[1].legend(fontsize=8)\n'
            'fig.tight_layout()'
        ),
        md(
            "Median distance is near 1, so radii of 1.25–3 — the paper's search "
            "range — do discriminate between near and far comparatives, while "
            "$\\delta = 100$ flattens the kernel to a simple average. The two "
            "extremes the paper warns about are both inside the grid."
        ),
        md("## 3. The RMSE grid (paper Table 4)"),
        code(
            'grid_table = experiments.radius_grid(train, test)\n'
            'grid_table'
        ),
        code(
            'paper_t4 = pd.DataFrame(\n'
            '    [[7.98, 7.95, 7.77, 7.68, 7.78, 7.80, 7.82],\n'
            '     [7.98, 7.83, 7.80, 8.23, 8.30, 8.35, 8.45],\n'
            '     [7.98, 7.95, 7.81, 7.73, 7.74, 7.82, 7.87],\n'
            '     [7.98, 7.92, 7.91, 8.12, 8.27, 8.35, 8.39]],\n'
            '    index=grid_table.index, columns=config.EFFECT_RADII,\n'
            ')\n'
            'print("paper Table 4, Sindian:")\n'
            'paper_t4'
        ),
        md(
            "Two differences are worth stating plainly.\n\n"
            "**The level is better here** — about 7.3 against the paper's 7.7–8.5. "
            "The split differs, so some gap is expected, but the whole grid sits "
            "below the published one.\n\n"
            "**The grid is far flatter here.** The paper's RMSE moves ~10% across "
            "radii; ours moves under 1%. The reason is visible in notebook 02: the "
            "decomposition model already predicts the test set at RMSE ≈ 7.33, so "
            "once prices are corrected by the coefficient ratios of Eq. (34) the "
            "adjusted prices are nearly the same whichever comparative they came "
            "from, and reweighting them changes little. A likely cause of the "
            "difference is conditioning: the paper fits the Eq. (25) surface on "
            "absolute metre coordinates (values near 2,760,000), where the "
            "quadratic terms are badly scaled, while this project centres them "
            "first. A weaker location model leaves more spatial structure for the "
            "kernel to recover, and a steeper radius curve.\n\n"
            "The ranking of methods and the shape of the curve are reproduced; the "
            "magnitude of the radius effect is not."
        ),
        code(
            'fig, ax = plt.subplots(figsize=(7.5, 4.4))\n'
            'plots.radius_curve(grid_table.drop(columns="Min"), ax=ax)\n'
            'ax.set_title("Test RMSE against the effect radius")'
        ),
        code(
            'method, radius = experiments.best_setting(grid_table)\n'
            'print(f"best cell: weighting method {method}, effect radius {radius}")'
        ),
        md(
            "⚠️ **This is the paper's procedure, and it is the paper's weakest "
            "step.** The radius and the weighting method are chosen by reading the "
            "minimum off a grid scored on the *test* set, so the reported RMSE is "
            "not an out-of-sample figure for the chosen configuration. Notebook 05 "
            "re-runs the comparison with the radius cross-validated inside the "
            "training set, which is the honest version."
        ),
        md("## 4. Table 5 of the paper"),
        code(
            'best = QuantitativeComparativeApproach(\n'
            '    weights=experiments.weights_for(train, method), effect_radius=radius\n'
            ').fit(train)\n'
            'qca_pred = best.predict(test)\n\n'
            'table5 = pd.Series(\n'
            '    metrics.evaluate(test[config.TARGET], qca_pred, mean_price)\n'
            ').to_frame("replication")\n'
            'table5["paper (Table 5)"] = [80.2, 52.9, 0.683, np.nan, 7.73,\n'
            '                             np.nan, np.nan, 37.98, 0.204]\n'
            'table5.round(4)'
        ),
        md(
            "### Which $R^2$?\n\n"
            "The paper never says. The two usual definitions disagree whenever "
            "predictions are biased, and the $R^2$/RMSE pairs printed across "
            "Tables 5–7 are only mutually consistent under the squared-correlation "
            "reading — the three tables share one test set, yet $1-SSE/SST$ would "
            "imply three different test variances for it. Both are reported "
            "throughout."
        ),
        code(
            'fig, ax = plt.subplots(figsize=(5, 5))\n'
            'plots.actual_vs_predicted(test[config.TARGET], qca_pred, ax=ax,\n'
            '                          title="QCA on the test set (paper Fig. 9)")'
        ),
        md(
            "## 5. A worked appraisal — the output that justifies the method\n\n"
            "Everything above evaluates the approach as a predictor. This is what "
            "it produces *as an appraisal*: the real transactions behind one "
            "valuation and the correction applied to each."
        ),
        code(
            'target = test.iloc[0]\n'
            'report = best.comparables(target, k=6)\n'
            'print("TARGET CASE")\n'
            'for f in config.DECOMPOSITION_ORDER:\n'
            '    print(f"  {config.LABELS[f]:<60s} {target[f]:>12,.2f}")\n'
            'print(f"\\n  estimated price  {report.attrs[\'estimate\']:.2f}")\n'
            'print(f"  actual price     {target[config.TARGET]:.2f}")\n'
            'print(f"  error            {report.attrs[\'estimate\'] - target[config.TARGET]:+.2f}")\n'
            'report.round(4)'
        ),
        md(
            "Read one row. A comparative case that traded at 32.9 but is slightly "
            "less well located carries a correction factor near 0.98, so it "
            "contributes an adjusted 32.1 — and it contributes heavily, because "
            "its case distance is 0.10. That is an appraisal a client can be "
            "walked through, which is precisely the advantage the paper claims "
            "over a hedonic regression.\n\n"
            "It also shows the method's exposure. The 117.5 outlier identified in "
            "notebook 01 appears among the nearest comparatives for some targets; "
            "because Eq. (35) is a weighted mean rather than a weighted median, a "
            "single anomalous trade moves the estimate."
        ),
        code(
            '_, best_detail = best.predict(test, return_detail=True)\n'
            'contrib = pd.DataFrame({\n'
            '    "case weight": best_detail["weight"][0],\n'
            '    "adjusted price": best_detail["adjusted"][0],\n'
            '})\n'
            'contrib["share of the estimate"] = (\n'
            '    contrib["case weight"] / contrib["case weight"].sum()\n'
            ')\n'
            'top = contrib.sort_values("share of the estimate", ascending=False)\n'
            'print(f"comparatives carrying half the weight: "\n'
            '      f"{int((top[\'share of the estimate\'].cumsum() <= 0.5).sum()) + 1}"\n'
            '      f" of {len(top)}")\n'
            'print(f"largest single share: {top[\'share of the estimate\'].iloc[0]:.1%}")'
        ),
        md(
            "## 6. Does the comparison step earn its keep?\n\n"
            "Against the plain decomposition model of notebook 02:"
        ),
        code(
            'pd.DataFrame({\n'
            '    "decomposition model only":\n'
            '        metrics.evaluate(test[config.TARGET],\n'
            '                         best.decomposition.predict_price(test), mean_price),\n'
            '    "full QCA":\n'
            '        metrics.evaluate(test[config.TARGET], qca_pred, mean_price),\n'
            '}).T.round(4)'
        ),
        md(
            "On this split the gain in RMSE is small. The case for the approach "
            "rests less on accuracy than on the auditable output shown in "
            "section 5 — which is, in fairness, the argument the paper itself "
            "makes in its Table 8."
        ),
    ],
    "The Quantitative Comparative Approach",
)


# ===========================================================================
# 04 - benchmarks
# ===========================================================================
nb04 = notebook(
    [
        md(
            "# 04 · The hedonic benchmarks (paper Tables 6–7)\n\n"
            + CITE
            + "\n---\n\n"
            "Section 5.3 compares the approach against two classical hedonic "
            "price models on the same split:\n\n"
            "* **Multivariate regression** — Eq. (36), OLS on the four factors "
            "plus northing, easting, their squares and their interaction.\n"
            "* **Neural network** — a multi-layer perceptron, tuned by the authors "
            "to one hidden layer of five units, learning rate 0.6, 1500 learning "
            "cycles.\n\n"
            "The MLP is written out in NumPy rather than taken from a library, so "
            "that the published hyper-parameters mean what they meant in the "
            "paper."
        ),
        code(HEADER),
        code(
            'df = data.load()\n'
            'train, test = data.train_test_split_paper(df, config.RANDOM_STATE)\n'
            'mean_price = float(df[config.TARGET].mean())\n'
            'y_test = test[config.TARGET]'
        ),
        md(
            "## 1. Multivariate regression (Eq. 36)\n\n"
            "$$Y = a_0 + a_1x_1 + a_2x_2 + a_3x_3 + a_4x_4 + a_5N + a_6E + "
            "a_7N^2 + a_8E^2 + a_9NE$$"
        ),
        code(
            'mra = MultivariateRegression().fit(train)\n'
            'mra_pred = mra.predict(test)\n'
            'mra.summary().to_frame().round(6)'
        ),
        md(
            "The signs are the expected ones: negative on MRT distance and house "
            "age, positive on convenience stores and the transaction date. The "
            "quadratic coordinate terms are the model's only way to bend to "
            "location, and they are what the paper's Eq. (25) surface does more "
            "directly."
        ),
        md(
            "## 2. The neural network\n\n"
            "One detail decides whether this benchmark is fair. A **learning "
            "cycle** in the back-propagation literature the paper follows is one "
            "pass of *pattern-by-pattern* updates, not one batch-averaged step. "
            "Implemented as batch updates, 1500 cycles at learning rate 0.6 leaves "
            "the network far short of convergence and the benchmark looks far "
            "worse than the paper's — about RMSE 10.9 instead of 7.1. Implemented "
            "per pattern, it reproduces."
        ),
        code(
            'ann = PaperMLP(random_state=0).fit(train)\n'
            'ann_pred = ann.predict(test)\n\n'
            'fig, ax = plt.subplots(figsize=(7, 3.6))\n'
            'ax.plot(ann.history_, color=plots.PALETTE[0])\n'
            'ax.set_xlabel("Learning cycle")\n'
            'ax.set_ylabel("Training RMSE")\n'
            'ax.set_title("Convergence of the 5-unit MLP")\n'
            'print(f"training RMSE after {ann.n_cycles} cycles: {ann.history_[-1]:.3f}")\n'
            'print(f"test RMSE: {metrics.rmse(y_test, ann_pred):.3f}")'
        ),
        md(
            "### Is five hidden units the right choice?\n\n"
            "The paper reports it as the best of a number of trials without "
            "showing them. Re-running the search on the training set:"
        ),
        code(
            'rows = []\n'
            'for units in [2, 3, 5, 8, 12]:\n'
            '    net = PaperMLP(hidden_units=units, random_state=0).fit(train)\n'
            '    rows.append({\n'
            '        "hidden units": units,\n'
            '        "training RMSE": net.history_[-1],\n'
            '        "test RMSE": metrics.rmse(y_test, net.predict(test)),\n'
            '    })\n'
            'pd.DataFrame(rows).set_index("hidden units").round(4)'
        ),
        md(
            "## 3. The three approaches side by side (paper Tables 5–7)"
        ),
        code(
            'grid_table = experiments.radius_grid(train, test)\n'
            'method, radius = experiments.best_setting(grid_table)\n'
            'preds = experiments.fit_all_models(train, test, method, radius,\n'
            '                                   include_modern=False)\n'
            'comparison = experiments.evaluation_table(test, preds, mean_price)\n'
            'comparison.round(4)'
        ),
        code(
            'paper = pd.DataFrame(\n'
            '    [[80.2, 52.9, 0.683, 7.73, 0.204],\n'
            '     [74.3, 41.8, 0.530, 8.65, 0.228],\n'
            '     [78.0, 48.8, 0.627, 8.06, 0.212]],\n'
            '    index=comparison.index,\n'
            '    columns=["20% error hit rate", "10% error hit rate", "R2",\n'
            '             "RMSE", "RMSE/Avg. house price"],\n'
            ')\n'
            'print("as printed in the paper:")\n'
            'paper'
        ),
        code(
            'side = pd.DataFrame({\n'
            '    "RMSE — replication": comparison["RMSE"],\n'
            '    "RMSE — paper": paper["RMSE"],\n'
            '    "rank — replication": comparison["RMSE"].rank().astype(int),\n'
            '    "rank — paper": paper["RMSE"].rank().astype(int),\n'
            '})\n'
            'side.round(3)'
        ),
        md(
            "**Half of the paper's finding (3) replicates on this split, and half "
            "does not.**\n\n"
            "*Replicates.* The Quantitative Comparative Approach beats "
            "multivariate regression, on RMSE and on both hit rates. This is the "
            "larger of the two gaps the paper reports, and it survives.\n\n"
            "*Does not replicate.* The neural network is **not** worse than the "
            "QCA here — on RMSE it is marginally better, reversing the paper's "
            "ordering of the two. The margin is small in both directions: the "
            "paper has the QCA ahead by 0.33 RMSE, this split has the network "
            "ahead by about 0.18.\n\n"
            "Neither number deserves to be read as a ranking. A gap of 0.2 RMSE on "
            "138 test cases drawn from one random split is not evidence of "
            "anything, and that applies to the paper's result as much as to this "
            "one. Notebook 05 settles it properly by repeating the whole "
            "comparison over 30 independent splits and testing the differences in "
            "pairs — where the QCA's advantage over regression holds up and its "
            "difference from the network does not."
        ),
        code(
            'fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))\n'
            'lim = float(max(y_test.max(), 80))\n'
            'for ax, (name, pred) in zip(axes, preds.items()):\n'
            '    plots.actual_vs_predicted(\n'
            '        y_test, pred, ax=ax, lim=lim,\n'
            '        title=f"{name}\\nRMSE {metrics.rmse(y_test, pred):.2f}",\n'
            '    )\n'
            'fig.tight_layout()'
        ),
        md(
            "## 4. Where the models disagree\n\n"
            "All three share the same errors on the same handful of properties, "
            "which says the residual is in the data rather than in any one model."
        ),
        code(
            'errors = pd.DataFrame({n: p - y_test.to_numpy() for n, p in preds.items()},\n'
            '                      index=test.index)\n'
            'print("correlation between the models\' errors:")\n'
            'print(errors.corr().round(3).to_string())\n\n'
            'worst = errors.abs().mean(axis=1).nlargest(5).index\n'
            'print("\\nthe five hardest test properties:")\n'
            'cols = config.DECOMPOSITION_ORDER + [config.TARGET]\n'
            'test.loc[worst, cols].join(errors.loc[worst].round(2), rsuffix=" error").round(3)'
        ),
        code(
            'fig, ax = plt.subplots(figsize=(7.5, 4))\n'
            'for name in errors:\n'
            '    ax.scatter(y_test, errors[name], s=18, alpha=0.6,\n'
            '               label=name.replace(" (paper spec)", ""))\n'
            'ax.axhline(0, color="0.4", lw=1)\n'
            'ax.set_xlabel("Actual price per unit area")\n'
            'ax.set_ylabel("Prediction error")\n'
            'ax.set_title("All three approaches under-predict the most expensive properties")\n'
            'ax.legend(fontsize=8)\n'
            'fig.tight_layout()'
        ),
        md(
            "Every approach shrinks towards the middle of the price range, the "
            "standard behaviour of a model fitted on squared error with few "
            "observations in the tails. Neither the paper's method nor its "
            "benchmarks address it."
        ),
    ],
    "Hedonic benchmarks",
)


# ===========================================================================
# 05 - extensions
# ===========================================================================
nb05 = notebook(
    [
        md(
            "# 05 · Extensions beyond the paper\n\n"
            + CITE
            + "\n---\n\n"
            "Notebooks 01–04 reproduce the paper. This one asks whether its "
            "conclusions survive scrutiny, and takes up two of the three "
            "questions the authors themselves list as future work in Section 6.\n\n"
            "| # | Question | Why it matters |\n"
            "|---|---|---|\n"
            "| 1 | Does the ranking survive repeated splits? | The paper reports "
            "one split of 138 test cases |\n"
            "| 2 | What if the effect radius is tuned honestly? | The paper reads "
            "it off the test set |\n"
            "| 3 | Does the decomposition order matter? | The authors' own future "
            "work #1 |\n"
            "| 4 | Is each chosen curve family the best one? | The paper states "
            "its choices without the competition |\n"
            "| 5 | How does the approach compare to modern learners? | The "
            "benchmarks are of 2018 |\n"
        ),
        code(HEADER),
        code(
            'df = data.load()\n'
            'train, test = data.train_test_split_paper(df, config.RANDOM_STATE)\n'
            'mean_price = float(df[config.TARGET].mean())\n'
            'grid_table = experiments.radius_grid(train, test)\n'
            'method, radius = experiments.best_setting(grid_table)\n'
            'N_REPEATS = 30\n'
            'print(f"paper-style setting: method {method}, radius {radius}")'
        ),
        md(
            "## 1. Does the ranking survive repeated splits?\n\n"
            "The paper's conclusion rests on one random 2:1 split. With 138 test "
            "cases, an RMSE difference of a few tenths is well inside sampling "
            "noise. Re-running everything on 30 independent splits turns a single "
            "number into a distribution, and because every model sees the same "
            "splits the comparison can be **paired**."
        ),
        code(
            'repeats = experiments.repeated_splits(\n'
            '    df, n_repeats=N_REPEATS, method=method, radius=radius,\n'
            '    include_modern=True,\n'
            ')\n'
            'experiments.summarise_repeats(repeats, "RMSE").round(4)'
        ),
        code(
            'fig, ax = plt.subplots(figsize=(9, 4.6))\n'
            'order = repeats.groupby("model")["RMSE"].mean().sort_values().index\n'
            'ax.boxplot([repeats.loc[repeats.model == m, "RMSE"] for m in order],\n'
            '           tick_labels=[m.replace(" (paper spec)", "") for m in order],\n'
            '           widths=0.6)\n'
            'ax.set_ylabel("Test RMSE")\n'
            'ax.set_title(f"Test RMSE across {N_REPEATS} random 2/3–1/3 splits")\n'
            'plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)\n'
            'fig.tight_layout()'
        ),
        md(
            "The boxes overlap heavily. The spread **within** one model across "
            "splits is larger than the gaps **between** models that the paper "
            "reports as its result."
        ),
        code(
            'experiments.paired_tests(\n'
            '    repeats, "Quantitative Comparative Approach"\n'
            ').round(4)'
        ),
        md(
            "Read the sign: a **negative** mean difference means that model beats "
            "the QCA.\n\n"
            "**What holds.** The QCA beats multivariate regression — about 0.37 "
            "RMSE, losing on only a handful of the 30 splits, with both tests at "
            "p ≈ 0.0001. It also beats plain OLS and k-NN. The paper's comparison "
            "against the weaker hedonic benchmark replicates, and comfortably.\n\n"
            "**What does not.** The paper has the QCA ahead of the neural "
            "network. Across 30 splits the network is ahead instead, by about "
            "0.14 RMSE on 19 of 30 splits. The evidence that this difference is "
            "real is mixed — the paired *t*-test does not reach 5% (p ≈ 0.10) "
            "while the Wilcoxon test does (p ≈ 0.036) — so the fair reading is "
            "that the two are close. What there is **no** support for is the "
            "paper's ordering of them: every piece of evidence here points the "
            "other way, or at best to a tie.\n\n"
            "The defensible version of the headline claim is: *the Quantitative "
            "Comparative Approach is at least competitive with the best hedonic "
            "benchmark, and reliably better than ordinary regression.* That is a "
            "weaker claim than the paper's, and it still supports its argument, "
            "because the approach also produces an auditable appraisal that "
            "neither benchmark can."
        ),
        md(
            "## 2. What if the effect radius is tuned honestly?\n\n"
            "In Table 4 the radius and the weighting method are chosen by taking "
            "the minimum of a grid scored on the **test** set, so the reported "
            "RMSE is optimistic for the configuration it reports. Here the radius "
            "is chosen by 5-fold cross-validation **inside the training set**, "
            "with the test set untouched until the final score."
        ),
        code(
            'wide_radii = sorted([0.25, 0.4, 0.5, 0.6, 0.75, 1.0] + list(config.EFFECT_RADII))\n'
            'honest = experiments.repeated_splits(\n'
            '    df, n_repeats=N_REPEATS, method=method, include_modern=False,\n'
            '    tune_radius=True, radii=wide_radii,\n'
            ')\n'
            'experiments.summarise_repeats(honest, "RMSE").round(4)'
        ),
        code(
            'chosen = honest.loc[\n'
            '    honest.model == "Quantitative Comparative Approach", "effect_radius"\n'
            ']\n'
            'print("radius chosen by cross-validation, by split:")\n'
            'print(chosen.value_counts().sort_index().to_string())'
        ),
        md(
            "Cross-validation keeps the ordering and costs little accuracy, so the "
            "paper's conclusion is not an artefact of its tuning — but the "
            "procedure should be stated as cross-validated, not as a test-set "
            "result.\n\n"
            "It also lands consistently **below** the paper's grid, which only "
            "went down to 1.25. Extending the search:"
        ),
        code(
            'experiments.radius_grid(train, test, radii=wide_radii).round(4)'
        ),
        md(
            "The minimum sits near $\\delta \\approx 0.4$, outside the range the "
            "paper searched. A tighter kernel — weighting genuinely similar "
            "properties more sharply — is better here than any value the paper "
            "tried."
        ),
        md(
            "## 3. Does the decomposition order matter? *(the authors' future "
            "work #1)*\n\n"
            "Section 6 reads: *\"the sequence of decomposition of factors in this "
            "study, no matter in which area, are the same. Maybe the sequence "
            "affects the prediction. Therefore, optimizing the sequence may "
            "further improve the prediction.\"*\n\n"
            "With four factors there are only 24 orderings, so this can be settled "
            "exhaustively rather than guessed at."
        ),
        code(
            'orders = experiments.order_sensitivity(train, test, method, radius)\n'
            'print(f"best  {orders.RMSE.min():.4f}")\n'
            'print(f"worst {orders.RMSE.max():.4f}  "\n'
            '      f"({orders.RMSE.max() / orders.RMSE.min() - 1:.1%} worse)")\n'
            'rank = int(orders.index[orders["is paper order"]][0]) + 1\n'
            'print(f"the paper\'s order ranks {rank} of 24")\n'
            'orders.head(8).round(4)'
        ),
        code(
            'fig, ax = plt.subplots(figsize=(8, 5.6))\n'
            'colours = ["#c2703d" if p else "#2f5d8a" for p in orders["is paper order"]]\n'
            'ax.barh(range(len(orders)), orders["RMSE"], color=colours)\n'
            'ax.set_yticks(range(len(orders)))\n'
            'ax.set_yticklabels(orders["order"], fontsize=7)\n'
            'ax.invert_yaxis()\n'
            'ax.set_xlim(orders.RMSE.min() * 0.97, orders.RMSE.max() * 1.01)\n'
            'ax.set_xlabel("Test RMSE")\n'
            'ax.set_title("All 24 decomposition orders (the paper\'s in orange)")\n'
            'fig.tight_layout()'
        ),
        md(
            "**The answer is yes, strongly — and the paper's rule already "
            "captures nearly all of it.**\n\n"
            "The spread across orderings is about 19% of RMSE, far larger than any "
            "difference between the three approaches in Tables 5–7. What matters "
            "is almost entirely **which factor goes first**: every ordering led by "
            "MRT distance beats every ordering that is not. That follows from how "
            "the decomposition works — the first factor is credited with the whole "
            "aggregate coefficient, so putting a weak factor first forces it to "
            "absorb variation it cannot explain, and the later steps cannot undo "
            "it.\n\n"
            "The paper's importance-ranked order comes third of 24 and is within "
            "0.03% of the best. So the authors were right that sequence matters, "
            "and their sorting quantile heuristic was already close to optimal — "
            "the remaining gain from optimising it is negligible."
        ),
        md(
            "## 4. Is each chosen curve family the best one?\n\n"
            "Section 5.1 reports the selected family per factor without showing "
            "the alternatives. All five of Eqs. (27)–(31), refitted:"
        ),
        code(
            'families = experiments.family_comparison(train)\n'
            'pivot = families.pivot(index="factor", columns="family", values="RMSE")\n'
            'chosen_family = {config.LABELS[f]: c for f, c in config.PAPER_CURVES.items()}\n'
            'pivot["paper\'s choice"] = pd.Series(chosen_family)\n'
            'pivot["best available"] = pivot.drop(columns="paper\'s choice").idxmin(axis=1)\n'
            'pivot.round(5)'
        ),
        md(
            "The quadratic model fits the portion points better for **all four** "
            "factors. That is not a reason to adopt it. A quadratic in MRT "
            "distance turns back upwards beyond its vertex, so the adjustment "
            "coefficient would start *rising* with distance from the station, and "
            "it is unbounded in extrapolation. The logarithmic and linear forms "
            "the paper chose are monotone and economically interpretable over the "
            "whole range, and with only ten points to fit, a better in-sample fit "
            "from a more flexible family is weak evidence. **The paper's choices "
            "are defensible; they are just not the lowest-RMSE ones, and the paper "
            "does not say so.**"
        ),
        code(
            'fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n'
            'from qca.curves import fit_curve\n'
            'x = train["mrt"].to_numpy()\n'
            'k = train[config.TARGET].to_numpy() / train[config.TARGET].mean()\n'
            'idx = importance.portion_indices(x)\n'
            'xp = np.array([x[i].mean() for i in idx])\n'
            'kp = np.array([k[i].mean() for i in idx])\n'
            'grid = np.linspace(20, 6500, 400)\n'
            'for ax, span in zip(axes, [(20, 6500), (20, 12000)]):\n'
            '    ax.scatter(xp, kp, s=40, zorder=3, color=plots.PALETTE[0])\n'
            '    g = np.linspace(*span, 400)\n'
            '    for fam in ["logarithmic", "quadratic"]:\n'
            '        ax.plot(g, fit_curve(xp, kp, fam)(g), lw=2, label=fam)\n'
            '    ax.axhline(0, color="0.5", lw=0.8)\n'
            '    ax.set_xlabel(config.LABELS["mrt"])\n'
            '    ax.legend(fontsize=8)\n'
            'axes[0].set_ylabel("Adjustment coefficient")\n'
            'axes[0].set_title("Over the observed range")\n'
            'axes[1].set_title("Extrapolated: the quadratic turns back up")\n'
            'fig.tight_layout()'
        ),
        md(
            "## 5. How does the approach compare to modern learners?\n\n"
            "The paper's benchmarks are a linear regression and a five-unit MLP. "
            "A fair question in 2026 is whether the approach still earns its place "
            "against a gradient-boosted tree ensemble that needs no decomposition "
            "and no hand-chosen functional forms. These are the same 30 splits as "
            "section 1."
        ),
        code(
            'summary = experiments.summarise_repeats(repeats, "RMSE")\n'
            'hits = experiments.summarise_repeats(repeats, "20% error hit rate")\n'
            'pd.DataFrame({\n'
            '    "mean RMSE": summary["mean"],\n'
            '    "sd": summary["sd"],\n'
            '    "mean 20% hit rate": hits["mean"],\n'
            '}).round(3)'
        ),
        md(
            "Here the extension bites. Going back to the paired tests of "
            "section 1:\n\n"
            "* **Random forest beats the QCA decisively** — about 0.61 RMSE, "
            "winning on 26 of 30 splits, with both p-values below 0.0001. That is "
            "a *larger* margin, with stronger evidence, than the QCA's own win "
            "over multivariate regression (0.37 RMSE).\n"
            "* **Gradient boosting also beats it**, by about 0.23 RMSE on 21 of "
            "30 splits, significant at 5% on both tests.\n\n"
            "So the honest 2026 verdict is narrower than the paper's. Against the "
            "benchmarks it chose, the approach holds up: clearly better than "
            "regression, level with the neural network. Against off-the-shelf "
            "ensembles that need no decomposition, no hand-chosen functional "
            "forms and no effect radius, it is **beaten on accuracy**.\n\n"
            "That does not retire the method, because accuracy was never the "
            "argument for it. A random forest returns a number; this returns a "
            "number with the comparable sales and the priced reason for every "
            "adjustment attached. The fair claim in 2026 is that the approach buys "
            "its interpretability at a real, measurable cost in accuracy — which "
            "is a more useful thing to know than the paper's unqualified "
            "superiority claim, and still an argument for using it where an "
            "appraisal has to be defended."
        ),
        md(
            "## 6. What this replication concludes\n\n"
            "| Claim in the paper | Verdict |\n"
            "|---|---|\n"
            "| The released data is the Sindian circle of Table 1 | **Confirmed** "
            "to the second decimal |\n"
            "| Factor ranking: MRT > stores > age > date | **Confirmed** |\n"
            "| Adjustment coefficient models of Table 2 | **Confirmed** for three "
            "of four factors; the transaction-date intercept as printed is "
            "internally inconsistent |\n"
            "| Signs of all four coefficient models | **Confirmed** |\n"
            "| QCA beats multivariate regression | **Confirmed**, and significant "
            "across 30 splits |\n"
            "| QCA beats the neural network | **Not confirmed** — across 30 "
            "splits the network is marginally ahead, and no test supports the "
            "paper's ordering |\n"
            "| *(not claimed)* QCA against modern ensembles | Random forest beats "
            "it by a wider margin than it beats regression |\n"
            "| Effect radius best around 1.5–3 | **Not confirmed** — the optimum "
            "is near 0.4, below the range searched |\n"
            "| Decomposition sequence may matter *(future work #1)* | **Confirmed "
            "and quantified** — 19% spread, though the paper's own order is "
            "near-optimal |\n\n"
            "The method reproduces. The comparison against the stronger benchmark "
            "does not survive repeated splits, and two tuning choices were made on "
            "the test set. Neither finding undermines the paper's contribution, "
            "which is a way to compute correction coefficients that appraisers "
            "currently set by judgement."
        ),
    ],
    "Extensions",
)


NOTEBOOKS = {
    "01_data_and_factor_importance.ipynb": nb01,
    "02_stepwise_decomposition.ipynb": nb02,
    "03_quantitative_comparative_approach.ipynb": nb03,
    "04_hedonic_benchmarks.ipynb": nb04,
    "05_extensions.ipynb": nb05,
}


def main(run: bool = False, only: str | None = None) -> None:
    selected = {
        name: nb for name, nb in NOTEBOOKS.items() if only is None or only in name
    }
    if not selected:
        raise SystemExit(f"no notebook matches {only!r}; have {list(NOTEBOOKS)}")

    for name, nb in selected.items():
        path = NB_DIR / name
        nbf.write(nb, path)
        print(f"wrote {path.relative_to(ROOT)}  ({len(nb.cells)} cells)")

    if run:
        from nbclient import NotebookClient

        for name in selected:
            path = NB_DIR / name
            print(f"executing {name} ...", flush=True)
            nb = nbf.read(path, as_version=4)
            NotebookClient(
                nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}
            ).execute()
            nbf.write(nb, path)
            print(f"  done {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute after writing")
    parser.add_argument(
        "--only",
        metavar="SUBSTRING",
        help="restrict to notebooks whose filename contains this, e.g. --only 05",
    )
    main(**vars(parser.parse_args()))
