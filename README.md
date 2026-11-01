# Replicating *Building real estate valuation models with comparative approach through case-based reasoning*

A full, from-scratch Python replication of **Yeh & Hsu (2018)**, *Applied Soft
Computing* **65**, 260–271 — including the parts that do not reproduce.

> I-Cheng Yeh and Tzu-Kuang Hsu (2018). "Building real estate valuation models
> with comparative approach through case-based reasoning."
> *Applied Soft Computing* **65**, 260–271.
> [doi:10.1016/j.asoc.2018.01.029](https://doi.org/10.1016/j.asoc.2018.01.029)

**Data** · UCI Machine Learning Repository, *Real Estate Valuation*
([doi:10.24432/C5J30W](https://doi.org/10.24432/C5J30W)), CC BY 4.0 — 414
transactions from Sindian District, New Taipei City, June 2012 – May 2013.

---

## The problem the paper attacks

Property appraisers mostly use the **comparative approach**: find recent sales
of similar properties and adjust their prices for the differences. It is the
method clients accept, because the valuation is backed by real trades they can
look at.

Its weak point is the adjustment. *How much* is a property worth for being 300 m
closer to a metro station, or eight years newer? In practice the appraiser
decides by judgement, so two appraisers given the same property and the same
comparables return different numbers.

A hedonic regression removes the judgement but throws away the comparables — it
returns a number with no sales behind it.

Yeh & Hsu's **Quantitative Comparative Approach** keeps both. It estimates the
correction coefficients from the market data, then applies them to real
comparable sales. The valuation stays traceable to actual trades, and the
adjustments stop being subjective.

## How it works

Price per unit area is assumed to be the market average scaled by a product of
dimensionless **adjustment coefficients**, one per factor (Eq. 9):

$$Y_j = \bar{Y} \cdot k_{1j} k_{2j} \cdots k_{mj}$$

Only their product $k_j = Y_j/\bar{Y}$ is observable, so the paper separates
them greedily — **stepwise decomposition regression**. Rank the factors by the
**sorting quantile method** (Eq. 8), credit the whole product to the most
important one, fit a curve through ten portion means, divide it out, and repeat
down the ranking. Location goes last, as a quadratic surface (Eq. 25).

To value a target property, every comparative sale is scored by a weighted
standardised distance (Eq. 32), turned into a weight by a Gaussian kernel of
*effect radius* $\delta$ (Eq. 33), price-corrected factor by factor (Eq. 34),
and averaged (Eq. 35).

Steps 1–2 are kernel regression. **Step 3 is the contribution** — it is what
turns a prediction into an appraisal.

## What this replication found

### Reproduces

| Claim | Result |
|---|---|
| The released file is the Sindian circle of Table 1 | Every statistic matches to the second decimal |
| Factor ranking: **MRT distance > convenience stores > house age > transaction date** | Reproduced exactly |
| Adjustment coefficient models (Table 2) | Three of four factors land within 0.002 of the published coefficients; for the most important factor, `a = -0.23882` against the paper's `-0.23843` |
| All four coefficient signs | Reproduced, and all four are the sign economic reasoning predicts |
| **QCA beats multivariate regression** | Reproduced on the single split *and* significant across 30 repeated splits (0.37 RMSE, p ≈ 0.0001) |

### Does not reproduce

| Claim | Result |
|---|---|
| **QCA beats the neural network** | Not supported. On the replication split the network is marginally *ahead*, and across 30 splits it stays ahead (by 0.14 RMSE, on 19 of 30). No test supports the paper's ordering |
| Effect radius optimal around 1.5–3 | The optimum is near **0.4**, below the range the paper searched |
| Table 2, transaction-date row | The printed pair `a = 0.21170, b = -20.62480` is internally inconsistent — see below |

### Found along the way

**A reporting error in Table 2.** An adjustment coefficient is dimensionless and
sits near 1. With the paper's printed `a = 0.21170` and `b = -20.62480`, the
transaction-date model evaluates to **≈ 405** at the mean transaction date — it
would multiply every price by four hundred. The *slope* matches this
replication closely (0.21170 against 0.22280); an intercept near **−425** is what
that slope requires. The printed intercept has lost digits in typesetting. Pinned
by `tests/test_qca.py::test_transaction_date_intercept_contradicts_the_printed_table`.

**Two parameters are tuned on the test set.** Table 4 picks the effect radius and
the weighting method by taking the minimum of a grid scored on the test data, so
the headline RMSE is not out-of-sample for the configuration it reports.
Re-tuning by cross-validation *inside* the training set leaves the conclusions
standing — so this is a reporting problem, not a broken result.

**The decomposition order matters, and the paper's heuristic already handles it.**
Section 6 lists optimising the sequence as future work. With four factors there
are 24 orderings, so it can be settled exhaustively: RMSE spans **19%** across
them — a larger effect than any difference between the three approaches in
Tables 5–7. Almost all of it is *which factor goes first*: every ordering led by
MRT distance beats every ordering that is not. The paper's importance-ranked
order comes **3rd of 24** and within 0.03% of the best, so the authors' heuristic
was already near-optimal and little is left to gain.

**The "learning cycle" ambiguity decides the benchmark.** Read as batch-averaged
updates, the paper's MLP spec (5 hidden units, learning rate 0.6, 1500 cycles)
leaves the network far short of convergence and scores RMSE ≈ 10.9 — making the
benchmark look much worse than the paper's. Read as pattern-by-pattern updates,
the convention of the back-propagation literature the paper follows, it
reproduces at ≈ 7.1. A replication that got this wrong would have "confirmed"
the paper's ANN comparison for the wrong reason.

**Which $R^2$?** The paper never says. The two usual definitions disagree under
bias, and the $R^2$/RMSE pairs across Tables 5–7 are only mutually consistent
under the squared-correlation reading — the three tables share one test set, yet
$1-SSE/SST$ would imply three different variances for it. Both are reported
throughout.

**A random forest beats it by a wider margin than it beats regression.** Adding
modern learners over the same 30 splits: a random forest is ahead of the QCA by
0.61 RMSE, winning 26 of 30 splits at p < 0.0001 — a larger effect, with stronger
evidence, than the QCA's own 0.37 RMSE win over multivariate regression.
Gradient boosting is ahead too (0.23 RMSE, p ≈ 0.04). The paper could not have
known this in 2018, but it is the relevant comparison now.

### The honest summary

The method reproduces, and the code reproduces the paper's own numbers closely
enough to be confident the implementation is faithful.

Its advantage over ordinary regression is real and statistically supported. Its
advantage over a neural network is not — that conclusion rests on a single split
of 138 cases and reverses under repetition. Against modern ensembles it loses on
accuracy outright.

None of this retires the method, because accuracy was never the argument for it.
A random forest returns a number; this returns a number **with the comparable
sales and the priced reason for every adjustment attached**, which is what an
appraisal has to be able to show and what no model it is compared against can do.
The fair 2026 claim is that the approach buys its interpretability at a real,
measurable cost in accuracy — a more useful thing to know than an unqualified
superiority claim, and still a reason to use it where a valuation has to be
defended to the person it affects.

---

## Repository layout

```
.
├── data/
│   ├── raw/                     the UCI .xlsx release (CC BY 4.0, committed)
│   └── README.md                variable dictionary and preparation notes
├── src/qca/                     the replication package
│   ├── config.py                every constant taken from the paper, in one place
│   ├── data.py                  loading, age flooring, local metric projection
│   ├── importance.py            sorting quantile method (Eq. 8), weights (Table 3)
│   ├── curves.py                the five regression families (Eqs. 27–31)
│   ├── decomposition.py         stepwise decomposition (Eqs. 9–26)
│   ├── estimator.py             the approach itself (Eqs. 32–35)
│   ├── benchmarks.py            MRA (Eq. 36), the paper-spec MLP, modern learners
│   ├── metrics.py               hit rates, both R² definitions, RMSE
│   ├── experiments.py           the paper's tables and the extensions
│   └── plots.py                 figures
├── notebooks/                   the narrative, in order
│   ├── 01_data_and_factor_importance.ipynb
│   ├── 02_stepwise_decomposition.ipynb
│   ├── 03_quantitative_comparative_approach.ipynb
│   ├── 04_hedonic_benchmarks.ipynb
│   └── 05_extensions.ipynb
├── scripts/
│   ├── run_all.py               regenerate every table and figure
│   └── build_notebooks.py       regenerate the notebooks from source
├── tests/test_qca.py            27 tests
├── results/{tables,figures}/    generated output, committed for reference
└── docs/
    ├── paper/README.md          how to obtain the article (not redistributable)
    └── REPLICATION_REPORT.md    the written report
```

## Running it

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_all.py
```

Writes every table to `results/tables/` and every figure to `results/figures/`.
Takes roughly 15 minutes, almost all of it the MLP, which trains
pattern-by-pattern. For a 2-minute smoke test:

```bash
python scripts/run_all.py --quick
```

```bash
python -m pytest tests/ -q
```

```bash
jupyter lab notebooks/
```

The notebooks are committed **with their output**, so they can be read without
running anything. To regenerate and re-execute them:

```bash
python scripts/build_notebooks.py --run
```

Add `--only 05` to rebuild a single notebook. If only the prose changed, there
is nothing to recompute — this copies the text across and keeps every stored
output:

```bash
python scripts/sync_notebook_prose.py
```

## Design notes

**One source of truth.** Notebooks and `run_all.py` both call `qca.experiments`,
so a number in the report and the same number in a notebook cannot drift apart.

**Every choice the paper left open is named.** Equation (32) is printed without a
visible radical; the geographic coordinate is one factor with two components;
"learning cycle" has two readings; `R²` has two definitions. Each is resolved
explicitly in code comments and in the notebooks, with the reasoning, rather
than silently.

**Methods 3 and 4 use the published weights.** They average over all four
circles of supply and demand and only Sindian was released, so they cannot be
recomputed. `characteristic_weights` *raises* unless the published values are
passed in, so the substitution can never happen by accident.

**Tests check properties, not snapshots.** That the case weight is 1 at distance
zero and $1/e$ at the effect radius; that a large radius collapses the estimator
to the hedonic model; that adjustment coefficients average near 1. These stay
meaningful if a default is retuned.

## Citation

```bibtex
@article{yeh2018building,
  title   = {Building real estate valuation models with comparative approach
             through case-based reasoning},
  author  = {Yeh, I-Cheng and Hsu, Tzu-Kuang},
  journal = {Applied Soft Computing},
  volume  = {65},
  pages   = {260--271},
  year    = {2018},
  doi     = {10.1016/j.asoc.2018.01.029}
}
```

## Licence

Code released under the MIT Licence (`LICENSE`). The dataset is CC BY 4.0, from
the UCI Machine Learning Repository. The article itself is Elsevier's and is not
redistributed here — see `docs/paper/README.md`.
