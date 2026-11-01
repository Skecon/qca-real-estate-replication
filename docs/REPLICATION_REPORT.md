# Replication report

**Paper** — I-Cheng Yeh and Tzu-Kuang Hsu (2018), "Building real estate
valuation models with comparative approach through case-based reasoning",
*Applied Soft Computing* **65**, 260–271.
[doi:10.1016/j.asoc.2018.01.029](https://doi.org/10.1016/j.asoc.2018.01.029)

**Data** — UCI Machine Learning Repository, *Real Estate Valuation*
([doi:10.24432/C5J30W](https://doi.org/10.24432/C5J30W)), CC BY 4.0. 414
transactions, Sindian District, New Taipei City, June 2012 – May 2013.

**Scope** — the paper studies four circles of housing supply and demand
(Sindian, Danshuei, Wunshan, Beitou). Only Sindian was released, so this
replication covers the Sindian column of every published table. Everything
below is reproducible with `python scripts/run_all.py`.

---

## 1. What the paper proposes, and why it matters

Real estate appraisal has four standard methods. The **comparative approach** is
the one clients accept, because the valuation is anchored in real recent sales
of similar properties. Its weakness is the adjustment step: having found a
comparable sale, the appraiser must decide how much to add or subtract for every
difference — 300 m further from the metro, eight years older, two fewer
convenience stores nearby. Those **correction coefficients** are set by
judgement, so two appraisers with the same property and the same comparables
produce different valuations, and neither can fully defend their number.

The **hedonic price approach** — a regression of price on property attributes —
removes the judgement, and the paper acknowledges it is what industry uses for
mass appraisal. But it returns a number with no sales behind it. The paper is
blunt about why that matters in practice: comparative cases from real market
trading are what persuade a trader to accept an appraisal.

The **Quantitative Comparative Approach** is an attempt to have both. Estimate
the correction coefficients from market data, then apply them to real comparable
sales. The output is a valuation that is simultaneously objective and traceable.

This is worth taking seriously as an analytics contribution precisely because
its selling point is not accuracy. It is a method designed around the constraint
that the *user of the model must be able to explain its output to a third
party* — a constraint that shows up far beyond property valuation, in credit
decisions, insurance pricing and anywhere a model's output has to be justified
to the person it affects.

## 2. The method

### 2.1 The multiplicative model

Price per unit area is assumed to be the market average scaled by a product of
dimensionless **adjustment coefficients**, one per factor (Eq. 9):

$$Y_j = \bar{Y}\cdot k_{1j}k_{2j}\cdots k_{mj}$$

Each $k$ answers a single question: *how much does this one feature push this
property above or below the typical property in its market?* A coefficient of
1.1 means +10%.

Only the product $k_j = Y_j/\bar{Y}$ is observable (Eq. 10), so the individual
coefficients have to be separated.

### 2.2 Separating them: stepwise decomposition

The paper's device is greedy attribution. Rank the factors by importance, then:

1. Credit the **whole** product to the most important factor (Eq. 12).
2. Average it within ten portions of that factor, fit a curve through the ten
   points (Eq. 14).
3. Divide the fitted curve out; credit the remainder to the next factor
   (Eq. 16).
4. Repeat down the ranking (Eqs. 19, 21), and finish with location as a
   second-order surface in northing and easting (Eq. 25).

This is a multiplicative analogue of forward stagewise regression. **The order
is load-bearing** — the first factor absorbs everything, and later steps cannot
undo it. Section 5.3 of this report tests that.

Importance comes from the **sorting quantile method** (Eq. 8): sort by a factor,
cut into ten equal portions, and sum the squared deviations of the ten portion
mean prices from the grand mean. It uses only the ordering of the factor, so it
is indifferent to monotone rescaling and to functional form — which matters,
because half of the relationships here turn out to be non-linear, the most
important one most of all.

### 2.3 Valuing a target property

For each comparative sale $j$: a weighted standardised distance (Eq. 32), a
Gaussian kernel weight of effect radius $\delta$ (Eq. 33), a price corrected
factor by factor using $K_i = \hat{k}_i(\text{target})/\hat{k}_i(\text{comp})$
(Eq. 34), and a weighted average (Eq. 35).

Steps 1–2 are ordinary kernel regression. **Step 3 is the contribution**, and it
is the whole point: it is what makes the output an appraisal rather than a
prediction.

## 3. Replication decisions

The paper leaves five things genuinely ambiguous. Each is resolved explicitly
rather than silently, because a different choice would change the numbers.

| # | Ambiguity | Resolution | Why |
|---|---|---|---|
| 1 | **Eq. (32) has no visible radical** in the typeset article. Taken literally it is a squared distance, which the Eq. (33) kernel then squares again. | Take the square root. | Only then does the paper's own description of the kernel hold — weight 1 at $D=0$, $1/e$ at $D=\delta$ — and only then is $D$ in standard-deviation units. |
| 2 | **The geographic coordinate** is one factor with two components, but the paper assigns it a single weight of 3. | Northing and easting each enter the distance sum carrying weight 3. | The alternative (splitting 3 between them) halves location's influence relative to every scored factor, which contradicts fixing it high deliberately. |
| 3 | **"1500 learning cycles"** for the MLP. | One cycle = one pass of pattern-by-pattern updates. | See §5.5 — this single choice moves the benchmark from RMSE 10.9 to 7.1. |
| 4 | **Which $R^2$.** | Report both $1-SSE/SST$ and squared correlation. | Tables 5–7 share one test set, yet their $R^2$/RMSE pairs imply three different test variances under $1-SSE/SST$ — so the paper is using squared correlation. Stating this rather than guessing. |
| 5 | **Coordinate units.** The paper uses metres (Eq. 25); UCI ships degrees. | Equirectangular projection about the sample centroid, in metres. | Accurate to well under a metre over one district, and it keeps the quadratic surface well conditioned. See §5.2. |

Two further points of fidelity:

* **House age is floored at 0.10.** Table 1 of the paper reports a minimum of
  0.10 year for Sindian; the UCI release stores 17 of those as exactly 0.0, and
  the age coefficient is a logarithmic model, undefined at zero. Flooring at the
  paper's own reported minimum is the least arbitrary repair available.
* **Weighting methods 3 and 4 use the published weights.** They average over all
  four circles and only Sindian was released. `characteristic_weights` *raises*
  unless the Table 3 values are passed in, so the substitution cannot happen by
  accident and is visible at every call site.

## 4. Results

### 4.1 Table 1 — the dataset identification

Every descriptive statistic matches the Sindian row of the published Table 1 to
the second decimal: MRT distance mean 1083.89 (sd 1262.11), convenience stores
4.09 (2.95), house age 17.71 (11.39), transaction date 2013.15 (0.28), N = 414.

**The released file is the paper's Sindian circle.** The replication can
therefore be judged against the published numbers rather than against a
reconstruction.

### 4.2 Table 2 — the adjustment coefficient models

Fitted on all 414 rows, which is the closest well-defined comparison because the
paper's 2/3 training split was drawn with an unreported seed.

| Factor | Model | Paper $a$ | Ours $a$ | Paper $b$ | Ours $b$ |
|---|---|---|---|---|---|
| MRT distance | logarithmic | −0.23843 | **−0.23882** | 2.527609 | **2.52936** |
| Convenience stores | linear | 0.009452 | **0.00912** | 0.955591 | **0.95770** |
| House age | logarithmic | −0.06408 | **−0.06233** | 1.159766 | **1.15873** |
| Transaction date | linear | 0.21170 | **0.22280** | −20.62480 | **−447.54** |

Three of four factors land within 0.002 of the published slope and intercept.
For the most important factor the agreement is to the fourth decimal. **The
decomposition replicates.**

All four signs reproduce, and all four are the sign economic reasoning predicts:
price falls with distance from the metro and with age, rises with convenience
stores and over the study period (Sindian's price boom). The MRT curve's shape
matters as much as its sign — almost all of the effect is inside the first
500 m, because beyond about a kilometre nobody is walking to the station anyway.

#### A reporting error in the published Table 2

The transaction-date intercept does not agree, and it cannot. An adjustment
coefficient is dimensionless and sits near 1 by construction. With the printed
$a = 0.21170$ and $b = -20.62480$, the model evaluates at the mean transaction
date of 2013.15 to:

$$0.21170 \times 2013.15 - 20.62480 \approx 405$$

That would multiply every price by four hundred. The **slope** matches this
replication closely (0.21170 against 0.22280), and an intercept near **−425** is
what that slope requires for the coefficient to pass through 1 — which is almost
exactly what is obtained here. The printed intercept has lost digits in
typesetting.

This is a reporting error, not a different result, and it is pinned by
`tests/test_qca.py::test_transaction_date_intercept_contradicts_the_printed_table`
so the claim is checkable rather than asserted.

### 4.3 Table 3 — the characteristic weights

The *ranking* reproduces exactly: MRT distance > convenience stores > house age
> transaction date, which is the paper's finding (1).

The published *levels* differ, for a reason that is visible once the
normalisation is changed. All weights are expressed relative to the transaction
date, which has the smallest and noisiest Eq. (8) variance of the four — it is
the worst available denominator. Rescaling both sets against house age instead,
the replication and the paper agree closely. The weights are the same object;
only the published normaliser is unstable.

### 4.4 Tables 5–7 — the three approaches

On the replication split (seed 0, best cell of the Table 4 grid: weighting
method 4, radius 1.25):

| Approach | 20% hit | 10% hit | $R^2$ (corr²) | RMSE | RMSE/avg |
|---|---|---|---|---|---|
| Quantitative Comparative Approach | 76.1 | 53.6 | 0.692 | **7.301** | 0.192 |
| Multivariate regression (Eq. 36) | 75.4 | 48.6 | 0.649 | 7.751 | 0.204 |
| Neural network (paper spec) | 81.2 | 50.0 | 0.697 | **7.126** | 0.188 |

Against the paper: QCA 7.73, MRA 8.65, ANN 8.06.

**Half of finding (3) replicates.** The QCA beats multivariate regression on
RMSE and on both hit rates — the larger of the two gaps the paper reports, and
it survives. The neural network, however, is *not* worse than the QCA here; on
RMSE it is marginally better, reversing the paper's ordering of those two.

The margins are small in both directions — the paper has the QCA ahead by 0.33
RMSE, this split has the network ahead by 0.18 — and neither deserves to be read
as a ranking. A gap of 0.2 RMSE on 138 test cases from one random split is not
evidence of anything. That applies to the paper's result as much as to this one,
which is why §5.1 settles it by repetition.

## 5. Extensions

### 5.1 Does the ranking survive repeated splits?

The paper's conclusion rests on one split. Repeating the entire comparison over
30 independent 2/3–1/3 splits turns each single number into a distribution, and
because every model sees the same splits, the differences can be tested
**paired**.

The first thing the distributions show is that the spread *within* each model
across splits (sd ≈ 1.1 RMSE) is several times larger than the gaps *between*
models that the paper reports as its finding (0.33–0.92 RMSE). Pairing is what
makes the comparison recoverable at all.

Mean test RMSE over 30 splits, and the paired comparison against the QCA:

| Model | Mean RMSE | Difference vs QCA | Wins / 30 | paired *t* | Wilcoxon |
|---|---|---|---|---|---|
| Random forest | 7.366 | **−0.608** | 26 | <0.0001 | <0.0001 |
| Gradient boosting | 7.748 | **−0.227** | 21 | 0.040 | 0.033 |
| Neural network (paper spec) | 7.839 | −0.135 | 19 | 0.102 | 0.036 |
| **Quantitative Comparative Approach** | **7.974** | — | — | — | — |
| k-NN (k = 5) | 8.238 | +0.264 | 9 | 0.006 | 0.007 |
| Multivariate regression (Eq. 36) | 8.346 | **+0.372** | 6 | 0.0001 | 0.0001 |
| OLS, no polynomial terms | 8.772 | +0.798 | 2 | <0.0001 | <0.0001 |

A negative difference means the model beats the QCA.

* **QCA vs multivariate regression** — the QCA wins by 0.372 RMSE, losing on
  only 6 of 30 splits, with both tests at p ≈ 0.0001. **The paper's comparison
  against the weaker hedonic benchmark replicates, and comfortably.**
* **QCA vs the neural network** — the paper has the QCA ahead; here the network
  is ahead instead, by 0.135 RMSE on 19 of 30 splits. The evidence that this
  difference is real is mixed: the paired *t*-test does not reach 5%
  (p = 0.102) while the Wilcoxon test does (p = 0.036). The fair reading is that
  the two are close — but **nothing here supports the paper's ordering of them**,
  and the point estimate runs against it.

The claim that survives is: *the Quantitative Comparative Approach is at least
competitive with the best hedonic benchmark, and reliably better than ordinary
regression.* That is weaker than the paper's claim. It is still a result, and it
still supports the paper's argument, because the approach also produces an
auditable appraisal that neither benchmark can.

### 5.2 Two parameters are tuned on the test set

In Table 4 the effect radius and the weighting method are chosen by taking the
minimum of a grid scored on the **test** set. The reported RMSE is therefore not
an out-of-sample figure for the configuration it reports.

Re-selecting the radius by 5-fold cross-validation **inside the training set**,
with the test set untouched until the final score, does not hurt — it *helps*.
Mean RMSE over the 30 splits falls from **7.974 to 7.802**, which brings the QCA
level with the neural network (7.839). So this is a reporting problem rather
than a broken result: the paper's conclusions are not propped up by its tuning,
and a defensible procedure would have produced a slightly better number than the
one it published.

The reason the honest procedure wins is that cross-validation keeps choosing
radii **below the paper's grid**, which stopped at 1.25: across the 30 splits it
picks 0.25 or 0.4 two-thirds of the time. Extending the grid downwards confirms
it — the minimum sits near $\delta \approx 0.4$ (RMSE 7.02 against 7.33 at the
paper's best), so a tighter kernel that weights genuinely similar properties
more sharply beats every value the paper tried. The paper's own reasoning
pointed here: it argues that neither extreme works and that the radius should
let similar cases dominate. It simply did not search far enough in that
direction.

#### A note on the flat radius curve

The paper's RMSE moves about 10% across radii; here it moves under 1%. The
reason is traceable: the decomposition model alone already predicts the test set
at RMSE ≈ 7.33, so once prices are corrected by the Eq. (34) coefficient ratios,
the adjusted prices are nearly identical whichever comparative they came from,
and reweighting them changes little.

The most likely cause of the difference is numerical conditioning. The paper fits
the Eq. (25) quadratic surface on absolute metre coordinates — values near
2,760,000, whose squares are near $7.6\times10^{12}$ — where the design matrix is
severely ill-conditioned. This replication centres the coordinates first. A
weaker location model leaves more spatial structure for the kernel to recover,
and so a steeper radius curve. The shape and the ranking of methods reproduce;
the magnitude of the radius effect does not.

### 5.3 Does the decomposition order matter? *(the authors' future work #1)*

Section 6 of the paper lists this first among its suggestions: the sequence was
held fixed everywhere, and the authors expect optimising it may improve
prediction. With four factors there are only 24 orderings, so the question can
be settled exhaustively rather than guessed at.

**It matters a great deal.** RMSE spans roughly **19%** across the 24 orderings —
a larger effect than any difference between the three approaches in Tables 5–7,
and larger than every extension tested here.

Almost all of it is *which factor goes first*: **every** ordering led by MRT
distance beats **every** ordering that is not. That follows directly from the
mechanics — the first factor is credited with the entire aggregate coefficient,
so leading with a weak factor forces it to absorb variation it cannot explain,
and no later step can undo it.

**But the paper's heuristic already captures nearly all of the gain.** Its
importance-ranked order comes 3rd of 24 and sits within 0.03% of the best
available. So the authors were right that the sequence matters — and wrong to
expect much from optimising it, because their sorting quantile ranking was
already near-optimal. The honest answer to their own open question is: *yes, but
you already solved it.*

### 5.4 Are the chosen curve families the best ones?

Section 5.1 states the selected family per factor without showing the
alternatives. Refitting all five of Eqs. (27)–(31): the **quadratic** fits the
portion points better for **all four** factors.

That is not a reason to adopt it. A quadratic in MRT distance turns back upwards
beyond its vertex, so the adjustment coefficient would start *rising* with
distance from the station — economically absurd, and unbounded in extrapolation.
With only ten points to fit, a better in-sample fit from a more flexible family
is weak evidence. The paper's monotone choices are defensible on grounds the
paper does not state; the finding is that the choices were not purely empirical.

### 5.5 The "learning cycle" ambiguity decides the ANN benchmark

Read as **batch-averaged** updates, the paper's spec (5 hidden units, learning
rate 0.6, 1500 cycles) leaves the network far short of convergence: test RMSE
≈ **10.9**, far worse than the paper's 8.06, and the paper's benchmark would look
easy to beat.

Read as **pattern-by-pattern** updates — the convention of the back-propagation
literature the paper follows — it reproduces at ≈ **7.1**.

This is worth flagging as a replication hazard in its own right. A replication
that chose the first reading would have reported a much *larger* QCA advantage
than the paper claims, and would have "confirmed" the paper's conclusion for
entirely the wrong reason. Getting a benchmark wrong in the favourable direction
is harder to notice than getting it wrong in the unfavourable one.

### 5.6 How does it stand up to modern learners?

The paper's benchmarks are a linear regression and a five-unit MLP. Adding a
random forest and gradient boosting over the same 30 splits (table in §5.1):

* **Random forest beats the QCA by 0.608 RMSE**, winning 26 of 30 splits with
  both p-values below 0.0001. That is a *larger* margin, with stronger evidence,
  than the QCA's own 0.372 RMSE win over multivariate regression — the very
  comparison the paper rests its case on.
* **Gradient boosting also beats it**, by 0.227 RMSE on 21 of 30 splits,
  significant at 5% on both tests.

So the 2026 verdict is narrower than the paper's. Against the benchmarks it
chose, the approach holds up: clearly better than regression, level with the
neural network. Against off-the-shelf ensembles that need no decomposition, no
hand-chosen functional forms and no effect radius, it is **beaten on accuracy**.

This does not retire the method, for the reason the paper itself gives in its
Table 8. A random forest returns a number. The Quantitative Comparative Approach
returns a number with the comparable sales and the priced reason for every
adjustment attached. The defensible claim today is that it **buys its
interpretability at a real, measurable cost in accuracy** — which is a more
useful statement than unqualified superiority, and still an argument for using
it wherever a valuation has to be justified to the person it affects.

### 5.7 What the approach actually produces

Section 5 of notebook 03 shows the output that justifies the method: a target
property, the real comparable sales behind its valuation, each one's case
distance and weight, and the explicit correction factor applied to its price.

A comparative sale that traded at 32.9 but is slightly less well located carries
a correction factor near 0.98, contributing an adjusted 32.1 — and contributing
heavily, because its case distance is 0.10. That is an appraisal a client can be
walked through.

It also exposes the method's weakness. Equation (35) is a weighted **mean**, so a
single anomalous trade moves the estimate. The dataset's one 117.5 outlier
appears among the nearest comparatives for several targets. A weighted median,
or trimming, would be an obvious robustness improvement the paper does not
consider.

## 6. Verdict

| Claim | Verdict |
|---|---|
| The released file is the Sindian circle of Table 1 | **Confirmed** to the second decimal |
| Factor ranking MRT > stores > age > date | **Confirmed** |
| Adjustment coefficient models (Table 2) | **Confirmed** for three of four factors |
| Signs of all four coefficient models | **Confirmed** |
| Table 2 transaction-date intercept | **Reporting error** — internally inconsistent as printed |
| QCA beats multivariate regression | **Confirmed** — 0.37 RMSE, p ≈ 0.0001 across 30 splits |
| QCA beats the neural network | **Not confirmed** — the network is marginally ahead across 30 splits; no test supports the paper's ordering |
| Effect radius best around 1.5–3 | **Not confirmed** — optimum near 0.4, below the searched range |
| Sequence of decomposition may matter *(future work #1)* | **Confirmed and quantified** — 19% spread, but the paper's order is near-optimal |
| *(not claimed by the paper)* QCA against modern ensembles | A random forest beats it by 0.61 RMSE — a wider margin than it beats regression |

**The method reproduces**, closely enough on Table 2 to be confident the
implementation is faithful. Its advantage over ordinary regression is real and
statistically supported. Its advantage over a neural network is not: that
conclusion rests on a single split of 138 cases and reverses under repetition.
Against modern ensembles it loses on accuracy outright. Two tuning parameters
were selected on the test set, and one published coefficient is internally
inconsistent.

None of this undermines the paper's actual contribution. Accuracy was never its
main argument — Table 8 of the paper makes the case on *interpretability*, and
that case is untouched by anything found here. The Quantitative Comparative
Approach computes, from market data, the correction coefficients that appraisers
currently set by judgement, and it delivers them attached to the real sales they
apply to. That is a genuine methodological contribution, and it is one that a
more accurate black-box model would not provide.

## 7. Reproducing this report

```bash
pip install -r requirements.txt
python scripts/run_all.py        # every table and figure, ~15 min
python -m pytest tests/ -q       # 27 tests
jupyter lab notebooks/           # the narrative, in order
```

Every figure quoted above is in `results/figures/`, every table in
`results/tables/`, each alongside a `*_paper_reference.csv` holding the
published values for direct comparison.
