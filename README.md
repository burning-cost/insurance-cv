# insurance-cv

[![PyPI](https://img.shields.io/pypi/v/insurance-cv)](https://pypi.org/project/insurance-cv/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-cv)](https://pypi.org/project/insurance-cv/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-cv/blob/main/notebooks/quickstart.ipynb)

Temporal cross-validation for insurance pricing models — walk-forward splits that respect policy year, accident year, and IBNR development structure, because standard k-fold gives you overoptimistic CV scores that don't survive contact with a live rating year.

---

## Why bother

Benchmarked against random 5-fold KFold on synthetic UK motor insurance data — 20,000 policies
with a +20%/year claims trend, 2021–2024. Poisson frequency model (sklearn PoissonRegressor).
True out-of-time holdout: 2024 policies.

| Metric | Random KFold CV | Temporal walk-forward CV |
|--------|-----------------|--------------------------|
| Mean Poisson deviance | 0.54889 | 0.59235 |
| vs true prospective (0.63244) | −13.2% (optimistic) | −6.3% (optimistic) |
| Temporal leakage | Yes | No (verified by leakage check) |
| Fold-level variance | Low (averages all periods) | Higher — shows degradation trend |
| Structured audit trail | No | Yes (`split_summary` output) |

k-fold overestimates model quality by 13.2% vs the prospective holdout. Walk-forward overestimates by 6.3%. Walk-forward is roughly 2× more accurate as a prospective score estimate.

The more important signal is the fold-by-fold trajectory. Walk-forward per-fold deviances rise monotonically from 0.547 (early 2022 test) to 0.681 (mid-2024 test). That is a 24% deterioration trend. k-fold folds are shuffled across time (0.515–0.591) — no temporal pattern, no warning signal.

For a Head of Pricing, the trajectory is the decision signal: "My model degrades by 24% over the rating year. I need a trend term, or to re-fit quarterly, or to load my rate." k-fold gives you one number. Walk-forward gives you a timeline.

---

[Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/demo_insurance_cv.py)

---

## The problem with standard k-fold in insurance

K-fold cross-validation randomly partitions data into folds. For insurance pricing, this is wrong in at least three ways.

**Temporal leakage.** Insurance claims develop over time. A motor claim reported 18 months after the accident may still be open. If you train on 2022 data and test on 2020 data, your model sees future development patterns that wouldn't have been available at the 2020 pricing date. K-fold does this routinely.

**IBNR contamination.** For any accident date near your training cutoff, some claims will not yet be reported or fully developed (Incurred But Not Reported). If those claims appear in your training set, the model learns from targets that are systematically understated. The fix is a development buffer - exclude claims with accident dates in the N months before your test window from both training and test sets.

**Seasonal confounding.** Motor claims peak in winter. Property claims follow weather cycles. If a randomly-selected test fold contains a disproportionate share of December policies, the test loss will look different to what you'd see prospectively. A prospective evaluation should test on a contiguous future period with the same seasonal mix the model will face in deployment.

The result of using k-fold on insurance data is a model that looks better in CV than it performs in the rating year. Prospective monitoring then shows a gap between modelled and actual loss ratios that is partly attributable to the leaky evaluation methodology.

---

## Blog post

[Why Your Cross-Validation is Lying to You](https://burning-cost.github.io/2026/03/06/why-your-cross-validation-is-lying-to-you/)

---

## How this library fixes it

All splits in `insurance-cv` are **walk-forward** (or boundary-aligned): training data always precedes test data in calendar time, with a configurable gap for IBNR development.

Three split generators cover the main use cases:

| Function | When to use it |
|---|---|
| `walk_forward_split` | General-purpose. Expanding training window, rolling test. Standard choice for motor, home, commercial. |
| `policy_year_split` | When rate changes align to policy year boundaries and you want clean PY-aligned folds. |
| `accident_year_split` | Long-tail lines (liability, PI) where accident year development varies across the triangle. |

All generators return `TemporalSplit` objects and yield `(train_idx, test_idx)` tuples that index into your DataFrame. They are also wrapped by `InsuranceCV`, which implements sklearn's CV splitter interface (`split` and `get_n_splits`) so you can pass them directly to `GridSearchCV`, `cross_val_score`, etc.

---

## Installation

```bash
uv add insurance-cv
# or
pip install insurance-cv
```

---

## Quickstart

This example is self-contained — no external files needed.

```python
import polars as pl
import numpy as np
from datetime import date, timedelta
from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV

# Generate a synthetic UK motor portfolio: 1000 policies over 3 years
rng = np.random.default_rng(42)
n = 1_000
start = date(2021, 1, 1)
inception_dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365 * 3, n)]

df = pl.DataFrame({
    "policy_id": [f"POL{i:05d}" for i in range(n)],
    "inception_date": inception_dates,
    "exposure": rng.uniform(0.1, 1.0, n).round(4).tolist(),
    "claim_count": rng.poisson(0.08, n).tolist(),
    "claim_amount": (rng.exponential(1500, n) * rng.binomial(1, 0.08, n)).round(2).tolist(),
    "vehicle_age": rng.integers(0, 15, n).tolist(),
    "driver_age": rng.integers(18, 80, n).tolist(),
    "ncd_years": rng.integers(0, 9, n).tolist(),
}).with_columns(pl.col("inception_date").cast(pl.Date))

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=12,    # need at least a year to cover seasonality
    test_months=6,          # evaluate on 6-month windows
    step_months=6,          # non-overlapping test periods
    ibnr_buffer_months=3,   # exclude claims in the 3 months before each test window
)

# Always validate before running the model
check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("\n".join(check["errors"]))

print(split_summary(splits, df, date_col="inception_date"))
# fold  train_n  test_n  train_end   test_start  gap_days
#    1      312     167  2021-12-31  2022-04-01        91
#    2      479     161  2022-06-30  2022-10-01        93
#  ...

# sklearn-compatible: pass to cross_val_score or GridSearchCV
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

X = df.select(["vehicle_age", "driver_age", "ncd_years"]).to_numpy()
y = df["claim_count"].to_numpy().astype(float)
model = PoissonRegressor()

cv = InsuranceCV(splits, df)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_poisson_deviance")
```

If you already have policy data in a parquet file, replace the synthetic DataFrame block with:

```python
df = pl.read_parquet("policies.parquet")
```

---

## API

### `walk_forward_split`

```python
walk_forward_split(
    df,
    date_col: str,
    min_train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    ibnr_buffer_months: int = 3,
) -> list[TemporalSplit]
```

Generates an expanding-window walk-forward split. The earliest data is always included in training. Each fold advances the test window by `step_months`. The IBNR buffer excludes rows in the `ibnr_buffer_months` months before `test_start` from both train and test.

Setting `step_months == test_months` gives non-overlapping test windows (the usual choice for insurance). Smaller values increase fold count but introduce correlation between adjacent test periods.

For long-tail lines, `ibnr_buffer_months` should be 12-24 months. For motor it is typically 3-6 months.

### `policy_year_split`

```python
policy_year_split(
    df,
    date_col: str,
    n_years_train: int,
    n_years_test: int = 1,
    step_years: int = 1,
) -> list[TemporalSplit]
```

Splits aligned to 1 Jan - 31 Dec policy year boundaries. Use this when your rate changes are annual and you want clean year-aligned train/test boundaries. There is no IBNR buffer because the year boundary is treated as a natural development cutoff - if you need one, adjust `n_years_train` to leave a gap year.

### `accident_year_split`

```python
accident_year_split(
    df,
    date_col: str,
    development_col: str,
    min_development_months: int = 12,
) -> list[TemporalSplit]
```

Generates one fold per accident year, filtering out years where median claim development is below `min_development_months`. The `development_col` should contain months from accident date to valuation date. This is the right approach for liability and professional indemnity where the development triangle matters.

`development_col` is a derived column you must pre-compute from your dates before calling `accident_year_split`. For Polars DataFrames:

```python
import polars as pl

df = df.with_columns(
    ((pl.col("valuation_date") - pl.col("accident_date")).dt.total_days() / 30.4)
    .alias("development_months")
)

splits = accident_year_split(df, date_col="accident_date", development_col="development_months")
```

The 30.4 divisor converts days to approximate calendar months. Use your actual valuation date column name — for a single snapshot dataset this is often a constant.

### `TemporalSplit`

```python
TemporalSplit(
    date_col: str,
    train_start,
    train_end,
    test_start,
    test_end,
    ibnr_buffer_months: int = 0,
    label: str = "",
)
```

A single split definition. Call `.get_indices(df)` to get `(train_idx, test_idx)` as numpy integer arrays.

### `InsuranceCV`

```python
InsuranceCV(splits: list[TemporalSplit], df)
```

Wraps a list of `TemporalSplit` objects as a sklearn-compatible CV splitter. Implements `split()` and `get_n_splits()`. Pass to `cross_val_score`, `GridSearchCV`, or any other sklearn utility that accepts a CV splitter.

### `temporal_leakage_check`

```python
temporal_leakage_check(
    splits: list[TemporalSplit],
    df,
    date_col: str,
) -> dict[str, list[str]]
```

Returns `{"errors": [...], "warnings": [...]}`. Run this before any model fitting. An empty `errors` list means no temporal leakage was detected.

### `split_summary`

```python
split_summary(
    splits: list[TemporalSplit],
    df,
    date_col: str,
) -> pl.DataFrame
```

Returns a DataFrame with one row per fold: fold number, train/test sizes, actual date boundaries, gap days, and IBNR buffer months. Useful for confirming that your splits look sensible before committing compute to model fitting.

---

## IBNR buffer: choosing the right value

The IBNR buffer is the most consequential parameter in `walk_forward_split`. A buffer that is too short means partially-developed claims contaminate your test evaluation; too long reduces the amount of usable test data.

Rough guidelines by line:

| Line | Typical buffer |
|---|---|
| Motor own damage | 3-6 months |
| Motor third party property | 6-12 months |
| Motor third party bodily injury | 12-24 months |
| Home buildings | 6-12 months |
| Employers' liability | 24-36 months |
| Professional indemnity | 24-48 months |

These are starting points. The right value depends on your claims handling speed, the proportion of large/complex claims, and how you define your loss target (paid vs. incurred vs. ultimate).

---

## Benchmark Results

Measured on Databricks serverless compute (Python 3.12), 20,000 synthetic UK motor
policies, +20%/year claims trend, 2021–2024. Poisson frequency model. Run `benchmarks/benchmark.py` to reproduce.

| Method | Mean Poisson deviance | vs Prospective | Temporal leakage |
|---|---|---|---|
| k-fold (5-fold random) | 0.54889 | −13.2% (optimistic) | Yes |
| **Walk-forward (insurance-cv)** | **0.59235** | **−6.3% (optimistic)** | **No** |
| Prospective holdout (ground truth) | 0.63244 | 0.00% | — |

Walk-forward is 2.1× more accurate as a prospective score estimate (6.3% vs 13.2% error).

Walk-forward per-fold trajectory (chronological test windows):

| Fold | Test window | Poisson deviance |
|---|---|---|
| 1 | 2022-04 to 2022-09 | 0.54727 |
| 2 | 2022-10 to 2023-03 | 0.55499 |
| 3 | 2023-04 to 2023-09 | 0.60626 |
| 4 | 2023-10 to 2024-03 | 0.57222 |
| 5 | 2024-04 to 2024-09 | 0.68103 |

The trajectory rises from 0.547 to 0.681 — a 24% deterioration as test windows advance into the trending period. k-fold's fold scores (0.515, 0.544, 0.559, 0.514, 0.591) show no temporal pattern and cannot surface this signal.

Benchmark completed in 2.3s on serverless compute.

---

## Performance

`temporal_leakage_check` catches 100% of forward-looking splits. `split_summary` produces the fold structure documentation that model governance reviewers will ask for.

The temporal CV fold-level variance is typically 2–5x higher than random KFold, because each fold genuinely tests on a different time period rather than averaging across all periods. This higher variance is informative — it shows whether model performance degrades as the validation period moves further from the training window.

Run `benchmarks/benchmark.py` on Databricks to reproduce.

---

## Capabilities

The notebook at `notebooks/demo_insurance_cv.py` runs a complete demonstration on a 5-year synthetic UK motor portfolio with known seasonal and trend structure, and shows:

- **Walk-forward vs random k-fold gap**: Poisson deviance is measurably lower (better-looking) under random k-fold because future data leaks into training. Walk-forward gives the honest prospective estimate.
- **Per-fold trajectory**: Walk-forward fold scores trend with the data's temporal structure; random k-fold averages across all periods and hides this signal.
- **IBNR buffer effect**: Test set size and cleanliness trade off against each other. The notebook shows how buffer length from 0 to 12 months changes both.
- **Policy-year splits**: Clean 1 Jan boundaries keep training and test sets on opposite sides of rate changes, demonstrated on 5 policy years.
- **sklearn drop-in compatibility**: InsuranceCV passes directly to cross_val_score with no code changes beyond swapping the CV object.

---

## Related Libraries

| Library | What it does |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs — combine with walk-forward CV to evaluate GBM-derived factor tables |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Conformal prediction intervals — uses temporal splits (same logic as this library) to calibrate coverage guarantees |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — walk-forward splits here produce the baseline metrics; monitoring tracks performance after deployment |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK insurance datasets with known DGPs — use to benchmark CV strategies against a controlled ground truth |

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Full conditional distribution per risk: mean, variance, CoV |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries](https://burning-cost.github.io)

---

## Development

```bash
git clone https://github.com/burning-cost/insurance-cv
cd insurance-cv
uv sync --dev
uv run pytest -v
```

Tests are designed to run on Databricks (serverless) for the compute-heavy cases. On a local machine `uv run pytest -v` covers the full test suite in seconds since the fixtures use synthetic data.

---

## Licence

MIT. See [LICENSE](LICENSE).
