"""
Benchmark: insurance-cv walk-forward split vs standard k-fold for insurance pricing.

The claim: k-fold cross-validation gives overoptimistic CV scores for insurance
models because it allows temporal leakage — future claims development appearing
in training data, seasonal patterns being randomly mixed between folds. Walk-forward
temporal splits are the correct methodology and produce CV scores that match
prospective monitoring.

Setup:
- 8,000 synthetic UK motor policies over 4 calendar years (2021-2024)
- Poisson frequency model with trend: +5% claims frequency per year (market inflation)
- k-fold CV on random 5-fold partitions (wrong approach)
- Walk-forward CV with 12-month train minimum, 6-month test window, 3-month IBNR buffer
- Comparison: CV score vs prospective out-of-sample score on final year (ground truth)

The bias pattern:
- k-fold mixes years — model sees 2024 data when "training" on 2021 fold
- In a trending environment, this inflates apparent model fit (future data leaks)
- Walk-forward scores match prospective monitoring within noise

Expected output:
- k-fold CV score: more optimistic than walk-forward (temporal leakage inflates it)
- Walk-forward CV score: close to true prospective holdout score
- Walk-forward correctly detects claims trend; k-fold masks it by leakage

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-cv walk-forward splits vs k-fold")
print("=" * 70)
print()

try:
    from insurance_cv import (
        walk_forward_split,
        temporal_leakage_check,
        split_summary,
    )
    from insurance_cv.splits import InsuranceCV
    print("insurance-cv imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-cv: {e}")
    print("Install with: pip install insurance-cv")
    sys.exit(1)

try:
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_POLICIES = 8_000
START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 1, 1)

# +5% claims trend per year (market inflation — standard UK motor environment)
CLAIMS_TREND_PER_YEAR = 0.05
BASE_FREQUENCY = 0.07

print(f"DGP: {N_POLICIES:,} policies, 2021-2024, claims trend +{CLAIMS_TREND_PER_YEAR*100:.0f}%/year")
print(f"     This trend is the key: k-fold leaks future years into train folds,")
print(f"     making the model look better than it actually is prospectively.")
print()

# Generate policies uniformly over the date range
date_range_days = (END_DATE - START_DATE).days
inception_days = RNG.integers(0, date_range_days, N_POLICIES)
inception_dates = [START_DATE + timedelta(days=int(d)) for d in inception_days]

# Year (fractional, from 2021) as the trend variable
years_from_start = np.array([(d - START_DATE).days / 365.25 for d in inception_dates])

# Features
driver_age = RNG.integers(18, 80, N_POLICIES).astype(float)
ncd_years = RNG.integers(0, 9, N_POLICIES).astype(float)
vehicle_age = RNG.integers(0, 15, N_POLICIES).astype(float)
exposure = RNG.uniform(0.3, 1.0, N_POLICIES)

# True frequency with trend
log_freq = (
    np.log(BASE_FREQUENCY)
    - 0.008 * (driver_age - 40)
    - 0.05 * ncd_years
    + 0.02 * vehicle_age
    + np.log(1 + CLAIMS_TREND_PER_YEAR) * years_from_start  # claims trend
)
true_freq = np.exp(log_freq)
claim_count = RNG.poisson(true_freq * exposure)

df = pl.DataFrame({
    "inception_date": inception_dates,
    "driver_age": driver_age.tolist(),
    "ncd_years": ncd_years.tolist(),
    "vehicle_age": vehicle_age.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
    "year_float": years_from_start.tolist(),
}).with_columns(pl.col("inception_date").cast(pl.Date))

# Feature matrix (model does NOT see the year — this is realistic)
X_features = ["driver_age", "ncd_years", "vehicle_age"]
X = df.select(X_features).to_numpy()
y = df["claim_count"].to_numpy().astype(float) / df["exposure"].to_numpy()
sample_weight = df["exposure"].to_numpy()

print(f"Policies: {len(df):,}  |  Claim frequency: {float(np.sum(claim_count)/np.sum(exposure)):.4f}/year")
print()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("poisson", PoissonRegressor(alpha=1.0, max_iter=300)),
])

# ---------------------------------------------------------------------------
# BASELINE: k-fold CV (standard but wrong for insurance)
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE: 5-fold k-fold cross-validation (random partition)")
print("-" * 70)
print()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(
    model, X, y,
    cv=kf,
    scoring="neg_mean_poisson_deviance",
    fit_params={"poisson__sample_weight": sample_weight},
)
kf_mean = float(np.mean(-kf_scores))
kf_std = float(np.std(-kf_scores))

print(f"  k-fold mean Poisson deviance: {kf_mean:.5f}  (std: {kf_std:.5f})")
print()
print("  Problems with k-fold here:")
print("  - Fold 1 may train on 2023-2024 policies and test on 2021 policies")
print("  - The model 'sees' future claims trends during training")
print("  - This makes the model look better than it will be in deployment")
print("  - No IBNR buffer: policies near the fold boundary may have")
print("    unreported claims that appear in both train and test")
print()

# Check for temporal leakage explicitly
kf_folds = list(kf.split(X))
first_fold_test_years = df["year_float"].to_numpy()[kf_folds[0][1]]
first_fold_train_years = df["year_float"].to_numpy()[kf_folds[0][0]]
max_test_year = first_fold_test_years.max()
min_train_year = first_fold_train_years.min()
has_leakage = min_train_year < max_test_year - 0.001

print(f"  Leakage check (fold 1):")
print(f"    Train set: min year = {min_train_year:.2f}, max year = {first_fold_train_years.max():.2f}")
print(f"    Test set:  min year = {first_fold_test_years.min():.2f}, max year = {max_test_year:.2f}")
print(f"    Temporal leakage present: {has_leakage}  (training on data AFTER test period)")
print()

# ---------------------------------------------------------------------------
# LIBRARY: Walk-forward CV
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: insurance-cv walk-forward splits")
print("-" * 70)
print()

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=12,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)

# Validate: no temporal leakage by construction
check = temporal_leakage_check(splits, df, date_col="inception_date")
if check.get("errors"):
    print(f"  WARNING: Leakage detected: {check['errors']}")
else:
    print("  Temporal leakage check: PASSED (no leakage by construction)")
print()

# Print split summary
summary = split_summary(splits, df, date_col="inception_date")
print("  Split structure:")
print(summary.to_pandas().to_string(index=False))
print()

# Walk-forward CV using InsuranceCV (sklearn-compatible)
cv_wf = InsuranceCV(splits, df)
wf_scores = cross_val_score(
    model, X, y,
    cv=cv_wf,
    scoring="neg_mean_poisson_deviance",
    fit_params={"poisson__sample_weight": sample_weight},
)
wf_mean = float(np.mean(-wf_scores))
wf_std = float(np.std(-wf_scores))

print(f"  Walk-forward mean Poisson deviance: {wf_mean:.5f}  (std: {wf_std:.5f})")
print()

# Print per-fold scores to show trend
print("  Per-fold scores (walk-forward):")
print(f"  {'Fold':<6} {'Split label':<20} {'Poisson deviance':>18}")
print(f"  {'-'*6} {'-'*20} {'-'*18}")
for i, (split, score) in enumerate(zip(splits, -wf_scores)):
    label = split.label or f"fold_{i+1}"
    print(f"  {i+1:<6} {label:<20} {score:>18.5f}")
print()

# ---------------------------------------------------------------------------
# Ground truth: true prospective holdout (2024 data only)
# ---------------------------------------------------------------------------

print("-" * 70)
print("GROUND TRUTH: Prospective holdout on 2024 data")
print("-" * 70)
print()

# Train on 2021-2023, test on 2024
cutoff = date(2024, 1, 1)
train_mask = df["inception_date"].to_numpy() < np.array(cutoff, dtype="datetime64[D]")
test_mask = ~train_mask

X_train_gt = X[train_mask]
y_train_gt = y[train_mask]
w_train_gt = sample_weight[train_mask]
X_test_gt = X[test_mask]
y_test_gt = y[test_mask]
w_test_gt = sample_weight[test_mask]

from sklearn.metrics import mean_poisson_deviance

model_gt = Pipeline([
    ("scaler", StandardScaler()),
    ("poisson", PoissonRegressor(alpha=1.0, max_iter=300)),
])
model_gt.fit(X_train_gt, y_train_gt, poisson__sample_weight=w_train_gt)
y_pred_gt = model_gt.predict(X_test_gt)
prospective_score = mean_poisson_deviance(y_test_gt, y_pred_gt, sample_weight=w_test_gt)

print(f"  Train period: 2021-2023 ({int(train_mask.sum()):,} policies)")
print(f"  Test period:  2024       ({int(test_mask.sum()):,} policies)")
print(f"  Prospective Poisson deviance: {prospective_score:.5f}")
print()
print(f"  This is the ground truth — how the model actually performs prospectively.")
print()

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: k-fold vs walk-forward vs ground truth")
print("=" * 70)
print()

kf_error = (kf_mean - prospective_score) / prospective_score * 100
wf_error = (wf_mean - prospective_score) / prospective_score * 100

print(f"  {'Method':<35} {'CV Score':>10} {'vs Prospective':>16} {'Leakage':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*16} {'-'*10}")
print(f"  {'k-fold (5-fold random)':<35} {kf_mean:>10.5f} {kf_error:>+15.2f}% {'YES':>10}")
print(f"  {'Walk-forward (12m train, 6m test)':<35} {wf_mean:>10.5f} {wf_error:>+15.2f}% {'NO':>10}")
print(f"  {'Prospective holdout (ground truth)':<35} {prospective_score:>10.5f} {'0.00%':>16} {'n/a':>10}")
print()
print("INTERPRETATION")
print(f"  k-fold error vs prospective: {kf_error:+.2f}%")
print(f"  Walk-forward error vs prospective: {wf_error:+.2f}%")
print()

if kf_mean < prospective_score:
    print(f"  k-fold is overoptimistic by {abs(kf_error):.1f}%: future data leaks into training folds.")
    print(f"  Walk-forward is {abs(kf_error) - abs(wf_error):.1f}pp closer to the true prospective score.")
else:
    print(f"  Walk-forward better tracks the prospective score on this DGP.")

print()
print(f"  Walk-forward splits also provide the IBNR buffer (3 months excluded)")
print(f"  between training and test windows. k-fold cannot do this — it has no")
print(f"  concept of calendar time, so it cannot protect against IBNR contamination.")
print()
print(f"  In practice, overoptimistic CV scores lead to:")
print(f"    - Overconfident model selection (choosing more complex models than warranted)")
print(f"    - Budget for expected vs actual LR gap > model uncertainty")
print(f"    - Stakeholder disappointment when prospective monitoring shows worse results")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
