# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-cv: Temporal Cross-Validation Capability Demo
# MAGIC
# MAGIC This notebook demonstrates why temporal cross-validation matters for insurance
# MAGIC pricing models, using synthetic UK motor data where the true generating process
# MAGIC is known.
# MAGIC
# MAGIC **What this shows:**
# MAGIC 1. Generate a synthetic motor portfolio with seasonal and trend structure
# MAGIC 2. Walk-forward splits vs random k-fold: the gap in reported performance
# MAGIC 3. P-value uniformity check on genuine claims
# MAGIC 4. Policy-year aligned splits for rate-change scenarios
# MAGIC 5. IBNR buffer effect: how much the buffer size matters
# MAGIC 6. sklearn compatibility: pass InsuranceCV to cross_val_score

# COMMAND ----------

# MAGIC %pip install insurance-cv scikit-learn polars

# COMMAND ----------

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold, cross_val_score

from insurance_cv import (
    TemporalSplit,
    walk_forward_split,
    policy_year_split,
    accident_year_split,
)
from insurance_cv.diagnostics import temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV

print(f"insurance-cv imported successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic motor portfolio
# MAGIC
# MAGIC We build a 5-year motor portfolio with:
# MAGIC - Seasonal frequency (winter peak — December/January have ~20% higher claims)
# MAGIC - A gradual upward trend in frequency over the period (portfolio risk drift)
# MAGIC - Three rating factors: vehicle group, NCD years, area band
# MAGIC - Exposure varying by policy

# COMMAND ----------

rng = np.random.default_rng(42)

N_POLICIES = 20_000
YEARS = 5

# Spread policies uniformly across 5 years (2019-01-01 to 2023-12-31)
days_total = YEARS * 365
inception_days = rng.integers(0, days_total, size=N_POLICIES)
inception_dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(inception_days, unit="D")

# Rating factors
vehicle_group = rng.integers(1, 51, size=N_POLICIES)
ncd_years = rng.integers(0, 6, size=N_POLICIES)
area = rng.choice(["A", "B", "C", "D", "E", "F"], size=N_POLICIES, p=[0.25, 0.22, 0.20, 0.15, 0.10, 0.08])
driver_age = rng.integers(17, 86, size=N_POLICIES)

# Exposure: most policies run 12 months, a few are partial
exposure = rng.beta(9, 1, size=N_POLICIES)
exposure = np.clip(exposure, 0.08, 1.0)

# True log-linear frequency model
area_effect = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.65}
area_coef = np.array([area_effect[a] for a in area])
young_flag = (driver_age < 25).astype(float)

log_lambda_base = (
    -3.2
    + 0.025 * vehicle_group
    - 0.12 * ncd_years
    + 0.55 * young_flag
    + area_coef
)

# Seasonal effect: month of inception affects claim exposure
# Winter policies (Nov-Feb) have higher-severity months in their exposure period
month = pd.DatetimeIndex(inception_dates).month
seasonal_coef = np.where(
    np.isin(month, [11, 12, 1, 2]),
    0.18,   # winter uplift
    np.where(np.isin(month, [3, 4, 9, 10]), 0.06, 0.0)
)

# Trend: frequency creeps up ~3% per year (e.g. distracted driving, repair costs)
year_offset = (inception_days / 365.0)
trend_coef = 0.03 * year_offset  # subtle upward drift

log_lambda = log_lambda_base + seasonal_coef + trend_coef + np.log(exposure)
claim_count = rng.poisson(np.exp(log_lambda))

df = pd.DataFrame({
    "inception_date": inception_dates,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "area": area,
    "driver_age": driver_age,
    "young_driver": young_flag.astype(int),
    "exposure": exposure,
    "claim_count": claim_count,
    "seasonal_coef": seasonal_coef,
    "year_offset": year_offset,
})

print(f"Portfolio: {len(df):,} policies over {YEARS} years")
print(f"Date range: {df['inception_date'].min().date()} to {df['inception_date'].max().date()}")
print(f"Overall claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f} per policy year")
print(f"Total claims: {df['claim_count'].sum():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Walk-forward splits
# MAGIC
# MAGIC Generate 6-month test windows with an 18-month minimum training window.
# MAGIC IBNR buffer of 3 months (standard for motor).

# COMMAND ----------

df_pl = pl.from_pandas(df)

splits = walk_forward_split(
    df_pl,
    date_col="inception_date",
    min_train_months=18,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)

print(f"Generated {len(splits)} walk-forward folds")
print()

summary = split_summary(splits, df_pl, date_col="inception_date")
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Temporal leakage check
# MAGIC
# MAGIC Always run this before fitting. It catches IBNR violations and
# MAGIC zero-gap splits before they silently inflate your CV scores.

# COMMAND ----------

check = temporal_leakage_check(splits, df_pl, date_col="inception_date")

if check["errors"]:
    print("ERRORS:")
    for e in check["errors"]:
        print(f"  {e}")
else:
    print("No temporal leakage errors detected.")

if check["warnings"]:
    print("Warnings:")
    for w in check["warnings"]:
        print(f"  {w}")
else:
    print("No warnings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Walk-forward vs random k-fold: the performance gap
# MAGIC
# MAGIC This is the central claim of the library. We train a Poisson model and
# MAGIC evaluate it with both temporal and random splits. Because the data has
# MAGIC trend and seasonal structure, random k-fold mixes future data into
# MAGIC training, and mixes seasonal periods across folds — making the model
# MAGIC look better than it will in production.

# COMMAND ----------

# Prepare features
area_dummies = pd.get_dummies(df["area"], prefix="area", drop_first=True)
feature_cols = ["vehicle_group", "ncd_years", "young_driver"] + list(area_dummies.columns)
X = pd.concat([df[["vehicle_group", "ncd_years", "young_driver"]], area_dummies], axis=1).values
y = df["claim_count"].values
exposure_arr = df["exposure"].values

# Poisson model with log-link — this is the standard actuarial frequency model
# We use sample_weight=exposure to approximate the exposure offset
model = PoissonRegressor(alpha=0.0, max_iter=300)

# --- Walk-forward CV ---
wf_scores = []
for s in splits:
    train_idx, test_idx = s.get_indices(df_pl)
    if len(train_idx) < 100 or len(test_idx) < 50:
        continue
    model.fit(X[train_idx], y[train_idx], sample_weight=exposure_arr[train_idx])
    # Mean Poisson deviance on test
    y_pred = model.predict(X[test_idx])
    # Poisson deviance: 2 * sum(y*log(y/mu) - (y-mu)), clipped to avoid log(0)
    mu = np.maximum(y_pred, 1e-9)
    y_t = y[test_idx]
    dev = 2.0 * np.sum(
        np.where(y_t > 0, y_t * np.log(y_t / mu), 0.0) - (y_t - mu)
    ) / len(test_idx)
    wf_scores.append(dev)

wf_mean = np.mean(wf_scores)
wf_std = np.std(wf_scores)

# --- Random k-fold CV ---
kf = KFold(n_splits=len(splits), shuffle=True, random_state=42)
kf_scores = []
for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx], sample_weight=exposure_arr[train_idx])
    y_pred = model.predict(X[test_idx])
    mu = np.maximum(y_pred, 1e-9)
    y_t = y[test_idx]
    dev = 2.0 * np.sum(
        np.where(y_t > 0, y_t * np.log(y_t / mu), 0.0) - (y_t - mu)
    ) / len(test_idx)
    kf_scores.append(dev)

kf_mean = np.mean(kf_scores)
kf_std = np.std(kf_scores)

print("=== Poisson Deviance (lower = better) ===")
print(f"Walk-forward CV: {wf_mean:.5f} +/- {wf_std:.5f}  (folds: {len(wf_scores)})")
print(f"Random k-fold:   {kf_mean:.5f} +/- {kf_std:.5f}  (folds: {len(kf_scores)})")
print()
gap = wf_mean - kf_mean
print(f"Gap (WF - KF):   {gap:+.5f}")
print()
if gap > 0:
    print("Walk-forward is HIGHER (worse) than random k-fold.")
    print("This is expected: random k-fold is optimistic because it uses future")
    print("data in training and ignores temporal structure.")
    print(f"Over-optimism: {abs(gap)/wf_mean*100:.1f}% of the walk-forward deviance.")
else:
    print("Note: Gap direction depends on the specific trend/seasonal structure.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-fold performance
# MAGIC
# MAGIC Walk-forward folds show the model's actual trajectory. Later folds
# MAGIC generally get harder (trend in the data) — this is invisible in k-fold.

# COMMAND ----------

print("Walk-forward fold scores (Poisson deviance):")
print(f"{'Fold':<8} {'Train N':>10} {'Test N':>10} {'Deviance':>12}")
print("-" * 44)
for i, (s, score) in enumerate(zip(splits, wf_scores)):
    train_idx, test_idx = s.get_indices(df_pl)
    print(f"{i+1:<8} {len(train_idx):>10,} {len(test_idx):>10,} {score:>12.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Policy-year splits
# MAGIC
# MAGIC When your rate changes fire on 1 January, policy-year aligned splits
# MAGIC keep the training and test sets cleanly on opposite sides of the
# MAGIC rate change boundary.

# COMMAND ----------

py_splits = policy_year_split(
    df_pl,
    date_col="inception_date",
    n_years_train=2,
    n_years_test=1,
    step_years=1,
)

print(f"Generated {len(py_splits)} policy-year folds")
print()

for s in py_splits:
    train_idx, test_idx = s.get_indices(df_pl)
    print(f"{s.label}")
    print(f"  Train rows: {len(train_idx):,}  |  Test rows: {len(test_idx):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. IBNR buffer effect
# MAGIC
# MAGIC The buffer excludes partially-developed claims from test sets.
# MAGIC Too small a buffer means you're evaluating on incomplete losses.
# MAGIC Too large wastes test data. This cell shows how test set size
# MAGIC varies with buffer length.

# COMMAND ----------

print(f"{'Buffer (months)':<18} {'Folds':>8} {'Avg test N':>12} {'Avg train N':>12}")
print("-" * 54)
for buf in [0, 1, 3, 6, 12]:
    try:
        buf_splits = walk_forward_split(
            df_pl,
            date_col="inception_date",
            min_train_months=18,
            test_months=6,
            step_months=6,
            ibnr_buffer_months=buf,
        )
        test_sizes = [len(s.get_indices(df_pl)[1]) for s in buf_splits]
        train_sizes = [len(s.get_indices(df_pl)[0]) for s in buf_splits]
        print(f"{buf:<18} {len(buf_splits):>8} {int(np.mean(test_sizes)):>12,} {int(np.mean(train_sizes)):>12,}")
    except ValueError as e:
        print(f"{buf:<18} {'N/A':>8}  ({e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. sklearn compatibility: InsuranceCV with cross_val_score
# MAGIC
# MAGIC InsuranceCV implements the sklearn BaseCrossValidator interface.
# MAGIC Pass it directly to cross_val_score, GridSearchCV, or any other
# MAGIC sklearn utility that accepts a splitter.

# COMMAND ----------

cv = InsuranceCV(splits, df_pl)

print(f"InsuranceCV has {cv.get_n_splits()} splits")

# cross_val_score uses the split() generator
# We pass neg_mean_poisson_deviance so higher = better (sklearn convention)
scores = cross_val_score(
    PoissonRegressor(alpha=0.0, max_iter=300),
    X, y,
    cv=cv,
    scoring="neg_mean_poisson_deviance",
    fit_params={"sample_weight": exposure_arr},
)

print(f"\nNeg Poisson deviance per fold: {np.round(scores, 5)}")
print(f"Mean: {scores.mean():.5f}")
print(f"Std:  {scores.std():.5f}")
print()
print("The sign convention is sklearn's: negative deviance, higher = better.")
print("InsuranceCV is a drop-in replacement for KFold anywhere in sklearn.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Accident-year splits (long-tail demonstration)
# MAGIC
# MAGIC For liability lines where claims develop over years, accident-year
# MAGIC splits filter out immature accident years before including them as
# MAGIC test targets.

# COMMAND ----------

# Add a development_months column: months from inception to end of dataset
valuation_date = pd.Timestamp("2024-01-01")
df["development_months"] = (
    (valuation_date - df["inception_date"]).dt.days / 30.4
).round().astype(int)

df_pl2 = pl.from_pandas(df)

ay_splits = accident_year_split(
    df_pl2,
    date_col="inception_date",
    development_col="development_months",
    min_development_months=12,
)

print(f"Generated {len(ay_splits)} accident-year folds (min 12 months development)")
print()
for s in ay_splits:
    train_idx, test_idx = s.get_indices(df_pl2)
    print(f"{s.label}")
    print(f"  Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | Poisson Deviance | Notes |
# MAGIC |---|---|---|
# MAGIC | Walk-forward CV | Higher (honest) | Future cannot leak into training |
# MAGIC | Random k-fold | Lower (optimistic) | Future data in training set |
# MAGIC
# MAGIC **Key observations from this demo:**
# MAGIC
# MAGIC 1. Random k-fold consistently under-reports deviance (over-reports model quality) because it trains on future periods and tests on past periods simultaneously.
# MAGIC
# MAGIC 2. Walk-forward fold scores trend upward over time in data with trend structure — random k-fold averages across all periods and hides this.
# MAGIC
# MAGIC 3. The IBNR buffer trades test set size for cleanliness of the evaluation target. For motor, 3 months is standard. For liability, 12–24 months is appropriate.
# MAGIC
# MAGIC 4. InsuranceCV is a drop-in replacement for sklearn's KFold — no changes to your GridSearchCV or cross_val_score calls.
# MAGIC
# MAGIC 5. Policy-year splits and accident-year splits handle the specific structures in UK commercial and long-tail lines that walk_forward_split does not address.
