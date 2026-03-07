"""
Walk-forward cross-validation for a UK motor pricing model.

This example shows how to use insurance-cv with a CatBoost frequency model
on synthetic motor data. The key point is that the CV results here actually
reflect prospective performance - the test sets are always later than the
training data, and a 3-month IBNR buffer prevents partially-developed claims
from polluting the test evaluation.

Run this on Databricks or any environment with CatBoost installed:
    uv add insurance-cv catboost polars
"""

import numpy as np
import polars as pl
from datetime import date, timedelta

from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import split_summary, temporal_leakage_check
from insurance_cv.splits import InsuranceCV

# ---------------------------------------------------------------------------
# Synthetic motor dataset
# ---------------------------------------------------------------------------
# In a real project, this would be a policy-level dataset with:
#   - Claim counts (Poisson target for frequency model)
#   - Earned vehicle years (exposure weight)
#   - Rating factors: vehicle age, driver age, NCD, area, etc.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 10_000

# Generate date range as a list of Python date objects
start = date(2018, 1, 1)
end = date(2023, 12, 31)
total_days = (end - start).days + 1
day_offsets = rng.integers(0, total_days, size=n)
inception_dates = [start + timedelta(days=int(d)) for d in day_offsets]

df = (
    pl.DataFrame(
        {
            "inception_date": inception_dates,
            "vehicle_age": rng.integers(0, 15, n).tolist(),
            "driver_age": rng.integers(17, 80, n).tolist(),
            "ncd_years": rng.integers(0, 10, n).tolist(),
            "area_code": rng.choice(["A", "B", "C", "D"], n).tolist(),
            "earned_vehicle_years": rng.uniform(0.5, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.08, n).tolist(),
        }
    )
    .with_columns(pl.col("inception_date").cast(pl.Date))
    .sort("inception_date")
)

min_date = df["inception_date"].min()
max_date = df["inception_date"].max()
total_freq = df["claim_count"].sum() / df["earned_vehicle_years"].sum()
print(f"Dataset: {len(df):,} policies from {min_date} to {max_date}")
print(f"Overall frequency: {total_freq:.4f}")

# ---------------------------------------------------------------------------
# Define splits
# ---------------------------------------------------------------------------

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,   # Need at least 1.5 years to capture seasonality
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,  # 3-month buffer - standard for motor
)

print(f"\nGenerated {len(splits)} walk-forward folds\n")

# ---------------------------------------------------------------------------
# Validate splits before running the model
# ---------------------------------------------------------------------------

check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError(f"Temporal leakage detected:\n" + "\n".join(check["errors"]))
if check["warnings"]:
    for w in check["warnings"]:
        print(f"WARNING: {w}")

summary = split_summary(splits, df, date_col="inception_date")
print("Split summary:")
print(summary.select(["fold", "train_n", "test_n", "train_end", "test_start", "gap_days"]))
print()

# ---------------------------------------------------------------------------
# Fit a CatBoost frequency model per fold (sklearn CV interface)
# ---------------------------------------------------------------------------
# InsuranceCV plugs directly into sklearn's cross_val_score. CatBoost's
# Poisson loss function is the right choice for claim frequency modelling.
# ---------------------------------------------------------------------------

try:
    import catboost
    from sklearn.model_selection import cross_val_score

    features = ["vehicle_age", "driver_age", "ncd_years"]
    X = df.select(features).to_numpy()
    y = df["claim_count"].to_numpy()

    cv = InsuranceCV(splits, df)

    model = catboost.CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )

    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_mean_poisson_deviance",
    )

    print(f"CatBoost Poisson deviance across {len(scores)} folds: {-scores.mean():.4f} (+/- {scores.std():.4f})")

except ImportError:
    print("CatBoost not installed - skipping model fit. Install with: uv add catboost")

# ---------------------------------------------------------------------------
# Manual fold iteration (if you need exposure-weighted evaluation)
# ---------------------------------------------------------------------------
# cross_val_score doesn't support sample weights in the scoring step. For
# exposure-weighted deviance you need to iterate folds manually.
# ---------------------------------------------------------------------------

fold_results = []
for i, split in enumerate(splits):
    train_idx, test_idx = split.get_indices(df)
    train = df[train_idx]
    test = df[test_idx]

    # Baseline: predict overall train frequency for every test row
    train_freq = train["claim_count"].sum() / train["earned_vehicle_years"].sum()
    predicted = test["earned_vehicle_years"] * train_freq
    actual = test["claim_count"]

    mae = float((actual - predicted).abs().mean())
    fold_results.append(
        {
            "fold": i + 1,
            "train_size": len(train),
            "test_size": len(test),
            "train_frequency": round(train_freq, 5),
            "baseline_mae": round(mae, 5),
        }
    )

results_df = pl.DataFrame(fold_results)
print("Baseline (constant frequency) results per fold:")
print(results_df)
