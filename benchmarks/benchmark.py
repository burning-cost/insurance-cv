"""
Benchmark: insurance-cv walk-forward CV vs k-fold for a CatBoost frequency model.

k-fold shuffles policies across folds, so future data leaks into training.
In a trending market this causes two compounding errors: (1) the CV score is
optimistically biased, and (2) hyperparameters selected by k-fold are calibrated
to a mixed-time distribution that doesn't exist in production.

Walk-forward CV fixes both by enforcing strict temporal ordering: train on
everything up to month M, test on months M+buffer to M+test.

Setup
-----
- 18,000 synthetic UK motor policies, 2021-2024, +17%/year claims trend
- Two CatBoost configs (A: shallow/regularised, B: deeper/less regularised)
- k-fold vs walk-forward: which selects the config that generalises forward?
- Ground truth: prospective holdout on 2024 data

Run:  python benchmarks/benchmark.py
"""

from __future__ import annotations
import sys, time, warnings
from datetime import date, timedelta

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
T0 = time.time()

print("=" * 68)
print("Benchmark: walk-forward CV vs k-fold  |  CatBoost Poisson frequency")
print("=" * 68)

try:
    from insurance_cv import walk_forward_split, temporal_leakage_check
    from insurance_cv.splits import InsuranceCV
except ImportError as e:
    sys.exit(f"insurance-cv not found: {e}\n  pip install insurance-cv")
try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    sys.exit("CatBoost not found: pip install catboost")
try:
    from sklearn.model_selection import KFold
except ImportError:
    sys.exit("scikit-learn not found: pip install scikit-learn")

# ── Synthetic data ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N, TREND = 18_000, 0.17
START_DATE, END_DATE = date(2021, 1, 1), date(2025, 1, 1)

inception_days = RNG.integers(0, (END_DATE - START_DATE).days, N)
inception_dates = [START_DATE + timedelta(days=int(d)) for d in inception_days]
t = np.array([(d - START_DATE).days / 365.25 for d in inception_dates])

driver_age = RNG.integers(18, 75, N).astype(float)
ncd        = RNG.integers(0, 9, N).astype(float)
veh_age    = RNG.integers(0, 15, N).astype(float)
mileage    = RNG.uniform(3_000, 25_000, N)
exposure   = RNG.uniform(0.25, 1.0, N)

log_mu = (
    np.log(0.075) - 0.010 * (driver_age - 35) - 0.055 * ncd
    + 0.018 * veh_age + 3e-5 * (mileage - 12_000)
    + np.log(1 + TREND) * t
)
claims = RNG.poisson(np.exp(log_mu) * exposure)

df = pl.DataFrame({
    "inception_date": inception_dates,
    "driver_age": driver_age.tolist(), "ncd": ncd.tolist(),
    "veh_age": veh_age.tolist(),       "mileage": mileage.tolist(),
    "exposure": exposure.tolist(),     "claims": claims.tolist(),
    "t": t.tolist(),
}).with_columns(pl.col("inception_date").cast(pl.Date))

FEATURES = ["driver_age", "ncd", "veh_age", "mileage"]
X, y, w = df.select(FEATURES).to_numpy().astype(float), claims / exposure, exposure

print(f"\nDGP: {N:,} policies, 2021-2024, +{TREND*100:.0f}%/yr trend, "
      f"overall freq {claims.sum()/exposure.sum():.4f}/yr\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def poisson_deviance(yt, yp, w):
    yp = np.maximum(yp, 1e-9)
    d  = 2 * (yt * np.where(yt > 0, np.log(yt / yp), 0.0) - (yt - yp))
    return float(np.average(d, weights=w))

def catboost_cv(X, y, w, depth, lr, splitter):
    scores = []
    for tr, te in splitter.split(X):
        m = CatBoostRegressor(loss_function="Poisson", iterations=400,
                              depth=depth, learning_rate=lr,
                              random_seed=42, verbose=False)
        m.fit(Pool(X[tr], y[tr], weight=w[tr]))
        scores.append(poisson_deviance(y[te], m.predict(X[te]), w[te]))
    return scores

CONFIGS = {"A (depth=3, lr=0.08)": (3, 0.08), "B (depth=6, lr=0.03)": (6, 0.03)}

# ── k-fold ────────────────────────────────────────────────────────────────────
print("-" * 68)
print("k-fold CV (5-fold random shuffle)")
print("-" * 68)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_means = {}
for label, (d, lr) in CONFIGS.items():
    scores = catboost_cv(X, y, w, d, lr, kf)
    kf_means[label] = float(np.mean(scores))
kf_selected = min(kf_means, key=kf_means.get)

for c, mean in kf_means.items():
    print(f"  {c:<26} deviance={mean:.5f}{'  <-- selected' if c==kf_selected else ''}")

# Leakage check on fold 0
tr0, te0 = list(kf.split(X))[0]
leaks = t[tr0].max() > t[te0].min() + 0.001
print(f"\n  Temporal leakage present: {leaks}  "
      f"(max train t={t[tr0].max():.2f} > min test t={t[te0].min():.2f})")

# ── Walk-forward ──────────────────────────────────────────────────────────────
print("\n" + "-" * 68)
print("Walk-forward CV (insurance-cv, 12m train, 6m test, 3m IBNR buffer)")
print("-" * 68)
splits = walk_forward_split(
    df, date_col="inception_date",
    min_train_months=12, test_months=6, step_months=6, ibnr_buffer_months=3,
)
check = temporal_leakage_check(splits, df, date_col="inception_date")
print(f"  Leakage check: {'PASSED' if not check.get('errors') else check['errors']}")
print(f"  Folds: {len(splits)}\n")

cv      = InsuranceCV(splits, df)
wf_means = {}
wf_folds = {}
for label, (d, lr) in CONFIGS.items():
    scores = catboost_cv(X, y, w, d, lr, cv)
    wf_means[label] = float(np.mean(scores))
    wf_folds[label] = scores
wf_selected = min(wf_means, key=wf_means.get)

for c, folds in wf_folds.items():
    fold_str = "  ".join(f"{v:.4f}" for v in folds)
    print(f"  {c:<26} mean={wf_means[c]:.5f}{'  <-- selected' if c==wf_selected else ''}")
    print(f"  {'':26}   fold scores: {fold_str}")

baseline_folds = wf_folds[list(CONFIGS.keys())[0]]
print(f"\n  Drift signal: {baseline_folds[0]:.4f} -> {baseline_folds[-1]:.4f} "
      f"({(baseline_folds[-1]/baseline_folds[0]-1)*100:+.1f}%) across folds. "
      "k-fold cannot show this.")

# ── Prospective holdout ───────────────────────────────────────────────────────
print("\n" + "-" * 68)
print("Prospective holdout: train 2021-2023, test 2024 (ground truth)")
print("-" * 68)
cutoff = np.datetime64(date(2024, 1, 1), "D")
mask = df["inception_date"].cast(pl.Date).to_numpy().astype("datetime64[D]") < cutoff
tr, te = np.where(mask)[0], np.where(~mask)[0]
prosp = {}
for label, (d, lr) in CONFIGS.items():
    m = CatBoostRegressor(loss_function="Poisson", iterations=400, depth=d,
                          learning_rate=lr, random_seed=42, verbose=False)
    m.fit(Pool(X[tr], y[tr], weight=w[tr]))
    prosp[label] = poisson_deviance(y[te], m.predict(X[te]), w[te])
    print(f"  {label:<26} prospective deviance: {prosp[label]:.5f}")
prosp_selected = min(prosp, key=prosp.get)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"\n  {'Config':<26} {'kf CV':>9} {'wf CV':>9} {'Prospective':>12} "
      f"{'kf bias':>9} {'wf bias':>9}")
print(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*12} {'-'*9} {'-'*9}")
for c in CONFIGS:
    kfb, wfb = kf_means[c] - prosp[c], wf_means[c] - prosp[c]
    print(f"  {c:<26} {kf_means[c]:>9.5f} {wf_means[c]:>9.5f} "
          f"{prosp[c]:>12.5f} {kfb:>+9.5f} {wfb:>+9.5f}")

print(f"\n  k-fold selected:       {kf_selected}")
print(f"  Walk-forward selected: {wf_selected}")
print(f"  Best prospectively:    {prosp_selected}")
print(f"\n  k-fold picks right config:       {kf_selected == prosp_selected}")
print(f"  Walk-forward picks right config: {wf_selected == prosp_selected}")

kf_abs_bias = np.mean([abs(kf_means[c] - prosp[c]) for c in CONFIGS])
wf_abs_bias = np.mean([abs(wf_means[c] - prosp[c]) for c in CONFIGS])
print(f"\n  Mean |CV bias| — k-fold: {kf_abs_bias:.5f}  walk-forward: {wf_abs_bias:.5f}")
if wf_abs_bias < kf_abs_bias:
    print(f"  Walk-forward is {kf_abs_bias/max(wf_abs_bias,1e-8):.1f}x more accurate prospectively.")

print(f"\nCompleted in {time.time()-T0:.1f}s")
