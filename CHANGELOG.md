# Changelog

## v0.3.0 (2026-04-01)
- Add `SupportPointSplit`: distributional train/test splitting via energy distance
  minimisation (greedy swap algorithm). Ensures rare events are proportionally
  represented in test sets. Based on Mak & Joseph (2018) and Guo et al. (2026).
- Add `ChatterjeeSelector`: feature screening using Chatterjee's Xi coefficient.
  Detects nonlinear dependence (U-shapes, threshold effects) that Pearson and
  Spearman miss. Implements Xi natively via scipy without external dependencies.
- Add `tests/test_distributional.py` and `tests/test_feature_selection.py`
- Bump version: 0.2.4 -> 0.3.0
- Update package description to reflect broader scope

## [0.2.4] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)

## v0.2.3 (2026-03-22) [unreleased]
- Add quickstart notebook and Colab badge
- Rebuild benchmark: honest trajectory story, fix leakage check logic
- fix: sync __version__ with pyproject.toml (0.2.1 -> 0.2.3) and export InsuranceCV

## v0.2.3 (2026-03-16)
- Add Benchmark Results section with real numbers from Databricks
- QA batch 9 fixes: README accuracy, development_col example, version bump
- Add PyPI classifiers for financial/insurance audience
- Add benchmark: walk-forward CV vs k-fold for insurance temporal validation
- Polish flagship README: badges, benchmark table, problem statement
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix: README quick-start cross_val_score used undefined model, X, y
- fix: update cross-references to consolidated repos
- docs: add Performance section with benchmark summary
- Add benchmark notebook: insurance-cv temporal CV vs random KFold
- docs: add self-contained synthetic data quickstart
- fix: update polars floor to >=1.0 and fix project URLs
- Add capability demo and README Capabilities section

## v0.2.0 (2026-03-09)
- Polish README - blog link, badges, cross-references table
- Add GitHub Actions CI workflow and test badge
- fix: update URLs to burning-cost org
- Replace uv pip install with uv add in examples
- Migrate examples to CatBoost and Polars
- docs: switch examples to polars/uv, fix tone
- fix: standardise on CatBoost, uv, clean up style
- fix: catboost, uv, polars references
- Migrate to Polars
- Initial release: walk-forward, policy-year, and accident-year CV splitters
