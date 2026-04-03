"""
Tests for ChatterjeeSelector (feature_selection.py).

The key properties to verify:
- Xi correctly identifies nonlinear dependence (U-shaped, threshold effects)
  that Pearson/Spearman miss
- Xi is near 0 for independent variables
- Feature selection (k and threshold) works correctly
- sklearn transformer API is complete (fit, transform, fit_transform)
- Pandas and numpy input both work
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_cv.feature_selection import ChatterjeeSelector, _xi


# ---------------------------------------------------------------------------
# Unit tests for the _xi function
# ---------------------------------------------------------------------------


class TestXiFunction:
    def test_xi_near_zero_for_independent(self) -> None:
        """
        For two independent standard normals, Xi should be close to 0.
        With N=2000 the sampling variability is low enough to detect this.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        y = rng.standard_normal(2000)
        xi_val = _xi(x, y, ties="average")
        # Under independence, E[xi] = 0 and SD ~ sqrt(2/5)/sqrt(N)
        # For N=2000, 3 SD = 3 * sqrt(2/5) / sqrt(2000) ≈ 0.030
        assert abs(xi_val) < 0.06, (
            f"Xi should be near 0 for independent variables; got {xi_val:.4f}"
        )

    def test_xi_high_for_deterministic_relationship(self) -> None:
        """
        When y is a measurable function of x, Xi should approach 1.
        """
        rng = np.random.default_rng(1)
        x = rng.standard_normal(1000)
        y = x ** 2  # deterministic
        xi_val = _xi(x, y, ties="average")
        assert xi_val > 0.7, (
            f"Xi should be high for y=x^2; got {xi_val:.4f}"
        )

    def test_xi_nonlinear_beats_pearson(self) -> None:
        """
        Xi should detect U-shaped dependence that Pearson misses.
        This mimics an age curve in motor insurance (young and old = higher risk).
        """
        rng = np.random.default_rng(2)
        age = rng.uniform(18, 80, 2000)
        # Frequency increases towards young and old ends — U-shaped
        frequency = 0.05 + 0.01 * (age - 49) ** 2 / 100 + rng.standard_normal(2000) * 0.02
        frequency = np.clip(frequency, 0, None)

        xi_val = _xi(age, frequency, ties="average")
        pearson_corr = abs(np.corrcoef(age, frequency)[0, 1])

        # Xi should detect the dependence; Pearson should underestimate it
        assert xi_val > 0.1, f"Xi should detect U-shaped dependence; got {xi_val:.4f}"
        # The U-shape means pearson_corr may actually be near zero
        assert xi_val > pearson_corr, (
            f"Xi ({xi_val:.4f}) should exceed Pearson ({pearson_corr:.4f}) for U-shaped data"
        )

    def test_xi_with_binary_x_ties(self) -> None:
        """
        Binary x produces maximum ties. Xi should still return a finite value.
        """
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=500).astype(float)
        y = x * 2 + rng.standard_normal(500) * 0.5
        xi_val = _xi(x, y, ties="random", n_reps=20)
        assert np.isfinite(xi_val), f"Xi should be finite for binary x; got {xi_val}"
        assert xi_val > 0.1, f"Xi should detect binary x -> y signal; got {xi_val:.4f}"

    def test_xi_range_non_negative(self) -> None:
        """
        Xi should be non-negative for any reasonable dataset size.
        (Slightly negative values are possible but close to 0 for small N;
        for N>=100 this should not occur with any real dependence.)
        """
        rng = np.random.default_rng(4)
        for _ in range(5):
            x = rng.standard_normal(300)
            y = rng.standard_normal(300)
            xi_val = _xi(x, y, ties="average")
            assert xi_val >= -0.1, f"Xi should be near 0 or positive; got {xi_val:.4f}"

    def test_xi_threshold_dependence(self) -> None:
        """
        Xi should detect a threshold effect (step function), which is common
        near deductible levels in insurance data.
        """
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 10000, 1000)
        # Claim only occurs above deductible
        y = (x > 5000).astype(float) + rng.standard_normal(1000) * 0.1
        xi_val = _xi(x, y, ties="average")
        assert xi_val > 0.2, (
            f"Xi should detect threshold (deductible) effect; got {xi_val:.4f}"
        )


# ---------------------------------------------------------------------------
# ChatterjeeSelector fit / scores_
# ---------------------------------------------------------------------------


class TestChatterjeeSelectorFit:
    def _make_insurance_X_y(self, seed: int = 0, n: int = 1000) -> tuple[pd.DataFrame, np.ndarray]:
        """Synthetic insurance features: age (nonlinear signal), sum_insured (log signal),
        noise1, noise2 (independent)."""
        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 80, n)
        sum_insured = np.exp(rng.uniform(9, 13, n))  # £8k–£440k
        noise1 = rng.standard_normal(n)
        noise2 = rng.integers(0, 5, n).astype(float)
        # Target: nonlinear function of age and log(sum_insured)
        y = (
            0.05
            + 0.001 * (age - 45) ** 2
            + 0.3 * np.log(sum_insured / 1000)
            + rng.standard_normal(n) * 0.5
        )
        X = pd.DataFrame({
            "age": age,
            "sum_insured": sum_insured,
            "noise1": noise1,
            "noise2": noise2,
        })
        return X, y

    def test_fit_sets_scores(self) -> None:
        X, y = self._make_insurance_X_y()
        sel = ChatterjeeSelector()
        sel.fit(X, y)
        assert hasattr(sel, "scores_")
        assert set(sel.scores_.keys()) == {"age", "sum_insured", "noise1", "noise2"}

    def test_signal_features_rank_above_noise(self) -> None:
        """Age and sum_insured should have higher Xi than pure noise features."""
        X, y = self._make_insurance_X_y(n=2000, seed=42)
        sel = ChatterjeeSelector(n_reps=20, random_state=42)
        sel.fit(X, y)

        xi_age = sel.scores_["age"]
        xi_si = sel.scores_["sum_insured"]
        xi_noise1 = sel.scores_["noise1"]
        xi_noise2 = sel.scores_["noise2"]

        assert xi_age > xi_noise1, (
            f"age Xi ({xi_age:.4f}) should exceed noise1 Xi ({xi_noise1:.4f})"
        )
        assert xi_si > xi_noise1, (
            f"sum_insured Xi ({xi_si:.4f}) should exceed noise1 Xi ({xi_noise1:.4f})"
        )

    def test_independent_features_xi_near_zero(self) -> None:
        """Pure noise features should have Xi close to 0."""
        rng = np.random.default_rng(99)
        X = pd.DataFrame({
            "a": rng.standard_normal(1500),
            "b": rng.standard_normal(1500),
        })
        y = rng.standard_normal(1500)
        sel = ChatterjeeSelector()
        sel.fit(X, y)
        for feat, xi_val in sel.scores_.items():
            assert abs(xi_val) < 0.1, (
                f"Independent feature '{feat}' has Xi={xi_val:.4f}; expected near 0"
            )

    def test_fit_returns_self(self) -> None:
        X, y = self._make_insurance_X_y(n=200)
        sel = ChatterjeeSelector()
        result = sel.fit(X, y)
        assert result is sel

    def test_fit_with_numpy_array(self) -> None:
        rng = np.random.default_rng(10)
        X_np = rng.standard_normal((300, 4))
        y = X_np[:, 0] ** 2 + rng.standard_normal(300) * 0.5
        sel = ChatterjeeSelector()
        sel.fit(X_np, y)
        assert set(sel.scores_.keys()) == {0, 1, 2, 3}

    def test_n_features_in_set(self) -> None:
        X, y = self._make_insurance_X_y(n=200)
        sel = ChatterjeeSelector()
        sel.fit(X, y)
        assert sel.n_features_in_ == 4


# ---------------------------------------------------------------------------
# ChatterjeeSelector transform / selection
# ---------------------------------------------------------------------------


class TestChatterjeeSelectorTransform:
    def test_k_selection(self) -> None:
        rng = np.random.default_rng(20)
        X = pd.DataFrame({
            "strong": rng.uniform(0, 10, 500) ** 2,
            "weak": rng.standard_normal(500),
            "noise": rng.standard_normal(500),
        })
        y = X["strong"].to_numpy() + rng.standard_normal(500)
        sel = ChatterjeeSelector(k=1)
        sel.fit(X, y)
        X_out = sel.transform(X)
        assert isinstance(X_out, pd.DataFrame)
        assert list(X_out.columns) == ["strong"]

    def test_threshold_selection(self) -> None:
        rng = np.random.default_rng(21)
        X = pd.DataFrame({
            "signal": rng.standard_normal(500) * np.sign(rng.standard_normal(500)),
            "noise": rng.standard_normal(500),
        })
        y = X["signal"].to_numpy() ** 2 + rng.standard_normal(500) * 0.1
        sel = ChatterjeeSelector(threshold=0.05)
        sel.fit(X, y)
        X_out = sel.transform(X)
        # signal should survive threshold; noise should typically not
        assert "signal" in X_out.columns

    def test_no_selection_keeps_all(self) -> None:
        rng = np.random.default_rng(22)
        X = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
        y = rng.standard_normal(200)
        sel = ChatterjeeSelector(k=None, threshold=None)
        sel.fit(X, y)
        X_out = sel.transform(X)
        assert set(X_out.columns) == {"a", "b"}

    def test_fit_transform_equivalent(self) -> None:
        rng = np.random.default_rng(23)
        X = pd.DataFrame({
            "x1": rng.standard_normal(300),
            "x2": rng.standard_normal(300),
        })
        y = X["x1"].to_numpy() ** 2
        sel = ChatterjeeSelector(k=1)
        X_ft = sel.fit_transform(X, y)
        sel2 = ChatterjeeSelector(k=1)
        sel2.fit(X, y)
        X_t = sel2.transform(X)
        assert list(X_ft.columns) == list(X_t.columns)

    def test_transform_without_fit_raises(self) -> None:
        sel = ChatterjeeSelector()
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(Exception):  # NotFittedError or similar
            sel.transform(X)

    def test_numpy_transform(self) -> None:
        rng = np.random.default_rng(24)
        X_np = rng.standard_normal((200, 3))
        y = X_np[:, 0] ** 2 + rng.standard_normal(200) * 0.2
        sel = ChatterjeeSelector(k=1)
        sel.fit(X_np, y)
        X_out = sel.transform(X_np)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape[1] == 1

    def test_polars_transform(self) -> None:
        rng = np.random.default_rng(25)
        X_np = rng.standard_normal((200, 2))
        X_pl = pl.from_numpy(X_np, schema=["feat_a", "feat_b"])
        y = X_np[:, 0] + rng.standard_normal(200) * 0.5
        sel = ChatterjeeSelector(k=1)
        sel.fit(X_pl, y)
        X_out = sel.transform(X_pl)
        assert isinstance(X_out, pl.DataFrame)
        assert X_out.shape[1] == 1

    def test_get_feature_names_out(self) -> None:
        rng = np.random.default_rng(26)
        X = pd.DataFrame({"age": rng.uniform(18, 80, 200), "noise": rng.standard_normal(200)})
        y = (X["age"].to_numpy() - 49) ** 2
        sel = ChatterjeeSelector(k=1)
        sel.fit(X, y)
        names = sel.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert "age" in names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_mismatched_X_y_length_raises(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = np.array([1, 2, 3])
        sel = ChatterjeeSelector()
        with pytest.raises(ValueError, match="rows"):
            sel.fit(X, y)

    def test_k_larger_than_features_clipped(self) -> None:
        rng = np.random.default_rng(30)
        X = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)})
        y = rng.standard_normal(100)
        sel = ChatterjeeSelector(k=10)  # k > n_features
        sel.fit(X, y)
        assert len(sel.selected_features_) == 2  # all features kept

    def test_binary_target_works(self) -> None:
        """Claim indicator (0/1) is a common insurance target."""
        rng = np.random.default_rng(31)
        age = rng.uniform(18, 80, 500)
        y_binary = (rng.uniform(size=500) < 0.1 + 0.005 * abs(age - 49)).astype(float)
        X = pd.DataFrame({"age": age, "noise": rng.standard_normal(500)})
        sel = ChatterjeeSelector(k=1)
        sel.fit(X, y_binary)
        assert "age" in sel.selected_features_

    def test_zero_inflated_target_works(self) -> None:
        """Compound Poisson / Tweedie targets are common in insurance."""
        rng = np.random.default_rng(32)
        n = 800
        x = rng.uniform(0, 100, n)
        # 90% zeros, rest positive correlated with x
        mask = rng.uniform(size=n) < 0.1
        y = np.where(mask, x * rng.exponential(1, n), 0.0)
        X = pd.DataFrame({"x": x, "noise": rng.standard_normal(n)})
        sel = ChatterjeeSelector(k=1)
        sel.fit(X, y)
        # x should score higher than noise given the signal
        assert sel.scores_["x"] >= sel.scores_["noise"] - 0.05
