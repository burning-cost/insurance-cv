"""
Tests for SupportPointSplit (distributional.py).

The key property to verify is that support-point splitting produces a test set
with lower energy distance to the full dataset than a random split — this is
the whole point of the algorithm. We test this on small synthetic datasets
where the effect is detectable without excessive compute.

We also verify sklearn API compatibility (get_n_splits returns 1, split accepts
X and optional y, output indices are disjoint and cover all rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_cv.distributional import SupportPointSplit, _energy_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_skewed_dataset(
    n: int = 400,
    n_features: int = 3,
    rare_fraction: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Synthetic dataset with a small cluster of rare high-value observations.

    This mimics catastrophe-exposed insurance data where a handful of extreme
    risks need to be represented proportionally in both train and test.
    """
    rng = np.random.default_rng(seed)
    n_rare = max(2, round(n * rare_fraction))
    n_common = n - n_rare

    common = rng.standard_normal((n_common, n_features))
    rare = rng.standard_normal((n_rare, n_features)) * 0.1 + 10.0  # far cluster

    return np.vstack([common, rare])


# ---------------------------------------------------------------------------
# Core property tests
# ---------------------------------------------------------------------------


class TestEnergyDistanceReduction:
    """Support-point splitting should yield lower energy distance than random."""

    def test_support_split_beats_random(self) -> None:
        """
        On a dataset with a small rare cluster, SPlit should achieve lower
        energy distance than a random split in most runs.

        We repeat 10 random splits and check that the SPlit distance is below
        the median random distance. This is a probabilistic test — if the
        algorithm is working, it should pass reliably.
        """
        X = make_skewed_dataset(n=300, n_features=4, rare_fraction=0.05, seed=42)
        N = len(X)
        test_size = 0.2
        n_test = round(N * test_size)

        splitter = SupportPointSplit(
            test_size=test_size, n_iter=150, random_state=42, standardize=True
        )
        train_idx, test_idx = splitter.split(X)
        X_test_sps = X[test_idx]
        ed_sps = _energy_distance(X_test_sps, X)

        # Compare against 10 random splits
        rng = np.random.default_rng(0)
        ed_random_vals = []
        for _ in range(10):
            rand_test = rng.choice(N, size=n_test, replace=False)
            ed_random_vals.append(_energy_distance(X[rand_test], X))

        median_random_ed = float(np.median(ed_random_vals))

        assert ed_sps < median_random_ed, (
            f"SPlit energy distance ({ed_sps:.6f}) should be below median "
            f"random energy distance ({median_random_ed:.6f})."
        )

    def test_energy_distance_decreases_over_iterations(self) -> None:
        """
        More iterations should not make the result worse — the greedy swap
        only accepts moves that decrease energy distance.
        """
        X = make_skewed_dataset(n=200, seed=7)

        splitter_few = SupportPointSplit(
            test_size=0.2, n_iter=10, random_state=1, standardize=True
        )
        splitter_many = SupportPointSplit(
            test_size=0.2, n_iter=200, random_state=1, standardize=True
        )

        _, test_few = splitter_few.split(X)
        _, test_many = splitter_many.split(X)

        ed_few = _energy_distance(X[test_few], X)
        ed_many = _energy_distance(X[test_many], X)

        # More iterations should give equal or better (lower) energy distance
        assert ed_many <= ed_few + 1e-9, (
            f"More iterations gave higher energy distance: {ed_many:.6f} > {ed_few:.6f}"
        )


# ---------------------------------------------------------------------------
# Index correctness
# ---------------------------------------------------------------------------


class TestSplitIndices:
    def test_indices_cover_all_rows(self) -> None:
        X = make_skewed_dataset(n=200, seed=0)
        splitter = SupportPointSplit(test_size=0.2, n_iter=50, random_state=0)
        train_idx, test_idx = splitter.split(X)
        combined = sorted(np.concatenate([train_idx, test_idx]).tolist())
        assert combined == list(range(len(X)))

    def test_train_test_disjoint(self) -> None:
        X = make_skewed_dataset(n=200, seed=1)
        splitter = SupportPointSplit(test_size=0.2, n_iter=50, random_state=1)
        train_idx, test_idx = splitter.split(X)
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0

    def test_test_size_approximately_correct(self) -> None:
        X = make_skewed_dataset(n=500, seed=2)
        test_size = 0.25
        splitter = SupportPointSplit(test_size=test_size, n_iter=30, random_state=2)
        train_idx, test_idx = splitter.split(X)
        actual_frac = len(test_idx) / len(X)
        assert abs(actual_frac - test_size) <= 0.01, (
            f"Expected test fraction ~{test_size}, got {actual_frac:.3f}"
        )

    def test_returns_numpy_arrays(self) -> None:
        X = make_skewed_dataset(n=100, seed=3)
        splitter = SupportPointSplit(test_size=0.2, n_iter=10, random_state=3)
        train_idx, test_idx = splitter.split(X)
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert train_idx.dtype == int or np.issubdtype(train_idx.dtype, np.integer)


# ---------------------------------------------------------------------------
# y-augmentation
# ---------------------------------------------------------------------------


class TestYAugmentation:
    def test_split_with_y_changes_result(self) -> None:
        """
        Including y should change which observations end up in test.
        The result is not guaranteed to be better or worse — just different.
        """
        X = make_skewed_dataset(n=200, seed=5)
        y = np.random.default_rng(5).standard_normal(200)

        splitter = SupportPointSplit(test_size=0.2, n_iter=50, random_state=5)
        _, test_no_y = splitter.split(X)
        _, test_with_y = splitter.split(X, y=y)

        # Results should not be identical — y adds information
        assert not np.array_equal(test_no_y, test_with_y)

    def test_split_y_indices_still_disjoint(self) -> None:
        X = make_skewed_dataset(n=200, seed=6)
        y = np.random.default_rng(6).exponential(1000, 200)  # simulate loss amounts

        splitter = SupportPointSplit(test_size=0.2, n_iter=50, random_state=6)
        train_idx, test_idx = splitter.split(X, y=y)
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Input type acceptance
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_accepts_pandas_dataframe(self) -> None:
        X_np = make_skewed_dataset(n=100, seed=8)
        X_pd = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
        splitter = SupportPointSplit(test_size=0.2, n_iter=20, random_state=8)
        train_idx, test_idx = splitter.split(X_pd)
        assert len(train_idx) + len(test_idx) == len(X_pd)

    def test_accepts_polars_dataframe(self) -> None:
        X_np = make_skewed_dataset(n=100, seed=9)
        X_pl = pl.from_numpy(X_np, schema=[f"f{i}" for i in range(X_np.shape[1])])
        splitter = SupportPointSplit(test_size=0.2, n_iter=20, random_state=9)
        train_idx, test_idx = splitter.split(X_pl)
        assert len(train_idx) + len(test_idx) == len(X_pl)

    def test_accepts_1d_array(self) -> None:
        x = np.random.default_rng(10).standard_normal(100)
        splitter = SupportPointSplit(test_size=0.2, n_iter=20, random_state=10)
        train_idx, test_idx = splitter.split(x)
        assert len(train_idx) + len(test_idx) == 100


# ---------------------------------------------------------------------------
# sklearn API compatibility
# ---------------------------------------------------------------------------


class TestSklearnAPI:
    def test_get_n_splits_returns_1(self) -> None:
        splitter = SupportPointSplit()
        assert splitter.get_n_splits() == 1

    def test_get_n_splits_accepts_X(self) -> None:
        X = np.zeros((100, 3))
        splitter = SupportPointSplit()
        assert splitter.get_n_splits(X=X) == 1

    def test_standardize_false_accepted(self) -> None:
        X = make_skewed_dataset(n=100, seed=11)
        splitter = SupportPointSplit(
            test_size=0.2, n_iter=20, random_state=11, standardize=False
        )
        train_idx, test_idx = splitter.split(X)
        assert len(train_idx) + len(test_idx) == 100

    def test_reproducibility(self) -> None:
        X = make_skewed_dataset(n=200, seed=12)
        splitter = SupportPointSplit(test_size=0.2, n_iter=50, random_state=42)
        train_a, test_a = splitter.split(X)
        train_b, test_b = splitter.split(X)
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)

    def test_raises_if_test_size_too_large(self) -> None:
        X = np.random.default_rng(0).standard_normal((10, 2))
        splitter = SupportPointSplit(test_size=0.95, n_iter=10)
        with pytest.raises(ValueError, match="training observations"):
            splitter.split(X)
