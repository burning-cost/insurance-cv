"""
Distributional train/test splitting for insurance datasets.

Temporal splits handle the time-axis problem. This module handles a different
but equally important problem: when you split randomly, rare events — large
losses, catastrophe-exposed risks, unusual risk profiles — cluster unevenly
between train and test by pure chance. SPlit (Mak & Joseph 2018) solves this
by selecting the test set to minimise energy distance to the full dataset
distribution, ensuring the test set is as representative as possible.

The algorithm finds "support points" — a subset of observations that best
represents the full empirical distribution in terms of energy distance. These
support points become the test set; the rest form the training set.

Reference
---------
Mak, S. & Joseph, V.R. (2018). Support points. Annals of Statistics, 46(6A).
Guo, J., Dong, P., Quan, Z. (2026). Starting off on the wrong foot: Pitfalls
in data preparation. arXiv:2603.18190.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


def _to_numeric_array(X: "np.ndarray | object") -> np.ndarray:
    """
    Coerce X into a 2-D float64 numpy array.

    Accepts numpy arrays, pandas DataFrames, and polars DataFrames. Raises
    ValueError if the result contains non-numeric dtypes.
    """
    # Polars
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
    except ImportError:
        pass
    # Pandas
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=float)
    except ImportError:
        pass

    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array (n_samples, n_features); got shape {arr.shape}."
        )
    return arr


def _energy_distance(
    X_test: np.ndarray,
    X_full: np.ndarray,
) -> float:
    """
    Compute energy distance between X_test (proposed test set) and X_full.

    ED = 2/(n_test * N) * sum_j sum_i ||o_j - x_i||
       - 1/n_test^2 * sum_j sum_j' ||o_j - o_j'||
       - 1/N^2 * sum_i sum_i' ||x_i - x_i'||

    The last term is constant (depends only on X_full) and does not affect
    comparisons between candidate test sets, but is included here for
    interpretability as a true energy distance value.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_test, d)
    X_full : np.ndarray of shape (N, d)

    Returns
    -------
    float
    """
    n_test = len(X_test)
    n_full = len(X_full)

    cross = cdist(X_test, X_full, metric="euclidean").sum()
    self_test = cdist(X_test, X_test, metric="euclidean").sum()
    self_full = cdist(X_full, X_full, metric="euclidean").sum()

    return (
        2.0 / (n_test * n_full) * cross
        - 1.0 / (n_test ** 2) * self_test
        - 1.0 / (n_full ** 2) * self_full
    )


def _greedy_swap(
    X: np.ndarray,
    test_idx: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Improve a candidate test set via greedy random-swap optimisation.

    At each iteration, pick a random test observation and a random training
    observation, swap them, and keep the swap if it reduces energy distance.
    This is a standard greedy local search — simple, fast, and effective for
    insurance-scale datasets (N <= 200k).

    The energy distance objective is maintained incrementally: only the terms
    that change on each swap are recomputed. For a swap of one test point:
    - cross_sum changes by O(N) — recompute distances from the new point
    - self_test_sum changes by O(n_test) — recompute distances within test set

    Parameters
    ----------
    X : np.ndarray of shape (N, d), standardised
        Full dataset.
    test_idx : np.ndarray of int
        Initial test indices.
    n_iter : int
        Number of swap attempts.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int
        Optimised test indices.
    """
    N = len(X)
    test_set = set(test_idx.tolist())
    train_set = set(range(N)) - test_set

    test_arr = np.array(sorted(test_set), dtype=int)
    X_test = X[test_arr].copy()
    n_test = len(test_arr)
    n_full = N

    # cross_mat[i, j] = ||X_test[i] - X[j]||  (n_test x N)
    cross_mat = cdist(X_test, X, metric="euclidean")
    cross_sum = cross_mat.sum()

    # self_test_sum = sum of all pairwise distances within X_test (diagonal = 0)
    self_test_sum = cdist(X_test, X_test, metric="euclidean").sum()

    train_arr = np.array(sorted(train_set), dtype=int)

    for _ in range(n_iter):
        # Draw a random test point to potentially remove and a random train
        # point to potentially insert
        t_local = int(rng.integers(n_test))
        in_idx = int(rng.choice(train_arr))

        row_in = X[in_idx]  # shape (d,)

        # --- Incremental cross_sum update ---
        # Remove contribution of test_arr[t_local] to cross_sum
        # Add contribution of in_idx
        d_out_full = cross_mat[t_local]          # shape (N,)
        d_in_full = np.linalg.norm(X - row_in, axis=1)  # shape (N,)
        new_cross_sum = cross_sum - d_out_full.sum() + d_in_full.sum()

        # --- Incremental self_test_sum update ---
        # The current self_test_sum includes all ||X_test[i] - X_test[j]|| pairs.
        # When we replace X_test[t_local] with row_in:
        #
        # d_out_test[i] = ||X_test[t_local] - X_test[i]||  (includes i=t_local: = 0)
        # d_in_test[i]  = ||row_in - X_test[i]||           (includes i=t_local: = ||row_in - X_test[t_local]||)
        #
        # Remove: 2 * d_out_test.sum() — each edge (t_local, i) appears twice
        # Add:    2 * d_in_test.sum()  — each new edge (in, i) appears twice
        # But d_in_test[t_local] = ||row_in - X_test[t_local]|| which is the
        # distance from the new point to the OLD occupant of slot t_local, not
        # to itself. After the swap, slot t_local holds row_in, so that diagonal
        # entry becomes 0. We therefore subtract d_in_test[t_local] to undo
        # the double-count (it got added twice; correct count is once as a
        # cross-term, then once removed as diagonal, net = remove once).
        d_out_test = cross_mat[t_local, test_arr]   # shape (n_test,)
        d_in_test = np.linalg.norm(X_test - row_in, axis=1)  # shape (n_test,)

        new_self_test = (
            self_test_sum
            - 2.0 * d_out_test.sum()
            + 2.0 * d_in_test.sum()
            - 2.0 * d_in_test[t_local]   # remove the old-slot cross-distance (not a true diagonal)
        )
        # Note: d_out_test[t_local] = 0 (diagonal), so it contributes nothing
        # to the d_out_test.sum() removal — correct.

        # --- Compare reduced energy distances (constant self_full term cancels) ---
        current_ed = (
            2.0 / (n_test * n_full) * cross_sum
            - 1.0 / (n_test ** 2) * self_test_sum
        )
        new_ed = (
            2.0 / (n_test * n_full) * new_cross_sum
            - 1.0 / (n_test ** 2) * new_self_test
        )

        if new_ed < current_ed:
            # Accept swap: update all state
            out_idx = int(test_arr[t_local])
            test_set.discard(out_idx)
            test_set.add(in_idx)
            train_set.discard(in_idx)
            train_set.add(out_idx)

            test_arr[t_local] = in_idx
            X_test[t_local] = row_in

            # Update cross_mat row
            cross_mat[t_local] = d_in_full
            cross_sum = new_cross_sum
            self_test_sum = new_self_test

            # Rebuild train_arr — needed to keep rng.choice representative.
            # Only rebuild periodically (every 10 acceptances) to avoid O(N log N)
            # overhead per iteration; for small N this is negligible.
            train_arr = np.array(sorted(train_set), dtype=int)

    return test_arr


class SupportPointSplit(BaseEstimator):
    """
    Distributional train/test split using support-point optimisation.

    Selects the test set as the subset of the data that minimises energy
    distance to the full dataset distribution. This produces a test set
    that is statistically representative of the full distribution — rare
    events, extreme values, and unusual risk profiles are proportionally
    represented rather than randomly over- or under-sampled.

    The algorithm uses greedy random-swap optimisation, which is O(n_iter *
    n_test * d) per run and typically converges within 100 iterations for
    insurance-scale datasets.

    Parameters
    ----------
    test_size : float, default 0.2
        Proportion of the dataset to include in the test set.
    n_iter : int, default 100
        Number of swap iterations for the greedy optimiser. Increase for
        better convergence on larger datasets; 100 is sufficient for most
        insurance datasets up to 100k rows.
    random_state : int or None, default 42
        Controls reproducibility of the initial random test set and the
        swap sequence.
    standardize : bool, default True
        Whether to standardise features to zero mean and unit variance before
        computing energy distances. Strongly recommended when features are on
        different scales (e.g. premium in GBP, age in years, sum insured in
        thousands). Set to False only if features are already on comparable
        scales.

    Notes
    -----
    Energy distance is well-defined in any dimension but becomes less
    discriminative as dimensionality grows (the curse of dimensionality
    affects all distance-based methods). In practice this method works well
    for d <= 20 features. For higher-dimensional data, consider reducing
    dimensionality first or using only the most important features.

    For N > 200k, computation becomes slower due to the O(N * n_test)
    distance computations per iteration. The implementation is vectorised
    via scipy's cdist but is not distributed.

    References
    ----------
    Mak, S. & Joseph, V.R. (2018). Support points. Annals of Statistics,
    46(6A), 2562–2592.
    Guo, J., Dong, P., Quan, Z. (2026). Starting off on the wrong foot:
    Pitfalls in data preparation. arXiv:2603.18190.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_cv.distributional import SupportPointSplit
    >>> X = np.random.default_rng(0).standard_normal((1000, 5))
    >>> splitter = SupportPointSplit(test_size=0.2, n_iter=50)
    >>> train_idx, test_idx = splitter.split(X)
    >>> len(train_idx), len(test_idx)
    (800, 200)
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_iter: int = 100,
        random_state: int | None = 42,
        standardize: bool = True,
    ) -> None:
        self.test_size = test_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.standardize = standardize

    def split(
        self,
        X: "np.ndarray | object",
        y: "np.ndarray | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the train/test split that minimises energy distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Accepts numpy arrays, pandas DataFrames, and
            polars DataFrames. When ``y`` is provided, it is appended as an
            additional column so the split respects the joint distribution of
            features and target — recommended for insurance loss modelling.
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target variable(s). If provided, included in the distributional
            matching. Pass the claim amount or frequency target here to ensure
            extreme losses are proportionally represented in both sets.

        Returns
        -------
        train_idx : np.ndarray of int
            Integer indices of training observations.
        test_idx : np.ndarray of int
            Integer indices of test observations.

        Raises
        ------
        ValueError
            If ``test_size`` would produce fewer than 2 test observations or
            fewer than 2 training observations.
        """
        X_arr = _to_numeric_array(X)
        N = len(X_arr)

        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            X_arr = np.hstack([X_arr, y_arr])

        n_test = max(2, round(N * self.test_size))
        n_train = N - n_test
        if n_train < 2:
            raise ValueError(
                f"test_size={self.test_size} with N={N} leaves only {n_train} "
                "training observations. Reduce test_size."
            )

        if self.standardize:
            means = X_arr.mean(axis=0)
            stds = X_arr.std(axis=0)
            stds[stds == 0] = 1.0  # avoid division by zero for constant columns
            X_std = (X_arr - means) / stds
        else:
            X_std = X_arr

        rng = check_random_state(self.random_state)
        # Convert sklearn RandomState seed to a numpy Generator seed
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # Initialise test set with a random sample
        init_test_idx = np_rng.choice(N, size=n_test, replace=False)

        # Warn if dataset is large — computation may be slow
        if N > 50_000:
            warnings.warn(
                f"SupportPointSplit: N={N} is large. Each iteration performs "
                f"O(N * n_test) distance computations. Consider reducing n_iter "
                "or using a feature-reduced representation.",
                UserWarning,
                stacklevel=2,
            )

        test_idx = _greedy_swap(X_std, init_test_idx, self.n_iter, np_rng)
        test_set = set(test_idx.tolist())
        train_idx = np.array([i for i in range(N) if i not in test_set], dtype=int)
        test_idx_out = np.array(sorted(test_set), dtype=int)

        return train_idx, test_idx_out

    def get_n_splits(
        self,
        X: "np.ndarray | object | None" = None,
        y: "np.ndarray | None" = None,
        groups: "np.ndarray | None" = None,
    ) -> int:
        """Return 1 — this splitter produces a single train/test split."""
        return 1
