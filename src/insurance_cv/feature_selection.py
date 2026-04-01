"""
Feature screening for insurance datasets using Chatterjee's Xi coefficient.

Standard correlation measures fail for insurance loss data. Pearson is linear
only. Spearman is monotone only. Both miss the relationships that dominate
insurance: age U-curves (young and old drivers both have higher frequency),
sum insured log-curves (larger risks have lower severity per unit), and
threshold effects near deductible levels.

Chatterjee's Xi is consistent under any dependence structure. It equals 0 when
X and Y are independent and approaches 1 when Y is a measurable function of X.
It has O(N log N) complexity and handles heavy-tailed and zero-inflated data
(compound Poisson, Tweedie) without modification.

Reference
---------
Chatterjee, S. (2021). A new coefficient of correlation. Journal of the
American Statistical Association, 116(536), 2009–2022.
Guo, J., Dong, P., Quan, Z. (2026). Starting off on the wrong foot: Pitfalls
in data preparation. arXiv:2603.18190.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _xi(
    x: np.ndarray,
    y: np.ndarray,
    ties: str = "random",
    n_reps: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute Chatterjee's Xi coefficient between x and y.

    Uses the formula from Chatterjee (2021):
        xi_N = 1 - (3 * sum |r_{i+1} - r_i|) / (N^2 - 1)

    where the data are sorted by x, and r_i is the rank of y_{(i)} in the
    sorted order.

    When x contains ties (common in insurance: binary flags, capped variables,
    large zero masses), the paper recommends averaging over multiple random
    tie-breaking runs. ``n_reps`` runs are averaged when ties are detected.

    Parameters
    ----------
    x : np.ndarray of shape (N,)
    y : np.ndarray of shape (N,)
    ties : {'random', 'average'}
        Tie-breaking strategy for x. 'random' (default) shuffles tied values
        before sorting; 'average' uses average ranks. The paper recommends
        'random' with averaging over n_reps runs.
    n_reps : int
        Number of repetitions when ties are present and ties='random'.
    rng : np.random.Generator or None
        Random number generator for tie-breaking. If None, uses default_rng(0).

    Returns
    -------
    float
        Xi value in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(x)
    if N < 4:
        return float("nan")

    y_ranks = rankdata(y, method="average")

    has_ties_x = len(np.unique(x)) < N

    if has_ties_x and ties == "random":
        # Average over n_reps random tie-breaking runs
        xi_vals = []
        for _ in range(n_reps):
            noise = rng.random(N)
            sort_order = np.lexsort((noise, x))
            r = y_ranks[sort_order]
            xi_val = 1.0 - (3.0 * np.sum(np.abs(np.diff(r)))) / (N ** 2 - 1.0)
            xi_vals.append(xi_val)
        return float(np.mean(xi_vals))
    else:
        # No ties or average rank method
        sort_order = np.argsort(x, kind="stable")
        r = y_ranks[sort_order]
        return float(1.0 - (3.0 * np.sum(np.abs(np.diff(r)))) / (N ** 2 - 1.0))


class ChatterjeeSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using Chatterjee's Xi coefficient.

    Screens features by computing xi(X_j, y) for each column X_j against the
    target y. Features with Xi below a threshold or outside the top-k are
    dropped. The direction is always feature -> target (not symmetric), which
    is the correct direction for feature screening.

    Parameters
    ----------
    k : int or None, default None
        Keep the top-k features by Xi score. If None, ``threshold`` is used
        instead. If both are specified, k takes precedence: the top-k features
        are kept provided their scores also exceed ``threshold``.
    threshold : float or None, default None
        Minimum Xi score for a feature to be retained. If both ``k`` and
        ``threshold`` are None, all features are kept (selector is a no-op
        but still computes and exposes scores via ``scores_``).
    ties : {'random', 'average'}, default 'random'
        Tie-breaking strategy for x values. Insurance data frequently contains
        binary flags and capped variables that produce many ties. 'random' with
        averaging (see ``n_reps``) is the approach recommended by Chatterjee.
    n_reps : int, default 20
        Number of random tie-breaking repetitions to average when ties are
        present. Higher values reduce variance at the cost of compute time.
        20 is sufficient for most datasets up to 100k rows.
    random_state : int or None, default 42
        Seed for reproducibility of tie-breaking.

    Attributes
    ----------
    scores_ : dict of str -> float
        Xi scores for each feature, keyed by column name (or integer index
        if X was passed as a numpy array). Available after ``fit()``.
    feature_names_in_ : np.ndarray of str
        Column names seen during fit, if X was a DataFrame.
    selected_features_ : list
        Feature names or indices retained by the selector.
    n_features_in_ : int
        Number of features seen during fit.

    Notes
    -----
    Xi is not symmetric: xi(X, Y) != xi(Y, X). The direction used here is
    xi(feature, target), which measures how well the feature predicts the
    target. This is the appropriate direction for feature screening.

    For binary or heavily zero-inflated targets (e.g. claim indicator), Xi
    is well-defined and handles the degenerate distribution correctly because
    it operates on ranks rather than values.

    References
    ----------
    Chatterjee, S. (2021). A new coefficient of correlation. JASA,
    116(536), 2009–2022.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame({'age': rng.uniform(18, 80, 500),
    ...                   'noise': rng.standard_normal(500)})
    >>> y = (X['age'] - 40) ** 2 + rng.standard_normal(500) * 10
    >>> sel = ChatterjeeSelector(k=1)
    >>> sel.fit(X, y)
    ChatterjeeSelector(k=1)
    >>> sel.transform(X).columns.tolist()
    ['age']
    """

    def __init__(
        self,
        k: int | None = None,
        threshold: float | None = None,
        ties: str = "random",
        n_reps: int = 20,
        random_state: int | None = 42,
    ) -> None:
        self.k = k
        self.threshold = threshold
        self.ties = ties
        self.n_reps = n_reps
        self.random_state = random_state

    def fit(
        self,
        X: "np.ndarray | object",
        y: "np.ndarray | object",
    ) -> "ChatterjeeSelector":
        """
        Compute Xi scores for all features against the target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Pandas or Polars DataFrames preserve column names;
            numpy arrays use integer indices as keys in ``scores_``.
        y : array-like of shape (n_samples,)
            Target variable. Passed directly to Xi without transformation.

        Returns
        -------
        self
        """
        # Extract column names if available
        feature_names: list[str] | None = None
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
                self.feature_names_in_ = np.array(feature_names, dtype=object)
                X_arr = X.to_numpy(dtype=float)
            else:
                X_arr = np.asarray(X, dtype=float)
        except ImportError:
            X_arr = np.asarray(X, dtype=float)

        if feature_names is None:
            try:
                import polars as pl
                if isinstance(X, pl.DataFrame):
                    feature_names = list(X.columns)
                    self.feature_names_in_ = np.array(feature_names, dtype=object)
                    X_arr = X.to_numpy()
            except ImportError:
                pass

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_arr.shape

        if len(y_arr) != n_samples:
            raise ValueError(
                f"X has {n_samples} rows but y has {len(y_arr)} elements."
            )

        self.n_features_in_ = n_features

        # Set up RNG for tie-breaking
        from sklearn.utils import check_random_state as _crs
        sklearn_rng = _crs(self.random_state)
        seed_val = sklearn_rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        scores: dict = {}
        for j in range(n_features):
            key = feature_names[j] if feature_names is not None else j
            xi_val = _xi(
                X_arr[:, j],
                y_arr,
                ties=self.ties,
                n_reps=self.n_reps,
                rng=np_rng,
            )
            scores[key] = xi_val

        self.scores_ = scores

        # Determine selected features
        all_keys = list(scores.keys())
        sorted_keys = sorted(all_keys, key=lambda k: scores[k], reverse=True)

        if self.k is not None:
            k_eff = min(self.k, n_features)
            candidates = sorted_keys[:k_eff]
            if self.threshold is not None:
                candidates = [f for f in candidates if scores[f] >= self.threshold]
            self.selected_features_ = candidates
        elif self.threshold is not None:
            self.selected_features_ = [
                f for f in all_keys if scores[f] >= self.threshold
            ]
        else:
            self.selected_features_ = all_keys

        return self

    def transform(
        self,
        X: "np.ndarray | object",
    ) -> "np.ndarray | object":
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must have the same features as were passed to ``fit()``.

        Returns
        -------
        X_reduced : same type as X
            Dataset with only the selected features. If X was a DataFrame,
            a DataFrame is returned. If X was a numpy array, a numpy array
            is returned (columns indexed by integer position).
        """
        check_is_fitted(self, "scores_")

        # Pandas path — return DataFrame subset
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                missing = set(self.selected_features_) - set(X.columns)
                if missing:
                    raise ValueError(
                        f"Columns present during fit but missing from X: {missing}"
                    )
                return X[self.selected_features_]
        except ImportError:
            pass

        # Polars path — return DataFrame subset
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                missing = set(self.selected_features_) - set(X.columns)
                if missing:
                    raise ValueError(
                        f"Columns present during fit but missing from X: {missing}"
                    )
                return X.select(self.selected_features_)
        except ImportError:
            pass

        # Numpy path — selected_features_ are integer indices
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        cols = [j for j in self.selected_features_ if isinstance(j, int)]
        return X_arr[:, cols]

    def get_feature_names_out(self) -> np.ndarray:
        """
        Return feature names for the selected features.

        Returns
        -------
        np.ndarray of str
        """
        check_is_fitted(self, "scores_")
        return np.array([str(f) for f in self.selected_features_], dtype=object)
