"""
insurance-cv: Temporal cross-validation for insurance pricing models.

Walk-forward splits that respect policy year, accident year, and IBNR
development structure. Drop-in replacement for sklearn CV splitters
wherever you need time-aware splits.
"""

from .splits import (
    InsuranceCV,
    TemporalSplit,
    walk_forward_split,
    policy_year_split,
    accident_year_split,
)
from .diagnostics import temporal_leakage_check, split_summary

__all__ = [
    "InsuranceCV",
    "TemporalSplit",
    "walk_forward_split",
    "policy_year_split",
    "accident_year_split",
    "temporal_leakage_check",
    "split_summary",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-cv")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
