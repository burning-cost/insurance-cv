"""
insurance-cv: Temporal and distributional cross-validation for insurance
pricing models.

Walk-forward splits that respect policy year, accident year, and IBNR
development structure. Distributional splits that ensure rare events are
proportionally represented in test sets. Feature screening via Chatterjee's
Xi coefficient for nonlinear dependence detection.

All splitters are drop-in replacements for sklearn CV splitters wherever
you need time-aware or distribution-aware splits.
"""

from .splits import (
    InsuranceCV,
    TemporalSplit,
    walk_forward_split,
    policy_year_split,
    accident_year_split,
)
from .diagnostics import temporal_leakage_check, split_summary
from .distributional import SupportPointSplit
from .feature_selection import ChatterjeeSelector

__all__ = [
    "InsuranceCV",
    "TemporalSplit",
    "walk_forward_split",
    "policy_year_split",
    "accident_year_split",
    "temporal_leakage_check",
    "split_summary",
    "SupportPointSplit",
    "ChatterjeeSelector",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-cv")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
