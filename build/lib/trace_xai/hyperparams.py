"""Automatic hyperparameter selection for TRACE surrogates.

Addresses the criticism that TRACE has 6+ hyperparameters with no
automated selection procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class HyperparamPreset:
    """Pre-defined hyperparameter configuration.

    Attributes
    ----------
    name : str
        Human-readable preset name.
    max_depth : int
        Maximum surrogate tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    ccp_alpha : float
        Cost-complexity pruning parameter.
    description : str
        Short description of when to use this preset.
    """

    name: str
    max_depth: int
    min_samples_leaf: int
    ccp_alpha: float
    description: str


# Pre-defined presets
PRESET_INTERPRETABLE = HyperparamPreset(
    name="interpretable",
    max_depth=3,
    min_samples_leaf=20,
    ccp_alpha=0.01,
    description="Maximum interpretability: shallow trees, few rules, easy to audit.",
)

PRESET_BALANCED = HyperparamPreset(
    name="balanced",
    max_depth=5,
    min_samples_leaf=10,
    ccp_alpha=0.005,
    description="Balance between fidelity and interpretability.",
)

PRESET_FAITHFUL = HyperparamPreset(
    name="faithful",
    max_depth=8,
    min_samples_leaf=5,
    ccp_alpha=0.0,
    description="Maximum fidelity: deeper trees, more rules, closer to black-box.",
)

PRESETS: Dict[str, HyperparamPreset] = {
    "interpretable": PRESET_INTERPRETABLE,
    "balanced": PRESET_BALANCED,
    "faithful": PRESET_FAITHFUL,
}


def get_preset(name: str) -> HyperparamPreset:
    """Return a named hyperparameter preset.

    Parameters
    ----------
    name : str
        One of ``"interpretable"``, ``"balanced"``, ``"faithful"``.

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown preset {name!r}. Available: {available}"
        )
    return PRESETS[name]


@dataclass(frozen=True)
class AutoDepthResult:
    """Result of automatic depth selection.

    Attributes
    ----------
    best_depth : int
        Selected max_depth.
    fidelity_scores : dict
        Mapping from max_depth to mean CV fidelity.
    selected_fidelity : float
        Mean CV fidelity at the selected depth.
    target_fidelity : float
        Target fidelity threshold used for selection.
    """

    best_depth: int
    fidelity_scores: Dict[int, float]
    selected_fidelity: float
    target_fidelity: float


def auto_select_depth(
    explainer,
    X: ArrayLike,
    *,
    y: Optional[ArrayLike] = None,
    min_depth: int = 2,
    max_depth: int = 10,
    target_fidelity: float = 0.85,
    n_folds: int = 5,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> AutoDepthResult:
    """Automatically select the minimum tree depth that achieves target fidelity.

    Uses cross-validated fidelity to find the shallowest tree
    (= most interpretable) that meets the target fidelity threshold.

    Parameters
    ----------
    explainer : Explainer
        Configured explainer instance.
    X : array-like
        Feature matrix.
    y : array-like, optional
        True labels (for accuracy metrics).
    min_depth : int, default 2
        Minimum depth to test.
    max_depth : int, default 10
        Maximum depth to test.
    target_fidelity : float, default 0.85
        Minimum acceptable mean fidelity.
    n_folds : int, default 5
        Number of CV folds.
    min_samples_leaf : int, default 5
        Minimum samples per leaf.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    AutoDepthResult
    """
    fidelity_scores: Dict[int, float] = {}

    for depth in range(min_depth, max_depth + 1):
        cv_report = explainer.cross_validate_fidelity(
            X, y=y, n_folds=n_folds,
            max_depth=depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        fidelity_scores[depth] = cv_report.mean_fidelity

        # Early stop: once we meet the target, this is the minimum depth
        if cv_report.mean_fidelity >= target_fidelity:
            return AutoDepthResult(
                best_depth=depth,
                fidelity_scores=fidelity_scores,
                selected_fidelity=cv_report.mean_fidelity,
                target_fidelity=target_fidelity,
            )

    # Target not met — return the depth with highest fidelity
    best_depth = max(fidelity_scores, key=lambda d: fidelity_scores[d])
    return AutoDepthResult(
        best_depth=best_depth,
        fidelity_scores=fidelity_scores,
        selected_fidelity=fidelity_scores[best_depth],
        target_fidelity=target_fidelity,
    )


@dataclass(frozen=True)
class SensitivityResult:
    """Result of hyperparameter sensitivity analysis.

    Each entry in ``results`` maps a configuration tuple to its metrics.
    """

    results: list[Dict]
    best_config: Dict
    best_fidelity: float


def sensitivity_analysis(
    explainer,
    X: ArrayLike,
    *,
    y: Optional[ArrayLike] = None,
    depth_range: Sequence[int] = (3, 5, 7),
    min_samples_leaf_range: Sequence[int] = (5, 10, 20),
    ccp_alpha_range: Sequence[float] = (0.0, 0.005, 0.01),
    n_folds: int = 3,
    random_state: int = 42,
) -> SensitivityResult:
    """Grid search over hyperparameters reporting fidelity and complexity.

    Parameters
    ----------
    explainer : Explainer
        Configured explainer instance.
    X : array-like
        Feature matrix.
    y : array-like, optional
        True labels.
    depth_range, min_samples_leaf_range, ccp_alpha_range : sequences
        Parameter grids.
    n_folds : int, default 3
        Number of CV folds.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    SensitivityResult
    """
    X = np.asarray(X)
    results: list[Dict] = []
    best_fidelity = -1.0
    best_config: Dict = {}

    for depth in depth_range:
        for min_leaf in min_samples_leaf_range:
            cv = explainer.cross_validate_fidelity(
                X, y=y, n_folds=n_folds,
                max_depth=depth, min_samples_leaf=min_leaf,
                random_state=random_state,
            )
            # Also get rule count from a single extraction
            result = explainer.extract_rules(
                X, max_depth=depth, min_samples_leaf=min_leaf,
            )
            entry = {
                "max_depth": depth,
                "min_samples_leaf": min_leaf,
                "mean_fidelity": cv.mean_fidelity,
                "std_fidelity": cv.std_fidelity,
                "num_rules": result.rules.num_rules,
                "avg_rule_length": result.rules.avg_conditions,
            }
            results.append(entry)

            if cv.mean_fidelity > best_fidelity:
                best_fidelity = cv.mean_fidelity
                best_config = entry

    return SensitivityResult(
        results=results,
        best_config=best_config,
        best_fidelity=best_fidelity,
    )


def compute_adaptive_tolerance(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    scale: float = 0.05,
) -> dict[str, float]:
    """Compute per-feature tolerance as a fraction of feature standard deviation.

    Addresses the criticism that the fuzzy matching tolerance delta
    is an arbitrary parameter with no guideline for selection.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    feature_names : sequence of str
        Feature names.
    scale : float, default 0.05
        Fraction of std to use as tolerance (e.g. 0.05 = 5% of std).

    Returns
    -------
    dict mapping feature name to tolerance value
    """
    stds = np.std(X, axis=0)
    tolerances = {}
    for i, name in enumerate(feature_names):
        tol = float(stds[i] * scale)
        tolerances[name] = max(tol, 1e-6)  # floor to avoid zero tolerance
    return tolerances
