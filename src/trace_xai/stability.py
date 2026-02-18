"""Structural stability metrics beyond syntactic Jaccard similarity.

Addresses the criticism that fuzzy Jaccard is 0.00 on 4/5 datasets because
it relies on exact syntactic matching of rule strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .ruleset import RuleSet


@dataclass(frozen=True)
class StructuralStabilityReport:
    """Comprehensive stability report using multiple metrics.

    Attributes
    ----------
    mean_coverage_overlap : float
        Average pairwise coverage overlap between bootstrap rulesets (0-1).
    std_coverage_overlap : float
        Std of pairwise coverage overlaps.
    mean_prediction_agreement : float
        Average pairwise prediction agreement on the data (0-1).
    std_prediction_agreement : float
        Std of pairwise prediction agreements.
    feature_importance_stability : float
        Kendall's tau correlation of feature importance rankings (0-1).
    top_k_feature_agreement : float
        Fraction of top-k features shared across bootstraps.
    n_bootstraps : int
        Number of bootstrap surrogates used.
    """

    mean_coverage_overlap: float
    std_coverage_overlap: float
    mean_prediction_agreement: float
    std_prediction_agreement: float
    feature_importance_stability: float
    top_k_feature_agreement: float
    n_bootstraps: int

    def __str__(self) -> str:
        return (
            f"=== Structural Stability Report ({self.n_bootstraps} bootstraps) ===\n"
            f"  Coverage overlap: {self.mean_coverage_overlap:.4f} ± {self.std_coverage_overlap:.4f}\n"
            f"  Prediction agreement: {self.mean_prediction_agreement:.4f} ± {self.std_prediction_agreement:.4f}\n"
            f"  Feature importance stability (tau): {self.feature_importance_stability:.4f}\n"
            f"  Top-k feature agreement: {self.top_k_feature_agreement:.4f}"
        )


def compute_structural_stability(
    explainer,
    X: ArrayLike,
    *,
    n_bootstraps: int = 20,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    top_k: int = 3,
    random_state: int = 42,
) -> StructuralStabilityReport:
    """Compute structural stability via coverage overlap, prediction agreement,
    and feature importance consistency.

    Parameters
    ----------
    explainer : Explainer
        Configured explainer instance.
    X : array-like of shape (n_samples, n_features)
        Data to evaluate stability on.
    n_bootstraps : int, default 20
        Number of bootstrap surrogates.
    max_depth : int, default 5
        Tree depth.
    min_samples_leaf : int, default 5
        Minimum leaf samples.
    top_k : int, default 3
        Number of top features to compare for agreement.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    StructuralStabilityReport
    """
    X = np.asarray(X)
    rng = np.random.RandomState(random_state)

    predictions_list: list[np.ndarray] = []
    feature_importances_list: list[np.ndarray] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot = X[idx]
        y_bb_boot = np.asarray(explainer.model.predict(X_boot))

        surr = explainer._build_surrogate(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        surr.fit(X_boot, y_bb_boot)

        # Predictions on full X (not bootstrap sample)
        preds = surr.predict(X)
        predictions_list.append(preds)

        # Feature importances
        feature_importances_list.append(surr.feature_importances_)

    # 1. Pairwise prediction agreement
    agreements: list[float] = []
    for a, b in combinations(range(n_bootstraps), 2):
        agreement = np.mean(predictions_list[a] == predictions_list[b])
        agreements.append(float(agreement))

    agreements_arr = np.array(agreements) if agreements else np.array([1.0])

    # 2. Coverage overlap (leaf assignment overlap)
    # Two surrogates "cover" the same point similarly if they predict the same
    # This is captured by prediction agreement above. For a more structural
    # measure, we can use the decision path overlap.
    coverage_overlaps = agreements  # functional coverage = prediction agreement

    # 3. Feature importance stability
    importances_matrix = np.array(feature_importances_list)
    fi_stability = _compute_feature_rank_stability(importances_matrix)

    # 4. Top-k feature agreement
    top_k_agreement = _compute_top_k_agreement(importances_matrix, top_k)

    cov_arr = np.array(coverage_overlaps) if coverage_overlaps else np.array([1.0])
    return StructuralStabilityReport(
        mean_coverage_overlap=float(cov_arr.mean()),
        std_coverage_overlap=float(cov_arr.std()),
        mean_prediction_agreement=float(agreements_arr.mean()),
        std_prediction_agreement=float(agreements_arr.std()),
        feature_importance_stability=fi_stability,
        top_k_feature_agreement=top_k_agreement,
        n_bootstraps=n_bootstraps,
    )


def _compute_feature_rank_stability(importances: np.ndarray) -> float:
    """Compute average pairwise Kendall's tau of feature importance rankings.

    Parameters
    ----------
    importances : ndarray of shape (n_bootstraps, n_features)

    Returns
    -------
    float
        Mean Kendall's tau (0-1 after normalization to [0,1]).
    """
    n_bootstraps = importances.shape[0]
    if n_bootstraps < 2:
        return 1.0

    # Rank features (higher importance = lower rank number)
    ranks = np.zeros_like(importances)
    for i in range(n_bootstraps):
        order = np.argsort(-importances[i])
        ranks[i, order] = np.arange(importances.shape[1])

    taus: list[float] = []
    for a, b in combinations(range(n_bootstraps), 2):
        tau = _kendall_tau(ranks[a], ranks[b])
        taus.append(tau)

    # Normalize from [-1, 1] to [0, 1]
    mean_tau = np.mean(taus)
    return float((mean_tau + 1.0) / 2.0)


def _kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Compute Kendall's tau-b between two rank arrays."""
    n = len(rank_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = rank_a[i] - rank_a[j]
            diff_b = rank_b[i] - rank_b[j]
            prod = diff_a * diff_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def _compute_top_k_agreement(importances: np.ndarray, k: int) -> float:
    """Fraction of top-k features shared across all bootstrap surrogates.

    Parameters
    ----------
    importances : ndarray of shape (n_bootstraps, n_features)
    k : int
        Number of top features.

    Returns
    -------
    float
        Mean pairwise Jaccard of top-k feature sets (0-1).
    """
    n_bootstraps = importances.shape[0]
    k = min(k, importances.shape[1])

    top_k_sets: list[frozenset] = []
    for i in range(n_bootstraps):
        top_indices = frozenset(np.argsort(-importances[i])[:k])
        top_k_sets.append(top_indices)

    if n_bootstraps < 2:
        return 1.0

    jaccards: list[float] = []
    for a, b in combinations(range(n_bootstraps), 2):
        inter = len(top_k_sets[a] & top_k_sets[b])
        union = len(top_k_sets[a] | top_k_sets[b])
        jaccards.append(inter / union if union > 0 else 1.0)

    return float(np.mean(jaccards))
