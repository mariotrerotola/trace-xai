"""Fidelity report: how well the surrogate tree mimics the black-box model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# ── Confidence Interval ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfidenceInterval:
    """A bootstrap confidence interval for a metric."""

    lower: float
    upper: float
    point_estimate: float
    confidence_level: float


def compute_bootstrap_ci(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute a percentile bootstrap confidence interval from resampled values."""
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    point = float(np.mean(values))
    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=point,
        confidence_level=confidence_level,
    )


# ── Fidelity Report (classification) ────────────────────────────────────

@dataclass(frozen=True)
class FidelityReport:
    """Quantitative summary of surrogate faithfulness.

    Attributes
    ----------
    fidelity : float
        Agreement between the surrogate and the black-box predictions (0-1).
    accuracy : float or None
        Surrogate accuracy against the true labels (None if y_true unavailable).
    blackbox_accuracy : float or None
        Black-box accuracy against the true labels (None if y_true unavailable).
    num_rules : int
        Number of leaves (= rules) in the surrogate tree.
    avg_rule_length : float
        Average number of conditions per rule.
    max_rule_length : int
        Maximum number of conditions across all rules.
    surrogate_depth : int
        Depth of the fitted surrogate tree.
    surrogate_n_leaves : int
        Number of leaves in the surrogate tree.
    num_samples : int
        Number of samples used to fit the surrogate.
    class_fidelity : dict
        Per-class fidelity (agreement on samples where the black-box predicts
        that class).
    evaluation_type : str
        ``"in_sample"``, ``"hold_out"``, or ``"validation_split"``.
    avg_conditions_per_feature : float or None
        Average conditions per rule normalised by feature count.
    interaction_strength : float or None
        Fraction of rules that use more than one distinct feature.
    fidelity_ci : ConfidenceInterval or None
        Bootstrap CI for fidelity (populated by ``compute_confidence_intervals``).
    accuracy_ci : ConfidenceInterval or None
        Bootstrap CI for accuracy (populated by ``compute_confidence_intervals``).
    fidelity_r2 : float or None
        R² between surrogate and black-box (regression only).
    fidelity_mse : float or None
        MSE between surrogate and black-box (regression only).
    accuracy_r2 : float or None
        R² of surrogate vs true labels (regression only).
    accuracy_mse : float or None
        MSE of surrogate vs true labels (regression only).
    """

    fidelity: float
    accuracy: Optional[float]
    blackbox_accuracy: Optional[float]
    num_rules: int
    avg_rule_length: float
    max_rule_length: int
    surrogate_depth: int
    surrogate_n_leaves: int
    num_samples: int
    class_fidelity: Dict[str, float]
    # ---- new fields (all with defaults for backward compat) ----
    evaluation_type: str = "in_sample"
    avg_conditions_per_feature: Optional[float] = None
    interaction_strength: Optional[float] = None
    fidelity_ci: Optional[ConfidenceInterval] = None
    accuracy_ci: Optional[ConfidenceInterval] = None
    fidelity_r2: Optional[float] = None
    fidelity_mse: Optional[float] = None
    accuracy_r2: Optional[float] = None
    accuracy_mse: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            "=== Fidelity Report ===",
            f"  Evaluation type: {self.evaluation_type}",
            f"  Fidelity (surrogate vs black-box): {self.fidelity:.4f}",
        ]
        if self.accuracy is not None:
            lines.append(f"  Surrogate accuracy (vs true labels): {self.accuracy:.4f}")
        if self.blackbox_accuracy is not None:
            lines.append(f"  Black-box accuracy (vs true labels): {self.blackbox_accuracy:.4f}")
        if self.fidelity_r2 is not None:
            lines.append(f"  Fidelity R²: {self.fidelity_r2:.4f}")
        if self.fidelity_mse is not None:
            lines.append(f"  Fidelity MSE: {self.fidelity_mse:.4f}")
        if self.accuracy_r2 is not None:
            lines.append(f"  Surrogate R² (vs true labels): {self.accuracy_r2:.4f}")
        if self.accuracy_mse is not None:
            lines.append(f"  Surrogate MSE (vs true labels): {self.accuracy_mse:.4f}")
        lines += [
            f"  Number of rules: {self.num_rules}",
            f"  Avg rule length: {self.avg_rule_length:.2f}",
            f"  Max rule length: {self.max_rule_length}",
            f"  Surrogate depth: {self.surrogate_depth}",
            f"  Surrogate leaves: {self.surrogate_n_leaves}",
            f"  Samples used: {self.num_samples}",
        ]
        if self.avg_conditions_per_feature is not None:
            lines.append(f"  Avg conditions/feature: {self.avg_conditions_per_feature:.4f}")
        if self.interaction_strength is not None:
            lines.append(f"  Interaction strength: {self.interaction_strength:.4f}")
        if self.fidelity_ci is not None:
            ci = self.fidelity_ci
            lines.append(
                f"  Fidelity CI ({ci.confidence_level:.0%}): "
                f"[{ci.lower:.4f}, {ci.upper:.4f}]"
            )
        if self.accuracy_ci is not None:
            ci = self.accuracy_ci
            lines.append(
                f"  Accuracy CI ({ci.confidence_level:.0%}): "
                f"[{ci.lower:.4f}, {ci.upper:.4f}]"
            )
        if self.class_fidelity:
            lines.append("  Per-class fidelity:")
            for cls, fid in self.class_fidelity.items():
                lines.append(f"    {cls}: {fid:.4f}")
        return "\n".join(lines)


# ── CV Fidelity Report ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CVFidelityReport:
    """Aggregated cross-validated fidelity report."""

    mean_fidelity: float
    std_fidelity: float
    mean_accuracy: Optional[float]
    std_accuracy: Optional[float]
    fold_reports: List[FidelityReport]
    n_folds: int

    def __str__(self) -> str:
        lines = [
            f"=== Cross-Validated Fidelity ({self.n_folds}-fold) ===",
            f"  Mean fidelity: {self.mean_fidelity:.4f} ± {self.std_fidelity:.4f}",
        ]
        if self.mean_accuracy is not None and self.std_accuracy is not None:
            lines.append(
                f"  Mean accuracy:  {self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}"
            )
        return "\n".join(lines)


# ── Stability Report ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StabilityReport:
    """Bootstrap stability analysis of extracted rules."""

    mean_jaccard: float
    std_jaccard: float
    pairwise_jaccards: List[float]
    n_bootstraps: int

    def __str__(self) -> str:
        return (
            f"=== Stability Report ({self.n_bootstraps} bootstraps) ===\n"
            f"  Mean Jaccard: {self.mean_jaccard:.4f} ± {self.std_jaccard:.4f}"
        )


# ── Factory functions ───────────────────────────────────────────────────

def compute_fidelity_report(
    surrogate,
    X: ArrayLike,
    y_bb: np.ndarray,
    y_true: Optional[np.ndarray],
    class_names: tuple[str, ...],
    num_rules: int,
    avg_rule_length: float,
    max_rule_length: int,
    *,
    evaluation_type: str = "in_sample",
    avg_conditions_per_feature: Optional[float] = None,
    interaction_strength: Optional[float] = None,
) -> FidelityReport:
    """Compute all fidelity metrics (classification).

    Parameters
    ----------
    surrogate
        The fitted surrogate tree (classifier).
    X : array-like
        Feature matrix used for evaluation.
    y_bb : ndarray
        Black-box predictions (integer-encoded).
    y_true : ndarray or None
        Ground-truth labels (integer-encoded), if available.
    class_names : tuple of str
        Human-readable class names.
    num_rules, avg_rule_length, max_rule_length
        Rule-complexity statistics (already computed from RuleSet).
    evaluation_type : str
        How the data was split for evaluation.
    avg_conditions_per_feature : float or None
        Normalised complexity metric.
    interaction_strength : float or None
        Multi-feature interaction metric.
    """
    X = np.asarray(X)
    y_surr = surrogate.predict(X)

    fidelity = accuracy_score(y_bb, y_surr)

    accuracy: Optional[float] = None
    blackbox_accuracy: Optional[float] = None
    if y_true is not None:
        accuracy = accuracy_score(y_true, y_surr)
        blackbox_accuracy = accuracy_score(y_true, y_bb)

    # Per-class fidelity
    class_fidelity: dict[str, float] = {}
    unique_bb = np.unique(y_bb)
    for cls_idx in unique_bb:
        mask = y_bb == cls_idx
        if mask.sum() == 0:
            continue
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        class_fidelity[name] = accuracy_score(y_bb[mask], y_surr[mask])

    return FidelityReport(
        fidelity=fidelity,
        accuracy=accuracy,
        blackbox_accuracy=blackbox_accuracy,
        num_rules=num_rules,
        avg_rule_length=avg_rule_length,
        max_rule_length=max_rule_length,
        surrogate_depth=surrogate.get_depth(),
        surrogate_n_leaves=surrogate.get_n_leaves(),
        num_samples=len(X),
        class_fidelity=class_fidelity,
        evaluation_type=evaluation_type,
        avg_conditions_per_feature=avg_conditions_per_feature,
        interaction_strength=interaction_strength,
    )


def compute_regression_fidelity_report(
    surrogate,
    X: ArrayLike,
    y_bb: np.ndarray,
    y_true: Optional[np.ndarray],
    num_rules: int,
    avg_rule_length: float,
    max_rule_length: int,
    *,
    evaluation_type: str = "in_sample",
    avg_conditions_per_feature: Optional[float] = None,
    interaction_strength: Optional[float] = None,
) -> FidelityReport:
    """Compute fidelity metrics for a regression surrogate."""
    X = np.asarray(X)
    y_surr = surrogate.predict(X)

    fidelity_r2 = float(r2_score(y_bb, y_surr))
    fidelity_mse = float(mean_squared_error(y_bb, y_surr))
    # Use 1 - normalised MSE as a [0,1] fidelity score
    var_bb = float(np.var(y_bb))
    fidelity = fidelity_r2 if var_bb > 0 else 1.0

    accuracy_r2: Optional[float] = None
    accuracy_mse: Optional[float] = None
    accuracy: Optional[float] = None
    blackbox_accuracy: Optional[float] = None
    if y_true is not None:
        accuracy_r2 = float(r2_score(y_true, y_surr))
        accuracy_mse = float(mean_squared_error(y_true, y_surr))
        accuracy = accuracy_r2
        blackbox_r2 = float(r2_score(y_true, y_bb))
        blackbox_accuracy = blackbox_r2

    return FidelityReport(
        fidelity=fidelity,
        accuracy=accuracy,
        blackbox_accuracy=blackbox_accuracy,
        num_rules=num_rules,
        avg_rule_length=avg_rule_length,
        max_rule_length=max_rule_length,
        surrogate_depth=surrogate.get_depth(),
        surrogate_n_leaves=surrogate.get_n_leaves(),
        num_samples=len(X),
        class_fidelity={},
        evaluation_type=evaluation_type,
        avg_conditions_per_feature=avg_conditions_per_feature,
        interaction_strength=interaction_strength,
        fidelity_r2=fidelity_r2,
        fidelity_mse=fidelity_mse,
        accuracy_r2=accuracy_r2,
        accuracy_mse=accuracy_mse,
    )
