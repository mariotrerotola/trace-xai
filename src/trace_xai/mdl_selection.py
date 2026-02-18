"""MDL-based rule selection using the Minimum Description Length principle.

Selects an optimal subset of rules by minimising the total description length:

    MDL = L(model) + L(data | model)

where L(model) is the cost in bits to encode the ruleset structure and
L(data | model) is the cost to encode misclassifications given the rules.

This provides an information-theoretic alternative to frequency-based ensemble
selection.  Rules that are expensive to describe (many conditions, rare
feature splits) and that poorly compress the data (high error rate on covered
samples) are pruned, yielding a compact, high-fidelity ruleset.

References
----------
Rissanen, J. (1978). Modeling by shortest data description.
    *Automatica*, 14(5), 465-471.
Grunwald, P. (2007). *The Minimum Description Length Principle*. MIT Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ruleset import Rule, RuleSet


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleMDLScore:
    """MDL cost decomposition for a single rule."""

    rule: Rule
    rule_index: int
    model_cost: float
    data_cost: float
    total_mdl: float
    coverage: int
    error_rate: float


@dataclass(frozen=True)
class MDLSelectionReport:
    """Result of MDL-based rule selection."""

    rule_scores: tuple[RuleMDLScore, ...]
    selected_ruleset: RuleSet
    selection_method: str
    total_mdl_before: float
    total_mdl_after: float
    mdl_reduction: float
    n_rules_original: int
    n_rules_selected: int
    precision_bits: int

    def __str__(self) -> str:
        return (
            f"=== MDL Selection Report ===\n"
            f"  Method: {self.selection_method}\n"
            f"  Precision bits: {self.precision_bits}\n"
            f"  Rules: {self.n_rules_original} -> {self.n_rules_selected}\n"
            f"  Total MDL: {self.total_mdl_before:.2f} -> {self.total_mdl_after:.2f} bits\n"
            f"  MDL reduction: {self.mdl_reduction:.2f} bits"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def binary_entropy(p: float) -> float:
    """Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).

    Returns 0.0 for p in {0, 1}.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def compute_rule_model_cost(
    rule: Rule,
    n_features: int,
    n_classes: int,
    *,
    precision_bits: int = 16,
) -> float:
    """Compute L(rule): bits to encode the rule structure.

    Formula: n_conditions * (log2(n_features) + precision_bits) + log2(n_classes)
    """
    n_conditions = len(rule.conditions)
    feature_bits = math.log2(max(n_features, 1)) if n_features > 0 else 0.0
    condition_cost = n_conditions * (feature_bits + precision_bits)
    prediction_cost = math.log2(max(n_classes, 1)) if n_classes > 1 else 0.0
    return condition_cost + prediction_cost


# ---------------------------------------------------------------------------
# Coverage and data cost
# ---------------------------------------------------------------------------

def _compute_rule_coverage_mask(
    rule: Rule,
    X: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Boolean mask of samples satisfying all conditions in a rule."""
    feature_index_map: dict[str, int] = {
        name: i for i, name in enumerate(feature_names)
    }
    mask = np.ones(len(X), dtype=bool)
    for cond in rule.conditions:
        feat_idx = feature_index_map.get(cond.feature)
        if feat_idx is None:
            continue
        if cond.operator == "<=":
            mask &= X[:, feat_idx] <= cond.threshold
        else:
            mask &= X[:, feat_idx] > cond.threshold
    return mask


def compute_rule_data_cost(
    rule: Rule,
    model,
    X: np.ndarray,
    feature_names: tuple[str, ...],
    class_names: tuple[str, ...],
) -> tuple[float, int, float]:
    """Compute L(data | rule): bits to encode misclassifications.

    Returns (data_cost, coverage, error_rate).
    """
    mask = _compute_rule_coverage_mask(rule, X, feature_names)
    coverage = int(mask.sum())

    if coverage == 0:
        return 0.0, 0, 0.0

    X_covered = X[mask]
    bb_preds = np.asarray(model.predict(X_covered))

    # rule.prediction is a class name string; bb_preds may be int indices
    if class_names:
        try:
            rule_class_idx = list(class_names).index(rule.prediction)
            agreement = bb_preds == rule_class_idx
        except ValueError:
            agreement = bb_preds.astype(str) == rule.prediction
    else:
        agreement = bb_preds.astype(str) == rule.prediction
    error_rate = 1.0 - float(np.mean(agreement))

    data_cost = coverage * binary_entropy(error_rate)
    return data_cost, coverage, error_rate


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ruleset_mdl(
    ruleset: RuleSet,
    model,
    X: np.ndarray,
    *,
    n_classes: int = 2,
    precision_bits: int = 16,
) -> tuple[RuleMDLScore, ...]:
    """Compute per-rule MDL scores for the entire RuleSet."""
    X = np.asarray(X)
    n_features = len(ruleset.feature_names)
    scores: list[RuleMDLScore] = []

    for i, rule in enumerate(ruleset.rules):
        model_cost = compute_rule_model_cost(
            rule, n_features, n_classes, precision_bits=precision_bits
        )
        data_cost, coverage, error_rate = compute_rule_data_cost(
            rule, model, X, ruleset.feature_names, ruleset.class_names,
        )
        scores.append(RuleMDLScore(
            rule=rule,
            rule_index=i,
            model_cost=model_cost,
            data_cost=data_cost,
            total_mdl=model_cost + data_cost,
            coverage=coverage,
            error_rate=error_rate,
        ))

    return tuple(scores)


# ---------------------------------------------------------------------------
# Selection algorithms
# ---------------------------------------------------------------------------

def _greedy_forward_selection(
    rule_scores: tuple[RuleMDLScore, ...],
    n_classes: int,
) -> list[int]:
    """Select rules greedily by ascending total MDL vs null cost."""
    null_cost_per_sample = math.log2(max(n_classes, 2))
    selected: list[int] = []

    for i, rs in sorted(enumerate(rule_scores), key=lambda t: t[1].total_mdl):
        if rs.coverage == 0:
            continue
        null_savings = null_cost_per_sample * rs.coverage
        if rs.total_mdl < null_savings:
            selected.append(i)

    return selected


def _greedy_backward_elimination(
    rule_scores: tuple[RuleMDLScore, ...],
    n_classes: int,
) -> list[int]:
    """Eliminate rules whose cost exceeds their null-hypothesis savings."""
    null_cost_per_sample = math.log2(max(n_classes, 2))
    selected = list(range(len(rule_scores)))

    changed = True
    while changed:
        changed = False
        best_gain = 0.0
        best_idx: Optional[int] = None

        for i in selected:
            rs = rule_scores[i]
            null_savings = null_cost_per_sample * rs.coverage
            gain = rs.total_mdl - null_savings  # positive → removing is beneficial
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx is not None:
            selected.remove(best_idx)
            changed = True

    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_rules_mdl(
    ruleset: RuleSet,
    model,
    X: np.ndarray,
    *,
    n_classes: int = 2,
    precision_bits: int = 16,
    method: str = "forward",
) -> MDLSelectionReport:
    """Select an optimal subset of rules by minimising total MDL.

    Parameters
    ----------
    ruleset : RuleSet
        The full RuleSet to select from.
    model : object
        The black-box model with a ``.predict()`` method.
    X : array-like of shape (n_samples, n_features)
        Data used to compute coverage and error rates.
    n_classes : int, default 2
        Number of output classes.
    precision_bits : int, default 16
        Bits per threshold for L(model) computation.
    method : str, default "forward"
        ``"forward"`` (greedy forward selection),
        ``"backward"`` (greedy backward elimination), or
        ``"score_only"`` (compute scores without selection).

    Returns
    -------
    MDLSelectionReport
    """
    valid_methods = ("forward", "backward", "score_only")
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got {method!r}"
        )

    X = np.asarray(X)
    rule_scores = score_ruleset_mdl(
        ruleset, model, X,
        n_classes=n_classes, precision_bits=precision_bits,
    )

    total_mdl_before = sum(rs.total_mdl for rs in rule_scores)

    if method == "forward":
        selected_indices = _greedy_forward_selection(rule_scores, n_classes)
    elif method == "backward":
        selected_indices = _greedy_backward_elimination(rule_scores, n_classes)
    else:  # score_only
        selected_indices = list(range(len(rule_scores)))

    selected_rules = tuple(rule_scores[i].rule for i in selected_indices)
    total_mdl_after = sum(rule_scores[i].total_mdl for i in selected_indices)

    selected_ruleset = RuleSet(
        rules=selected_rules,
        feature_names=ruleset.feature_names,
        class_names=ruleset.class_names,
    )

    return MDLSelectionReport(
        rule_scores=rule_scores,
        selected_ruleset=selected_ruleset,
        selection_method=method,
        total_mdl_before=total_mdl_before,
        total_mdl_after=total_mdl_after,
        mdl_reduction=total_mdl_before - total_mdl_after,
        n_rules_original=len(rule_scores),
        n_rules_selected=len(selected_rules),
        precision_bits=precision_bits,
    )
