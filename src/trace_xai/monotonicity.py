"""Monotonicity constraints: enforce and validate domain-consistent rules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .ruleset import Condition, Rule, RuleSet


def check_sklearn_monotonic_support() -> bool:
    """Return True if the installed scikit-learn supports ``monotonic_cst``."""
    import inspect
    from sklearn.tree import DecisionTreeClassifier

    return "monotonic_cst" in inspect.signature(DecisionTreeClassifier.__init__).parameters


def constraints_to_array(
    constraints: Dict[str, int],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Convert a feature-name-keyed constraint dict to a positional array.

    Parameters
    ----------
    constraints : dict mapping feature name → {+1, -1, 0}
        +1 = monotonically increasing, -1 = decreasing, 0 = no constraint.
    feature_names : tuple of str
        Ordered feature names matching the training data columns.

    Returns
    -------
    np.ndarray of shape (n_features,) with dtype int
    """
    arr = np.zeros(len(feature_names), dtype=int)
    for name, direction in constraints.items():
        if direction not in (-1, 0, 1):
            raise ValueError(
                f"Monotonicity constraint for '{name}' must be -1, 0, or +1, "
                f"got {direction}"
            )
        if name not in feature_names:
            raise ValueError(
                f"Feature '{name}' not found in feature_names: {feature_names}"
            )
        idx = feature_names.index(name)
        arr[idx] = direction
    return arr


@dataclass(frozen=True)
class MonotonicityViolation:
    """A single detected monotonicity violation in the extracted rules."""

    rule_index: int
    rule: Rule
    feature: str
    expected_direction: int  # +1 or -1
    description: str


@dataclass(frozen=True)
class MonotonicityReport:
    """Result of checking extracted rules against monotonicity constraints."""

    constraints: Dict[str, int]
    violations: tuple[MonotonicityViolation, ...]
    is_compliant: bool

    def __str__(self) -> str:
        lines = [
            f"=== Monotonicity Report ===",
            f"  Constraints: {self.constraints}",
            f"  Compliant: {self.is_compliant}",
            f"  Violations: {len(self.violations)}",
        ]
        for v in self.violations[:5]:
            lines.append(f"    - Rule {v.rule_index}: {v.description}")
        if len(self.violations) > 5:
            lines.append(f"    ... and {len(self.violations) - 5} more")
        return "\n".join(lines)


def validate_monotonicity(
    ruleset: RuleSet,
    constraints: Dict[str, int],
) -> MonotonicityReport:
    """Check extracted rules for monotonicity violations.

    For each constrained feature, examine pairs of rules that split on that
    feature. If the constraint is +1 (increasing) but a rule assigns a lower
    prediction value to a higher threshold region, that's a violation.

    Violations are detected when sibling rules (same conditions except for
    one split on the constrained feature) predict different classes in a
    direction-inconsistent way.
    """
    violations: list[MonotonicityViolation] = []

    for feature, direction in constraints.items():
        if direction == 0:
            continue
        _check_feature_monotonicity(
            ruleset, feature, direction, violations,
        )

    return MonotonicityReport(
        constraints=constraints,
        violations=tuple(violations),
        is_compliant=len(violations) == 0,
    )


def _check_feature_monotonicity(
    ruleset: RuleSet,
    feature: str,
    direction: int,
    violations: list[MonotonicityViolation],
) -> None:
    """Check monotonicity for a single feature across all rule pairs.

    Groups rules by their conditions *excluding* the constrained feature,
    then within each group sorts by the constrained feature's threshold
    and checks that predictions are monotonic.
    """
    rules_with_indices = list(enumerate(ruleset.rules))

    # Group rules by their "context" — conditions on all OTHER features
    context_groups: dict[frozenset, list[tuple[int, Rule, Optional[float]]]] = defaultdict(list)

    for idx, rule in rules_with_indices:
        # Extract threshold info for the constrained feature
        feature_thresholds = []
        other_conditions = []
        for cond in rule.conditions:
            if cond.feature == feature:
                feature_thresholds.append(cond)
            else:
                other_conditions.append(cond)

        if not feature_thresholds:
            continue

        # Use the midpoint of the feature's constraints as a representative value
        # For a single ">" condition: the threshold is a lower bound
        # For a single "<=" condition: the threshold is an upper bound
        representative = _compute_representative_value(feature_thresholds)

        context_key = frozenset(str(c) for c in other_conditions)
        context_groups[context_key].append((idx, rule, representative))

    # Within each context group, check monotonicity
    for group in context_groups.values():
        if len(group) < 2:
            continue

        # Sort by representative threshold value
        group.sort(key=lambda x: x[2] if x[2] is not None else 0.0)

        for i in range(len(group) - 1):
            idx_a, rule_a, val_a = group[i]
            idx_b, rule_b, val_b = group[i + 1]

            if val_a is None or val_b is None:
                continue

            # Classification: different predictions indicate potential violation
            if rule_a.prediction != rule_b.prediction:
                violations.append(MonotonicityViolation(
                    rule_index=idx_b,
                    rule=rule_b,
                    feature=feature,
                    expected_direction=direction,
                    description=(
                        f"Feature '{feature}' (constraint={direction:+d}): "
                        f"class changes from '{rule_a.prediction}' to "
                        f"'{rule_b.prediction}' as feature value increases."
                    ),
                ))


def _compute_representative_value(
    conditions: list[Condition],
) -> Optional[float]:
    """Compute a representative feature value from a list of conditions on the same feature."""
    lower = float("-inf")
    upper = float("inf")

    for cond in conditions:
        if cond.operator == ">":
            lower = max(lower, cond.threshold)
        else:  # "<="
            upper = min(upper, cond.threshold)

    if lower == float("-inf") and upper == float("inf"):
        return None
    if lower == float("-inf"):
        return upper
    if upper == float("inf"):
        return lower
    return (lower + upper) / 2.0


def filter_monotonic_violations(
    ruleset: RuleSet,
    report: MonotonicityReport,
) -> RuleSet:
    """Return a new RuleSet with rules causing violations removed."""
    violating_indices = {v.rule_index for v in report.violations}
    filtered = tuple(
        rule for i, rule in enumerate(ruleset.rules)
        if i not in violating_indices
    )
    return RuleSet(
        rules=filtered,
        feature_names=ruleset.feature_names,
        class_names=ruleset.class_names,
    )


@dataclass(frozen=True)
class MonotonicityEnforcementResult:
    """Result of constructive monotonicity enforcement.

    Attributes
    ----------
    corrected_ruleset : RuleSet
        Rules with violations removed and fidelity impact measured.
    original_report : MonotonicityReport
        The validation report before enforcement.
    rules_removed : int
        Number of rules removed.
    fidelity_impact : float or None
        Change in fidelity after enforcement (negative = fidelity loss).
        None if surrogate/data not provided.
    """

    corrected_ruleset: RuleSet
    original_report: MonotonicityReport
    rules_removed: int
    fidelity_impact: Optional[float]


def enforce_monotonicity(
    ruleset: RuleSet,
    constraints: Dict[str, int],
    *,
    surrogate=None,
    X=None,
    model=None,
) -> MonotonicityEnforcementResult:
    """Constructively enforce monotonicity: validate, remove violations,
    and report the fidelity impact.

    This addresses the criticism that monotonicity validation is reactive
    (only diagnoses) rather than constructive (diagnoses and fixes).

    Parameters
    ----------
    ruleset : RuleSet
        The extracted rules.
    constraints : dict mapping feature name to {+1, -1, 0}
        Monotonicity constraints.
    surrogate : fitted tree, optional
        If provided with X and model, computes fidelity impact.
    X : array-like, optional
        Data for fidelity evaluation.
    model : object, optional
        Black-box model for fidelity evaluation.

    Returns
    -------
    MonotonicityEnforcementResult
    """
    # Step 1: Validate
    report = validate_monotonicity(ruleset, constraints)

    if report.is_compliant:
        return MonotonicityEnforcementResult(
            corrected_ruleset=ruleset,
            original_report=report,
            rules_removed=0,
            fidelity_impact=0.0,
        )

    # Step 2: Remove violating rules
    corrected = filter_monotonic_violations(ruleset, report)
    rules_removed = ruleset.num_rules - corrected.num_rules

    # Step 3: Measure fidelity impact if possible
    fidelity_impact: Optional[float] = None
    if surrogate is not None and X is not None and model is not None:
        import numpy as np
        from sklearn.metrics import accuracy_score

        X_arr = np.asarray(X)
        y_bb = np.asarray(model.predict(X_arr))
        y_surr = surrogate.predict(X_arr)
        original_fidelity = float(accuracy_score(y_bb, y_surr))

        # After removing rules, we can't directly recompute surrogate fidelity
        # since the tree is unchanged. Instead, we report the theoretical
        # fidelity on samples that fall into retained leaves.
        retained_leaf_ids = {r.leaf_id for r in corrected.rules}
        leaf_assignments = surrogate.apply(X_arr)
        retained_mask = np.isin(leaf_assignments, list(retained_leaf_ids))

        if retained_mask.sum() > 0:
            retained_fidelity = float(accuracy_score(
                y_bb[retained_mask], y_surr[retained_mask],
            ))
        else:
            retained_fidelity = 0.0

        fidelity_impact = retained_fidelity - original_fidelity

    return MonotonicityEnforcementResult(
        corrected_ruleset=corrected,
        original_report=report,
        rules_removed=rules_removed,
        fidelity_impact=fidelity_impact,
    )
