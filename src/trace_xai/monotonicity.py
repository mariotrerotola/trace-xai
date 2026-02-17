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

    Works for regression rules (compares ``prediction_value``). For
    classification rules, violations are detected when sibling rules
    (same conditions except for one split on the constrained feature) predict
    different classes in a direction-inconsistent way.
    """
    violations: list[MonotonicityViolation] = []
    is_regression = any(r.prediction_value is not None for r in ruleset.rules)

    for feature, direction in constraints.items():
        if direction == 0:
            continue
        _check_feature_monotonicity(
            ruleset, feature, direction, is_regression, violations,
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
    is_regression: bool,
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

            if is_regression:
                pred_a = rule_a.prediction_value or 0.0
                pred_b = rule_b.prediction_value or 0.0

                if direction == 1 and pred_b < pred_a:
                    violations.append(MonotonicityViolation(
                        rule_index=idx_b,
                        rule=rule_b,
                        feature=feature,
                        expected_direction=direction,
                        description=(
                            f"Feature '{feature}' should be increasing, but "
                            f"prediction drops from {pred_a:.4f} to {pred_b:.4f} "
                            f"as feature value increases."
                        ),
                    ))
                elif direction == -1 and pred_b > pred_a:
                    violations.append(MonotonicityViolation(
                        rule_index=idx_b,
                        rule=rule_b,
                        feature=feature,
                        expected_direction=direction,
                        description=(
                            f"Feature '{feature}' should be decreasing, but "
                            f"prediction rises from {pred_a:.4f} to {pred_b:.4f} "
                            f"as feature value increases."
                        ),
                    ))
            else:
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
