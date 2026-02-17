"""Data classes for representing extracted rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Condition:
    """A single split condition, e.g. ``petal_width <= 0.8``."""

    feature: str
    operator: str  # "<=" or ">"
    threshold: float

    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


@dataclass(frozen=True)
class Rule:
    """An IF-THEN rule extracted from a decision-tree leaf.

    Parameters
    ----------
    conditions : tuple of Condition
        The conjunction of split predicates leading to this leaf.
    prediction : str
        The predicted class name (or formatted value for regression).
    samples : int
        Number of training samples that reached this leaf.
    confidence : float
        Fraction of the dominant class at this leaf (0-1).
        For regression rules this is always 1.0.
    leaf_id : int
        Node id inside the surrogate tree.
    prediction_value : float or None
        Numeric prediction value (regression only).
    """

    conditions: tuple[Condition, ...]
    prediction: str
    samples: int
    confidence: float
    leaf_id: int
    prediction_value: Optional[float] = None

    def __str__(self) -> str:
        if self.conditions:
            antecedent = " AND ".join(str(c) for c in self.conditions)
        else:
            antecedent = "TRUE"
        if self.prediction_value is not None:
            return (
                f"IF {antecedent} THEN value = {self.prediction_value:.4f}"
                f"  [samples={self.samples}]"
            )
        return (
            f"IF {antecedent} THEN class = {self.prediction}"
            f"  [confidence={self.confidence:.2%}, samples={self.samples}]"
        )


@dataclass(frozen=True)
class RuleSet:
    """An ordered collection of rules extracted from a surrogate tree."""

    rules: tuple[Rule, ...]
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]

    @property
    def num_rules(self) -> int:
        return len(self.rules)

    @property
    def avg_conditions(self) -> float:
        if not self.rules:
            return 0.0
        return sum(len(r.conditions) for r in self.rules) / len(self.rules)

    @property
    def max_conditions(self) -> int:
        if not self.rules:
            return 0
        return max(len(r.conditions) for r in self.rules)

    @property
    def avg_conditions_per_feature(self) -> float:
        """Average number of conditions per rule normalised by total features.

        Returns 0.0 when there are no rules or no feature names.
        """
        n_features = len(self.feature_names)
        if not self.rules or n_features == 0:
            return 0.0
        return self.avg_conditions / n_features

    @property
    def interaction_strength(self) -> float:
        """Fraction of rules that reference more than one distinct feature.

        A value close to 0 means the surrogate rarely combines features;
        a value close to 1 means most rules are multi-feature interactions.
        """
        if not self.rules:
            return 0.0
        multi = sum(
            1
            for r in self.rules
            if len({c.feature for c in r.conditions}) > 1
        )
        return multi / len(self.rules)

    def rule_signatures(self) -> frozenset:
        """Canonical string representations of each rule (for Jaccard similarity)."""
        sigs: list[str] = []
        for rule in self.rules:
            parts = sorted(str(c) for c in rule.conditions)
            sig = " AND ".join(parts) + " -> " + str(rule.prediction)
            sigs.append(sig)
        return frozenset(sigs)

    def filter_by_class(self, class_name: str) -> RuleSet:
        """Return a new RuleSet containing only rules for *class_name*."""
        filtered = tuple(r for r in self.rules if r.prediction == class_name)
        return RuleSet(
            rules=filtered,
            feature_names=self.feature_names,
            class_names=self.class_names,
        )

    def to_text(self) -> str:
        """Render every rule as a human-readable string."""
        lines: list[str] = []
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"Rule {i}: {rule}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()
