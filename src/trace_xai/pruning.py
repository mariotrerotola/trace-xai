"""Regulatory rule pruning: simplify and filter extracted rules for audit compliance."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .ruleset import Condition, Rule, RuleSet


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for post-hoc rule pruning.

    Parameters
    ----------
    min_confidence : float
        Remove rules with confidence below this threshold (0-1).
    min_samples : int
        Remove rules covering fewer than this many samples.
    min_samples_fraction : float
        Remove rules covering less than this fraction of total samples (0-1).
        Requires ``total_samples`` to be set.
    max_conditions : int or None
        Truncate rules with more conditions than this limit.
    remove_redundant : bool
        Simplify redundant conditions on the same feature
        (e.g. ``A > 5 AND A > 3`` becomes ``A > 5``).
    total_samples : int or None
        Total number of samples in the dataset. Required for
        ``min_samples_fraction`` filtering.
    """

    min_confidence: float = 0.0
    min_samples: int = 0
    min_samples_fraction: float = 0.0
    max_conditions: Optional[int] = None
    remove_redundant: bool = False
    total_samples: Optional[int] = None


@dataclass(frozen=True)
class PruningReport:
    """Summary of what was removed or simplified during pruning."""

    original_count: int
    pruned_count: int
    removed_low_confidence: int
    removed_low_samples: int
    removed_over_max_conditions: int
    conditions_simplified: int


def prune_ruleset(
    ruleset: RuleSet,
    config: PruningConfig,
) -> tuple[RuleSet, PruningReport]:
    """Apply pruning filters to a RuleSet and return a new (pruned) RuleSet.

    The original RuleSet is never mutated.

    Returns
    -------
    tuple of (RuleSet, PruningReport)
    """
    removed_low_confidence = 0
    removed_low_samples = 0
    removed_over_max_conditions = 0
    conditions_simplified = 0

    min_abs_samples = config.min_samples
    if config.min_samples_fraction > 0 and config.total_samples is not None:
        fraction_threshold = int(config.min_samples_fraction * config.total_samples)
        min_abs_samples = max(min_abs_samples, fraction_threshold)

    kept: list[Rule] = []

    for rule in ruleset.rules:
        # 1. Confidence filter
        if rule.confidence < config.min_confidence:
            removed_low_confidence += 1
            continue

        # 2. Sample count filter
        if rule.samples < min_abs_samples:
            removed_low_samples += 1
            continue

        conditions = rule.conditions

        # 3. Simplify redundant conditions
        if config.remove_redundant:
            simplified = _simplify_conditions(conditions)
            if len(simplified) < len(conditions):
                conditions_simplified += len(conditions) - len(simplified)
                conditions = simplified

        # 4. Truncate to max_conditions
        if config.max_conditions is not None and len(conditions) > config.max_conditions:
            conditions = _truncate_conditions(conditions, config.max_conditions)

        # Build new rule if conditions changed
        if conditions is not rule.conditions:
            rule = Rule(
                conditions=conditions,
                prediction=rule.prediction,
                samples=rule.samples,
                confidence=rule.confidence,
                leaf_id=rule.leaf_id,
            )

        kept.append(rule)

    # Count rules removed due to max_conditions *after* truncation
    # (we truncate rather than remove, so this stays 0 unless we add a remove mode)
    pruned_ruleset = RuleSet(
        rules=tuple(kept),
        feature_names=ruleset.feature_names,
        class_names=ruleset.class_names,
    )

    report = PruningReport(
        original_count=ruleset.num_rules,
        pruned_count=pruned_ruleset.num_rules,
        removed_low_confidence=removed_low_confidence,
        removed_low_samples=removed_low_samples,
        removed_over_max_conditions=removed_over_max_conditions,
        conditions_simplified=conditions_simplified,
    )

    return pruned_ruleset, report


def _simplify_conditions(
    conditions: tuple[Condition, ...],
) -> tuple[Condition, ...]:
    """Remove redundant conditions on the same feature.

    For ``<=`` operators on the same feature, keep the tightest (smallest threshold).
    For ``>`` operators on the same feature, keep the tightest (largest threshold).
    """
    # Group by (feature, operator)
    groups: dict[tuple[str, str], list[Condition]] = defaultdict(list)
    for cond in conditions:
        groups[(cond.feature, cond.operator)].append(cond)

    simplified: list[Condition] = []
    for (feature, operator), conds in groups.items():
        if operator == "<=":
            # Keep only the smallest threshold (tightest upper bound)
            best = min(conds, key=lambda c: c.threshold)
        else:
            # operator == ">"  → keep the largest threshold (tightest lower bound)
            best = max(conds, key=lambda c: c.threshold)
        simplified.append(best)

    # Preserve original ordering: sort by first appearance in the original conditions
    original_order = {(c.feature, c.operator): i for i, c in enumerate(conditions)}
    simplified.sort(key=lambda c: original_order.get((c.feature, c.operator), 0))

    return tuple(simplified)


def _truncate_conditions(
    conditions: tuple[Condition, ...],
    max_conditions: int,
) -> tuple[Condition, ...]:
    """Keep only the first ``max_conditions`` conditions.

    Conditions closer to the root of the decision tree appear first and tend
    to be the most discriminative, so we keep the earliest ones.
    """
    return conditions[:max_conditions]
