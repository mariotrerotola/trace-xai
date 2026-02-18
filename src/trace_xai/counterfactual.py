"""Counterfactual-guided rule scoring.

Scores each extracted rule by probing whether the black-box model's decision
boundaries align with the surrogate's split thresholds.  For every condition
in a rule the algorithm generates a paired sample straddling the threshold and
checks whether the black-box also changes its prediction.  Rules whose
conditions correspond to real black-box boundaries receive high scores; rules
with "phantom" splits that the black-box ignores receive low scores and can
optionally be filtered out.

This addresses the fundamental limitation of surrogate-based explainers: a
high-fidelity surrogate may still place individual splits at locations that do
not reflect meaningful black-box decision boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ruleset import Condition, Rule, RuleSet


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionValidity:
    """Validity assessment for a single rule condition."""

    condition: Condition
    is_valid: bool
    bb_changes: bool
    delta_below: float
    delta_above: float


@dataclass(frozen=True)
class RuleCounterfactualScore:
    """Counterfactual validity score for a single rule.

    Attributes
    ----------
    score : float
        Fraction of conditions that are counterfactually valid (0.0-1.0).
    """

    rule: Rule
    rule_index: int
    score: float
    condition_validities: tuple[ConditionValidity, ...]
    n_conditions: int
    n_valid_conditions: int


@dataclass(frozen=True)
class CounterfactualReport:
    """Result of counterfactual validity scoring over a full RuleSet."""

    rule_scores: tuple[RuleCounterfactualScore, ...]
    filtered_ruleset: Optional[RuleSet]
    validity_threshold: Optional[float]
    mean_score: float
    std_score: float
    n_rules_total: int
    n_rules_retained: int
    noise_scale: float
    random_state: int

    def __str__(self) -> str:
        lines = [
            "=== Counterfactual Validity Report ===",
            f"  Rules scored: {self.n_rules_total}",
            f"  Mean validity score: {self.mean_score:.4f} \u00b1 {self.std_score:.4f}",
        ]
        if self.validity_threshold is not None:
            lines.append(f"  Validity threshold: {self.validity_threshold:.2f}")
            lines.append(
                f"  Rules retained: {self.n_rules_retained}/{self.n_rules_total}"
            )
        lines.append(f"  Noise scale: {self.noise_scale}")
        for rs in self.rule_scores[:5]:
            lines.append(
                f"  Rule {rs.rule_index}: score={rs.score:.2f} "
                f"({rs.n_valid_conditions}/{rs.n_conditions} valid)"
            )
        if len(self.rule_scores) > 5:
            lines.append(f"  ... and {len(self.rule_scores) - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def score_rule_counterfactual(
    rule: Rule,
    rule_index: int,
    model,
    X: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    noise_scale: float = 0.01,
    n_probes: int = 20,
    rng: np.random.RandomState,
) -> RuleCounterfactualScore:
    """Score a single rule by counterfactual boundary probing.

    For each condition, generates *n_probes* paired samples straddling the
    threshold, queries the black-box on both sides, and records whether the
    prediction changed in at least one probe.  Using multiple probes makes the
    method robust to ensemble models whose boundaries are not axis-aligned.

    Parameters
    ----------
    n_probes : int, default 20
        Number of random probe pairs per condition.  A condition is considered
        valid if the black-box prediction changes in at least one probe.
    """
    if not rule.conditions:
        return RuleCounterfactualScore(
            rule=rule,
            rule_index=rule_index,
            score=1.0,
            condition_validities=(),
            n_conditions=0,
            n_valid_conditions=0,
        )

    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)

    feature_index_map: dict[str, int] = {
        name: i for i, name in enumerate(feature_names)
    }

    condition_validities: list[ConditionValidity] = []

    for cond in rule.conditions:
        feat_idx = feature_index_map.get(cond.feature)
        if feat_idx is None:
            # Feature not found — mark as invalid (safe default)
            condition_validities.append(
                ConditionValidity(cond, False, False, 0.0, 0.0)
            )
            continue

        delta = noise_scale * feature_std[feat_idx]
        delta_below = float(np.clip(
            cond.threshold - delta, feature_mins[feat_idx], feature_maxs[feat_idx]
        ))
        delta_above = float(np.clip(
            cond.threshold + delta, feature_mins[feat_idx], feature_maxs[feat_idx]
        ))

        # Edge case: threshold at feature boundary — can't probe both sides
        if delta_below == delta_above:
            condition_validities.append(
                ConditionValidity(cond, False, False, delta_below, delta_above)
            )
            continue

        # Generate n_probes random base points and probe each
        bb_changes = False
        for _ in range(n_probes):
            base = rng.uniform(feature_mins, feature_maxs)

            sample_below = base.copy()
            sample_below[feat_idx] = delta_below

            sample_above = base.copy()
            sample_above[feat_idx] = delta_above

            pair = np.vstack([sample_below.reshape(1, -1), sample_above.reshape(1, -1)])
            bb_preds = np.asarray(model.predict(pair))

            if bb_preds[0] != bb_preds[1]:
                bb_changes = True
                break

        condition_validities.append(
            ConditionValidity(
                condition=cond,
                is_valid=bb_changes,
                bb_changes=bb_changes,
                delta_below=delta_below,
                delta_above=delta_above,
            )
        )

    n_conditions = len(condition_validities)
    n_valid = sum(1 for cv in condition_validities if cv.is_valid)
    score = n_valid / n_conditions if n_conditions > 0 else 1.0

    return RuleCounterfactualScore(
        rule=rule,
        rule_index=rule_index,
        score=score,
        condition_validities=tuple(condition_validities),
        n_conditions=n_conditions,
        n_valid_conditions=n_valid,
    )


def score_rules_counterfactual(
    ruleset: RuleSet,
    model,
    X: np.ndarray,
    *,
    validity_threshold: Optional[float] = None,
    noise_scale: float = 0.01,
    n_probes: int = 20,
    random_state: int = 42,
) -> CounterfactualReport:
    """Score all rules in a RuleSet by counterfactual validity.

    Parameters
    ----------
    ruleset : RuleSet
        The extracted rules to score.
    model : object
        The black-box model with a ``.predict()`` method.
    X : array-like of shape (n_samples, n_features)
        Data used to determine feature ranges.
    validity_threshold : float or None, default None
        If given, rules with score < threshold are filtered out.
    noise_scale : float, default 0.01
        Relative perturbation around each threshold.
    n_probes : int, default 20
        Number of random probe pairs per condition.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    CounterfactualReport
    """
    X = np.asarray(X)
    rng = np.random.RandomState(random_state)

    rule_scores: list[RuleCounterfactualScore] = []
    for i, rule in enumerate(ruleset.rules):
        score = score_rule_counterfactual(
            rule, i, model, X, ruleset.feature_names,
            noise_scale=noise_scale, n_probes=n_probes, rng=rng,
        )
        rule_scores.append(score)

    scores_arr = np.array([rs.score for rs in rule_scores])
    mean_score = float(scores_arr.mean()) if rule_scores else 0.0
    std_score = float(scores_arr.std()) if rule_scores else 0.0

    filtered_ruleset: Optional[RuleSet] = None
    n_retained = len(rule_scores)

    if validity_threshold is not None:
        retained_rules = [
            rule_scores[i].rule
            for i in range(len(rule_scores))
            if rule_scores[i].score >= validity_threshold
        ]
        n_retained = len(retained_rules)
        filtered_ruleset = RuleSet(
            rules=tuple(retained_rules),
            feature_names=ruleset.feature_names,
            class_names=ruleset.class_names,
        )

    return CounterfactualReport(
        rule_scores=tuple(rule_scores),
        filtered_ruleset=filtered_ruleset,
        validity_threshold=validity_threshold,
        mean_score=mean_score,
        std_score=std_score,
        n_rules_total=len(rule_scores),
        n_rules_retained=n_retained,
        noise_scale=noise_scale,
        random_state=random_state,
    )
