"""Ensemble rule extraction: stabilise rules via bagging of surrogate trees."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Union

import numpy as np

from .ruleset import Rule, RuleSet


@dataclass(frozen=True)
class StableRule:
    """A rule that appeared frequently across bootstrap surrogates."""

    rule: Rule
    frequency: float       # Fraction of trees containing this rule (0-1)
    signature: str         # Fuzzy signature used for matching
    variant_count: int     # Number of slightly different variants merged


@dataclass(frozen=True)
class EnsembleReport:
    """Summary of the ensemble rule extraction process."""

    n_estimators: int
    frequency_threshold: float
    total_unique_rules: int     # Distinct fuzzy signatures across all trees
    stable_rule_count: int      # Rules meeting the frequency threshold
    mean_rules_per_tree: float
    tolerance: Union[float, dict[str, float]]

    def __str__(self) -> str:
        return (
            f"=== Ensemble Report ({self.n_estimators} surrogates) ===\n"
            f"  Frequency threshold: {self.frequency_threshold:.0%}\n"
            f"  Tolerance: {self.tolerance}\n"
            f"  Total unique rules: {self.total_unique_rules}\n"
            f"  Stable rules: {self.stable_rule_count}\n"
            f"  Mean rules per tree: {self.mean_rules_per_tree:.1f}"
        )


def round_to_tolerance(value: float, tolerance: float) -> float:
    """Round *value* to the nearest multiple of *tolerance*."""
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def fuzzy_signature(
    rule: Rule,
    tolerance: float | dict[str, float],
) -> str:
    """Create a canonical rule signature with rounded thresholds.

    Parameters
    ----------
    tolerance : float or dict
        If float, a global absolute tolerance.
        If dict, maps feature names to specific absolute tolerances.
    """
    parts: list[str] = []
    for cond in rule.conditions:
        tol = tolerance
        if isinstance(tolerance, dict):
            tol = tolerance.get(cond.feature, 0.0)
            # If key missing, fallback to 0.0 (exact match) or a default? 
            # Let's assume 0.0 (exact) to be safe, or we could require it.
        
        # Determine tolerance value
        t_val = float(tol)
        
        rounded = round_to_tolerance(cond.threshold, t_val)
        parts.append(f"{cond.feature} {cond.operator} {rounded:.6f}")
    parts.sort()
    return " AND ".join(parts) + " -> " + str(rule.prediction)


def extract_ensemble_rules(
    rulesets: list[RuleSet],
    *,
    frequency_threshold: float = 0.5,
    tolerance: float | dict[str, float] = 0.01,
) -> tuple[list[StableRule], EnsembleReport]:
    """Given rule sets from N bootstrap surrogates, extract stable rules.

    Parameters
    ----------
    rulesets : list of RuleSet
        One RuleSet per bootstrap surrogate tree.
    frequency_threshold : float
        Minimum fraction of trees in which a rule must appear.
    tolerance : float or dict
        Threshold rounding tolerance for fuzzy matching.

    Returns
    -------
    tuple of (list[StableRule], EnsembleReport)
    """
    n_estimators = len(rulesets)
    if n_estimators == 0:
        report = EnsembleReport(
            n_estimators=0,
            frequency_threshold=frequency_threshold,
            total_unique_rules=0,
            stable_rule_count=0,
            mean_rules_per_tree=0.0,
            tolerance=tolerance,
        )
        return [], report

    # Collect all rules and their fuzzy signatures
    sig_counts: Counter[str] = Counter()
    sig_to_rules: dict[str, list[Rule]] = defaultdict(list)
    total_rules = 0

    for ruleset in rulesets:
        # Use a set to avoid counting duplicate signatures within one tree
        seen_in_tree: set[str] = set()
        total_rules += ruleset.num_rules

        for rule in ruleset.rules:
            sig = fuzzy_signature(rule, tolerance)
            sig_to_rules[sig].append(rule)
            if sig not in seen_in_tree:
                sig_counts[sig] += 1
                seen_in_tree.add(sig)

    # Select stable rules
    stable_rules: list[StableRule] = []
    for sig, count in sig_counts.items():
        freq = count / n_estimators
        if freq >= frequency_threshold:
            representative = _select_representative(sig_to_rules[sig])
            stable_rules.append(StableRule(
                rule=representative,
                frequency=freq,
                signature=sig,
                variant_count=len(sig_to_rules[sig]),
            ))

    # Sort by frequency descending, then by sample count descending
    stable_rules.sort(key=lambda sr: (-sr.frequency, -sr.rule.samples))

    report = EnsembleReport(
        n_estimators=n_estimators,
        frequency_threshold=frequency_threshold,
        total_unique_rules=len(sig_counts),
        stable_rule_count=len(stable_rules),
        mean_rules_per_tree=total_rules / n_estimators,
        tolerance=tolerance,
    )

    return stable_rules, report


def extract_ensemble_rules_adaptive(
    rulesets: list[RuleSet],
    *,
    tolerance: float | dict[str, float] = 0.01,
    min_rules: int = 1,
    quantile: float = 0.75,
) -> tuple[list[StableRule], EnsembleReport]:
    """Extract stable rules with an adaptive frequency threshold.

    Instead of a fixed phi, automatically selects the threshold as a quantile
    of the observed frequency distribution, guaranteeing at least *min_rules*
    rules in the output.

    Parameters
    ----------
    rulesets : list of RuleSet
        One RuleSet per bootstrap surrogate tree.
    tolerance : float or dict
        Threshold rounding tolerance for fuzzy matching.
    min_rules : int, default 1
        Minimum number of rules to return. If the quantile-based threshold
        would produce fewer rules, the threshold is lowered.
    quantile : float, default 0.75
        Quantile of rule frequencies to use as the threshold.
        E.g. 0.75 means the top 25% most frequent rules are kept.

    Returns
    -------
    tuple of (list[StableRule], EnsembleReport)
    """
    n_estimators = len(rulesets)
    if n_estimators == 0:
        report = EnsembleReport(
            n_estimators=0,
            frequency_threshold=0.0,
            total_unique_rules=0,
            stable_rule_count=0,
            mean_rules_per_tree=0.0,
            tolerance=tolerance,
        )
        return [], report

    # Collect signatures and frequencies
    sig_counts: Counter[str] = Counter()
    sig_to_rules: dict[str, list[Rule]] = defaultdict(list)
    total_rules = 0

    for ruleset in rulesets:
        seen_in_tree: set[str] = set()
        total_rules += ruleset.num_rules
        for rule in ruleset.rules:
            sig = fuzzy_signature(rule, tolerance)
            sig_to_rules[sig].append(rule)
            if sig not in seen_in_tree:
                sig_counts[sig] += 1
                seen_in_tree.add(sig)

    # Compute adaptive threshold
    frequencies = np.array([count / n_estimators for count in sig_counts.values()])
    if len(frequencies) == 0:
        adaptive_threshold = 0.0
    else:
        adaptive_threshold = float(np.quantile(frequencies, quantile))

    # Ensure min_rules are returned
    sorted_freqs = np.sort(frequencies)[::-1]
    if len(sorted_freqs) >= min_rules:
        min_freq_for_min_rules = sorted_freqs[min_rules - 1]
        adaptive_threshold = min(adaptive_threshold, min_freq_for_min_rules)

    # Select stable rules using adaptive threshold
    stable_rules: list[StableRule] = []
    for sig, count in sig_counts.items():
        freq = count / n_estimators
        if freq >= adaptive_threshold:
            representative = _select_representative(sig_to_rules[sig])
            stable_rules.append(StableRule(
                rule=representative,
                frequency=freq,
                signature=sig,
                variant_count=len(sig_to_rules[sig]),
            ))

    stable_rules.sort(key=lambda sr: (-sr.frequency, -sr.rule.samples))

    report = EnsembleReport(
        n_estimators=n_estimators,
        frequency_threshold=adaptive_threshold,
        total_unique_rules=len(sig_counts),
        stable_rule_count=len(stable_rules),
        mean_rules_per_tree=total_rules / n_estimators,
        tolerance=tolerance,
    )

    return stable_rules, report


def rank_rules_by_frequency(
    rulesets: list[RuleSet],
    *,
    tolerance: float | dict[str, float] = 0.01,
) -> list[StableRule]:
    """Rank all rules by their frequency without any threshold cutoff.

    Returns all rules sorted by frequency descending. This is the "soft
    filtering" alternative to hard threshold-based ensemble filtering.

    Parameters
    ----------
    rulesets : list of RuleSet
        One RuleSet per bootstrap surrogate tree.
    tolerance : float or dict
        Threshold rounding tolerance for fuzzy matching.

    Returns
    -------
    list of StableRule
        All rules ranked by frequency (no filtering applied).
    """
    n_estimators = len(rulesets)
    if n_estimators == 0:
        return []

    sig_counts: Counter[str] = Counter()
    sig_to_rules: dict[str, list[Rule]] = defaultdict(list)

    for ruleset in rulesets:
        seen_in_tree: set[str] = set()
        for rule in ruleset.rules:
            sig = fuzzy_signature(rule, tolerance)
            sig_to_rules[sig].append(rule)
            if sig not in seen_in_tree:
                sig_counts[sig] += 1
                seen_in_tree.add(sig)

    ranked: list[StableRule] = []
    for sig, count in sig_counts.items():
        freq = count / n_estimators
        representative = _select_representative(sig_to_rules[sig])
        ranked.append(StableRule(
            rule=representative,
            frequency=freq,
            signature=sig,
            variant_count=len(sig_to_rules[sig]),
        ))

    ranked.sort(key=lambda sr: (-sr.frequency, -sr.rule.samples))
    return ranked


def _select_representative(variants: list[Rule]) -> Rule:
    """Pick the best representative from a group of similar rules.

    Heuristic: pick the one with the highest sample count (most data support).
    """
    return max(variants, key=lambda r: r.samples)
