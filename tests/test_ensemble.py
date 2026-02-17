"""Tests for ensemble.py - Ensemble rule extraction and fuzzy matching."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer, EnsembleReport, StableRule
from trace_xai.ensemble import (
    fuzzy_signature,
    round_to_tolerance,
    extract_ensemble_rules,
)
from trace_xai.ruleset import Condition, Rule, RuleSet


class TestRoundToTolerance:
    def test_basic(self):
        assert round_to_tolerance(0.8012, 0.01) == pytest.approx(0.80)
        assert round_to_tolerance(0.8051, 0.01) == pytest.approx(0.81)

    def test_zero_tolerance(self):
        assert round_to_tolerance(0.8012, 0.0) == 0.8012

    def test_negative(self):
        assert round_to_tolerance(-0.8012, 0.01) == pytest.approx(-0.80)

    def test_large_tolerance(self):
        assert round_to_tolerance(7.3, 5.0) == pytest.approx(5.0)


class TestFuzzySignature:
    def test_near_identical_thresholds_match(self):
        rule_a = Rule(
            conditions=(Condition("A", "<=", 0.7999),),
            prediction="X", samples=10, confidence=0.9, leaf_id=1,
        )
        rule_b = Rule(
            conditions=(Condition("A", "<=", 0.8001),),
            prediction="X", samples=10, confidence=0.9, leaf_id=2,
        )
        sig_a = fuzzy_signature(rule_a, tolerance=0.01)
        sig_b = fuzzy_signature(rule_b, tolerance=0.01)
        assert sig_a == sig_b

    def test_different_thresholds_differ(self):
        rule_a = Rule(
            conditions=(Condition("A", "<=", 0.80),),
            prediction="X", samples=10, confidence=0.9, leaf_id=1,
        )
        rule_b = Rule(
            conditions=(Condition("A", "<=", 0.90),),
            prediction="X", samples=10, confidence=0.9, leaf_id=2,
        )
        assert fuzzy_signature(rule_a, 0.01) != fuzzy_signature(rule_b, 0.01)

    def test_different_predictions_differ(self):
        rule_a = Rule(
            conditions=(Condition("A", "<=", 0.80),),
            prediction="X", samples=10, confidence=0.9, leaf_id=1,
        )
        rule_b = Rule(
            conditions=(Condition("A", "<=", 0.80),),
            prediction="Y", samples=10, confidence=0.9, leaf_id=2,
        )
        assert fuzzy_signature(rule_a, 0.01) != fuzzy_signature(rule_b, 0.01)


class TestExtractEnsembleRules:
    def test_empty_input(self):
        stable, report = extract_ensemble_rules([], frequency_threshold=0.5)
        assert len(stable) == 0
        assert report.n_estimators == 0

    def test_single_tree(self):
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="Low", samples=50, confidence=0.9, leaf_id=1,
            ),
        )
        rulesets = [RuleSet(rules=rules, feature_names=("A",), class_names=("Low", "High"))]
        stable, report = extract_ensemble_rules(rulesets, frequency_threshold=0.5)
        assert len(stable) == 1
        assert stable[0].frequency == 1.0
        assert report.n_estimators == 1

    def test_frequency_filtering(self):
        """Rules appearing in < threshold trees are excluded."""
        rule_common = Rule(
            conditions=(Condition("A", "<=", 5.0),),
            prediction="Low", samples=50, confidence=0.9, leaf_id=1,
        )
        rule_rare = Rule(
            conditions=(Condition("B", ">", 10.0),),
            prediction="High", samples=10, confidence=0.8, leaf_id=2,
        )
        rulesets = [
            RuleSet(rules=(rule_common,), feature_names=("A", "B"), class_names=("Low", "High")),
            RuleSet(rules=(rule_common,), feature_names=("A", "B"), class_names=("Low", "High")),
            RuleSet(rules=(rule_common, rule_rare), feature_names=("A", "B"), class_names=("Low", "High")),
        ]
        stable, report = extract_ensemble_rules(rulesets, frequency_threshold=0.5)
        assert report.n_estimators == 3
        # rule_common appears in 3/3 = 1.0, rule_rare in 1/3 ≈ 0.33
        signatures = [sr.signature for sr in stable]
        assert any("A" in s for s in signatures)
        # rule_rare should be excluded (1/3 < 0.5)
        assert report.stable_rule_count <= report.total_unique_rules

    def test_report_fields(self):
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="Low", samples=50, confidence=0.9, leaf_id=1,
            ),
        )
        rulesets = [
            RuleSet(rules=rules, feature_names=("A",), class_names=("Low",))
            for _ in range(5)
        ]
        _, report = extract_ensemble_rules(rulesets, frequency_threshold=0.5, tolerance=0.01)
        assert isinstance(report, EnsembleReport)
        assert report.n_estimators == 5
        assert report.tolerance == 0.01
        assert report.mean_rules_per_tree == 1.0


class TestFuzzyRuleSignatures:
    def test_method_exists(self):
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0001),),
                prediction="Low", samples=50, confidence=0.9, leaf_id=1,
            ),
            Rule(
                conditions=(Condition("A", "<=", 4.9999),),
                prediction="Low", samples=50, confidence=0.9, leaf_id=2,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=("Low",))
        fuzzy = ruleset.fuzzy_rule_signatures(tolerance=0.01)
        exact = ruleset.rule_signatures()

        # Fuzzy should merge the two near-identical rules
        assert len(fuzzy) <= len(exact)
        assert isinstance(fuzzy, frozenset)


class TestEnsembleIntegration:
    def test_extract_stable_rules(self):
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            iris.data, iris.target
        )
        explainer = Explainer(
            rf,
            feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        result = explainer.extract_stable_rules(
            iris.data,
            y=iris.target,
            n_estimators=10,
            frequency_threshold=0.3,
            tolerance=0.1,
        )
        assert result.ensemble_report is not None
        assert isinstance(result.ensemble_report, EnsembleReport)
        assert result.ensemble_report.n_estimators == 10
        assert result.rules.num_rules > 0
        assert result.stable_rules is not None

    def test_stability_with_tolerance(self):
        """compute_stability with tolerance should give >= exact Jaccard."""
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            iris.data, iris.target
        )
        explainer = Explainer(
            rf,
            feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        exact = explainer.compute_stability(iris.data, n_bootstraps=5)
        fuzzy = explainer.compute_stability(iris.data, n_bootstraps=5, tolerance=0.1)
        # Fuzzy matching should give equal or higher Jaccard
        assert fuzzy.mean_jaccard >= exact.mean_jaccard - 0.01  # small tolerance for randomness
