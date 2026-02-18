"""Tests for new modules: hyperparams, stability, metrics, oblique tree, ensemble adaptive."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer
from trace_xai.hyperparams import (
    AutoDepthResult,
    SensitivityResult,
    auto_select_depth,
    compute_adaptive_tolerance,
    get_preset,
    sensitivity_analysis,
    PRESETS,
)
from trace_xai.stability import (
    StructuralStabilityReport,
    compute_structural_stability,
)
from trace_xai.metrics import (
    ComplementaryMetrics,
    compute_complementary_metrics,
)
from trace_xai.ensemble import (
    extract_ensemble_rules_adaptive,
    rank_rules_by_frequency,
)
from trace_xai.monotonicity import (
    enforce_monotonicity,
    MonotonicityEnforcementResult,
)
from trace_xai.ruleset import Condition, Rule, RuleSet


@pytest.fixture()
def iris_setup():
    iris = load_iris()
    X, y = iris.data, iris.target
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    explainer = Explainer(
        rf, feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )
    return explainer, X, y


class TestPresets:
    def test_get_preset(self):
        p = get_preset("interpretable")
        assert p.max_depth == 3
        assert p.min_samples_leaf == 20

    def test_get_preset_balanced(self):
        p = get_preset("balanced")
        assert p.max_depth == 5

    def test_get_preset_faithful(self):
        p = get_preset("faithful")
        assert p.max_depth == 8

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_all_presets_exist(self):
        assert "interpretable" in PRESETS
        assert "balanced" in PRESETS
        assert "faithful" in PRESETS


class TestAutoSelectDepth:
    def test_returns_result(self, iris_setup):
        explainer, X, y = iris_setup
        result = auto_select_depth(
            explainer, X, y=y, min_depth=2, max_depth=5, n_folds=3,
        )
        assert isinstance(result, AutoDepthResult)
        assert 2 <= result.best_depth <= 5
        assert result.selected_fidelity > 0
        assert len(result.fidelity_scores) > 0

    def test_explainer_method(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.auto_select_depth(
            X, y=y, min_depth=2, max_depth=4, n_folds=3,
        )
        assert isinstance(result, AutoDepthResult)


class TestSensitivityAnalysis:
    def test_returns_result(self, iris_setup):
        explainer, X, y = iris_setup
        result = sensitivity_analysis(
            explainer, X, y=y,
            depth_range=(3, 5), min_samples_leaf_range=(5, 10), n_folds=3,
        )
        assert isinstance(result, SensitivityResult)
        assert len(result.results) == 4  # 2 depths * 2 min_leaf
        assert result.best_fidelity > 0

    def test_explainer_method(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.sensitivity_analysis(
            X, y=y, depth_range=(3,), min_samples_leaf_range=(5,), n_folds=3,
        )
        assert len(result.results) == 1


class TestAdaptiveTolerance:
    def test_returns_per_feature_dict(self, iris_setup):
        _, X, _ = iris_setup
        feature_names = ["f1", "f2", "f3", "f4"]
        tols = compute_adaptive_tolerance(X, feature_names)
        assert len(tols) == 4
        for name in feature_names:
            assert tols[name] > 0


class TestStructuralStability:
    def test_compute(self, iris_setup):
        explainer, X, _ = iris_setup
        report = compute_structural_stability(
            explainer, X, n_bootstraps=5, max_depth=3,
        )
        assert isinstance(report, StructuralStabilityReport)
        assert 0.0 <= report.mean_prediction_agreement <= 1.0
        assert 0.0 <= report.feature_importance_stability <= 1.0
        assert 0.0 <= report.top_k_feature_agreement <= 1.0

    def test_explainer_method(self, iris_setup):
        explainer, X, _ = iris_setup
        report = explainer.compute_structural_stability(
            X, n_bootstraps=5, max_depth=3,
        )
        assert isinstance(report, StructuralStabilityReport)
        assert "Structural Stability" in str(report)


class TestComplementaryMetrics:
    def test_compute(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        metrics = compute_complementary_metrics(
            result.surrogate, explainer.model, X, result.rules,
            class_names=explainer.class_names or (),
        )
        assert isinstance(metrics, ComplementaryMetrics)
        assert 0.0 <= metrics.rule_coverage <= 1.0
        assert 0.0 <= metrics.effective_complexity <= 1.0

    def test_explainer_method(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        metrics = explainer.compute_complementary_metrics(result, X)
        assert isinstance(metrics, ComplementaryMetrics)
        assert "Complementary Metrics" in str(metrics)


class TestAdaptiveEnsemble:
    def test_adaptive_extraction(self):
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="Low", samples=50, confidence=0.9, leaf_id=1,
            ),
        )
        rulesets = [
            RuleSet(rules=rules, feature_names=("A",), class_names=("Low",))
            for _ in range(10)
        ]
        stable, report = extract_ensemble_rules_adaptive(
            rulesets, min_rules=1,
        )
        assert len(stable) >= 1
        assert report.n_estimators == 10

    def test_guarantees_min_rules(self):
        rule_a = Rule(
            conditions=(Condition("A", "<=", 5.0),),
            prediction="Low", samples=50, confidence=0.9, leaf_id=1,
        )
        rule_b = Rule(
            conditions=(Condition("B", ">", 10.0),),
            prediction="High", samples=10, confidence=0.8, leaf_id=2,
        )
        # rule_a in all, rule_b in only 1
        rulesets = [
            RuleSet(rules=(rule_a,), feature_names=("A", "B"), class_names=("Low", "High"))
            for _ in range(9)
        ] + [
            RuleSet(rules=(rule_a, rule_b), feature_names=("A", "B"), class_names=("Low", "High")),
        ]
        stable, _ = extract_ensemble_rules_adaptive(rulesets, min_rules=2)
        assert len(stable) >= 2

    def test_rank_rules(self):
        rule = Rule(
            conditions=(Condition("A", "<=", 5.0),),
            prediction="Low", samples=50, confidence=0.9, leaf_id=1,
        )
        rulesets = [
            RuleSet(rules=(rule,), feature_names=("A",), class_names=("Low",))
            for _ in range(5)
        ]
        ranked = rank_rules_by_frequency(rulesets)
        assert len(ranked) > 0
        assert ranked[0].frequency == 1.0


class TestMonotonicityEnforcement:
    def test_no_violations(self):
        rules = (
            Rule(
                conditions=(Condition("income", "<=", 50000),),
                prediction="low", samples=50, confidence=0.9, leaf_id=1,
            ),
            Rule(
                conditions=(Condition("income", ">", 50000),),
                prediction="low", samples=50, confidence=0.9, leaf_id=2,
            ),
        )
        ruleset = RuleSet(
            rules=rules, feature_names=("income",), class_names=("low", "high"),
        )
        result = enforce_monotonicity(ruleset, {"income": 1})
        assert isinstance(result, MonotonicityEnforcementResult)
        assert result.rules_removed == 0

    def test_with_violations(self):
        rules = (
            Rule(
                conditions=(Condition("income", "<=", 50000),),
                prediction="high", samples=50, confidence=0.9, leaf_id=1,
            ),
            Rule(
                conditions=(Condition("income", ">", 50000),),
                prediction="low", samples=50, confidence=0.9, leaf_id=2,
            ),
        )
        ruleset = RuleSet(
            rules=rules, feature_names=("income",), class_names=("low", "high"),
        )
        result = enforce_monotonicity(ruleset, {"income": 1})
        assert result.rules_removed > 0


class TestExtractRulesWithPreset:
    def test_preset_interpretable(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, preset="interpretable")
        assert result.report.surrogate_depth <= 3

    def test_preset_faithful(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, preset="faithful")
        assert result.report.surrogate_depth <= 8


class TestExtractRulesWithAugmentation:
    def test_perturbation_augmentation(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(
            X, y=y, max_depth=3, augmentation="perturbation",
        )
        assert result.rules.num_rules > 0
        assert result.report.fidelity > 0.5


class TestObliqueTree:
    def test_oblique_surrogate(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(
            X, y=y, max_depth=3, surrogate_type="oblique_tree",
        )
        assert result.rules.num_rules > 0
        # Oblique tree may have interaction features in rules
        all_features = set()
        for rule in result.rules.rules:
            for cond in rule.conditions:
                all_features.add(cond.feature)
        assert len(all_features) > 0

    def test_invalid_surrogate_type(self, iris_setup):
        explainer, X, y = iris_setup
        with pytest.raises(ValueError, match="Supported surrogates"):
            explainer.extract_rules(X, y=y, surrogate_type="invalid")
