"""Tests for counterfactual-guided rule scoring."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer
from trace_xai.counterfactual import (
    ConditionValidity,
    CounterfactualReport,
    RuleCounterfactualScore,
    score_rule_counterfactual,
    score_rules_counterfactual,
)
from trace_xai.ruleset import Condition, Rule


@pytest.fixture()
def iris_setup():
    iris = load_iris()
    X, y = iris.data, iris.target
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    explainer = Explainer(
        rf,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )
    return explainer, X, y


class TestConditionValidity:
    def test_is_frozen_dataclass(self):
        cv = ConditionValidity(
            condition=Condition("x", "<=", 1.0),
            is_valid=True,
            bb_changes=True,
            delta_below=0.99,
            delta_above=1.01,
        )
        assert isinstance(cv.is_valid, bool)
        assert cv.delta_below < cv.delta_above

    def test_invalid_condition(self):
        cv = ConditionValidity(
            condition=Condition("x", "<=", 1.0),
            is_valid=False,
            bb_changes=False,
            delta_below=0.99,
            delta_above=1.01,
        )
        assert not cv.is_valid


class TestScoreRuleCounterfactual:
    def test_empty_conditions_scores_1(self, iris_setup):
        explainer, X, y = iris_setup
        rule = Rule(
            conditions=(), prediction="setosa",
            samples=50, confidence=1.0, leaf_id=0,
        )
        rng = np.random.RandomState(42)
        score = score_rule_counterfactual(
            rule, 0, explainer.model, X,
            tuple(explainer.feature_names),
            noise_scale=0.01, rng=rng,
        )
        assert score.score == 1.0
        assert score.n_conditions == 0
        assert score.n_valid_conditions == 0

    def test_score_in_range(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        for rs in report.rule_scores:
            assert 0.0 <= rs.score <= 1.0
            assert rs.n_valid_conditions <= rs.n_conditions

    def test_rule_index_matches_position(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        for i, rs in enumerate(report.rule_scores):
            assert rs.rule_index == i


class TestCounterfactualReport:
    def test_returns_report_type(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        assert isinstance(report, CounterfactualReport)

    def test_mean_score_in_range(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        assert 0.0 <= report.mean_score <= 1.0

    def test_no_threshold_no_filtered_ruleset(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        assert report.filtered_ruleset is None
        assert report.validity_threshold is None

    def test_threshold_zero_retains_all(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(
            result.rules, explainer.model, X, validity_threshold=0.0,
        )
        assert report.filtered_ruleset is not None
        assert report.n_rules_retained == report.n_rules_total

    def test_high_threshold_filters_rules(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=5)
        report = score_rules_counterfactual(
            result.rules, explainer.model, X, validity_threshold=1.0,
        )
        assert report.n_rules_retained <= report.n_rules_total

    def test_str_representation(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        s = str(report)
        assert "Counterfactual" in s
        assert "Mean validity score" in s

    def test_n_rules_matches_ruleset(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = score_rules_counterfactual(result.rules, explainer.model, X)
        assert report.n_rules_total == result.rules.num_rules
        assert len(report.rule_scores) == report.n_rules_total


class TestExplainerCounterfactualMethod:
    def test_method_exists(self, iris_setup):
        explainer, X, y = iris_setup
        assert hasattr(explainer, "score_rules_counterfactual")

    def test_method_returns_report(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = explainer.score_rules_counterfactual(result, X)
        assert isinstance(report, CounterfactualReport)

    def test_integrated_via_extract_rules(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(
            X, y=y, max_depth=3,
            counterfactual_validity_threshold=0.0,
        )
        assert result.counterfactual_report is not None
        assert result.counterfactual_report.n_rules_retained == result.counterfactual_report.n_rules_total

