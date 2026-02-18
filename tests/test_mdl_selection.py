"""Tests for MDL-based rule selection."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer
from trace_xai.mdl_selection import (
    MDLSelectionReport,
    RuleMDLScore,
    _auto_precision_bits,
    binary_entropy,
    compute_rule_model_cost,
    score_ruleset_mdl,
    select_rules_mdl,
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


class TestBinaryEntropy:
    def test_zero_entropy_at_zero(self):
        assert binary_entropy(0.0) == 0.0

    def test_zero_entropy_at_one(self):
        assert binary_entropy(1.0) == 0.0

    def test_max_entropy_at_half(self):
        h = binary_entropy(0.5)
        assert abs(h - 1.0) < 1e-6

    def test_entropy_in_range(self):
        for p in [0.1, 0.3, 0.7, 0.9]:
            assert 0.0 <= binary_entropy(p) <= 1.0

    def test_symmetry(self):
        assert abs(binary_entropy(0.3) - binary_entropy(0.7)) < 1e-10


class TestComputeRuleModelCost:
    def test_more_conditions_higher_cost(self):
        rule_1 = Rule(
            conditions=(Condition("x", "<=", 1.0),),
            prediction="A", samples=10, confidence=1.0, leaf_id=0,
        )
        rule_3 = Rule(
            conditions=(
                Condition("x", "<=", 1.0),
                Condition("y", ">", 2.0),
                Condition("z", "<=", 3.0),
            ),
            prediction="A", samples=10, confidence=1.0, leaf_id=1,
        )
        cost_1 = compute_rule_model_cost(rule_1, n_features=4, n_classes=3)
        cost_3 = compute_rule_model_cost(rule_3, n_features=4, n_classes=3)
        assert cost_3 > cost_1

    def test_more_features_higher_cost(self):
        rule = Rule(
            conditions=(Condition("x", "<=", 1.0),),
            prediction="A", samples=10, confidence=1.0, leaf_id=0,
        )
        cost_10 = compute_rule_model_cost(rule, n_features=10, n_classes=2)
        cost_100 = compute_rule_model_cost(rule, n_features=100, n_classes=2)
        assert cost_100 > cost_10

    def test_no_conditions_only_prediction_cost(self):
        rule = Rule(
            conditions=(), prediction="A",
            samples=10, confidence=1.0, leaf_id=0,
        )
        cost = compute_rule_model_cost(rule, n_features=4, n_classes=4)
        # log2(4) = 2.0 bits for the prediction
        assert cost == pytest.approx(2.0, abs=0.01)



class TestScoreRulesetMDL:
    def test_returns_correct_types(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        scores = score_ruleset_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
        )
        for s in scores:
            assert isinstance(s, RuleMDLScore)
            assert s.model_cost >= 0.0
            assert s.data_cost >= 0.0
            assert s.total_mdl == pytest.approx(s.model_cost + s.data_cost)
            assert 0.0 <= s.error_rate <= 1.0


class TestMDLSelectionReport:
    def test_returns_report_type(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
        )
        assert isinstance(report, MDLSelectionReport)

    def test_selected_leq_original(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=5)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
            method="forward",
        )
        assert report.n_rules_selected <= report.n_rules_original

    def test_score_only_returns_all(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
            method="score_only",
        )
        assert report.n_rules_selected == report.n_rules_original

    def test_backward_method(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
            method="backward",
        )
        assert report.n_rules_selected <= report.n_rules_original

    def test_str_representation(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
        )
        s = str(report)
        assert "MDL" in s
        assert "Rules:" in s

    def test_forward_mdl_reduction_nonnegative(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
            method="forward",
        )
        assert report.total_mdl_after <= report.total_mdl_before + 1e-6

    def test_invalid_method_raises(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        with pytest.raises(ValueError, match="method"):
            select_rules_mdl(
                result.rules, explainer.model, X,
                n_classes=3, method="invalid_method",
            )


class TestExplainerMDLMethod:
    def test_method_exists(self, iris_setup):
        explainer, X, y = iris_setup
        assert hasattr(explainer, "select_rules_mdl")

    def test_method_returns_report(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = explainer.select_rules_mdl(result, X)
        assert isinstance(report, MDLSelectionReport)

    def test_integrated_via_extract_rules(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(
            X, y=y, max_depth=3, mdl_selection="forward",
        )
        assert result.mdl_report is not None
        assert result.rules.num_rules == result.mdl_report.n_rules_selected


class TestAutoPrecisionBits:
    def test_returns_int_in_range(self, iris_setup):
        _, X, _ = iris_setup
        bits = _auto_precision_bits(X)
        assert isinstance(bits, int)
        assert 4 <= bits <= 32

    def test_fewer_bits_than_default(self, iris_setup):
        _, X, _ = iris_setup
        bits = _auto_precision_bits(X)
        assert bits < 16

    def test_auto_default_works(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
        )
        assert isinstance(report, MDLSelectionReport)
        assert isinstance(report.precision_bits, int)

    def test_explicit_int_still_works(self, iris_setup):
        explainer, X, y = iris_setup
        result = explainer.extract_rules(X, y=y, max_depth=3)
        report = select_rules_mdl(
            result.rules, explainer.model, X,
            n_classes=len(explainer.class_names),
            precision_bits=16,
        )
        assert report.precision_bits == 16

