"""Tests for monotonicity.py - Monotonicity constraints and validation."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from trace_xai import (
    Explainer,
    MonotonicityReport,
    MonotonicityViolation,
    validate_monotonicity,
    filter_monotonic_violations,
)
from trace_xai.monotonicity import (
    check_sklearn_monotonic_support,
    constraints_to_array,
)
from trace_xai.ruleset import Condition, Rule, RuleSet


class TestConstraintsToArray:
    def test_basic(self):
        features = ("income", "age", "debt")
        constraints = {"income": 1, "debt": -1}
        arr = constraints_to_array(constraints, features)
        assert arr.tolist() == [1, 0, -1]

    def test_empty_constraints(self):
        features = ("a", "b")
        arr = constraints_to_array({}, features)
        assert arr.tolist() == [0, 0]

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="must be -1, 0, or \\+1"):
            constraints_to_array({"a": 2}, ("a",))

    def test_unknown_feature(self):
        with pytest.raises(ValueError, match="not found"):
            constraints_to_array({"unknown": 1}, ("a", "b"))


class TestCheckSklearnSupport:
    def test_returns_bool(self):
        result = check_sklearn_monotonic_support()
        assert isinstance(result, bool)


class TestValidateMonotonicity:
    def test_no_violations_empty(self):
        ruleset = RuleSet(rules=(), feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 1})
        assert report.is_compliant
        assert len(report.violations) == 0

    def test_regression_violation_detected(self):
        """Increasing constraint on A, but higher A → lower prediction."""
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="10.0000",
                samples=50,
                confidence=1.0,
                leaf_id=1,
                prediction_value=10.0,
            ),
            Rule(
                conditions=(Condition("A", ">", 5.0),),
                prediction="3.0000",
                samples=50,
                confidence=1.0,
                leaf_id=2,
                prediction_value=3.0,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 1})
        assert not report.is_compliant
        assert len(report.violations) > 0
        assert report.violations[0].feature == "A"
        assert report.violations[0].expected_direction == 1

    def test_regression_no_violation(self):
        """Increasing constraint on A, higher A → higher prediction: OK."""
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="3.0000",
                samples=50,
                confidence=1.0,
                leaf_id=1,
                prediction_value=3.0,
            ),
            Rule(
                conditions=(Condition("A", ">", 5.0),),
                prediction="10.0000",
                samples=50,
                confidence=1.0,
                leaf_id=2,
                prediction_value=10.0,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 1})
        assert report.is_compliant

    def test_decreasing_violation(self):
        """Decreasing constraint on A, but higher A → higher prediction."""
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="3.0000",
                samples=50,
                confidence=1.0,
                leaf_id=1,
                prediction_value=3.0,
            ),
            Rule(
                conditions=(Condition("A", ">", 5.0),),
                prediction="10.0000",
                samples=50,
                confidence=1.0,
                leaf_id=2,
                prediction_value=10.0,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": -1})
        assert not report.is_compliant

    def test_zero_constraint_ignored(self):
        """Constraint 0 should produce no violations."""
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="10.0000",
                samples=50,
                confidence=1.0,
                leaf_id=1,
                prediction_value=10.0,
            ),
            Rule(
                conditions=(Condition("A", ">", 5.0),),
                prediction="3.0000",
                samples=50,
                confidence=1.0,
                leaf_id=2,
                prediction_value=3.0,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 0})
        assert report.is_compliant

    def test_report_str(self):
        ruleset = RuleSet(rules=(), feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 1})
        text = str(report)
        assert "Monotonicity Report" in text
        assert "Compliant: True" in text


class TestFilterViolations:
    def test_removes_violating_rules(self):
        rules = (
            Rule(
                conditions=(Condition("A", "<=", 5.0),),
                prediction="10.0000",
                samples=50,
                confidence=1.0,
                leaf_id=1,
                prediction_value=10.0,
            ),
            Rule(
                conditions=(Condition("A", ">", 5.0),),
                prediction="3.0000",
                samples=50,
                confidence=1.0,
                leaf_id=2,
                prediction_value=3.0,
            ),
        )
        ruleset = RuleSet(rules=rules, feature_names=("A",), class_names=())
        report = validate_monotonicity(ruleset, {"A": 1})
        filtered = filter_monotonic_violations(ruleset, report)
        assert filtered.num_rules < ruleset.num_rules


class TestMonotonicityIntegration:
    @pytest.mark.skipif(
        not check_sklearn_monotonic_support(),
        reason="sklearn < 1.4, monotone_cst not supported",
    )
    def test_extract_rules_with_constraints(self):
        """End-to-end: extract rules with monotonic constraints."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = Explainer(
            rf,
            feature_names=["f1", "f2", "f3"],
            class_names=["low", "high"],
        )
        result = explainer.extract_rules(
            X, y=y,
            monotonic_constraints={"f1": 1, "f3": -1},
        )
        assert result.monotonicity_report is not None
        assert isinstance(result.monotonicity_report, MonotonicityReport)
