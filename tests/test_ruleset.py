"""Tests for ruleset.py data classes."""

import pytest

from trace_xai.ruleset import Condition, Rule, RuleSet


class TestCondition:
    def test_str(self):
        c = Condition("petal_width", "<=", 0.8)
        assert str(c) == "petal_width <= 0.8000"

    def test_frozen(self):
        c = Condition("f1", "<=", 1.0)
        with pytest.raises(AttributeError):
            c.feature = "f2"


class TestRule:
    def test_str_with_conditions(self):
        r = Rule(
            conditions=(
                Condition("f1", "<=", 1.0),
                Condition("f2", ">", 2.5),
            ),
            prediction="setosa",
            samples=50,
            confidence=0.98,
            leaf_id=3,
        )
        text = str(r)
        assert text.startswith("IF ")
        assert "f1 <= 1.0000 AND f2 > 2.5000" in text
        assert "THEN class = setosa" in text
        assert "confidence=98.00%" in text
        assert "samples=50" in text

    def test_str_no_conditions(self):
        r = Rule(
            conditions=(),
            prediction="A",
            samples=100,
            confidence=1.0,
            leaf_id=0,
        )
        assert "IF TRUE THEN class = A" in str(r)


class TestRuleSet:
    @pytest.fixture()
    def sample_ruleset(self):
        rules = (
            Rule(
                conditions=(Condition("f1", "<=", 1.0),),
                prediction="A",
                samples=30,
                confidence=0.9,
                leaf_id=1,
            ),
            Rule(
                conditions=(
                    Condition("f1", ">", 1.0),
                    Condition("f2", "<=", 3.0),
                ),
                prediction="B",
                samples=40,
                confidence=0.85,
                leaf_id=2,
            ),
            Rule(
                conditions=(
                    Condition("f1", ">", 1.0),
                    Condition("f2", ">", 3.0),
                ),
                prediction="A",
                samples=30,
                confidence=0.95,
                leaf_id=3,
            ),
        )
        return RuleSet(rules=rules, feature_names=("f1", "f2"), class_names=("A", "B"))

    def test_num_rules(self, sample_ruleset):
        assert sample_ruleset.num_rules == 3

    def test_avg_conditions(self, sample_ruleset):
        # (1 + 2 + 2) / 3
        assert abs(sample_ruleset.avg_conditions - 5 / 3) < 1e-9

    def test_max_conditions(self, sample_ruleset):
        assert sample_ruleset.max_conditions == 2

    def test_filter_by_class(self, sample_ruleset):
        filtered = sample_ruleset.filter_by_class("A")
        assert filtered.num_rules == 2
        assert all(r.prediction == "A" for r in filtered.rules)

    def test_filter_by_class_empty(self, sample_ruleset):
        filtered = sample_ruleset.filter_by_class("C")
        assert filtered.num_rules == 0

    def test_to_text(self, sample_ruleset):
        text = sample_ruleset.to_text()
        assert "Rule 1:" in text
        assert "Rule 3:" in text

    def test_avg_conditions_per_feature(self, sample_ruleset):
        # avg_conditions = 5/3, n_features = 2 → 5/6
        expected = (5 / 3) / 2
        assert abs(sample_ruleset.avg_conditions_per_feature - expected) < 1e-9

    def test_interaction_strength(self, sample_ruleset):
        # Rule 1: 1 feature (f1), Rule 2: 2 features (f1, f2), Rule 3: 2 features
        # 2 out of 3 rules have >1 distinct feature
        assert abs(sample_ruleset.interaction_strength - 2 / 3) < 1e-9

    def test_rule_signatures(self, sample_ruleset):
        sigs = sample_ruleset.rule_signatures()
        assert isinstance(sigs, frozenset)
        assert len(sigs) == 3  # one per rule

    def test_empty_ruleset_metrics(self):
        empty = RuleSet(rules=(), feature_names=("f1",), class_names=("A",))
        assert empty.avg_conditions_per_feature == 0.0
        assert empty.interaction_strength == 0.0
        assert empty.rule_signatures() == frozenset()


class TestRuleRegression:
    def test_regression_rule_str(self):
        r = Rule(
            conditions=(Condition("f1", "<=", 1.0),),
            prediction="42.1234",
            samples=50,
            confidence=1.0,
            leaf_id=1,
            prediction_value=42.1234,
        )
        text = str(r)
        assert "THEN value = 42.1234" in text
        assert "samples=50" in text
        assert "confidence" not in text

    def test_classification_rule_str_unchanged(self):
        r = Rule(
            conditions=(Condition("f1", "<=", 1.0),),
            prediction="setosa",
            samples=50,
            confidence=0.98,
            leaf_id=1,
        )
        text = str(r)
        assert "THEN class = setosa" in text
        assert "confidence=98.00%" in text
