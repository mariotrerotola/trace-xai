"""Tests for pruning.py - Regulatory rule pruning."""

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trace_xai import Explainer, PruningConfig, PruningReport, prune_ruleset
from trace_xai.pruning import _simplify_conditions, _truncate_conditions
from trace_xai.ruleset import Condition, Rule, RuleSet


@pytest.fixture()
def sample_ruleset():
    """A RuleSet with varied confidence and sample counts."""
    rules = (
        Rule(
            conditions=(Condition("A", ">", 5.0), Condition("B", "<=", 3.0)),
            prediction="High",
            samples=100,
            confidence=0.95,
            leaf_id=1,
        ),
        Rule(
            conditions=(Condition("A", "<=", 5.0),),
            prediction="Low",
            samples=10,
            confidence=0.55,
            leaf_id=2,
        ),
        Rule(
            conditions=(
                Condition("A", ">", 3.0),
                Condition("A", ">", 5.0),
                Condition("B", "<=", 10.0),
                Condition("B", "<=", 7.0),
            ),
            prediction="Medium",
            samples=50,
            confidence=0.80,
            leaf_id=3,
        ),
        Rule(
            conditions=(
                Condition("A", ">", 1.0),
                Condition("B", "<=", 9.0),
                Condition("C", ">", 2.0),
                Condition("D", "<=", 8.0),
                Condition("E", ">", 0.5),
            ),
            prediction="High",
            samples=30,
            confidence=0.70,
            leaf_id=4,
        ),
    )
    return RuleSet(
        rules=rules,
        feature_names=("A", "B", "C", "D", "E"),
        class_names=("Low", "Medium", "High"),
    )


class TestPruningConfig:
    def test_defaults(self):
        config = PruningConfig()
        assert config.min_confidence == 0.0
        assert config.min_samples == 0
        assert config.max_conditions is None
        assert config.remove_redundant is False

    def test_custom_values(self):
        config = PruningConfig(min_confidence=0.6, max_conditions=3)
        assert config.min_confidence == 0.6
        assert config.max_conditions == 3


class TestSimplifyConditions:
    def test_redundant_gt(self):
        """A > 5 AND A > 3 should simplify to A > 5."""
        conds = (Condition("A", ">", 3.0), Condition("A", ">", 5.0))
        result = _simplify_conditions(conds)
        assert len(result) == 1
        assert result[0].threshold == 5.0

    def test_redundant_le(self):
        """A <= 10 AND A <= 7 should simplify to A <= 7."""
        conds = (Condition("A", "<=", 10.0), Condition("A", "<=", 7.0))
        result = _simplify_conditions(conds)
        assert len(result) == 1
        assert result[0].threshold == 7.0

    def test_different_features_preserved(self):
        """Conditions on different features should all be kept."""
        conds = (Condition("A", ">", 5.0), Condition("B", "<=", 3.0))
        result = _simplify_conditions(conds)
        assert len(result) == 2

    def test_mixed_operators_same_feature(self):
        """A > 3 AND A <= 7 should keep both (valid range)."""
        conds = (Condition("A", ">", 3.0), Condition("A", "<=", 7.0))
        result = _simplify_conditions(conds)
        assert len(result) == 2

    def test_preserves_order(self):
        """Result should follow original ordering."""
        conds = (
            Condition("B", "<=", 5.0),
            Condition("A", ">", 3.0),
            Condition("A", ">", 1.0),
        )
        result = _simplify_conditions(conds)
        assert len(result) == 2
        assert result[0].feature == "B"
        assert result[1].feature == "A"


class TestTruncateConditions:
    def test_truncate(self):
        conds = tuple(Condition(f"f{i}", ">", float(i)) for i in range(5))
        result = _truncate_conditions(conds, 3)
        assert len(result) == 3
        assert result == conds[:3]

    def test_no_truncation_needed(self):
        conds = (Condition("A", ">", 1.0),)
        result = _truncate_conditions(conds, 3)
        assert result == conds


class TestPruneRuleset:
    def test_confidence_filter(self, sample_ruleset):
        config = PruningConfig(min_confidence=0.6)
        pruned, report = prune_ruleset(sample_ruleset, config)
        assert report.removed_low_confidence == 1
        assert pruned.num_rules == 3
        # The rule with confidence 0.55 should be removed
        for rule in pruned.rules:
            assert rule.confidence >= 0.6

    def test_samples_filter(self, sample_ruleset):
        config = PruningConfig(min_samples=20)
        pruned, report = prune_ruleset(sample_ruleset, config)
        assert report.removed_low_samples == 1
        assert pruned.num_rules == 3

    def test_samples_fraction_filter(self, sample_ruleset):
        total = sum(r.samples for r in sample_ruleset.rules)
        config = PruningConfig(min_samples_fraction=0.2, total_samples=total)
        pruned, _ = prune_ruleset(sample_ruleset, config)
        for rule in pruned.rules:
            assert rule.samples >= 0.2 * total

    def test_redundant_removal(self, sample_ruleset):
        config = PruningConfig(remove_redundant=True)
        pruned, report = prune_ruleset(sample_ruleset, config)
        assert report.conditions_simplified > 0
        # Rule 3 had A>3 AND A>5 → should now just have A>5
        rule3 = pruned.rules[2]
        a_gt_conditions = [c for c in rule3.conditions if c.feature == "A" and c.operator == ">"]
        assert len(a_gt_conditions) == 1
        assert a_gt_conditions[0].threshold == 5.0

    def test_max_conditions(self, sample_ruleset):
        config = PruningConfig(max_conditions=2)
        pruned, _ = prune_ruleset(sample_ruleset, config)
        for rule in pruned.rules:
            assert len(rule.conditions) <= 2

    def test_original_not_mutated(self, sample_ruleset):
        original_count = sample_ruleset.num_rules
        config = PruningConfig(min_confidence=0.9)
        prune_ruleset(sample_ruleset, config)
        assert sample_ruleset.num_rules == original_count

    def test_report_counts(self, sample_ruleset):
        config = PruningConfig(min_confidence=0.6, min_samples=20)
        _, report = prune_ruleset(sample_ruleset, config)
        assert isinstance(report, PruningReport)
        assert report.original_count == 4
        assert report.pruned_count == report.original_count - report.removed_low_confidence - report.removed_low_samples


class TestPruningIntegration:
    def test_extract_rules_with_pruning(self):
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            iris.data, iris.target
        )
        explainer = Explainer(
            rf, feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        config = PruningConfig(min_confidence=0.6, remove_redundant=True)
        result = explainer.extract_rules(iris.data, y=iris.target, pruning=config)

        assert result.pruned_rules is not None
        assert result.pruning_report is not None
        assert result.pruned_rules.num_rules <= result.rules.num_rules

    def test_prune_rules_method(self):
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            iris.data, iris.target
        )
        explainer = Explainer(
            rf, feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        result = explainer.extract_rules(iris.data, y=iris.target)
        assert result.pruned_rules is None

        pruned = explainer.prune_rules(result, PruningConfig(min_confidence=0.8))
        assert pruned.pruned_rules is not None
        assert pruned.pruning_report is not None
        # Original rules unchanged
        assert pruned.rules.num_rules == result.rules.num_rules

    def test_ccp_alpha(self):
        iris = load_iris()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            iris.data, iris.target
        )
        explainer = Explainer(
            rf, feature_names=list(iris.feature_names),
            class_names=list(iris.target_names),
        )
        result_no_ccp = explainer.extract_rules(iris.data, max_depth=5)
        result_ccp = explainer.extract_rules(iris.data, max_depth=5, ccp_alpha=0.05)

        assert result_ccp.rules.num_rules <= result_no_ccp.rules.num_rules
