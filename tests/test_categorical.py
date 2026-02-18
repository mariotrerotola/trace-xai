"""Tests for categorical.py - Categorical feature decoding."""

import pytest

from trace_xai.categorical import (
    CategoricalCondition,
    CategoricalMapping,
    decode_conditions,
    decode_ruleset,
)
from trace_xai.ruleset import Condition, Rule, RuleSet


class TestCategoricalCondition:
    def test_str_with_value(self):
        cond = CategoricalCondition("occupation", "=", 0.0, "Prof-specialty")
        assert str(cond) == "occupation = Prof-specialty"

    def test_str_with_set(self):
        cond = CategoricalCondition("occupation", "in", 0.0, "{A, B, C}")
        assert str(cond) == "occupation in {A, B, C}"

    def test_is_condition_subclass(self):
        cond = CategoricalCondition("occupation", "=", 0.0, "A")
        assert isinstance(cond, Condition)


class TestDecodeOnehot:
    def test_active_category(self):
        mapping = CategoricalMapping(
            original_name="color",
            encoding="onehot",
            encoded_columns=("color_red", "color_blue", "color_green"),
            categories=("red", "blue", "green"),
        )
        # col > 0.5 means active
        conditions = (Condition("color_red", ">", 0.5),)
        decoded = decode_conditions(conditions, [mapping])
        assert len(decoded) == 1
        assert str(decoded[0]) == "color = red"

    def test_inactive_categories(self):
        mapping = CategoricalMapping(
            original_name="color",
            encoding="onehot",
            encoded_columns=("color_red", "color_blue", "color_green"),
            categories=("red", "blue", "green"),
        )
        # col <= 0.5 means inactive
        conditions = (
            Condition("color_red", "<=", 0.5),
            Condition("color_blue", "<=", 0.5),
        )
        decoded = decode_conditions(conditions, [mapping])
        assert len(decoded) == 1
        assert "green" in str(decoded[0])

    def test_non_categorical_preserved(self):
        mapping = CategoricalMapping(
            original_name="color",
            encoding="onehot",
            encoded_columns=("color_red",),
            categories=("red",),
        )
        conditions = (
            Condition("age", "<=", 30.0),
            Condition("color_red", ">", 0.5),
        )
        decoded = decode_conditions(conditions, [mapping])
        assert len(decoded) == 2
        assert str(decoded[0]) == "age <= 30.0000"


class TestDecodeOrdinal:
    def test_ordinal_leq(self):
        mapping = CategoricalMapping(
            original_name="education",
            encoding="ordinal",
            encoded_columns=("education",),
            categories=("HS", "BS", "MS", "PhD"),
        )
        conditions = (Condition("education", "<=", 1.5),)
        decoded = decode_conditions(conditions, [mapping])
        assert len(decoded) == 1
        assert "HS" in str(decoded[0])
        assert "BS" in str(decoded[0])

    def test_ordinal_gt_single(self):
        mapping = CategoricalMapping(
            original_name="education",
            encoding="ordinal",
            encoded_columns=("education",),
            categories=("HS", "BS", "MS", "PhD"),
        )
        conditions = (Condition("education", ">", 2.5),)
        decoded = decode_conditions(conditions, [mapping])
        assert len(decoded) == 1
        assert "PhD" in str(decoded[0])


class TestDecodeRuleset:
    def test_full_decode(self):
        mapping = CategoricalMapping(
            original_name="color",
            encoding="onehot",
            encoded_columns=("color_red", "color_blue"),
            categories=("red", "blue"),
        )
        rules = (
            Rule(
                conditions=(
                    Condition("age", "<=", 30.0),
                    Condition("color_red", ">", 0.5),
                ),
                prediction="A",
                samples=50,
                confidence=0.9,
                leaf_id=1,
            ),
        )
        ruleset = RuleSet(
            rules=rules,
            feature_names=("age", "color_red", "color_blue"),
            class_names=("A", "B"),
        )
        decoded = decode_ruleset(ruleset, [mapping])
        assert "color" in decoded.feature_names
        assert "color_red" not in decoded.feature_names
        rule_str = str(decoded.rules[0])
        assert "color = red" in rule_str
