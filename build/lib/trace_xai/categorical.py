"""Categorical feature decoding: translate encoded rule conditions back to human-readable form.

Addresses the limitation that rules on one-hot or ordinal-encoded features
(e.g. "occupation_0 <= 0.5") are unreadable without decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .ruleset import Condition, Rule, RuleSet


@dataclass(frozen=True)
class CategoricalMapping:
    """Describes how a single categorical feature was encoded.

    Parameters
    ----------
    original_name : str
        The original feature name (e.g. ``"occupation"``).
    encoding : str
        ``"onehot"`` or ``"ordinal"``.
    encoded_columns : tuple of str
        The names of the encoded columns in the feature matrix.
        For one-hot: ``("occupation_Tech-support", "occupation_Prof-specialty", ...)``.
        For ordinal: a single column ``("occupation",)``.
    categories : tuple of str
        The original category labels in order.
    """

    original_name: str
    encoding: str  # "onehot" or "ordinal"
    encoded_columns: tuple[str, ...]
    categories: tuple[str, ...]


@dataclass(frozen=True)
class CategoricalCondition(Condition):
    """A condition that represents a categorical split with human-readable display.

    Extends Condition with a ``display_value`` field for readable categorical
    representation while maintaining compatibility with the rest of the framework.
    """

    display_value: str = ""

    def __str__(self) -> str:
        if self.display_value:
            return f"{self.feature} {self.operator} {self.display_value}"
        return super().__str__()

    def __repr__(self) -> str:
        return (
            f"CategoricalCondition(feature={self.feature!r}, "
            f"operator={self.operator!r}, threshold={self.threshold}, "
            f"display_value={self.display_value!r})"
        )


def _make_categorical(feature: str, operator: str, threshold: float, display_value: str) -> CategoricalCondition:
    """Create a CategoricalCondition."""
    return CategoricalCondition(
        feature=feature,
        operator=operator,
        threshold=threshold,
        display_value=display_value,
    )


def decode_conditions(
    conditions: tuple[Condition, ...],
    mappings: Sequence[CategoricalMapping],
) -> tuple[Condition, ...]:
    """Translate encoded conditions back to human-readable categorical conditions.

    Parameters
    ----------
    conditions : tuple of Condition
        The original (encoded) conditions from a rule.
    mappings : sequence of CategoricalMapping
        One mapping per categorical feature.

    Returns
    -------
    tuple of Condition
        Decoded conditions where encoded splits are replaced with
        readable categorical conditions.
    """
    # Build lookup: encoded_column -> mapping
    col_to_mapping: dict[str, CategoricalMapping] = {}
    for m in mappings:
        for col in m.encoded_columns:
            col_to_mapping[col] = m

    decoded: list[Condition] = []
    # Group one-hot conditions by original feature
    onehot_groups: dict[str, list[Condition]] = {}

    for cond in conditions:
        mapping = col_to_mapping.get(cond.feature)
        if mapping is None:
            decoded.append(cond)
            continue

        if mapping.encoding == "onehot":
            key = mapping.original_name
            if key not in onehot_groups:
                onehot_groups[key] = []
            onehot_groups[key].append(cond)
        elif mapping.encoding == "ordinal":
            decoded.append(_decode_ordinal_condition(cond, mapping))
        else:
            decoded.append(cond)

    # Process one-hot groups
    for orig_name, group_conds in onehot_groups.items():
        mapping = next(m for m in mappings if m.original_name == orig_name)
        decoded_cond = _decode_onehot_group(group_conds, mapping)
        decoded.append(decoded_cond)

    return tuple(decoded)


def _decode_onehot_group(
    conditions: list[Condition],
    mapping: CategoricalMapping,
) -> Condition:
    """Decode a group of one-hot conditions into a single categorical condition.

    Logic:
    - ``col_X <= 0.5`` means category X is NOT active
    - ``col_X > 0.5`` means category X IS active
    """
    active_categories: list[str] = []
    inactive_categories: list[str] = []

    for cond in conditions:
        cat_name = _extract_category_from_column(cond.feature, mapping)
        if cat_name is None:
            continue

        if cond.operator == ">" and cond.threshold <= 0.5:
            active_categories.append(cat_name)
        elif cond.operator == "<=" and cond.threshold >= 0.5:
            inactive_categories.append(cat_name)

    if len(active_categories) == 1:
        return _make_categorical(mapping.original_name, "=", 0.0, active_categories[0])

    if inactive_categories:
        remaining = [c for c in mapping.categories if c not in inactive_categories]
        if len(remaining) == 1:
            return _make_categorical(mapping.original_name, "=", 0.0, remaining[0])
        elif remaining:
            return _make_categorical(
                mapping.original_name, "in", 0.0,
                "{" + ", ".join(remaining) + "}",
            )

    # Fallback: return the first condition unchanged
    if conditions:
        return conditions[0]
    return Condition(mapping.original_name, "=", 0.0)


def _decode_ordinal_condition(
    condition: Condition,
    mapping: CategoricalMapping,
) -> Condition:
    """Decode an ordinal-encoded condition.

    For ordinal encoding, threshold splits correspond to category boundaries.
    E.g., ``education <= 2.5`` with categories [HS, BS, MS, PhD]
    means ``education in {HS, BS, MS}``.
    """
    idx = int(round(condition.threshold))
    categories = mapping.categories

    if condition.operator == "<=":
        included = list(categories[: idx + 1])
    else:  # ">"
        included = list(categories[idx + 1:])

    if len(included) == 1:
        return _make_categorical(mapping.original_name, "=", condition.threshold, included[0])
    elif included:
        return _make_categorical(
            mapping.original_name, "in", condition.threshold,
            "{" + ", ".join(included) + "}",
        )
    return condition


def _extract_category_from_column(
    col_name: str,
    mapping: CategoricalMapping,
) -> Optional[str]:
    """Extract the category name from an encoded column name."""
    prefix = mapping.original_name + "_"
    if col_name.startswith(prefix):
        return col_name[len(prefix):]
    # Try matching against known encoded columns
    for i, enc_col in enumerate(mapping.encoded_columns):
        if enc_col == col_name and i < len(mapping.categories):
            return mapping.categories[i]
    return None


def decode_ruleset(
    ruleset: RuleSet,
    mappings: Sequence[CategoricalMapping],
) -> RuleSet:
    """Decode all rules in a RuleSet, replacing encoded conditions with readable ones.

    Parameters
    ----------
    ruleset : RuleSet
        The original ruleset with encoded feature conditions.
    mappings : sequence of CategoricalMapping
        One mapping per categorical feature.

    Returns
    -------
    RuleSet
        New ruleset with decoded conditions.
    """
    decoded_rules = []
    for rule in ruleset.rules:
        new_conditions = decode_conditions(rule.conditions, mappings)
        decoded_rules.append(Rule(
            conditions=new_conditions,
            prediction=rule.prediction,
            samples=rule.samples,
            confidence=rule.confidence,
            leaf_id=rule.leaf_id,
            prediction_value=rule.prediction_value,
        ))

    # Update feature names to use original names
    encoded_cols: set[str] = set()
    for m in mappings:
        encoded_cols.update(m.encoded_columns)

    new_feature_names: list[str] = []
    seen_originals: set[str] = set()
    for fn in ruleset.feature_names:
        if fn in encoded_cols:
            for m in mappings:
                if fn in m.encoded_columns and m.original_name not in seen_originals:
                    new_feature_names.append(m.original_name)
                    seen_originals.add(m.original_name)
                    break
        else:
            new_feature_names.append(fn)

    return RuleSet(
        rules=tuple(decoded_rules),
        feature_names=tuple(new_feature_names),
        class_names=ruleset.class_names,
    )
