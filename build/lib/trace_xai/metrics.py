"""Complementary evaluation metrics beyond fidelity.

Addresses the circularity criticism: the surrogate is trained to minimize
error on pseudo-labels and then evaluated on the same criterion.
These metrics provide independent evaluation perspectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from .ruleset import RuleSet


@dataclass(frozen=True)
class ComplementaryMetrics:
    """Metrics that go beyond simple fidelity measurement.

    Attributes
    ----------
    rule_coverage : float
        Fraction of samples covered by at least one rule (should be 1.0
        for complete decision trees, but may be < 1 for pruned rulesets).
    boundary_agreement : float
        Agreement between surrogate and black-box on samples near
        decision boundaries (0-1).
    counterfactual_consistency : float
        Fraction of rules where flipping a boundary condition changes
        the prediction as expected (0-1).
    class_balance_fidelity : dict
        Per-class fidelity weighted by class prevalence.
    effective_complexity : float
        Ratio of rules actually used (covering > 0 test samples) to total rules.
    """

    rule_coverage: float
    boundary_agreement: float
    counterfactual_consistency: float
    class_balance_fidelity: Dict[str, float]
    effective_complexity: float

    def __str__(self) -> str:
        lines = [
            "=== Complementary Metrics ===",
            f"  Rule coverage: {self.rule_coverage:.4f}",
            f"  Boundary agreement: {self.boundary_agreement:.4f}",
            f"  Counterfactual consistency: {self.counterfactual_consistency:.4f}",
            f"  Effective complexity: {self.effective_complexity:.4f}",
        ]
        if self.class_balance_fidelity:
            lines.append("  Class-balanced fidelity:")
            for cls, fid in self.class_balance_fidelity.items():
                lines.append(f"    {cls}: {fid:.4f}")
        return "\n".join(lines)


def compute_complementary_metrics(
    surrogate,
    model,
    X: ArrayLike,
    ruleset: RuleSet,
    *,
    class_names: tuple[str, ...] = (),
    n_boundary_samples: int = 500,
    boundary_noise: float = 0.01,
    random_state: int = 42,
) -> ComplementaryMetrics:
    """Compute metrics complementary to standard fidelity.

    Parameters
    ----------
    surrogate : fitted tree
        The fitted surrogate model.
    model : object
        The black-box model.
    X : array-like
        Feature matrix.
    ruleset : RuleSet
        Extracted rules.
    class_names : tuple of str
        Class names for per-class metrics.
    n_boundary_samples : int, default 500
        Number of boundary perturbation samples.
    boundary_noise : float, default 0.01
        Noise scale for boundary perturbation (relative to feature std).
    random_state : int, default 42
        Random seed.
    """
    X = np.asarray(X)
    rng = np.random.RandomState(random_state)

    y_bb = np.asarray(model.predict(X))
    y_surr = surrogate.predict(X)

    # 1. Rule coverage
    coverage = _compute_rule_coverage(X, ruleset)

    # 2. Boundary agreement
    boundary_agr = _compute_boundary_agreement(
        X, model, surrogate, n_samples=n_boundary_samples,
        noise_scale=boundary_noise, rng=rng,
    )

    # 3. Counterfactual consistency
    cf_consistency = _compute_counterfactual_consistency(
        X, model, surrogate, rng=rng,
    )

    # 4. Class-balanced fidelity
    class_fid = _compute_class_balanced_fidelity(y_bb, y_surr, class_names)

    # 5. Effective complexity
    eff_complexity = _compute_effective_complexity(X, surrogate, ruleset)

    return ComplementaryMetrics(
        rule_coverage=coverage,
        boundary_agreement=boundary_agr,
        counterfactual_consistency=cf_consistency,
        class_balance_fidelity=class_fid,
        effective_complexity=eff_complexity,
    )


def _compute_rule_coverage(X: np.ndarray, ruleset: RuleSet) -> float:
    """Fraction of samples matched by at least one rule."""
    n = len(X)
    if n == 0 or not ruleset.rules:
        return 0.0

    covered = np.zeros(n, dtype=bool)
    for rule in ruleset.rules:
        mask = np.ones(n, dtype=bool)
        for cond in rule.conditions:
            feat_idx = ruleset.feature_names.index(cond.feature) if cond.feature in ruleset.feature_names else -1
            if feat_idx < 0:
                continue
            if cond.operator == "<=":
                mask &= X[:, feat_idx] <= cond.threshold
            else:
                mask &= X[:, feat_idx] > cond.threshold
        covered |= mask

    return float(covered.mean())


def _compute_boundary_agreement(
    X: np.ndarray,
    model,
    surrogate,
    *,
    n_samples: int = 500,
    noise_scale: float = 0.01,
    rng: np.random.RandomState,
) -> float:
    """Agreement between model and surrogate near decision boundaries.

    Identifies points where the surrogate's prediction changes with
    small perturbation, then checks if the black-box also changes.
    """
    tree_ = surrogate.tree_
    feature_ids = tree_.feature
    thresholds = tree_.threshold
    children_left = tree_.children_left

    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)

    # Collect split boundaries
    splits = []
    for node_id in range(tree_.node_count):
        if children_left[node_id] != -1:
            splits.append((feature_ids[node_id], thresholds[node_id]))

    if not splits:
        return 1.0  # Single-leaf tree: trivially agrees

    # Generate pairs near boundaries
    samples_per_split = max(1, n_samples // len(splits))
    agreements = []

    for feat_idx, thresh in splits:
        for _ in range(samples_per_split):
            base = rng.uniform(feature_mins, feature_maxs)

            # Point just below threshold
            below = base.copy()
            below[feat_idx] = thresh - noise_scale * feature_std[feat_idx]
            below = np.clip(below, feature_mins, feature_maxs)

            # Point just above threshold
            above = base.copy()
            above[feat_idx] = thresh + noise_scale * feature_std[feat_idx]
            above = np.clip(above, feature_mins, feature_maxs)

            pair = np.vstack([below, above])
            surr_preds = surrogate.predict(pair)
            bb_preds = np.asarray(model.predict(pair))

            # Check if boundary behavior matches
            surr_changes = surr_preds[0] != surr_preds[1]
            bb_changes = bb_preds[0] != bb_preds[1]

            if surr_changes:
                # When surrogate says there's a boundary, does BB agree?
                agreements.append(1.0 if bb_changes else 0.0)

    if not agreements:
        return 1.0

    return float(np.mean(agreements))


def _compute_counterfactual_consistency(
    X: np.ndarray,
    model,
    surrogate,
    *,
    rng: np.random.RandomState,
    n_test: int = 200,
) -> float:
    """Check counterfactual consistency of the surrogate.

    For a sample of points, find the nearest decision boundary in the
    surrogate tree and verify that crossing it changes the black-box
    prediction consistently with the surrogate.
    """
    tree_ = surrogate.tree_
    feature_ids = tree_.feature
    thresholds = tree_.threshold
    children_left = tree_.children_left

    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)

    # Get decision path for each point
    n_test = min(n_test, len(X))
    test_idx = rng.choice(len(X), size=n_test, replace=False)
    X_test = X[test_idx]

    consistent = 0
    tested = 0

    # For each point, find which splits it passes through
    node_indicator = surrogate.decision_path(X_test)

    for i in range(n_test):
        node_indices = node_indicator[i].indices
        # Find last internal node (the split closest to the leaf)
        for node_id in reversed(node_indices):
            if children_left[node_id] != -1:
                feat = feature_ids[node_id]
                thresh = thresholds[node_id]

                # Create counterfactual by flipping across this split
                x_orig = X_test[i].copy()
                x_cf = x_orig.copy()

                if x_orig[feat] <= thresh:
                    x_cf[feat] = thresh + 0.01 * feature_std[feat]
                else:
                    x_cf[feat] = thresh - 0.01 * feature_std[feat]

                x_cf = np.clip(x_cf, feature_mins, feature_maxs)

                pair = np.vstack([x_orig.reshape(1, -1), x_cf.reshape(1, -1)])
                surr_preds = surrogate.predict(pair)
                bb_preds = np.asarray(model.predict(pair))

                surr_changed = surr_preds[0] != surr_preds[1]
                bb_changed = bb_preds[0] != bb_preds[1]

                tested += 1
                if surr_changed == bb_changed:
                    consistent += 1
                break

    if tested == 0:
        return 1.0
    return consistent / tested


def _compute_class_balanced_fidelity(
    y_bb: np.ndarray,
    y_surr: np.ndarray,
    class_names: tuple[str, ...],
) -> Dict[str, float]:
    """Per-class fidelity weighted by inverse class frequency."""
    result: Dict[str, float] = {}
    unique_classes = np.unique(y_bb)

    for cls_idx in unique_classes:
        mask = y_bb == cls_idx
        if mask.sum() == 0:
            continue
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        fid = float(np.mean(y_bb[mask] == y_surr[mask]))
        result[name] = fid

    return result


def _compute_effective_complexity(
    X: np.ndarray,
    surrogate,
    ruleset: RuleSet,
) -> float:
    """Ratio of rules that actually cover test samples to total rules.

    A value much less than 1.0 means many rules are "dead" (never activated).
    """
    if not ruleset.rules:
        return 0.0

    # Use surrogate leaf assignment to count active leaves
    leaf_ids = surrogate.apply(X)
    active_leaves = set(leaf_ids)
    rule_leaf_ids = {rule.leaf_id for rule in ruleset.rules}
    active_rules = len(active_leaves & rule_leaf_ids)

    return active_rules / len(ruleset.rules)
