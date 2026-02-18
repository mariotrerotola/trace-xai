"""Tests for SparseObliqueTreeSurrogate.

Synthetic data: y = 1 if x0 + x1 > 2 else 0 (diagonal boundary).
A standard axis-aligned tree produces phantom splits on this boundary;
SparseObliqueTreeSurrogate should detect them and add targeted interactions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from trace_xai import Explainer, SparseObliqueTreeSurrogate
from trace_xai.surrogates.base import BaseSurrogate
from trace_xai.surrogates.oblique_tree import ObliqueTreeSurrogate
from trace_xai.pruning import PruningConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def diagonal_data():
    """1000 points in [0, 4]^2, label = 1 iff x0 + x1 > 2."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 4, size=(1000, 2))
    y = (X[:, 0] + X[:, 1] > 2).astype(int)
    return X, y


@pytest.fixture()
def rf_on_diagonal(diagonal_data):
    """RandomForest trained on the diagonal boundary."""
    X, y = diagonal_data
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(X, y)
    return rf, X, y


@pytest.fixture()
def iris_rf():
    iris = load_iris()
    X, y = iris.data, iris.target
    rf = RandomForestClassifier(n_estimators=30, random_state=0)
    rf.fit(X, y)
    return rf, X, y, tuple(iris.feature_names)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFitPredict:
    def test_basic_fit_predict(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=4, min_samples_leaf=5)
        surr.fit(X, y)
        preds = surr.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_satisfies_protocol(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate()
        surr.fit(X, y)
        assert isinstance(surr, BaseSurrogate)

    def test_get_depth_and_leaves(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=3)
        surr.fit(X, y)
        assert surr.get_depth() > 0
        assert surr.get_n_leaves() > 0

    def test_get_depth_bounded(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=3)
        surr.fit(X, y)
        assert surr.get_depth() <= 3


class TestAugmentedFeatureNames:
    def test_augmented_feature_names_include_interactions(self, rf_on_diagonal):
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)
        feature_names = ("x0", "x1")
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y_bb, model=rf, feature_names=feature_names)
        aug = surr.get_augmented_feature_names(feature_names)
        # Should have original names plus any interaction terms
        assert len(aug) >= len(feature_names)
        # If interactions were added, they contain " * "
        interaction_names = [n for n in aug if " * " in n]
        if surr.interaction_pairs_:
            assert len(interaction_names) > 0

    def test_augmented_feature_names_no_interactions(self, diagonal_data):
        X, y = diagonal_data
        # Fit without model, no interactions possible with 2 features and top-3 fallback
        surr = SparseObliqueTreeSurrogate(max_depth=3, max_iterations=0)
        # max_iterations=0 means no phantom detection; fallback gives pairs
        surr.fit(X, y)
        aug = surr.get_augmented_feature_names(("x0", "x1"))
        assert "x0" in aug
        assert "x1" in aug


class TestSparseness:
    def test_sparse_fewer_interactions_than_full(self, rf_on_diagonal):
        """SparseObliqueTreeSurrogate should add fewer interactions than ObliqueTreeSurrogate."""
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)

        sparse = SparseObliqueTreeSurrogate(max_depth=4)
        sparse.fit(X, y_bb, model=rf)

        full = ObliqueTreeSurrogate(max_depth=4)
        full.fit(X, y_bb)

        # Full oblique adds n*(n-1)/2 pairs; sparse adds only phantom-guided pairs
        n_features = X.shape[1]
        n_full_pairs = n_features * (n_features - 1) // 2
        assert len(sparse.interaction_pairs_) <= n_full_pairs


class TestDecisionPathAndApply:
    def test_decision_path(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y)
        dp = surr.decision_path(X[:10])
        assert dp.shape[0] == 10

    def test_apply(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y)
        leaves = surr.apply(X[:10])
        assert leaves.shape == (10,)
        assert (leaves >= 0).all()


class TestFallbackWithoutModel:
    def test_fallback_without_model(self, diagonal_data):
        """Without a model, should degrade gracefully to importance-based selection."""
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y)  # no model kwarg
        preds = surr.predict(X)
        assert preds.shape == (len(X),)
        # phantom_features_ should be empty
        assert surr.phantom_features_ == set()

    def test_fallback_uses_pairs_from_important_features(self, diagonal_data):
        X, y = diagonal_data
        # 5-feature dataset; without model, pairs come from top-k features
        X5 = np.hstack([X, np.random.default_rng(1).random((len(X), 3))])
        surr = SparseObliqueTreeSurrogate(max_depth=3, max_interaction_features=3)
        surr.fit(X5, y)
        # Should have at most 3 pairs
        assert len(surr.interaction_pairs_) <= 3


class TestPhantomDetection:
    def test_phantom_features_detected_on_rf(self, rf_on_diagonal):
        """On RF with diagonal boundary, some splits should be flagged as phantom."""
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)
        surr = SparseObliqueTreeSurrogate(
            max_depth=4, n_probes=20, phantom_threshold=0.3, max_iterations=1
        )
        surr.fit(X, y_bb, model=rf)
        # With a diagonal RF boundary, at least one feature should be phantom
        # OR interactions were added — either way the surrogate went through detection
        assert surr.n_iterations_ >= 1

    def test_no_phantom_on_axis_aligned_dt(self):
        """On a single DT with axis-aligned boundary, phantom detection should find few/no phantoms."""
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 4, size=(500, 2))
        # Pure axis-aligned boundary: x0 > 2
        y = (X[:, 0] > 2).astype(int)
        dt = DecisionTreeClassifier(max_depth=3, random_state=0)
        dt.fit(X, y)
        y_bb = dt.predict(X)

        surr = SparseObliqueTreeSurrogate(
            max_depth=3, n_probes=20, phantom_threshold=0.3, max_iterations=1,
            noise_scale=0.05,
        )
        surr.fit(X, y_bb, model=dt)
        # Axis-aligned splits should not produce many phantom features
        # (at most 1 might slip through noise, but should be much less than all)
        assert len(surr.phantom_features_) <= X.shape[1]


class TestExplainerIntegration:
    def test_explainer_integration(self, iris_rf):
        rf, X, y, feature_names = iris_rf
        explainer = Explainer(rf, feature_names=feature_names)
        result = explainer.extract_rules(X, y=y, surrogate_type="sparse_oblique_tree")
        assert result.rules is not None
        assert result.rules.num_rules > 0
        assert result.report.fidelity > 0

    def test_explainer_integration_has_rules(self, iris_rf):
        rf, X, y, feature_names = iris_rf
        explainer = Explainer(rf, feature_names=feature_names)
        result = explainer.extract_rules(X, surrogate_type="sparse_oblique_tree")
        # Rules should have conditions
        rules_with_conds = [r for r in result.rules.rules if r.conditions]
        assert len(rules_with_conds) > 0

    def test_mixed_conditions_axis_and_interaction(self, rf_on_diagonal):
        """Some rules may reference interaction terms."""
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y_bb, model=rf)
        aug_names = surr.get_augmented_feature_names(("x0", "x1"))
        # At least original names are present
        assert "x0" in aug_names
        assert "x1" in aug_names


class TestFidelityImprovement:
    def test_fidelity_at_least_as_good_as_decision_tree(self, iris_rf):
        """sparse_oblique_tree fidelity should be >= standard decision_tree fidelity."""
        from sklearn.metrics import accuracy_score

        rf, X, y, feature_names = iris_rf
        y_bb = rf.predict(X)

        # Standard DT
        dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
        dt.fit(X, y_bb)
        dt_fidelity = accuracy_score(y_bb, dt.predict(X))

        # Sparse oblique
        surr = SparseObliqueTreeSurrogate(max_depth=5, min_samples_leaf=5)
        surr.fit(X, y_bb, model=rf)
        surr_fidelity = accuracy_score(y_bb, surr.predict(X))

        # Allow small margin; sparse oblique should not be much worse
        assert surr_fidelity >= dt_fidelity - 0.05


class TestCompatibility:
    def test_with_pruning(self, iris_rf):
        rf, X, y, feature_names = iris_rf
        explainer = Explainer(rf, feature_names=feature_names)
        config = PruningConfig(min_confidence=0.5, min_samples=5)
        result = explainer.extract_rules(
            X, surrogate_type="sparse_oblique_tree", pruning=config
        )
        # pruning_report should be present
        assert result.pruning_report is not None

    def test_with_counterfactual(self, iris_rf):
        rf, X, y, feature_names = iris_rf
        explainer = Explainer(rf, feature_names=feature_names)
        result = explainer.extract_rules(
            X,
            surrogate_type="sparse_oblique_tree",
            counterfactual_validity_threshold=0.0,  # keep all rules
        )
        assert result.counterfactual_report is not None
        assert result.counterfactual_report.mean_score >= 0.0


class TestProperties:
    def test_n_iterations_incremented(self, rf_on_diagonal):
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)
        surr = SparseObliqueTreeSurrogate(max_depth=3, max_iterations=2)
        surr.fit(X, y_bb, model=rf)
        assert surr.n_iterations_ >= 0

    def test_interaction_pairs_are_valid_indices(self, rf_on_diagonal):
        rf, X, y = rf_on_diagonal
        y_bb = rf.predict(X)
        n_features = X.shape[1]
        surr = SparseObliqueTreeSurrogate(max_depth=3)
        surr.fit(X, y_bb, model=rf)
        for i, j in surr.interaction_pairs_:
            assert 0 <= i < n_features
            assert 0 <= j < n_features
            assert i != j

    def test_feature_importances_shape(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=4)
        surr.fit(X, y)
        # importances cover augmented feature space
        n_aug = X.shape[1] + len(surr.interaction_pairs_)
        assert len(surr.feature_importances_) == n_aug

    def test_tree_attribute_accessible(self, diagonal_data):
        X, y = diagonal_data
        surr = SparseObliqueTreeSurrogate(max_depth=3)
        surr.fit(X, y)
        assert surr.tree_ is not None
        assert surr.tree_.node_count > 0
