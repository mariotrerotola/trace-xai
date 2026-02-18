"""Tests for augmentation.py - Data augmentation and query synthesis."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from trace_xai.augmentation import (
    augment_data,
    boundary_augmentation,
    perturbation_augmentation,
    sparse_region_augmentation,
)


@pytest.fixture()
def iris_model():
    iris = load_iris()
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        iris.data, iris.target
    )
    return rf, iris.data, iris.target


class TestPerturbationAugmentation:
    def test_output_shape(self, iris_model):
        model, X, _ = iris_model
        X_aug, y_aug = perturbation_augmentation(X, model, n_neighbors=3)
        assert X_aug.shape[0] == X.shape[0] * 4  # original + 3x neighbors
        assert X_aug.shape[1] == X.shape[1]
        assert len(y_aug) == X_aug.shape[0]

    def test_original_data_preserved(self, iris_model):
        model, X, _ = iris_model
        X_aug, _ = perturbation_augmentation(X, model, n_neighbors=2)
        np.testing.assert_array_equal(X_aug[:len(X)], X)

    def test_reproducible(self, iris_model):
        model, X, _ = iris_model
        X1, y1 = perturbation_augmentation(X, model, random_state=42)
        X2, y2 = perturbation_augmentation(X, model, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestBoundaryAugmentation:
    def test_output_shape(self, iris_model):
        model, X, _ = iris_model
        surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
        surrogate.fit(X, model.predict(X))
        X_b, y_b = boundary_augmentation(X, model, surrogate, n_samples=50)
        assert X_b.shape[0] <= 50
        assert X_b.shape[1] == X.shape[1]
        assert len(y_b) == X_b.shape[0]


class TestSparseRegionAugmentation:
    def test_output_shape(self, iris_model):
        model, X, _ = iris_model
        surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
        surrogate.fit(X, model.predict(X))
        X_s, y_s = sparse_region_augmentation(X, model, surrogate, n_samples=50)
        assert X_s.shape[0] <= 50
        assert X_s.shape[1] == X.shape[1]
        assert len(y_s) == X_s.shape[0]


class TestAugmentData:
    def test_perturbation_strategy(self, iris_model):
        model, X, _ = iris_model
        X_aug, y_aug = augment_data(X, model, strategy="perturbation")
        assert X_aug.shape[0] > X.shape[0]

    def test_boundary_requires_surrogate(self, iris_model):
        model, X, _ = iris_model
        with pytest.raises(ValueError, match="requires a fitted surrogate"):
            augment_data(X, model, strategy="boundary")

    def test_combined_strategy(self, iris_model):
        model, X, _ = iris_model
        surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
        surrogate.fit(X, model.predict(X))
        X_aug, y_aug = augment_data(
            X, model, surrogate, strategy="combined",
            n_boundary_samples=20, n_sparse_samples=20,
        )
        assert X_aug.shape[0] > X.shape[0]

    def test_unknown_strategy(self, iris_model):
        model, X, _ = iris_model
        with pytest.raises(ValueError, match="Unknown augmentation"):
            augment_data(X, model, strategy="unknown")
