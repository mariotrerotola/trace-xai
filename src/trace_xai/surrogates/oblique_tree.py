"""Oblique decision tree surrogate using linear combinations of features.

Addresses the limited expressivity of axis-aligned splits by allowing
oblique (hyperplane) splits that can capture diagonal decision boundaries.

Uses sklearn's DecisionTree with a preprocessing step that creates
interaction features (pairwise products) to simulate oblique splits
within the standard axis-aligned tree framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier


class ObliqueTreeSurrogate:
    """Oblique decision tree surrogate via feature interaction augmentation.

    Approximates oblique splits by augmenting the feature space with
    pairwise interaction terms (products of feature pairs), then fitting
    a standard axis-aligned tree on the augmented space.

    This allows the tree to split on conditions like
    ``feature_A * feature_B <= threshold``, which corresponds to an
    oblique hyperplane in the original feature space.

    Parameters
    ----------
    task : str
        ``"classification"``.
    max_depth : int or None
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    max_interaction_features : int or None
        Maximum number of interaction features to add. If None, adds all
        pairwise interactions (n*(n-1)/2 features). Set to a lower value
        for high-dimensional data to control complexity.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        max_depth: Optional[int] = 5,
        min_samples_leaf: int = 5,
        max_interaction_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._max_interaction_features = max_interaction_features
        self._random_state = random_state

        self._tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._interaction_pairs: list[tuple[int, int]] = []
        self._n_original_features: int = 0

    def _create_interactions(self, X: np.ndarray) -> np.ndarray:
        """Augment X with pairwise interaction features."""
        n_features = X.shape[1]
        self._n_original_features = n_features

        # Generate all pairs
        all_pairs = [
            (i, j) for i in range(n_features) for j in range(i + 1, n_features)
        ]

        # Limit number of interaction features
        if self._max_interaction_features is not None:
            all_pairs = all_pairs[:self._max_interaction_features]

        self._interaction_pairs = all_pairs

        if not all_pairs:
            return X

        interactions = np.column_stack([
            X[:, i] * X[:, j] for i, j in all_pairs
        ])
        return np.hstack([X, interactions])

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Apply stored interaction pairs to new data."""
        if not self._interaction_pairs:
            return X
        interactions = np.column_stack([
            X[:, i] * X[:, j] for i, j in self._interaction_pairs
        ])
        return np.hstack([X, interactions])

    def fit(self, X: ArrayLike, y: np.ndarray) -> ObliqueTreeSurrogate:
        X = np.asarray(X)
        X_aug = self._create_interactions(X)
        self._tree.fit(X_aug, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        X_aug = self._augment(X)
        return self._tree.predict(X_aug)

    def get_depth(self) -> int:
        return self._tree.get_depth()

    def get_n_leaves(self) -> int:
        return self._tree.get_n_leaves()

    @property
    def tree_(self):
        """Access the underlying sklearn tree structure."""
        return self._tree.tree_

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._tree.feature_importances_

    def get_augmented_feature_names(
        self, feature_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return feature names including interaction terms.

        Parameters
        ----------
        feature_names : tuple of str
            Original feature names.

        Returns
        -------
        tuple of str
            Original names + interaction names (e.g. "feat_A * feat_B").
        """
        names = list(feature_names)
        for i, j in self._interaction_pairs:
            name_i = feature_names[i] if i < len(feature_names) else f"f{i}"
            name_j = feature_names[j] if j < len(feature_names) else f"f{j}"
            names.append(f"{name_i} * {name_j}")
        return tuple(names)

    def decision_path(self, X: ArrayLike):
        X = np.asarray(X)
        X_aug = self._augment(X)
        return self._tree.decision_path(X_aug)

    def apply(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        X_aug = self._augment(X)
        return self._tree.apply(X_aug)
