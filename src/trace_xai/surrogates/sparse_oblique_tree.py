"""Sparse oblique decision tree surrogate with phantom-guided interaction selection.

Addresses the "phantom split" problem of axis-aligned surrogates: when a black-box
model (e.g. Random Forest) has diagonal decision boundaries, standard decision trees
produce splits where crossing the threshold does not actually change the black-box
prediction. The result is low counterfactual validity.

This module implements an iterative algorithm that:
1. Fits a standard axis-aligned tree (tree_0).
2. Probes each internal node counterfactually to detect phantom splits.
3. Adds interaction features (products) ONLY for features involved in phantom splits.
4. Re-fits on the augmented space.
5. Iterates up to max_iterations.

When no black-box model is provided (fallback mode), interaction pairs are chosen
by ranking feature importances from tree_0 and selecting the top-k pairs of the
most important features.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier


class SparseObliqueTreeSurrogate:
    """Surrogate tree with sparse, phantom-guided oblique splits.

    Only adds interaction features for features that are involved in
    phantom splits (splits not supported by the black-box model), rather
    than adding all pairwise interactions like ObliqueTreeSurrogate.

    Parameters
    ----------
    task : str
        ``"classification"`` (only classification is supported).
    max_depth : int or None
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    max_iterations : int
        Maximum number of phantom-detection / augmentation iterations.
    phantom_threshold : float
        Fraction of probes that must *not* change the black-box prediction
        for a node to be labelled a phantom split. A value of 0.3 means
        that if fewer than 30% of probes trigger a BB prediction change the
        split is considered phantom.
    noise_scale : float
        Perturbation magnitude expressed as a multiple of each feature's
        standard deviation when generating counterfactual probe pairs.
    n_probes : int
        Number of random probe pairs per internal tree node.
    max_interaction_features : int or None
        Cap on the total number of interaction (product) features. When set,
        mutual-information ranking is used to keep only the top-k pairs.
        When None, all pairs involving phantom features are kept.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        max_depth: Optional[int] = 5,
        min_samples_leaf: int = 5,
        max_iterations: int = 2,
        phantom_threshold: float = 0.3,
        noise_scale: float = 0.01,
        n_probes: int = 20,
        max_interaction_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._max_iterations = max_iterations
        self._phantom_threshold = phantom_threshold
        self._noise_scale = noise_scale
        self._n_probes = n_probes
        self._max_interaction_features = max_interaction_features
        self._random_state = random_state

        self._tree: DecisionTreeClassifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._interaction_pairs: list[tuple[int, int]] = []
        self._n_original_features: int = 0
        self._phantom_features: set[int] = set()
        self._n_iterations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        *,
        model=None,
        feature_names=None,
    ) -> "SparseObliqueTreeSurrogate":
        """Fit the sparse oblique surrogate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels (black-box predictions).
        model : object, optional
            Black-box model with a ``.predict()`` method. When provided,
            phantom splits are detected via counterfactual probing. When
            None, falls back to importance-based interaction selection.
        feature_names : sequence of str, optional
            Feature names (used only internally; does not affect fitting).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._n_original_features = X.shape[1]

        # Precompute feature statistics for probing
        feature_std = np.std(X, axis=0)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        feature_mins = X.min(axis=0)
        feature_maxs = X.max(axis=0)

        rng = np.random.RandomState(self._random_state)

        # --- Step 1: fit axis-aligned tree_0 ---
        tree_0 = DecisionTreeClassifier(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
        )
        tree_0.fit(X, y)

        if model is None:
            # Fallback: select top-k interactions by feature importance
            self._interaction_pairs = self._fallback_interactions(
                tree_0, X.shape[1]
            )
            self._phantom_features = set()
        else:
            # Iterative phantom detection
            phantom_features: set[int] = set()
            interaction_pairs: list[tuple[int, int]] = []
            current_X = X
            current_pairs: list[tuple[int, int]] = []
            tree_current = tree_0

            for iteration in range(self._max_iterations):
                new_phantoms = self._detect_phantom_features(
                    tree_current,
                    current_X,
                    model,
                    feature_std=feature_std,
                    feature_mins=feature_mins,
                    feature_maxs=feature_maxs,
                    rng=rng,
                )
                # Map phantom indices back to original feature space
                # (first n_original_features columns are always original)
                original_phantoms = {
                    idx for idx in new_phantoms if idx < self._n_original_features
                }
                added = original_phantoms - phantom_features
                phantom_features |= original_phantoms

                if not added and iteration > 0:
                    # No new phantom features — converged
                    break

                # Build interaction pairs: phantom x all others
                new_pairs = self._build_interaction_pairs(
                    phantom_features, self._n_original_features
                )

                # Rank by mutual information if capping
                if self._max_interaction_features is not None:
                    new_pairs = self._rank_by_mutual_info(
                        new_pairs, X, y, self._max_interaction_features
                    )

                interaction_pairs = new_pairs

                # Augment and re-fit
                current_X = self._apply_interactions(X, interaction_pairs)
                current_pairs = interaction_pairs
                tree_current = DecisionTreeClassifier(
                    max_depth=self._max_depth,
                    min_samples_leaf=self._min_samples_leaf,
                    random_state=self._random_state,
                )
                tree_current.fit(current_X, y)
                self._n_iterations = iteration + 1

            self._phantom_features = phantom_features
            self._interaction_pairs = current_pairs

        # Final fit with selected interaction pairs
        X_aug = self._apply_interactions(X, self._interaction_pairs)
        self._tree = DecisionTreeClassifier(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
        )
        self._tree.fit(X_aug, y)

        # Ensure n_iterations is at least 1 when model was provided
        if model is not None and self._n_iterations == 0:
            self._n_iterations = 1

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_aug = self._apply_interactions(X, self._interaction_pairs)
        return self._tree.predict(X_aug)

    def get_depth(self) -> int:
        return self._tree.get_depth()

    def get_n_leaves(self) -> int:
        return self._tree.get_n_leaves()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tree_(self):
        """Underlying sklearn tree structure (augmented feature space)."""
        return self._tree.tree_

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._tree.feature_importances_

    @property
    def phantom_features_(self) -> set:
        """Indices (in original feature space) of features in phantom splits."""
        return self._phantom_features

    @property
    def interaction_pairs_(self) -> list:
        """Selected (sparse) interaction pairs as list of (i, j) tuples."""
        return list(self._interaction_pairs)

    @property
    def n_iterations_(self) -> int:
        """Number of phantom-detection iterations performed."""
        return self._n_iterations

    # ------------------------------------------------------------------
    # Feature name utilities
    # ------------------------------------------------------------------

    def get_augmented_feature_names(
        self, feature_names: tuple
    ) -> tuple:
        """Return feature names including interaction terms.

        Parameters
        ----------
        feature_names : tuple of str
            Original feature names.

        Returns
        -------
        tuple of str
            Original names + interaction names (e.g. ``"feat_A * feat_B"``).
        """
        names = list(feature_names)
        for i, j in self._interaction_pairs:
            name_i = feature_names[i] if i < len(feature_names) else f"f{i}"
            name_j = feature_names[j] if j < len(feature_names) else f"f{j}"
            names.append(f"{name_i} * {name_j}")
        return tuple(names)

    # ------------------------------------------------------------------
    # sklearn-compatible utilities
    # ------------------------------------------------------------------

    def decision_path(self, X: ArrayLike):
        X = np.asarray(X, dtype=float)
        X_aug = self._apply_interactions(X, self._interaction_pairs)
        return self._tree.decision_path(X_aug)

    def apply(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_aug = self._apply_interactions(X, self._interaction_pairs)
        return self._tree.apply(X_aug)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_interactions(
        self, X: np.ndarray, pairs: list
    ) -> np.ndarray:
        """Augment X with product interaction columns for the given pairs."""
        if not pairs:
            return X
        interactions = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
        return np.hstack([X, interactions])

    def _detect_phantom_features(
        self,
        tree: DecisionTreeClassifier,
        X: np.ndarray,
        model,
        *,
        feature_std: np.ndarray,
        feature_mins: np.ndarray,
        feature_maxs: np.ndarray,
        rng: np.random.RandomState,
    ) -> set:
        """Return feature indices used in phantom splits.

        A split at node n using feature f at threshold t is phantom when
        straddling probes across t do not cause the black-box to change
        its prediction — meaning the surrogate placed a split boundary
        where the black-box has none.

        Only original features (index < n_original_features) can be phantom;
        interaction columns are derived and do not map to single-feature probing.
        """
        tree_ = tree.tree_
        phantom = set()

        for node_id in range(tree_.node_count):
            left = tree_.children_left[node_id]
            right = tree_.children_right[node_id]
            # Skip leaves
            if left == right:
                continue

            feat_idx = int(tree_.feature[node_id])
            # Only probe original features — interaction columns cannot be
            # probed in the original feature space directly
            if feat_idx >= self._n_original_features:
                continue

            threshold = float(tree_.threshold[node_id])

            # Probe counterfactually
            std = float(feature_std[feat_idx])
            delta = self._noise_scale * std
            delta_below = float(
                np.clip(threshold - delta, feature_mins[feat_idx], feature_maxs[feat_idx])
            )
            delta_above = float(
                np.clip(threshold + delta, feature_mins[feat_idx], feature_maxs[feat_idx])
            )

            if delta_below == delta_above:
                # Cannot probe both sides — skip
                continue

            bb_changed_any = False
            for _ in range(self._n_probes):
                base = rng.uniform(feature_mins[:self._n_original_features],
                                   feature_maxs[:self._n_original_features])
                below = base.copy()
                below[feat_idx] = delta_below
                above = base.copy()
                above[feat_idx] = delta_above

                # model.predict expects original feature space
                pair = np.vstack([below.reshape(1, -1), above.reshape(1, -1)])
                preds = np.asarray(model.predict(pair))
                if preds[0] != preds[1]:
                    bb_changed_any = True
                    break

            if not bb_changed_any:
                phantom.add(feat_idx)

        return phantom

    def _build_interaction_pairs(
        self, phantom_features: set, n_features: int
    ) -> list:
        """Build (phantom_feat, other_feat) pairs for all phantom features."""
        pairs = []
        for p in sorted(phantom_features):
            for f in range(n_features):
                if f != p:
                    pair = (min(p, f), max(p, f))
                    if pair not in pairs:
                        pairs.append(pair)
        return pairs

    def _rank_by_mutual_info(
        self,
        pairs: list,
        X: np.ndarray,
        y: np.ndarray,
        top_k: int,
    ) -> list:
        """Keep top_k pairs ranked by mutual information of their product with y."""
        if len(pairs) <= top_k:
            return pairs

        mi_scores = []
        for i, j in pairs:
            product = (X[:, i] * X[:, j]).reshape(-1, 1)
            mi = float(mutual_info_classif(product, y, random_state=self._random_state)[0])
            mi_scores.append(mi)

        order = np.argsort(mi_scores)[::-1]
        return [pairs[k] for k in order[:top_k]]

    def _fallback_interactions(
        self, tree: DecisionTreeClassifier, n_features: int
    ) -> list:
        """Fallback: select interaction pairs from top-k important features."""
        importances = tree.feature_importances_
        if importances is None or len(importances) == 0:
            return []

        # Sort features by importance descending
        ranked = np.argsort(importances)[::-1]

        # Determine how many features to pair
        if self._max_interaction_features is not None:
            # We want roughly max_interaction_features pairs.
            # k*(k-1)/2 pairs from top-k features: solve for k
            k = 2
            while k * (k - 1) // 2 < self._max_interaction_features and k < n_features:
                k += 1
        else:
            # Default: top-3 features (3 pairs)
            k = min(3, n_features)

        top_k = ranked[:k].tolist()
        pairs = []
        for idx_i, i in enumerate(top_k):
            for j in top_k[idx_i + 1:]:
                pairs.append((min(i, j), max(i, j)))

        if self._max_interaction_features is not None:
            pairs = pairs[: self._max_interaction_features]

        return pairs
