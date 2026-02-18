"""Decision-tree surrogate (wraps sklearn's implementation)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeSurrogate:
    """Thin wrapper around sklearn decision trees that satisfies :class:`BaseSurrogate`."""

    def __init__(
        self,
        *,
        task: str = "classification",
        max_depth: Optional[int] = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self._tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: ArrayLike, y: np.ndarray) -> DecisionTreeSurrogate:
        self._tree.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._tree.predict(X)

    def get_depth(self) -> int:
        return self._tree.get_depth()

    def get_n_leaves(self) -> int:
        return self._tree.get_n_leaves()

    @property
    def tree_(self):
        """Access the underlying sklearn tree structure."""
        return self._tree.tree_
