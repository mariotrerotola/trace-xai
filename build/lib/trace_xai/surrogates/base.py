"""Protocol / base class for surrogate models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class BaseSurrogate(Protocol):
    """Minimal interface that a surrogate model must satisfy."""

    def fit(self, X: ArrayLike, y: np.ndarray) -> BaseSurrogate:
        """Fit the surrogate on training data."""
        ...

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict target values."""
        ...

    def get_depth(self) -> int:
        """Return the depth of the fitted model (tree depth or equivalent)."""
        ...

    def get_n_leaves(self) -> int:
        """Return the number of leaves / terminal nodes."""
        ...
